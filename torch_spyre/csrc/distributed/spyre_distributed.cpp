/*
 * Copyright 2025-2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>

#include <algorithm>
#include <exception>
#include <flex/flex.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <spyre_comms.hpp>
#include <spyre_comms_tensor.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../logging.h"
#include "../spyre_allocator.h"
#include "../spyre_composite_address.h"
#include "../spyre_gil.h"
#include "../spyre_stream.h"
#include "comm_stream.h"

namespace spyre {

enum class CollectiveKind { Broadcast, AllGather, AllReduce };

// Structure to hold pending async work
struct PendingWork {
  CollectiveKind kind;
  std::shared_ptr<spyre_comms::WorkSchedule> work;
  std::vector<at::Tensor> rank_outputs;
  int64_t chunk_size = 0;
  std::vector<at::Tensor> hold_tensors;
  bool work_waited = false;
  std::shared_ptr<flex::Event> producer_event;
  std::shared_ptr<flex::Event> consumer_event;
  // Immutable execution identity survives moves/copies into deferred records.
};

// Global map to track pending async operations.
// Key: SharedOwnerCtx* (stable per-allocation identity). PendingWork holds
// tensor references (hold_tensors) that prevent the key from being freed
// while communication is in flight.
static std::unordered_map<spyre::SharedOwnerCtx*, PendingWork>
    pending_work_map_;
static std::mutex work_map_mutex_;

namespace {
constexpr size_t kRetiredByteBudget = 8 * 1024 * 1024;

struct CommStreamState {
  // Admission is separate from lifecycle state: no state lock is held across
  // blocking initialization DMA or destruction of Python-linked Tensor owners.
  std::mutex initialization_mutex;
  std::mutex mutex;
  bool execution_started = false;
  std::vector<std::shared_ptr<const void>> initialization_owners;
  std::thread::id owner;
  c10::DeviceIndex device = -1;
  flex::RuntimeContext* runtime = nullptr;
  flex::RuntimeStream* dev = nullptr;
  flex::RuntimeStream* comm = nullptr;
  std::unordered_map<SharedOwnerCtx*, PendingWork> records;
  std::optional<PendingWork> deferred;
  std::optional<PendingWork> waiting;
  std::optional<PendingWork> launch_backup;
  std::vector<at::Tensor> retired_tensors;
  size_t retired_bytes = 0;
  size_t live_work = 0;
  std::exception_ptr failure;
};

CommStreamState& comm_state() {
  // Deliberately process-lived: failed streams cannot prove completion, so
  // their WorkSchedules, events and borrowed tensor allocations must stay
  // alive.
  static auto* state = new CommStreamState();
  return *state;
}

void rethrow_locked(const CommStreamState& state) {
  if (state.failure) std::rethrow_exception(state.failure);
}

void fail_locked(CommStreamState& state, std::exception_ptr error) {
  if (!state.failure) state.failure = error;
  auto poison = [&](PendingWork& work) {
    if (work.producer_event) work.producer_event->poison(state.failure);
    if (work.consumer_event) work.consumer_event->poison(state.failure);
  };
  for (auto& entry : state.records) poison(entry.second);
  if (state.deferred) poison(*state.deferred);
  if (state.waiting) poison(*state.waiting);
  if (state.launch_backup) poison(*state.launch_backup);
}

void check_owner_locked(const CommStreamState& state, c10::Device device) {
  TORCH_CHECK(
      state.owner == std::this_thread::get_id() &&
          state.device == device.index(),
      "TORCH_SPYRE_COMM_STREAM requires one fixed caller thread/device");
}

void bridge(const std::shared_ptr<flex::Event>& event,
            flex::RuntimeStream* producer, flex::RuntimeStream* consumer) {
  // Construct both sides before publishing either operation.
  std::unique_ptr<flex::EventSignalParams,
                  decltype(&flex::destroyEventSignalParams)>
      signal(flex::createEventSignalParams(event, true),
             &flex::destroyEventSignalParams);
  std::unique_ptr<flex::EventWaitParams,
                  decltype(&flex::destroyEventWaitParams)>
      wait(flex::createEventWaitParams(event, true),
           &flex::destroyEventWaitParams);
  producer->launchOperationEventSignal(signal.get());
  consumer->launchOperationEventWait(wait.get());
}

void wait_in_place(PendingWork& pending) {
  if (!pending.work_waited) {
    // Only this scope denotes a real WorkSchedule wait. Scope payloads are
    // copied integers; destruction of a pending record cannot invalidate them.
    SPYRE_RUNTIME_DEBUG() << "overlap actual collective wait";
    pending.work->wait();
    // A real wait must succeed before nested AllGather assembly copies may
    // sync.
    pending.work_waited = true;
  }
}

void raw_sync(flex::RuntimeStream* stream) {
  stream->synchronize();
  // Flex consumes deferred_error_ once and skips its drain after shutdown.
  TORCH_CHECK(!stream->needsShutdown(),
              "TORCH_SPYRE_COMM_STREAM cannot reuse a failed runtime stream");
}

void raw_drain_locked(CommStreamState& state) {
  raw_sync(state.comm);
  raw_sync(state.dev);
  synchronizePrepStreams(
      c10::Device(c10::DeviceType::PrivateUse1, state.device));
  reapOperationOwners();
  state.retired_tensors.clear();
  state.retired_bytes = 0;
}

size_t owner_bytes(const at::Tensor& tensor) {
  return get_composite_address(tensor)->total_size();
}

void retire_locked(CommStreamState& state, const PendingWork& pending) {
  TORCH_CHECK(pending.work_waited, "Cannot retire an incomplete WorkSchedule");
  TORCH_CHECK(state.retired_bytes <= kRetiredByteBudget,
              "Collective retention accounting exceeded its budget");
  size_t candidate_bytes = 0;
  auto fits = [&](const std::vector<at::Tensor>& tensors) {
    for (const auto& tensor : tensors) {
      const size_t bytes = owner_bytes(tensor);
      // Checked, conservative accounting of backing owners (including aliases).
      if (bytes > kRetiredByteBudget - candidate_bytes) return false;
      candidate_bytes += bytes;
    }
    return true;
  };
  if (!fits(pending.rank_outputs) || !fits(pending.hold_tensors) ||
      candidate_bytes > kRetiredByteBudget - state.retired_bytes) {
    // Keep the owning record intact until both streams have safely drained.
    // An oversized completed record is released directly, never retired.
    raw_drain_locked(state);
    return;
  }
  auto append = [&](const std::vector<at::Tensor>& tensors) {
    for (const auto& tensor : tensors) {
      const size_t bytes = owner_bytes(tensor);
      state.retired_tensors.push_back(tensor);
      // If an append throws, accounting still matches all successful appends;
      // the caller quarantines both the complete record and the earlier batch.
      state.retired_bytes += bytes;
    }
  };
  append(pending.rank_outputs);
  append(pending.hold_tensors);
}

void release_record_locked(CommStreamState& state) {
  TORCH_CHECK(state.live_work == 1, "Expected exactly one live WorkSchedule");
  state.live_work = 0;
}

void drain_deferred_locked(CommStreamState& state) {
  if (!state.deferred) return;
  wait_in_place(*state.deferred);
  retire_locked(state, *state.deferred);
  release_record_locked(state);
  state.deferred.reset();
}

void synchronize_locked(CommStreamState& state) {
  if (state.deferred) wait_in_place(*state.deferred);
  for (auto& entry : state.records) wait_in_place(entry.second);
  // An outer ordinary wait_work owns waiting during AllGather assembly. Its
  // copies may reenter here; never erase or release that record in this helper.
  if (state.waiting) {
    TORCH_CHECK(state.waiting->work_waited,
                "Cannot synchronize during an unfinished collective wait");
  }
  raw_drain_locked(state);
  if (state.deferred) {
    release_record_locked(state);
    state.deferred.reset();
  }
  // Map identity and launch ownership survive ordinary synchronization.
}

void begin_collective(c10::Device device) {
  ReleaseGilIfHeld release;
  validate_comm_stream_caller(device, 0, true);
  if (!comm_stream_enabled()) return;
  auto& state = comm_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  rethrow_locked(state);
  try {
    drain_deferred_locked(state);
  }
  catch (...) {
    fail_locked(state, std::current_exception());
    throw;
  }
  TORCH_CHECK(!state.live_work && !state.launch_backup && !state.waiting &&
                  state.records.empty(),
              "Previous WorkSchedule must be consumed by wait_work before "
              "creating another collective");
}

void start_collective(SharedOwnerCtx* ctx, PendingWork pending) {
  // Public WorkSchedule has no schedule/epoch accessor. Those payloads stay
  // zero here; nested Comms records supply its native schedule identity.
  ReleaseGilIfHeld release;
  if (!comm_stream_enabled()) {
    pending.work->start();
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "Collective allocation already has pending work");
    pending_work_map_.emplace(ctx, std::move(pending));
    return;
  }
  auto& state = comm_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  rethrow_locked(state);
  TORCH_CHECK(!state.live_work, "Only one live WorkSchedule is supported");
  state.launch_backup.emplace(std::move(pending));
  state.live_work = 1;
  try {
    TORCH_CHECK(state.comm && spyre_comms::get_runtime_stream() == state.comm,
                "Dedicated comms initialization changed");
    state.launch_backup->producer_event = flex::createEvent();
    bridge(state.launch_backup->producer_event, state.dev, state.comm);
    state.launch_backup->work->start();
    // Copy rather than move: failed map allocation must leave the complete
    // backup alive, including every tensor borrowed by start().
    const bool inserted =
        state.records.emplace(ctx, *state.launch_backup).second;
    TORCH_CHECK(inserted, "Collective allocation already has pending work");
    state.launch_backup.reset();
  }
  catch (...) {
    fail_locked(state, std::current_exception());
    throw;
  }
}
}  // namespace

void validate_comm_stream_caller(c10::Device device, c10::StreamId target,
                                 bool starts_execution) {
  if (!comm_stream_enabled()) return;
  auto& state = comm_state();
  ReleaseGilIfHeld release;
  std::unique_lock<std::mutex> admission(state.initialization_mutex,
                                        std::defer_lock);
  if (starts_execution) admission.lock();
  std::lock_guard<std::mutex> lock(state.mutex);
  rethrow_locked(state);
  TORCH_CHECK(target == 0 && getCurrentStream(device).id() == 0,
              "TORCH_SPYRE_COMM_STREAM requires the default Dev stream");
  TORCH_CHECK(state.device < 0 || state.device == device.index(),
              "TORCH_SPYRE_COMM_STREAM requires one fixed caller thread/device");
  if (state.owner == std::thread::id{}) {
    state.owner = std::this_thread::get_id();
    state.device = device.index();
  }
  check_owner_locked(state, device);
  // Rejected callers never seal initialization or poison the legitimate owner.
  if (starts_execution) state.execution_started = true;
}

bool run_initialization_copy(
    c10::Device device, c10::StreamId target,
    const std::function<void()>& operation,
    std::vector<std::shared_ptr<const void>> owners) {
  if (!comm_stream_enabled()) return false;
  auto& state = comm_state();
  ReleaseGilIfHeld release;
  std::lock_guard<std::mutex> admission(state.initialization_mutex);
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    rethrow_locked(state);
    if (state.execution_started) return false;
    TORCH_CHECK(target == 0 && getCurrentStream(device).id() == 0,
                "TORCH_SPYRE_COMM_STREAM requires the default Dev stream");
    TORCH_CHECK(state.device < 0 || state.device == device.index(),
                "TORCH_SPYRE_COMM_STREAM requires one fixed caller thread/device");
    state.device = device.index();
    TORCH_CHECK(state.initialization_owners.empty(),
                "Initialization DMA owners were not released");
    state.initialization_owners = std::move(owners);
  }
  try {
    operation();
  }
  catch (...) {
    std::lock_guard<std::mutex> lock(state.mutex);
    // Completion is unproven: keep the actual Tensor owners process-rooted.
    fail_locked(state, std::current_exception());
    throw;
  }
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    owners.swap(state.initialization_owners);
  }
  // Destruction happens outside the state lock, after checked DMA completion.
  return true;
}

void rethrow_comm_stream_failure() {
  if (!comm_stream_enabled()) return;
  auto& state = comm_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  rethrow_locked(state);
}

void record_comm_stream_failure(std::exception_ptr error) {
  if (!comm_stream_enabled()) return;
  auto& state = comm_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  fail_locked(state, error);
}

flex::RuntimeStream* get_comms_stream() {
  auto* dev = getDefaultStreamRuntimeHandle();
  validate_comm_stream_caller(getCurrentStream().device(), 0);
  if (!comm_stream_enabled()) return dev;
  auto& state = comm_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  rethrow_locked(state);
  auto* runtime = GlobalRuntime::get();
  if (!state.comm) {
    try {
      state.runtime = runtime;
      state.dev = dev;
      state.comm = runtime->createStream(
          flex::RuntimeStreamPriority::NORMAL,
          flex::RuntimeStreamMode::STRICT_ORDERING, false);
      TORCH_CHECK(state.comm != dev, "Dedicated comms stream aliases Dev");
      SPYRE_RUNTIME_DEBUG() << "overlap topology Dev=" << dev->streamId()
                            << " Comm=" << state.comm->streamId();
    }
    catch (...) {
      fail_locked(state, std::current_exception());
      throw;
    }
  }
  TORCH_CHECK(state.runtime == runtime && state.dev == dev,
              "Dedicated comms runtime identity changed");
  return state.comm;
}

void check_comms_stream() {
  if (!comm_stream_enabled()) return;
  TORCH_CHECK(spyre_comms::get_runtime_stream() == get_comms_stream(),
              "spyre-comms was initialized on a different stream");
}

void guard_legacy_collective() {
  TORCH_CHECK(!comm_stream_enabled(),
              "TORCH_SPYRE_COMM_STREAM supports functional collectives only");
}

bool synchronize_comm_and_dev(c10::Device device, flex::RuntimeStream* dev) {
  if (!comm_stream_enabled()) return false;
  auto& state = comm_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  rethrow_locked(state);
  if (!state.comm) return false;
  check_owner_locked(state, device);
  TORCH_CHECK(state.dev == dev, "Default Dev stream identity changed");
  try {
    synchronize_locked(state);
  }
  catch (...) {
    fail_locked(state, std::current_exception());
    throw;
  }
  return true;
}

bool drain_comms_for_finalize() noexcept {
  if (!comm_stream_enabled()) return true;
  try {
    auto& state = comm_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.failure || state.waiting || state.launch_backup) return false;
    if (!state.comm) return true;
    if (state.owner != std::this_thread::get_id()) return false;
    try {
      synchronize_locked(state);
      // Teardown is the final owner; no later functional wait is expected.
      state.records.clear();
      state.live_work = 0;
      return true;
    }
    catch (...) {
      fail_locked(state, std::current_exception());
      return false;
    }
  }
  catch (...) {
    return false;
  }
}

// Compile-time plan cache.
enum class PlanKind { Broadcast, AllReduce, AllGather };

struct CachedPlan {
  PlanKind kind;
  spyre_comms::TensorDataTypeEnum dtype;
  int64_t rank_param;  // src_rank (broadcast) or dst_rank (reduce); unused for
                       // allreduce
  spyre_comms::SpyreReductionOpType reduce_op;  // only for allreduce/reduce

  // tensor_info MUST outlive wsi — spyre_comms stores a non-owning reference
  // to it inside the WorkScheduleInfo's sentinel envelope.
  std::unique_ptr<spyre_comms::TensorInfo> tensor_info;
  std::unique_ptr<spyre_comms::WorkScheduleInfo> wsi;
  int64_t num_elems = 0;
  int64_t group_size = 0;  // allgather only
};
static std::vector<CachedPlan> wsi_cache_;
static std::mutex wsi_cache_mutex_;

// Helper to convert PyTorch ScalarType to spyre_comms TensorDataTypeEnum
spyre_comms::TensorDataTypeEnum torch_dtype_to_spyre_comms(
    c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return spyre_comms::TensorDataTypeEnum::float32;
    case c10::ScalarType::Double:
      return spyre_comms::TensorDataTypeEnum::float64;
    case c10::ScalarType::Half:
      return spyre_comms::TensorDataTypeEnum::float16;
    case c10::ScalarType::BFloat16:
      return spyre_comms::TensorDataTypeEnum::bfloat16;
    case c10::ScalarType::Int:
      return spyre_comms::TensorDataTypeEnum::int32;
    case c10::ScalarType::Long:
      return spyre_comms::TensorDataTypeEnum::int64;
    case c10::ScalarType::Short:
      return spyre_comms::TensorDataTypeEnum::int16;
    case c10::ScalarType::Char:
      return spyre_comms::TensorDataTypeEnum::int8;
    case c10::ScalarType::Byte:
      return spyre_comms::TensorDataTypeEnum::uint8;
    case c10::ScalarType::Bool:
      return spyre_comms::TensorDataTypeEnum::boolean;
    default:
      TORCH_CHECK(false, "Unsupported dtype for spyre_comms: ", dtype);
  }
}

// Ensure spyre_comms is initialized and return the world context.
std::shared_ptr<spyre_comms::Context> ensure_context() {
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    SPYRE_RUNTIME_DEBUG() << "Initializing spyre-comms library";
    spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                    spyre::get_comms_stream());
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  }
  check_comms_stream();
  return context;
}

// Helper to convert reduce_op string to SpyreReductionOpType
spyre_comms::SpyreReductionOpType parse_reduce_op(
    const std::string& reduce_op) {
  if (reduce_op == "sum") {
    return spyre_comms::SpyreReductionOpType::SUM;
  }
  TORCH_CHECK(false, "Unsupported reduce_op for spyre allreduce: ", reduce_op,
              ". Only 'sum' is currently supported.");
}

// ============================================================================
// Compile-time plan ops — store collective parameters at graph load + create
// WSI
// ============================================================================

// Search the cache for an existing plan that matches the given parameters.
// Must be called with wsi_cache_mutex_ held. Returns -1 if not found.
int64_t cache_lookup(PlanKind kind, spyre_comms::TensorDataTypeEnum dtype,
                     int64_t num_elems, int64_t rank_param,
                     spyre_comms::SpyreReductionOpType reduce_op,
                     int64_t group_size) {
  for (size_t i = 0; i < wsi_cache_.size(); i++) {
    auto& entry = wsi_cache_[i];
    if (entry.kind == kind && entry.dtype == dtype &&
        entry.num_elems == num_elems && entry.rank_param == rank_param &&
        entry.reduce_op == reduce_op && entry.group_size == group_size) {
      return static_cast<int64_t>(i);
    }
  }
  return -1;
}

// Ensure the WSI is created for a cached plan entry.
// Must be called with wsi_cache_mutex_ held.
void ensure_wsi(CachedPlan& plan, int64_t num_elems,
                std::shared_ptr<spyre_comms::Context>& context) {
  if (plan.wsi != nullptr) return;

  spyre_comms::TensorShape shape({num_elems});
  plan.tensor_info =
      std::make_unique<spyre_comms::TensorInfo>(plan.dtype, shape);
  plan.num_elems = num_elems;

  switch (plan.kind) {
    case PlanKind::Broadcast:
      plan.wsi = context->broadcast(
          *plan.tensor_info,
          static_cast<spyre_comms::process_id_t>(plan.rank_param));
      break;
    case PlanKind::AllReduce:
      plan.wsi = context->allreduce(*plan.tensor_info, plan.reduce_op);
      break;
    case PlanKind::AllGather: {
      std::vector<spyre_comms::TensorInfo> output_infos(
          static_cast<size_t>(plan.group_size), *plan.tensor_info);
      plan.wsi = context->allgather(output_infos, *plan.tensor_info);
      break;
    }
  }
  TORCH_CHECK(plan.wsi != nullptr, "Failed to create WSI");
}

int64_t spyre_broadcast_plan_impl(int64_t num_elems, int64_t dtype_code,
                                  int64_t src_rank,
                                  const std::string& group_name) {
  SPYRE_RUNTIME_DEBUG() << "called with num_elems=" << num_elems
                        << ", dtype=" << dtype_code
                        << ", src_rank=" << src_rank;

  auto context = ensure_context();

  TORCH_CHECK(
      src_rank >= 0 && src_rank < static_cast<int64_t>(context->getSize()),
      "src_rank out of range: ", src_rank, " (world size is ",
      context->getSize(), ")");

  auto dtype =
      torch_dtype_to_spyre_comms(static_cast<c10::ScalarType>(dtype_code));

  std::lock_guard<std::mutex> lock(wsi_cache_mutex_);
  int64_t handle = cache_lookup(PlanKind::Broadcast, dtype, num_elems, src_rank,
                                spyre_comms::SpyreReductionOpType::SUM, 0);
  if (handle >= 0) {
    SPYRE_RUNTIME_DEBUG() << "cache hit at handle=" << handle;
    return handle;
  }

  handle = static_cast<int64_t>(wsi_cache_.size());
  wsi_cache_.push_back(CachedPlan{PlanKind::Broadcast, dtype, src_rank,
                                  spyre_comms::SpyreReductionOpType::SUM,
                                  nullptr, nullptr, 0, 0});
  auto& plan = wsi_cache_.back();
  ensure_wsi(plan, num_elems, context);

  SPYRE_RUNTIME_DEBUG() << "created WSI at handle=" << handle;
  return handle;
}

int64_t spyre_allreduce_plan_impl(int64_t num_elems, int64_t dtype_code,
                                  const std::string& reduce_op,
                                  const std::string& group_name) {
  SPYRE_RUNTIME_DEBUG() << "called with num_elems=" << num_elems
                        << ", dtype=" << dtype_code
                        << ", reduce_op=" << reduce_op;

  auto context = ensure_context();
  auto op_type = parse_reduce_op(reduce_op);
  auto dtype =
      torch_dtype_to_spyre_comms(static_cast<c10::ScalarType>(dtype_code));

  std::lock_guard<std::mutex> lock(wsi_cache_mutex_);
  int64_t handle =
      cache_lookup(PlanKind::AllReduce, dtype, num_elems, 0, op_type, 0);
  if (handle >= 0) {
    SPYRE_RUNTIME_DEBUG() << "cache hit at handle=" << handle;
    return handle;
  }

  handle = static_cast<int64_t>(wsi_cache_.size());
  wsi_cache_.push_back(CachedPlan{PlanKind::AllReduce, dtype, 0, op_type,
                                  nullptr, nullptr, 0, 0});
  auto& plan = wsi_cache_.back();
  ensure_wsi(plan, num_elems, context);

  SPYRE_RUNTIME_DEBUG() << "created WSI at handle=" << handle;
  return handle;
}

int64_t spyre_allgather_plan_impl(int64_t num_elems, int64_t dtype_code,
                                  int64_t group_size,
                                  const std::string& group_name) {
  SPYRE_RUNTIME_DEBUG() << "called with num_elems=" << num_elems
                        << ", dtype=" << dtype_code
                        << ", group_size=" << group_size;

  auto context = ensure_context();

  TORCH_CHECK(
      group_size > 0 && group_size == static_cast<int64_t>(context->getSize()),
      "group_size must equal world size: got ", group_size, " (world size is ",
      context->getSize(), ")");

  auto dtype =
      torch_dtype_to_spyre_comms(static_cast<c10::ScalarType>(dtype_code));

  std::lock_guard<std::mutex> lock(wsi_cache_mutex_);
  int64_t handle =
      cache_lookup(PlanKind::AllGather, dtype, num_elems, 0,
                   spyre_comms::SpyreReductionOpType::SUM, group_size);
  if (handle >= 0) {
    SPYRE_RUNTIME_DEBUG() << "cache hit at handle=" << handle;
    return handle;
  }

  handle = static_cast<int64_t>(wsi_cache_.size());
  wsi_cache_.push_back(CachedPlan{PlanKind::AllGather, dtype, 0,
                                  spyre_comms::SpyreReductionOpType::SUM,
                                  nullptr, nullptr, 0, group_size});
  auto& plan = wsi_cache_.back();
  ensure_wsi(plan, num_elems, context);

  SPYRE_RUNTIME_DEBUG() << "created WSI at handle=" << handle;
  return handle;
}

// ============================================================================
// Runtime run ops
// ============================================================================

at::Tensor spyre_broadcast_run_impl(const at::Tensor& input,
                                    int64_t plan_handle, int64_t src_rank) {
  begin_collective(input.device());
  SPYRE_RUNTIME_DEBUG() << "called with plan_handle=" << plan_handle
                        << ", src_rank=" << src_rank;

  auto context = ensure_context();

  std::lock_guard<std::mutex> cache_lock(wsi_cache_mutex_);
  TORCH_CHECK(
      plan_handle >= 0 && plan_handle < static_cast<int64_t>(wsi_cache_.size()),
      "broadcast_run: invalid plan_handle=", plan_handle);
  auto& plan = wsi_cache_[static_cast<size_t>(plan_handle)];

  // Create output tensor
  at::Tensor output = at::empty_like(input);
  TORCH_CHECK(output.nbytes() > 0,
              "Tensor must have non-zero size for broadcast");

  // Copy input to output if we're the source rank
  int current_rank = context->getRank();
  if (current_rank == src_rank) {
    output.copy_(input);
  }

  // Get SharedOwnerCtx for map key
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      output.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for output tensor");

  // Build spyre_comms::Tensor using the plan's TensorInfo (must stay alive)
  spyre_comms::Tensor buffer_tensor(*plan.tensor_info);
  buffer_tensor.SetSpyreDeviceAddressBorrowed(get_composite_address(output));

  auto work_schedule = context->broadcast_applyTensor(*plan.wsi, buffer_tensor);
  TORCH_CHECK(work_schedule != nullptr,
              "broadcast_applyTensor operation failed to create WorkSchedule");

  start_collective(ctx, PendingWork{CollectiveKind::Broadcast,
                                    std::move(work_schedule),
                                    {output},
                                    0,
                                    {input}});

  return output;
}

at::Tensor spyre_allreduce_run_impl(const at::Tensor& input,
                                    int64_t plan_handle) {
  begin_collective(input.device());
  SPYRE_RUNTIME_DEBUG() << "called with plan_handle=" << plan_handle;

  auto context = ensure_context();

  std::lock_guard<std::mutex> cache_lock(wsi_cache_mutex_);
  TORCH_CHECK(
      plan_handle >= 0 && plan_handle < static_cast<int64_t>(wsi_cache_.size()),
      "allreduce_run: invalid plan_handle=", plan_handle);
  auto& plan = wsi_cache_[static_cast<size_t>(plan_handle)];

  TORCH_CHECK(input.is_privateuseone(),
              "Tensor must be on Spyre device for all_reduce");
  TORCH_CHECK(input.is_contiguous(),
              "Tensor must be contiguous for all_reduce");
  TORCH_CHECK(input.nbytes() > 0,
              "Tensor must have non-zero size for all_reduce");

  // Get SharedOwnerCtx
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      input.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for input tensor");

  // Build spyre_comms::Tensor using the plan's TensorInfo (must stay alive)
  spyre_comms::Tensor inout_tensor(*plan.tensor_info,
                                   input.storage().data_ptr().get());
  inout_tensor.SetSpyreDeviceAddressBorrowed(get_composite_address(input));

  auto work_schedule = context->allreduce_applyTensor(*plan.wsi, inout_tensor);
  TORCH_CHECK(work_schedule != nullptr,
              "allreduce_applyTensor operation failed to create WorkSchedule");

  start_collective(
      ctx, PendingWork{
               CollectiveKind::AllReduce, std::move(work_schedule), {input}});

  return input;
}

at::Tensor spyre_allgather_run_impl(const at::Tensor& input,
                                    int64_t plan_handle, int64_t group_size) {
  begin_collective(input.device());
  SPYRE_RUNTIME_DEBUG() << "called with plan_handle=" << plan_handle
                        << ", group_size=" << group_size;

  auto context = ensure_context();

  std::lock_guard<std::mutex> cache_lock(wsi_cache_mutex_);
  TORCH_CHECK(
      plan_handle >= 0 && plan_handle < static_cast<int64_t>(wsi_cache_.size()),
      "allgather_run: invalid plan_handle=", plan_handle);
  auto& plan = wsi_cache_[static_cast<size_t>(plan_handle)];

  TORCH_CHECK(input.is_privateuseone(),
              "Tensor must be on Spyre device for allgather");
  TORCH_CHECK(input.is_contiguous(), "Tensor must be contiguous for allgather");
  TORCH_CHECK(input.nbytes() > 0,
              "Tensor must have non-zero size for allgather");

  spyre_comms::Tensor input_tensor(*plan.tensor_info,
                                   input.storage().data_ptr().get());
  input_tensor.SetSpyreDeviceAddressBorrowed(get_composite_address(input));

  // Allocate per-rank output tensors (same shape/layout as input)
  std::vector<at::Tensor> rank_outputs;
  rank_outputs.reserve(group_size);
  for (int64_t i = 0; i < group_size; i++) {
    rank_outputs.push_back(at::empty_like(input));
  }

  // Build spyre_comms::Tensor vector for per-rank outputs
  std::vector<spyre_comms::Tensor> output_tensors;
  output_tensors.reserve(group_size);
  for (int64_t i = 0; i < group_size; i++) {
    spyre_comms::Tensor out_tensor(*plan.tensor_info,
                                   rank_outputs[i].storage().data_ptr().get());
    out_tensor.SetSpyreDeviceAddressBorrowed(
        get_composite_address(rank_outputs[i]));
    output_tensors.push_back(std::move(out_tensor));
  }

  auto work_schedule =
      context->allgather_applyTensors(*plan.wsi, output_tensors, input_tensor);
  TORCH_CHECK(work_schedule != nullptr,
              "allgather_applyTensors operation failed to create WorkSchedule");

  // Allocate final concatenated output
  auto output_sizes = input.sizes().vec();
  output_sizes[0] *= group_size;
  at::Tensor output = at::empty(output_sizes, input.options());

  auto* output_ctx = static_cast<spyre::SharedOwnerCtx*>(
      output.storage().data_ptr().get_context());
  TORCH_CHECK(output_ctx != nullptr,
              "SharedOwnerCtx is null for output tensor");

  start_collective(output_ctx, PendingWork{CollectiveKind::AllGather,
                                           std::move(work_schedule),
                                           std::move(rank_outputs),
                                           input.size(0),
                                           {output, input}});

  return output;
}

// Wait for async operation to complete.
at::Tensor spyre_wait_work_impl(const at::Tensor& tensor) {
  ReleaseGilIfHeld release;
  validate_comm_stream_caller(tensor.device(), 0);
  auto* ctx =
      static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for tensor");

  PendingWork local;
  PendingWork* active = &local;
  if (comm_stream_enabled()) {
    auto& state = comm_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    rethrow_locked(state);
    TORCH_CHECK(!state.deferred && !state.waiting,
                "Another functional collective wait is already active");
    auto it = state.records.find(ctx);
    TORCH_CHECK(it != state.records.end(),
                "No pending async work found for tensor; wait_work requires "
                "a tensor returned by a collective run operation");
    const bool defer = it->second.kind == CollectiveKind::AllReduce &&
                       tensor.nbytes() == 8192 &&
                       tensor.scalar_type() == c10::ScalarType::Half;
    auto& slot = defer ? state.deferred : state.waiting;
    slot.emplace(std::move(it->second));
    state.records.erase(it);
    active = &*slot;
    if (defer) {
      SPYRE_RUNTIME_DEBUG() << "overlap deferred 8192-byte FP16 AllReduce";
      try {
        active->consumer_event = flex::createEvent();
        bridge(active->consumer_event, state.comm, state.dev);
      }
      catch (...) {
        fail_locked(state, std::current_exception());
        throw;
      }
      return tensor;
    }
  } else {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    auto it = pending_work_map_.find(ctx);
    TORCH_CHECK(it != pending_work_map_.end(),
                "No pending async work found for tensor. "
                "wait_work must be called on a tensor returned from "
                "broadcast_run, allgather_run, or allreduce_run.");
    local = std::move(it->second);
    pending_work_map_.erase(it);
  }

  try {
    if (comm_stream_enabled()) {
      auto& state = comm_state();
      std::lock_guard<std::mutex> lock(state.mutex);
      wait_in_place(*active);
    } else {
      wait_in_place(*active);
    }
    // Assembly can synchronously copy through Dev. Keep the complete waiting
    // record alive and release the state mutex before entering Torch copies.
    if (active->kind == CollectiveKind::AllGather) {
      int64_t world = static_cast<int64_t>(active->rank_outputs.size());
      TORCH_CHECK(tensor.size(0) == world * active->chunk_size,
                  "wait_work: output dim 0 (", tensor.size(0),
                  ") != world_size * chunk_size (", world, " * ",
                  active->chunk_size,
                  "). all_gather_into_tensor must concatenate along dim 0.");
      for (size_t i = 0; i < active->rank_outputs.size(); i++) {
        tensor
            .narrow(0, static_cast<int64_t>(i) * active->chunk_size,
                    active->chunk_size)
            .copy_(active->rank_outputs[i]);
      }
    }
    if (comm_stream_enabled()) {
      auto& state = comm_state();
      std::lock_guard<std::mutex> lock(state.mutex);
      retire_locked(state, *active);
      release_record_locked(state);
      state.waiting.reset();
    }
  }
  catch (...) {
    record_comm_stream_failure(std::current_exception());
    throw;
  }
  return tensor;
}

}  // namespace spyre

// Define the spyre namespace and operations
TORCH_LIBRARY(spyre, m) {
  m.def(
      "broadcast_async(Tensor input, int src_rank, str group_name) -> Tensor");
  m.def(
      "all_gather_async(Tensor input, SymInt group_size=1, "
      "str group_name=\"default\") -> Tensor");
  m.def(
      "all_reduce_async(Tensor(a!) input, str reduce_op=\"sum\", "
      "str group_name=\"default\") -> Tensor(a)");
  m.def("wait_work(Tensor(a!) tensor) -> Tensor(a)");

  // Compile-time plan ops — scalar-only, registered with impl directly
  // so they dispatch via CompositeImplicitAutograd (no tensor to key off).
  m.def(
      "broadcast_plan(int num_elems, int dtype, int src_rank, "
      "str group_name) -> int",
      &spyre::spyre_broadcast_plan_impl);
  m.def(
      "allreduce_plan(int num_elems, int dtype, str reduce_op, "
      "str group_name) -> int",
      &spyre::spyre_allreduce_plan_impl);
  m.def(
      "allgather_plan(int num_elems, int dtype, int group_size, "
      "str group_name) -> int",
      &spyre::spyre_allgather_plan_impl);

  // Runtime run ops — bind cached WSI to a tensor and execute
  m.def(
      "broadcast_run(Tensor input, int plan_handle, int src_rank) "
      "-> Tensor");
  m.def("allreduce_run(Tensor(a!) input, int plan_handle) -> Tensor(a)");
  m.def(
      "allgather_run(Tensor input, int plan_handle, int group_size) "
      "-> Tensor");
}

// Register the implementations with PyTorch's dispatcher
TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
  m.impl("wait_work", &spyre::spyre_wait_work_impl);

  m.impl("broadcast_run", &spyre::spyre_broadcast_run_impl);
  m.impl("allreduce_run", &spyre::spyre_allreduce_run_impl);
  m.impl("allgather_run", &spyre::spyre_allgather_run_impl);
}
