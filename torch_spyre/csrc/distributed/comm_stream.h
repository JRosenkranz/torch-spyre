/*
 * Copyright 2026 The Torch-Spyre Authors.
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

#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <exception>
#include <functional>
#include <memory>
#include <vector>

namespace flex {
class RuntimeStream;
}

namespace spyre {

// Private opt-in collective lifecycle; never initializes comms during sync.
bool comm_stream_enabled();
#ifdef USE_SPYRE_CCL
void validate_comm_stream_caller(c10::Device device, c10::StreamId target,
                                 bool starts_execution = false);
bool run_initialization_copy(
    c10::Device device, c10::StreamId target,
    const std::function<void()>& operation,
    std::vector<std::shared_ptr<const void>> owners);
flex::RuntimeStream* get_comms_stream();
void check_comms_stream();
void guard_legacy_collective();
void rethrow_comm_stream_failure();
void record_comm_stream_failure(std::exception_ptr error);
bool synchronize_comm_and_dev(c10::Device device, flex::RuntimeStream* dev);
bool drain_comms_for_finalize() noexcept;

#else
inline void validate_comm_stream_caller(c10::Device, c10::StreamId,
                                         bool = false) {}
inline bool run_initialization_copy(
    c10::Device, c10::StreamId, const std::function<void()>&,
    std::vector<std::shared_ptr<const void>>) { return false; }
inline void rethrow_comm_stream_failure() {}
inline void record_comm_stream_failure(std::exception_ptr) {}
inline bool synchronize_comm_and_dev(c10::Device, flex::RuntimeStream*) {
  return false;
}
#endif

}  // namespace spyre
