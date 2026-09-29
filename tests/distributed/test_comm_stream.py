# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compiled 8-KiB AllReduce ordering and ordinary synchronization cleanup.

Run in a fresh four-rank process group with TORCH_SPYRE_COMM_STREAM=1,
SPYRE_HAZARD_TRACKER=1, and FORCE_SYNCHRONOUS_EXECUTION=NONE:

    torchrun --nproc-per-node 4 tests/distributed/test_comm_stream.py

Configuration tests use fresh child processes so runtime environment settings
cannot be inherited from an already initialized Spyre runtime.
"""

import gc
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import unittest

import pytest
import regex as re
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_utils import TestCase

import torch_spyre  # noqa: F401


if "RANK" not in os.environ or os.environ.get("WORLD_SIZE") != "4":
    pytest.skip(
        "Comm stream tests require torchrun with four ranks", allow_module_level=True
    )

if os.environ.get("TORCH_SPYRE_COMM_STREAM") != "1":
    pytest.skip("Comm stream tests require the opt-in feature", allow_module_level=True)

DEVICE = torch.device(f"spyre:{os.environ['RANK']}")
_GROUP_NAME = "comm_stream_test"
_NUMEL = 4096
_RETIRED_BYTE_BUDGET = 8 * 1024 * 1024


def _initialize_process_group(store_prefix=None):
    torch.spyre._impl._lazy_init()
    if not dist.is_initialized():
        if store_prefix is None:
            dist.init_process_group("cpu:gloo,spyre:spyreccl")
        else:
            store, rank, world_size = next(dist.rendezvous("env://"))
            dist.init_process_group(
                "cpu:gloo,spyre:spyreccl",
                store=dist.PrefixStore(store_prefix, store),
                rank=rank,
                world_size=world_size,
            )
    c10d._register_process_group(_GROUP_NAME, dist.group.WORLD)


def _allocator_usage():
    # Current live allocations, not reserved capacity or a reset peak counter.
    # This read does not synchronize or reap pending collective work.
    stats = torch.spyre._spyre_get_allocator_stats(0)
    return {
        "bytes": stats["allocated_bytes.all.current"],
        "allocations": stats["allocation.all.current"],
    }


def _synchronize(kind):
    if kind == "default":
        torch.spyre.default_stream(DEVICE).synchronize()
    else:
        torch.spyre.synchronize(DEVICE)


class AllReduceWithConsumer(torch.nn.Module):
    def forward(self, x):
        producer = x * 2.0
        reduced = torch.ops._c10d_functional.all_reduce(producer, "sum", _GROUP_NAME)
        ready = torch.ops._c10d_functional.wait_tensor(reduced)
        return ready * 0.5


class AllReduceBeforeWait(torch.nn.Module):
    def forward(self, x):
        # The async schema is a compile-time IR marker; its emitted runtime op is
        # allreduce_run. Return without wait_tensor so the caller can sync first.
        return torch.ops._c10d_functional.all_reduce(x * 2.0, "sum", _GROUP_NAME)


class SequentialAllReduceWithCompute(torch.nn.Module):
    def forward(self, x, bias):
        producer = x * 2.0
        first = torch.ops._c10d_functional.all_reduce(producer, "sum", _GROUP_NAME)
        first = torch.ops._c10d_functional.wait_tensor(first)
        second_input = first + bias
        second = torch.ops._c10d_functional.all_reduce(second_input, "sum", _GROUP_NAME)
        second = torch.ops._c10d_functional.wait_tensor(second)
        return second * 0.5 + bias


def _cpu_inputs(rank, iteration):
    # Small powers-of-two fractions keep the four-rank fp16 reductions exact.
    index = torch.arange(_NUMEL, dtype=torch.float32)
    x = index.remainder(16) / 8 + rank + 1 + iteration / 4
    bias = index.remainder(8) / 4 + rank / 8 + iteration / 8
    return x.to(torch.float16), bias.to(torch.float16)


def _cpu_expected(rank, iteration):
    inputs = [_cpu_inputs(r, iteration) for r in range(4)]
    first_sum = sum(x.float() * 2 for x, _ in inputs)
    second_sum = sum(first_sum + bias.float() for _, bias in inputs)
    return (second_sum * 0.5 + inputs[rank][1].float()).to(torch.float16)


def _guard_probe(kind):
    if kind == "hazard_retry":
        checks = unittest.TestCase()
        os.environ["SPYRE_HAZARD_TRACKER"] = "1"
        os.environ["FORCE_SYNCHRONOUS_EXECUTION"] = "H2D"
        with checks.assertRaisesRegex(RuntimeError, "(?i)synchronous"):
            torch.spyre._impl._lazy_init()
        del os.environ["SPYRE_HAZARD_TRACKER"]
        # Keeping H2D for the first retry diagnoses stale hazard state without
        # allowing a broken implementation to proceed to device initialization.
        for forced_sync in ("H2D", "NONE"):
            os.environ["FORCE_SYNCHRONOUS_EXECUTION"] = forced_sync
            with checks.assertRaisesRegex(RuntimeError, "(?i)hazard"):
                torch.spyre._impl._lazy_init()
        return
    pattern = "(?i)hazard" if kind == "hazard" else "(?i)synchronous|FORCE_SYNCHRONOUS"
    try:
        try:
            _initialize_process_group(f"comm_stream_guard_{kind}")
            compiled = torch.compile(AllReduceWithConsumer(), fullgraph=True)
            x = torch.ones(_NUMEL, dtype=torch.float16).to(DEVICE)
            compiled(x)
        except RuntimeError as error:
            if not re.search(pattern, str(error)):
                raise
            print(f"Rejected invalid {kind} setting: {error}", flush=True)
        else:
            raise AssertionError(
                f"TORCH_SPYRE_COMM_STREAM accepted invalid {kind} settings"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _initialization_load_probe():
    from torch_spyre.model_utils import (
        _dma_to_spyre_default,
        _dma_to_spyre_dim_order_swapped,
        _dma_to_spyre_indirect_access,
    )

    torch.spyre.set_device(DEVICE.index)
    _initialize_process_group()
    rank = dist.get_rank()
    sources = [
        torch.full((4096,), rank + 1, dtype=torch.float32),
        torch.arange(4096, dtype=torch.float32).remainder(8).reshape(64, 64),
        torch.arange(4096, dtype=torch.float32).remainder(31).reshape(64, 64) / 4
        + rank + 2,
        torch.arange(4096, dtype=torch.float32).remainder(29).reshape(64, 64) / 4
        + rank + 3,
    ]
    helpers = [
        _dma_to_spyre_default,
        _dma_to_spyre_default,
        _dma_to_spyre_dim_order_swapped,
        _dma_to_spyre_indirect_access,
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(helper, source, torch.float16, device=DEVICE)
            for helper, source in zip(helpers, sources)
        ]
        loaded = [future.result(timeout=60) for future in futures]
        for source, weight in zip(sources, loaded):
            torch.testing.assert_close(weight.cpu(), source.half(), rtol=0, atol=0)
        compiled = torch.compile(AllReduceWithConsumer(), fullgraph=True)
        expected = torch.full((4096,), 10, dtype=torch.float16)
        torch.testing.assert_close(compiled(loaded[0]).cpu(), expected, rtol=0, atol=0)
        torch.spyre.synchronize()
        future = pool.submit(
            _dma_to_spyre_default, sources[0], torch.float16, device=DEVICE
        )
        try:
            future.result(timeout=20)
        except RuntimeError as error:
            assert "fixed caller thread/device" in str(error), str(error)
        else:
            raise AssertionError("background copy accepted after active execution")
        torch.testing.assert_close(compiled(loaded[0]).cpu(), expected, rtol=0, atol=0)
    torch.spyre.synchronize()
    dist.destroy_process_group()
    print(f"INITIALIZATION_LOAD_PASS rank={rank} helpers=4 collective_calls=2", flush=True)


class TestACommStreamConfiguration(unittest.TestCase):
    # unittest sorts class names; run children before the parent's device opens.
    def _check_invalid_configuration(self, kind, overrides):
        env = os.environ.copy()
        env.update(
            TORCH_SPYRE_COMM_STREAM="1",
            SPYRE_HAZARD_TRACKER="1",
            FORCE_SYNCHRONOUS_EXECUTION="NONE",
        )
        env.update(overrides)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--guard-probe", kind],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)

    def test_requires_hazard_tracking(self):
        """Ignoring the opt-in flag must not silently bypass its safety guard."""
        self._check_invalid_configuration("hazard", {"SPYRE_HAZARD_TRACKER": "0"})

    def test_requires_asynchronous_runtime(self):
        self._check_invalid_configuration(
            "sync", {"FORCE_SYNCHRONOUS_EXECUTION": "H2D"}
        )

    def test_startup_retry_does_not_reuse_hazard_setting(self):
        self._check_invalid_configuration("hazard_retry", {})


class TestCommStream(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _initialize_process_group()
        cls.rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        torch.compiler.reset()
        gc.collect()
        gc.collect()

    def _warmed_workload(self, sync_kind):
        compiled = torch.compile(AllReduceWithConsumer(), fullgraph=True)
        x = _cpu_inputs(self.rank, 0)[0].to(DEVICE)
        self.assertEqual(x.nbytes, 8192)
        # Keep compiled code and input alive throughout allocator comparisons.
        for _ in range(3):
            result = compiled(x)
            del result
            gc.collect()
            _synchronize(sync_kind)
        gc.collect()
        return compiled, x, _allocator_usage()

    def _check_sync_releases_references(self, sync_kind):
        compiled, x, baseline = self._warmed_workload(sync_kind)
        for batch in range(3):
            for _ in range(8):
                result = compiled(x)
                del result
            gc.collect()
            # No tensor.cpu(), private drain, or model-forward hook precedes this.
            _synchronize(sync_kind)
            gc.collect()
            current = _allocator_usage()
            self.assertEqual(
                current,
                baseline,
                f"Rank {self.rank}: {sync_kind} sync retained collective references "
                f"after batch {batch}; baseline={baseline}, current={current}",
            )

    def test_default_stream_sync_releases_collective_references(self):
        """Missing the ordinary stream-sync drain leaks retired tensor storage."""
        self._check_sync_releases_references("default")

    def test_device_sync_releases_collective_references(self):
        """Device synchronization must also release completed collective owners."""
        self._check_sync_releases_references("device")

    def _check_sync_before_wait(self, sync_kind):
        compiled = torch.compile(AllReduceBeforeWait(), fullgraph=True)
        for iteration in range(3):
            x = _cpu_inputs(self.rank, iteration)[0].to(DEVICE)
            pending = compiled(x)
            _synchronize(sync_kind)
            # Synchronization may finish the WorkSchedule, but must leave the
            # allocation identity available for its subsequently requested wait.
            result = torch.ops.spyre.wait_work(pending)
            expected = sum(
                _cpu_inputs(r, iteration)[0].float() * 2 for r in range(4)
            ).to(torch.float16)
            self.assertEqual(result.cpu(), expected, rtol=0, atol=0)
            del result, pending, x
        gc.collect()
        _synchronize(sync_kind)

    def test_default_stream_sync_preserves_later_wait_work(self):
        """A sync drain must not consume the caller's pending-work identity."""
        self._check_sync_before_wait("default")

    def test_device_sync_preserves_later_wait_work(self):
        self._check_sync_before_wait("device")

    def _check_interleaved_correctness(self, sync_kind):
        compiled = torch.compile(SequentialAllReduceWithCompute(), fullgraph=True)
        inputs = [
            tuple(t.to(DEVICE) for t in _cpu_inputs(self.rank, i)) for i in range(8)
        ]
        for batch in range(3):
            # Enqueue different inputs before checking any result. The same compiled
            # correction programs and collective plans are reused across all calls.
            results = [compiled(*pair) for pair in inputs]
            _synchronize(sync_kind)
            for iteration, result in enumerate(results):
                self.assertEqual(
                    result.cpu(),
                    _cpu_expected(self.rank, iteration),
                    rtol=0,
                    atol=0,
                    msg=f"Rank {self.rank}, batch {batch}, iteration {iteration}",
                )
            del result, results
        gc.collect()
        _synchronize(sync_kind)

    def test_interleaved_collectives_default_stream_sync(self):
        """Missing Dev↔Comm ordering corrupts rank-varying reused-program results."""
        self._check_interleaved_correctness("default")

    def test_interleaved_collectives_device_sync(self):
        self._check_interleaved_correctness("device")

    def test_nondefault_caller_is_rejected_without_poisoning_next_collective(self):
        compiled, x, _ = self._warmed_workload("default")
        with torch.spyre.Stream(DEVICE):
            with self.assertRaisesRegex(
                RuntimeError, "(?i)default.*stream|stream.*default"
            ):
                compiled(x)
        # Rejection must occur before adding pending collective work.
        result = compiled(x)
        _synchronize("default")
        expected = sum(_cpu_inputs(r, 0)[0].float() for r in range(4)).to(torch.float16)
        self.assertEqual(result.cpu(), expected, rtol=0, atol=0)

    def test_nondefault_nonblocking_copy_is_rejected_without_poisoning_default_copy(
        self,
    ):
        compiled, x, _ = self._warmed_workload("default")
        result = compiled(x)
        _synchronize("default")
        # to(cpu, non_blocking=True) first requests a pinned allocator, which
        # Spyre does not provide. A preallocated destination reaches copyAsync.
        destination = torch.empty_like(result, device="cpu")
        with torch.spyre.Stream(DEVICE):
            with self.assertRaisesRegex(
                RuntimeError, "(?i)default.*stream|stream.*default"
            ):
                destination.copy_(result, non_blocking=True)
        expected = sum(_cpu_inputs(r, 0)[0].float() for r in range(4)).to(torch.float16)
        self.assertEqual(result.cpu(), expected, rtol=0, atol=0)

    def test_retained_storage_is_bounded_without_explicit_sync(self):
        """A missing budget drain grows retained storage beyond 8 MiB."""
        compiled, x, baseline = self._warmed_workload("device")
        result = compiled(x)
        one_call = _allocator_usage()
        del result
        gc.collect()
        _synchronize("device")
        gc.collect()
        self.assertEqual(_allocator_usage(), baseline)

        # Allow one live call's measured transient allocations above the retired
        # backing-storage budget. Each retained eligible tensor is at least 8 KiB.
        byte_limit = (
            baseline["bytes"]
            + _RETIRED_BYTE_BUDGET
            + max(0, one_call["bytes"] - baseline["bytes"])
        )
        count_limit = (
            baseline["allocations"]
            + _RETIRED_BYTE_BUDGET // x.nbytes
            + max(0, one_call["allocations"] - baseline["allocations"])
        )
        snapshots = []
        for _ in range(10):
            for _ in range(128):
                result = compiled(x)
                del result
            # No synchronization, result copies, or completion queries in this loop.
            gc.collect()
            snapshots.append(_allocator_usage())
        _synchronize("device")
        gc.collect()
        self.assertEqual(_allocator_usage(), baseline)
        self.assertLessEqual(
            max(s["bytes"] for s in snapshots), byte_limit, repr(snapshots)
        )
        self.assertLessEqual(
            max(s["allocations"] for s in snapshots), count_limit, repr(snapshots)
        )

    def test_z_process_group_teardown_releases_collective_references(self):
        """Process-group teardown must drain references without an explicit sync."""
        compiled, x, baseline = self._warmed_workload("device")
        for _ in range(8):
            result = compiled(x)
            del result
        gc.collect()
        dist.destroy_process_group()
        gc.collect()
        final = _allocator_usage()
        # Teardown may also release persistent process-group allocations.
        self.assertLessEqual(final["bytes"], baseline["bytes"], repr(final))
        self.assertLessEqual(final["allocations"], baseline["allocations"], repr(final))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--initialization-load-probe":
        _initialization_load_probe()
    elif len(sys.argv) == 3 and sys.argv[1] == "--guard-probe":
        _guard_probe(sys.argv[2])
    else:
        # common_utils.run_tests seeds every accelerator before running tests,
        # opening the parent's VF before configuration children can acquire it.
        unittest.main()
