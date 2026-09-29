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

"""Restored Prep/Comm caller, error and lifetime contracts."""

import gc
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import pytest


def _runtime_probe(kind):
    import regex
    import torch
    import torch_spyre
    from test_launch_jobplan import _build_d2h_jobplan
    from test_prepare_kernel import TestPrepareKernel

    cpu = torch.ones(256, dtype=torch.float16)
    device = torch.empty_like(cpu, device="spyre")
    if kind == "initialization_copy":
        # Process-group construction binds its control thread before HF starts
        # asynchronous loading. Synchronize models the same owner binding here.
        torch.spyre.synchronize()

        def rejected_copy(non_blocking=False, alternate_stream=False):
            stream = torch.Stream("spyre") if alternate_stream else None
            try:
                if stream is None:
                    torch_spyre._C.copy_tensor(cpu, device, non_blocking)
                else:
                    with stream:
                        torch_spyre._C.copy_tensor(cpu, device, non_blocking)
            except RuntimeError as error:
                assert regex.search("(?i)caller thread|default Dev", str(error))
            else:
                raise AssertionError("unsupported copy caller was accepted")

        with ThreadPoolExecutor(max_workers=4) as pool:
            # Rejection must neither close initialization nor poison its owner.
            pool.submit(rejected_copy, True).result(timeout=20)
            pool.submit(rejected_copy, False, True).result(timeout=20)

            def load_weight(value):
                source = torch.full((256,), value, dtype=torch.float16)
                target = torch.empty_like(source, device="spyre")
                torch_spyre._C.copy_tensor(source, target, False)
                return target

            loaded = list(pool.map(load_weight, range(1, 9)))
            # A nonblocking operation seals initialization, even after a drain.
            device.copy_(cpu, non_blocking=True)
            torch.spyre.synchronize()
            pool.submit(rejected_copy).result(timeout=20)
            for value, weight in enumerate(loaded, start=1):
                assert torch.equal(weight.cpu(), torch.full_like(cpu, value))
        device.copy_(cpu)
        torch.spyre.synchronize()
        return
    device.copy_(cpu)
    torch.spyre.synchronize()
    default = torch.spyre.default_stream()

    with tempfile.TemporaryDirectory() as directory:
        # Segment 7 is the real program allocation, valid with either symbolic
        # argument mode. This probe needs a harmless real launch, not I/O binding.
        plan = _build_d2h_jobplan(directory, 120259084288, 128)
        if kind.startswith("gil_"):
            for _ in range(16):
                torch_spyre._C.launch_jobplan(plan, [device])
                if kind == "gil_device":
                    torch.spyre.synchronize()
                elif kind == "gil_stream":
                    default.synchronize()
                elif kind == "gil_copy":
                    destination = torch.empty_like(cpu)
                    torch_spyre._C.copy_tensor(device, destination, False)
                elif kind == "gil_fill":
                    torch_spyre._C.fill_tensor(device, 0.0)
                elif kind == "gil_prepare":
                    with tempfile.TemporaryDirectory() as next_directory:
                        next_plan = _build_d2h_jobplan(
                            next_directory, 120259084288, 128
                        )
                        del next_plan
                else:
                    assert kind == "gil_launch"
            torch.spyre.synchronize()
            assert default.query()
            return
        torch_spyre._C.launch_jobplan(plan, [device])
        torch.spyre.synchronize()

        def actions(stream):
            return {
                "copy": lambda: device.copy_(cpu),
                "launch": lambda: torch_spyre._C.launch_jobplan(plan, [device]),
                "query": stream.query,
                "stream synchronize": stream.synchronize,
                "device synchronize": torch.spyre.synchronize,
            }

        if kind in ("caller", "stream"):
            failures = []
            expected = "(?i)caller thread" if kind == "caller" else "default Dev"

            def reject_actions(stream):
                for name, action in actions(stream).items():
                    try:
                        action()
                    except RuntimeError as error:
                        if not regex.search(expected, str(error)):
                            failures.append(f"{name}: wrong error: {error}")
                    else:
                        failures.append(f"{name}: unsupported caller accepted")

            if kind == "caller":
                thread = threading.Thread(target=reject_actions, args=(default,))
                thread.start()
                thread.join(timeout=20)
                assert not thread.is_alive(), "second caller did not terminate"
            else:
                stream = torch.Stream("spyre")
                with stream:
                    reject_actions(stream)
                    # A default target with a non-default current stream is also
                    # unsupported, so validating only target IDs is insufficient.
                    reject_actions(default)

            assert not failures, "\n".join(failures)
            # Rejected callers must not poison the legitimate caller's runtime.
            device.copy_(cpu)
            torch_spyre._C.launch_jobplan(plan, [device])
            default.synchronize()
            torch.spyre.synchronize()
            assert default.query()
            return

        assert kind == "failure"
        with tempfile.TemporaryDirectory() as invalid_directory:
            properties = {
                "ohandle": "output_buffer",
                "size": "1024",
                "ishape": ["0"],
                "ihandle": "",
                "hcm": {"vdci": {}, "senConstants": []},
            }
            code = TestPrepareKernel().create_mock_spyrecode(
                invalid_directory,
                exec_command="ComputeOnHost",
                exec_properties=properties,
            )
            invalid_plan = torch_spyre._C.prepare_kernel(code)
            try:
                torch_spyre._C.launch_jobplan(invalid_plan, [])
                torch.spyre.synchronize()
            except RuntimeError as error:
                first = str(error)
                assert "Expect one DCI" in first, first
            else:
                raise AssertionError("invalid zero-symbol correction did not execute")

            for _ in range(2):
                for name, action in actions(default).items():
                    try:
                        action()
                    except RuntimeError as error:
                        assert str(error) == first, f"{name}: first error was replaced"
                    else:
                        raise AssertionError(f"{name}: lost the first correction error")


def _run_runtime_probe(kind):
    env = os.environ.copy()
    env.update(
        FLEX_DEVICE="MOCK1p0",
        FLEX_COMPUTE="NULL",
        AIU_WORLD_SIZE="1",
        LOCAL_RANK="0",
        TORCH_SPYRE_COMM_STREAM="1",
        SPYRE_HAZARD_TRACKER="1",
        FORCE_SYNCHRONOUS_EXECUTION="NONE",
    )
    env.pop("SPYRE_DEVICES", None)
    env.pop("SPYRE_HOST_WORKER_THREADS", None)
    # Keep large sparse mock device files out of the source checkout. A caller
    # may provide an artifact directory when preserving diagnostic state.
    with tempfile.TemporaryDirectory(prefix="torch-spyre-mock-") as memory_dir:
        env.setdefault("FLEX_MOCK_DEVICE_MEMORY_PATH", memory_dir)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--runtime-probe", kind],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("kind", ["initialization_copy", "caller", "stream", "failure", "gil_device", "gil_stream", "gil_copy", "gil_fill", "gil_prepare", "gil_launch"])
def test_restored_runtime_contract(kind):
    _run_runtime_probe(kind)


def test_actual_correction_reuses_prepared_hcm_and_outlives_jobplan(monkeypatch):
    import torch
    import torch_spyre  # noqa: F401
    from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

    runners = set()
    real_run = SpyreSDSCKernelRunner.run

    def observe_real_run(runner, *args):
        runners.add(runner)
        return real_run(runner, *args)

    # Observe the real compiler-created runner; no execution method is mocked.
    monkeypatch.setattr(SpyreSDSCKernelRunner, "run", observe_real_run)
    torch.compiler.reset()
    compiled = torch.compile(torch.abs, fullgraph=True)
    warmup = compiled(torch.ones(256, dtype=torch.float16, device="spyre"))
    torch.spyre.synchronize()
    del warmup
    assert runners, "the compiled operation did not launch a real JobPlan"
    assert any(
        runner.jobplan.get_step_type(i) == "HostCompute"
        for runner in runners
        for i in range(runner.jobplan.num_steps())
    ), "the selected compiled operation does not exercise correction"

    for runner in runners:
        for i in range(runner.jobplan.num_steps()):
            assert runner.jobplan.get_step_pipeline_barrier(i)
            if runner.jobplan.get_step_type(i) == "HostCompute":
                assert runner.jobplan.get_step_stream_role(i) == "Prep"
    results = []
    for iteration in range(16):
        cpu = torch.arange(-128, 128, dtype=torch.float32).to(torch.float16)
        cpu += iteration / 4
        device = cpu.to("spyre", non_blocking=True)
        results.append((compiled(device), cpu.abs()))
        del device

    # The last launches can still be queued. Destroy the actual JobPlans,
    # including their HostCompute handles and owning program allocations.
    for runner in runners:
        runner._jobplan = None
    del compiled
    torch.compiler.reset()
    gc.collect()
    torch.spyre.synchronize()
    for actual, expected in results:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "--runtime-probe"
    _runtime_probe(sys.argv[2])
