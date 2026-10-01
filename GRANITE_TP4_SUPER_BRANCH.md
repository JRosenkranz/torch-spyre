# torch-spyre super-branch: best Granite TP4 decode (super/granite-tp4-best-decode)

Written 2026-10-01. This file is the only change on top of the tested commit, and it only adds docs.
Tags: **[measured]** = from a device run or a git check; **[code]** = read from the source on this
branch; **[inferred]** = reasoning that was not tested. Paths are relative to the repo root. Run
evidence is under `/tmp/devel/src/build/overlap-vs-hursey-ab-20260930/` (called `$EXP` below).

## 1. What this branch is

This is the torch-spyre side of the measured arm `hursey-tracker467-fix`, the fastest Granite 3.3 8B
TP4 decode run so far (decode forward 48.4-51.6 ms, rank-0 medians over 30 requests).

| Item | Value |
|---|---|
| Base | main `a99b5771` (2026-09-30, "feat(cache): record kernel names in cached kernel dirs (#4590)") |
| PR merged | #5040 "Relax wait_tensor overhead by removing the wait() and defer to the underlying steam", Joshua Hursey, tested head `68bbf076`. Two commits: `b6ba52b9` (Flavia Beo de Bayser, "distributed: skip wait_work host–device sync in single-stream mode", same subject as her closed #4846, so it seems to be carried over from there [inferred]) and `68bbf076` (Hursey, "Add TORCH_SPYRE_DIST_SKIP_WAIT, and retain the WorkSchedule objects"). 3 files, +47/-3 |
| Local commits | none apart from the merge `b238a6c2` (a local `--no-ff` merge, unsigned, fixed identity `overlap-vs-hursey-ab`) |
| Tested commit / tree | `b238a6c2` / tree `84be1d2e`. The tree is the same as the one built and measured [measured: tree hash and diff hash `9a8a0dec` match `$EXP/provenance/source-revisions.json`; the run's library receipts map the `_C.so` built from it, sha `43e80b39`, `$EXP/trackerfix/impl-report.md:128`] |
| Companion branches (same name) | flex `424ec94e` (tree `7e90fed1`) = main `9cf6b10e` + #1888 + #1894 + local tracker record-pruning commit. spyre-comms `6b2855ee` (tree `71a86a23`) = main `f11a5d9a` + #451 + #464 + #467 |
| Shared dependencies (not changed) | DeepTools `5a2d7b4d` (v2.0.0-rc.1), senlib `4e563485`, libaiupti `366ba0b5`, `common` submodule `8ff39153` (flex and spyre-comms only; torch-spyre has no submodules) |

Upstream state on 2026-10-01 [measured, gh REST]: #5040 is still open and was rebased to `0a6a4db1`;
its changed lines are the same as the tested diff (current patch in
`$EXP/super-branches/remote/pr5040-files-current.txt`). Its review asks for changes, with three
blockers (section 5, items 1-3; `/tmp/devel/src/torch-spyre/pr_review_5040.txt`). Companion PRs:
flex #1888 merged as `2a713cfc` (the tested change plus one line), spyre-comms #464 merged as
`a41ed0c0` (identical), spyre-comms #451 closed without merging (a different fix is planned, issue
#469), flex #1894 and spyre-comms #467 open. The local `origin/main` was not fetched, so how far
upstream main has moved is unknown.

## 2. Changes

### Already on main: correction-step split and HostCompute Prep role (#4101, #4864)

- **#4101** "feat: Generic hazard tracker" (Yue Zhu, `ea4603f0`). It reads `SPYRE_HAZARD_TRACKER`
  once at startup (`torch_spyre/csrc/module.cpp:106-113`; accepts `1`, `true`, `t`, `yes`, `y`, the
  same values flex accepts) and defaults off (`module.cpp:81-91`). When it is on, the launch router
  sends Prep-role steps to the host-compute stream S_prep and the rest to the default stream S_dev
  (`torch_spyre/csrc/spyre_stream.cpp:290-314`). The correction H2D carries the Prep role
  (`torch_spyre/csrc/job_plan.h:398`). Flex then adds the cross-stream H2D -> Compute ordering [code].
- **#4864** "change pipeline_barrier back to true as default" (Jordan Sullivan, co-authored by Joshua
  Rosenkranz, `8c4a2ce2`). HostCompute keeps its pipeline barrier and gets the Prep role
  (`torch_spyre/csrc/job_plan.h:541-546`), so the whole correction step (HostCompute, then H2D) goes
  to S_prep and the Compute stays on S_dev [code].
- **How it helped.** This split is what flex's tracker and #1894 work on. With
  `SPYRE_HAZARD_TRACKER=1` and `SPYRE_HAZARD_COMPLETED_READS=1` on the same binaries (arm
  `hursey-tracker467-fix`, pinned, ABBA order, 6 vs 6 launches of 30 requests, nothing else changed),
  the decode step drops from 65.2 to 54.3 ms (-10.9 ms, 95% CI -11.7 to -10.0; all 6 pairs negative,
  exact permutation p = 0.0022). Decode 1 goes from 60.6 to 49.5 ms, decode 2 from 61.7 to 52.8 ms,
  and prefill is 8.0 ms faster, with bit-identical outputs. Binaries, environment, pinning, CPU
  frequency and run order do not differ [measured, `$EXP/trackerab/run-report.md:14-29`,
  `$EXP/analysis/trackerab-verify-statistics/verify.md`, `$EXP/analysis/trackerab-verify-confounds/`].
  Without the tracker this stack does not reach 48-52 ms. Mechanism [measured, AB-P-ON vs AB-P-OFF
  traces, plus code; `$EXP/analysis/trackerab-verify-mechanism/verify.md`]: almost all of the gain is
  the gap from a collective's end to the next model kernel, median 168 us -> 0.4 us, 80 collectives
  per decode forward, about 13 ms per forward. With the tracker off every step goes to stream 0
  (`spyre_stream.cpp:306-314`), so the correction H2D sits on stream 0's DMA pipeline behind the
  collective's compute ops. Flex's pipeline-switch fence on the STRICT_ORDERING default stream (flex
  `src/runtime_stream/scheduler/runtime_scheduler.cpp:417-466`) makes it wait for the collective to
  finish, and the next kernel waits for the H2D. This chain would exist even with `pipeline_barrier=false`
  [code]. With it on, HostCompute and H2D go to prep stream 65 (`spyre_stream.cpp:102`), the upload
  goes out about 358 us before the collective ends, and the next kernel, on the same compute pipeline
  as the collective, is sent to hardware about 244 us before the collective ends and starts 0.4 us
  after it (`$EXP/trackerab/workflow-result.json:125`). Part of the saving is given back at
  layer starts (+0.3 / +2.7 ms per decode 1 / decode 2 forward). Host cost with it on: working time
  +4.3 to +5.3 ms per decode forward (PhaseLaunch 2.90 -> 4.71 ms, HostCompute +1.5 ms); blocked time
  31.5 -> 15.0 ms (`$EXP/analysis/trackerab-synthesis/synthesis.json`). Not separated: the share of
  `SPYRE_HAZARD_COMPLETED_READS` versus the tracker itself. Not tested: dropping both the tracker and
  #467 (the -7.4 ms against Hursey's stack without #467 is cross-build and weak). The earlier first
  study (without #467 and without the pruning, `report.md:83`) does not describe this stack.
- **Required?** Yes: without it the same binaries are 10.9 ms per decode step slower [measured]. It
  only works with the tracker on in both flex and torch-spyre. The flag must match in the two
  libraries: if torch-spyre splits while flex is not tracking, nothing orders H2D before Compute and
  results are wrong (`module.cpp:81-91`) [code].

### #5040, part 1: skip the host wait after each collective

- **What it does.** `spyre_wait_work_impl` (the wait used by compiled collectives) no longer calls
  `work->wait()` unless `TORCH_SPYRE_DIST_SKIP_WAIT=0`
  (`torch_spyre/csrc/distributed/spyre_distributed.cpp:513-532`). The reasoning, written in the code:
  spyre-comms is set up on the default flex stream (`spyre_distributed.cpp:122-131`), so the stream
  already orders the collective before later kernels on that stream [code]. On the host, the wait
  moves to the start of the next collective (`report.md:140`) [measured, profile].
- **How it helped (only in combination).** Together with flex #1888 (a completion callback on Null
  ops) and spyre-comms #464 (completes `wait()` through that callback, one WorkSchedule at a time),
  decode 1 went from 81.0-84.1 ms on main + #451 (T1-T4, unpinned) to 57.2-57.9 ms (E2-04, E2-05,
  pinned, a later session; pinning alone is worth about 1-2 ms on this stack,
  `$EXP/e1e2/e2-report.md:11-15`) [measured, `decode-stack-summary.md:62-63`]. The like-for-like
  figure is from the four-arm study, where both arms were unpinned in the same session: 22-25 ms
  faster per decode step than main in rounds T1-T3 (10.6 ms in T4) and 8-36 ms faster in prefill
  [measured, `report.md:80`]. Host time blocked per decode step
  fell from 62.5 to 27.5 ms (`report.md:139`) [measured, profile].
- **What it does alone.** Turning it off in the full stack (`TORCH_SPYRE_DIST_SKIP_WAIT=0`, CL-02 and
  CL-05, with the tracker and #467) puts decode 1 back at 82-85 ms, about the level of main + #451
  (81.0-84.1 ms, unpinned, earlier session) (`$EXP/climb/climb-report.md:18`) [measured]. So #5040 is the switch that turns the gain on. #1888
  and #464 do not help without it, because the blocking wait is still there [inferred]. #5040 was not
  measured without #1888 and #464. The #5040 review finds skipping the wait gave correct results only
  with #464 and #1888 present, everything on stream 0, and #467 whenever the tracker is on
  (`pr_review_5040.txt:152-159`).
- **Required.** Yes.

### #5040, part 2: keep every WorkSchedule until backend teardown

- **What it does.** When the wait is skipped, the WorkSchedule is moved into a process-wide list
  (`spyre_distributed.cpp:59-66`, `:523-530`) that is cleared only in the backend destructor
  (`torch_spyre/csrc/distributed/spyre_ccl.cpp:59-62`). The code comment explains why: the
  WorkSchedule destructor would wait, which would undo the gain (`spyre_distributed.cpp:523-527`)
  [code].
- **Effect.** It does not speed anything up; it exists so part 1 works. It costs memory and time.
  Memory [measured]: about 0.945 GiB of device memory per request (7.45 GiB at request 0 to
  34.85 GiB at request 29, CL-08, which is the stack before the flex pruning; device memory has not
  been re-measured on this stack) and 0.8-1.3 MiB of host memory per rank per request with retention
  on, against 0.09-0.13 MiB with it off (`$EXP/climb/climb-report.md:22-27`). Time [measured growth;
  link to retention inferred]: over 90 requests (FX-04, rank 0), request start to prefill start grew
  from 7.5 to 28.5 ms and last decode to request end from 5.5 to 53.3 ms, while the forwards stayed
  nearly flat. Runs without retention (skip-wait off, and main + #451) are flat, but skip-wait off
  also brings back the blocking wait (`decode-stack-summary.md:91-99`). With the tracker on and
  without flex record
  pruning, the retained buffers also made decode climb 0.25-1.0 ms per request; the flex
  companion commit `424ec94e` fixes that climb (+0.02 ms per request over 90 requests, 95% CI -0.04
  to +0.08) [measured, `decode-stack-summary.md:29`, `:73-74`].
- **Required?** Required by part 1 as written, but the #5040 review makes it blocker 1: it must be
  replaced by bounded retention before this can run as a server (section 5, item 1).

### Present but not exercised in these runs

- The eager ProcessGroup paths in `spyre_ccl.cpp` (for example `:455-457`) still call
  `work->wait()` for synchronous ops; #5040 changes only the compiled-path `wait_work` [code].
- `TORCH_SPYRE_DIST_SKIP_WAIT=0` is used only as a control.
- flex #1894 forwards a HostCompute callback to its correction H2D, but torch-spyre sets no
  HostCompute callback (`torch_spyre/csrc/job_plan.cpp:159-170`), so that part is not used [code].
- `TORCH_SPYRE_COMM_STREAM` (from #5037) is not on this branch. Pinned, the #5037 overlap stack was
  9-10 ms per decode step slower than Hursey's stack (decode 1 66.7-67.0 ms vs 57.2-57.9 ms,
  E2-02/E2-07 vs E2-04/E2-05; `decode-stack-summary.md:51-53`, `:63`, `:67`) [measured], and about
  15-18 ms slower than this stack (cross-session) [inferred].
- AllGather through `wait_work`: with the wait skipped, the reassembly copy from the per-rank buffers
  runs right away (`torch_spyre/csrc/distributed/spyre_distributed.cpp:534-552`) and depends on the
  same default-stream ordering as known issue 2 [code]. Granite uses only AllReduce
  (`$EXP/report/report.md:4-5`), so this path was not exercised [measured: not run]. With the wait
  skipped, `hold_tensors` and `rank_outputs` are freed while the collective may still run; this held
  only because Granite's AllReduce is in place (`pr_review_5040.txt:75-88`, `:192-194`) [code].

## 3. How to build and run the best configuration

Build the three companion branches with the canonical `torch-spyre-docs/scripts` (flex, then
spyre-comms, then libaiupti, then torch-spyre), each in a clean shell; run
`git submodule update --init` in flex and spyre-comms first (`common` at `8ff39153`). The measured
`_C.so` was built against flex `1287d452` (#1888 + #1894 without the pruning commit). The pruning
commit changes no installed flex header (`$EXP/trackerfix/include-diff.txt` is empty), so the run
swapped in the fixed libflex without rebuilding torch-spyre [measured]. Building torch-spyre against
flex `424ec94e` should give an equivalent `_C.so` [inferred].

Environment (FX launches: `$EXP/trackerfix/run-launch-trackerfix.sh:56-61`, which sources
`$EXP/trackerfix/device-env.sh` -> `$EXP/device-env.sh:10-20` with arm hursey-tracker, plus
`$EXP/activate.sh:28-29`; the harness also needs
`/tmp/devel/src/build/granite-tp4-prep-overlap-20260925/hf-adapters` on PYTHONPATH,
`$EXP/device-env.sh:11`; exact command in `$EXP/trackerfix/launches/FX-04/torchrun-command.txt`):

```bash
export SPYRE_HAZARD_TRACKER=1 SPYRE_HAZARD_COMPLETED_READS=1   # set before launch; both libs read it once
unset TORCH_SPYRE_DIST_SKIP_WAIT                              # #5040 default: skip the wait
export FORCE_SYNCHRONOUS_EXECUTION=NONE TORCH_SPYRE_NUM_HOST_COMPUTE_STREAMS=4
export FLEX_DEVICE=PF FLEX_COMPUTE=SENTIENT SPYRE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 BLIS_NUM_THREADS=4
taskset -c 0-35,72-107 torchrun --standalone --nproc-per-node=4 <harness>/run-measurement.py ...
```

- Pinning: run torchrun on the cards' NUMA node (NUMA0). The CPU list is for this pod only, and
  numactl is not installed, so there is no memory binding. Unpinned launches sometimes ran 10-12 ms
  slower per decode step (`$EXP/e1e2/e2-report.md:11-15`) [measured].
- Same in every arm: `SPYRE_KERNEL_CACHE=1` and `TORCH_COMPILE_DEBUG=1`; 2 warmups + 30 (or 90)
  requests per process; harness `$EXP/e1e2/harness-src-e2/timing/run-measurement.py`. Fresh compiles
  are not reproducible, so A/B runs must start from one seeded cache and compare against a seeded
  reference (`$EXP/report/report.md:53-57`).

## 4. Results for the whole stack

Granite 3.3 8B FP16, TP4, batch 1, 64 input tokens, 3 output tokens. Rank-0 medians over 30
requests, ms [measured, `decode-stack-summary.md:57-72`]. Token step = start of decode step 1 to
start of decode step 2 (time between tokens).

| Stack (cumulative) | Launches | Prefill | Decode 1 | Token step |
|---|---|---|---|---|
| main + #451 (unpinned) | T1-T4 base | 184.6-203.8 | 81.0-84.1 | 85.1-88.3 |
| + #1888, #464, #5040 | E2-04, E2-05 | 163.1-163.5 | 57.2-57.9 | 61.9-62.4 |
| + #467 | CL-04 | 163.2 | 61.1 | 65.8 |
| + #1894 and tracker, no pruning | E2-09, CL-01, CL-06, FX-02 | 154.9-155.8 | 53.7-57.4, climbing | 58.5-62.3 |
| + flex record pruning (this set of branches) | FX-01, FX-03, FX-04 | 154.7-155.3 | 48.4-51.6 | 53.3-56.9 |
| Same binaries, tracker off (A/B, mean of 6 launch medians) | AB OFF | 163.2 | 60.6 | 65.2 |
| Same binaries, tracker on (A/B, mean of 6 launch medians) | AB ON | 155.2 | 49.5 | 54.3 |

The A/B rows are from `$EXP/trackerab/run-report.md:14-20`. Outputs match the seeded reference
(rtol=atol=1e-3) on every rank, and tokens are exact; all 14 A/B launches matched it exactly
(`run-report.md:49-53`). Over 90 requests (FX-04) decode 1 stays flat (51.0 ms in requests 80-89).

## 5. Known issues and follow-ups (torch-spyre)

1. **Bound #5040's retention (review blocker 1).** Every WorkSchedule and its device buffers are kept
   until teardown (`spyre_distributed.cpp:529`). Suggested fix: in each `*_run_impl`, after the new
   schedule's `start()` returns, swap the retained list out under the lock and destroy it outside the
   lock; with #464's one-at-a-time queue the earlier schedules are done by then. Do not use `query()`
   (it checks the whole default stream) or destroy-when-DONE (it can free a condition variable that
   is still being notified, spyre-comms `work_schedule.cpp:823-830`) [code]. Then rerun 90 requests
   with the wait skipped, record device memory, and add a bounded-memory test. Until then device
   memory grows about 0.9 GiB per request (measured on CL-08, before the flex pruning; not
   re-measured on this stack) [measured growth; fix not built].
2. **#5040 skips the wait without checking the stream (review blocker 3).** spyre-comms is set up on
   the default stream (`spyre_distributed.cpp:130-131`), but `wait_work` does not check that the
   current stream is stream 0 before skipping the wait [code]. A consumer on another stream would
   read the result unordered [inferred]. Fix: skip only on stream 0, otherwise fall back to `wait()`.
3. **#467 is a prerequisite (review blocker 2), and the tracker flag must be on in both libraries.**
   The safety comment (`spyre_distributed.cpp:122`, `:508-512`) ignores cross-rank reuse of the host
   shared-memory slot [code]. Tracker on without #467: 6 of 9 launches failed (T1, T2, T4, E1-03,
   E1-06, CL-03), at decode step 1 with ranks split 3-1 (2-2 in E1-06). With #467: 0 of 22 counted,
   and none of the later A/B launches failed. Tracker off without #467: 0 of 12
   (`pr_review_5040.txt:161-172`, `$EXP/trackerab/run-report.md:49-53`) [measured].
4. **Tracker benefit is measured (section 2); two parts are open.** The share of
   `SPYRE_HAZARD_COMPLETED_READS` versus the tracker is not separated, and the stack without both the
   tracker and #467 was not run (`$EXP/analysis/trackerab-synthesis/synthesis.json`).
5. **Other #5040 review items:** on by default without checking for #464; `hold_tensors` and
   `rank_outputs` freed while the collective may still run; errors not surfaced when the wait is
   skipped; cleanup of all groups tied to each backend destructor; no tests; only `0` disables
   `TORCH_SPYRE_DIST_SKIP_WAIT` (`pr_review_5040.txt:62-141`).
6. **Before any push or PR:** the merge commit is local and unsigned with a fixed identity; redo it
   with sign-off. Rebase onto current main and take #5040's current head (`0a6a4db1`, same changes).
7. **Next lever:** about 5 ms per token still sits between decode forwards (A/B ON: 54.3 ms step vs
   49.5 ms decode 1), the tracker adds 4.3-5.3 ms of host work per decode forward and gives some
   saving back at layer starts, and host work at the start and end of each request grows with
   retention (`decode-stack-summary.md:89-114`, `:144-145`).
