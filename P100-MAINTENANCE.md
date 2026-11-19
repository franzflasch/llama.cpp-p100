# P100 fork maintenance guide (for the coding agent)

You keep the branch `p100-v2` up to date with upstream llama.cpp. Read this whole file before you start. Read `P100.md` too: it lists every P100 file and every hook.

## 1. What this fork is

- llama.cpp tuned for 2x Tesla P100 (sm_60). One model split over both GPUs (`-sm tensor`), q4_0 KV cache, MTP speculative decoding.
- `p100-v2` = upstream `llama_master/master` + a short stack of P100 commits (`git log --oneline llama_master/master..p100-v2`).
- Almost all P100 code is in NEW files (`*-p100.*`, `ggml/src/ggml-cuda/p100/`, `*.inc`). New files never conflict.
- Upstream files only have small hooks. Conflicts happen there.
- Repo: `/working_dir/work/local_llm/llama_rebase/llama_cpp_p100`
- Remotes: `llama_master` = upstream ggml-org/llama.cpp. `franzflasch` = the human's fork. Push only as rule 2 allows.

## 2. Hard rules

1. NEVER commit to, reset or rebase `p100-v2` directly. Work on a new branch `p100-next`.
2. NEVER `git push`. NEVER open PRs or issues.
   Exception: if the human explicitly asks, push to `franzflasch` (`master`, `p100-v2`). Never push to `llama_master` or `origin`. For `p100-v2` after a rebase, use `git push --force-with-lease=p100-v2:<old p100-v2 hash> franzflasch p100-v2` (old hash = the `p100-good-<date-time>` tag).
3. NEVER delete or disable P100 code to make a conflict or a compile error go away.
4. NEVER build into `build-opt`. The running server uses it. Build into `build-test`.
5. NEVER stop, restart or kill the running llama-server. The human does GPU tests. Never run GPU tests or the start scripts yourself.
6. If you are not sure how to resolve something: stop, run `git rebase --abort`, and report (section 9). A clean abort is a good result. A wrong guess is a bad result.
7. Keep the stack shape: same commits, same order, same messages. Do not squash, split or reword.
8. Only plain ASCII in code and comments. Keep comments short.

## 3. When to run

- Weekly. Small upstream gaps give small, easy conflicts.
- If the last run was aborted, do not try again on newer upstream until the human has looked at the report. Check the last report before a new run. If it says ABORTED: stop.

## 4. Procedure

### Step 1: prepare

```bash
cd /working_dir/work/local_llm/llama_rebase/llama_cpp_p100
git status --short | grep -v '^??'          # must print nothing; if not: stop and report
git config rerere.enabled                    # must print true
git fetch llama_master
git log --oneline p100-v2..llama_master/master | wc -l    # new upstream commits
git tag p100-good-$(date +%Y%m%d-%H%M) p100-v2           # safety point
git log --oneline p100-v2..p100-next                     # must print nothing; if not: stop and report (unmerged work)
git branch -f p100-next p100-v2
git switch p100-next
```

If there are 0 new upstream commits: stop, nothing to do.

### Step 2: rebase

```bash
git rebase llama_master/master
```

- No conflict: go to step 3.
- Conflict: follow section 5 for each file, then `git add <file>` and `git rebase --continue`. Repeat until done.
- rerere may resolve a conflict by itself (message "Resolved ... using previous resolution"). Check that result with `git diff --cached <file>` before you continue.

### Step 3: build (CPU check first, then CUDA)

```bash
cmake -B /tmp/p100-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON > /dev/null
cmake --build /tmp/p100-cpu -j --target llama-server llama-cli test-backend-ops 2>&1 | grep -E 'error' ; echo "cpu build rc=${PIPESTATUS[0]}"
./build_p100_llama.sh --clean build-test 2>&1 | tail -40
```

- Both builds must finish without errors.
- Warnings about unused static functions in `mmvq.cu` are expected (in sm_60 builds the upstream mmvq body is compiled out).
- Compile error: follow section 6.

### Step 4: CPU smoke test

```bash
./build-test/bin/test-backend-ops -b CPU -o MUL_MAT 2>&1 | tail -3
```

This only checks that the binary runs. Real checks need the GPUs (step 5).

### Step 5: hand over to the human

Write the report (section 9) and ask the human to run (server must be stopped for this). The agent does not run these commands:

```bash
for op in MUL_MAT MUL_MAT_ID FLASH_ATTN_EXT RMS_NORM GATED_DELTA_NET SSM_CONV CONCAT; do
    printf '%-16s ' "$op"; ./build-test/bin/test-backend-ops -o "$op" -b CUDA0 2>&1 | tail -1
done
sed 's#build-opt#build-test#' start_p100_llama.sh      > /tmp/t1.sh && bash /tmp/t1.sh   # expect ~50 t/s
sed 's#build-opt#build-test#' start_p100_llama_Q6_K.sh > /tmp/t2.sh && bash /tmp/t2.sh   # expect ~44 t/s
```

Pass criteria: no FAIL lines, t/s within 3% of the numbers above, answers make sense.

### Step 6: finish (only after the human says the tests passed)

```bash
git branch -f p100-v2 p100-next
git switch p100-v2
```

Then ask the human to stop the server and run `./build_p100_llama.sh build-opt` (or do it only after the human says the server is stopped). The human restarts the server afterwards. Skip the build if only docs changed.

Push only if the human asks (rule 2):

```bash
git fetch franzflasch
git push franzflasch master
git push --force-with-lease=p100-v2:<old p100-v2 hash> franzflasch p100-v2
```

If the tests failed: do not touch `p100-v2`. Report what failed. The human stays on the old build.

## 5. Resolving conflicts

First find out which pattern the conflicting spot belongs to. `P100.md` lists them. Look at the P100 side (the commit being replayed) and the upstream side (HEAD).

General rule: take the NEW upstream code as the base, then re-apply the small P100 change on top of it. Never the other way round.

### Pattern A: function hook

Looks like this at the top of an upstream function:
```cpp
    if (ggml_cuda_p100_rms_norm(ctx, dst)) {
        return;
    }
```
Files: `norm.cu`, `fattn.cu` (3 hooks + tile check), `ggml-cuda.cu` (try_fuse pre/post, peer copy, graph checks).
Resolution: keep the upstream version of the function, put the hook lines back at the same place (start of the function, or same position relative to the same upstream statement). If the upstream function was renamed or removed: stop and report.

### Pattern B: sm_60 build switch

```cpp
#if GGML_CUDA_P100_MMVQ
    ggml_cuda_p100_mul_mat_vec_q(ctx, src0, src1, ids, dst, fusion);
#else
    ... upstream body ...
#endif // GGML_CUDA_P100_MMVQ
```
Files: `mmvq.cu` (2 functions), `fattn-tile.cuh` and `fattn-vec.cuh` (whole file between `#if GGML_CUDA_P100_FATTN` / `#else` / `#endif`).
Resolution: the `#else` part must be EXACTLY the new upstream code. Take upstream's version of the file, then re-add the switch lines. If the function signature changed upstream, the P100 function signature must change the same way (that is a section 6 job).

### Pattern C: full replacement

`gated_delta_net.cu` and `gated_delta_net.cuh` start with:
```cpp
#include "gated_delta_net-p100.cuh"

#ifndef GGML_CUDA_GDN_P100
```
and end with `#endif // GGML_CUDA_GDN_P100`.
Resolution: take the upstream file completely, re-add the 3 lines at the top and the `#endif` at the end. Nothing else.

### Pattern D: textual include

One line such as `#include "p100/try-fuse.cuh"`, `#include "sampling-p100.inc"`, `#include "test-backend-ops-p100.inc"`.
Resolution: keep upstream code, put the include line back at the same spot (between the same two upstream functions). In `test-backend-ops.cpp` also keep the two calls `p100_add_test_cases_eval(test_cases);` and `p100_add_test_cases_perf(test_cases);` right before `return test_cases;`.

### Pattern E: real code changes inside upstream files

Files: `ggml-cuda.cu` (graph evaluate loop), `common.cuh` (fields of `ggml_backend_cuda_context`), `vecdotq.cuh`, `concat.cu`, `unary.cu`, `convert.cu`, `binbcast.cu`, `cpy.cu`, `allreduce.cu`, `ggml-backend*.cpp`, `common/speculative.cpp`, `common/sampling.cpp`, `tools/server/server-context.cpp`, `src/llama-*.cpp`, `src/models/qwen35*.cpp`, `qwen3next.cpp`.
Resolution:
- Added fields, added lines next to changed upstream lines: keep both. Easy.
- Upstream rewrote the code the P100 change sits in: STOP and report. Do not guess.

### Quick check after each resolved file

```bash
grep -n '<<<<<<<\|>>>>>>>\|=======' <file>     # must print nothing
git diff llama_master/master -- <file> | head -80   # P100 changes still small and present?
```

## 6. Compile errors

### In a P100 copy

Files: `mmvq-p100.cu`, `fattn-tile-p100.cuh`, `fattn-vec-p100.cuh`, `gated_delta_net-p100.cu`, `norm-p100.cu`, and the other `*-p100*` files.
These are private copies of upstream code. When upstream changes a shared interface (a struct field, a function parameter, a macro, a helper name), the copy no longer compiles.
How to fix:
1. Find what upstream changed: `git log -p --since=<last run date> llama_master/master -- <the upstream original, e.g. ggml/src/ggml-cuda/mmvq.cu>`.
2. Make the SAME small change in the copy (rename, new parameter, new field). Do not copy new upstream kernels into the copy.
3. If upstream changed the whole design of that file: stop and report.

### In an upstream file

Usually a hook calls something that was renamed. Fix the hook call, not the upstream code.

## 7. Risk map

| Risk | Files | What to do |
|---|---|---|
| low | hooks in `mmvq.cu`, `norm.cu`, `fattn*.cu/.cuh`, `gated_delta_net*`, `test-backend-ops.cpp`, `common.cpp`, `graph.cuh`, small headers | resolve with patterns A-D |
| medium | `concat.cu`, `unary.cu`, `convert.cu`, `binbcast.cu`, `cpy.cu`, `vecdotq.cuh`, `allreduce.cu`, `common/*`, `src/*`, `tools/server/*` | resolve if the change is next to upstream code; stop if upstream rewrote it |
| high | `ggml-cuda.cu` evaluate loop, `common.cuh`, `ggml-backend*.cpp`, all `*-p100` copies when upstream redesigns the original | stop and report unless the fix is an obvious rename |

## 8. Small P100 fixes later

If the human asks for a fix to a P100 feature:
```bash
git switch p100-next      # (create from p100-v2 if needed)
# edit, build into build-test, have the human test
git tag p100-good-$(date +%Y%m%d-%H%M) p100-v2
git commit --fixup=<hash of the stack commit for that area>
GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash llama_master/master
```
This keeps the stack the same length. Update `P100.md` if you add a file or a hook.

## 8a. Old safety tags

Keep the last 4 `p100-good-*` tags. Delete older ones only if the human asks: `git tag -d <tag>`.

## 9. Report template

```
P100 update report <date>
upstream: <old base hash> -> <new base hash> (<N> new upstream commits)
safety tag: p100-good-<date-time>
result: DONE / ABORTED / NEEDS HUMAN TEST
conflicts:
  - <file>: <pattern A-E>, <how resolved>  (or: rerere)
compile fixes:
  - <file>: <what changed upstream>, <what was changed in the copy>
stopped at (if aborted):
  - commit: <stack commit>, file: <file>, reason: <one line>
next step for the human:
  - <for example: run the GPU tests in step 5>
```

## 10. Emergency

- Something went wrong in the middle of a rebase: `git rebase --abort`, then `git switch p100-v2`.
- Branch is broken after the rebase: `git branch -f p100-next p100-good-<date-time>`.
- `p100-v2` itself must never be broken. If it is, restore it: `git branch -f p100-v2 p100-good-<date-time>`.
