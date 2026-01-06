# SPEC.md  Day1 (Colab, CUDA C++)

## Goal
- Build a minimal CUDA C++ benchmark binary (vadd) with CUDA-event timing.
- Log results to CSV with stable schema.

## Timing rule (critical)
- 	ime_ms_total measures TOTAL elapsed time over iters kernel launches.
- Therefore achieved throughput MUST use total work:
  - lops_total = flops_per_iter * iters
  - ytes_total = bytes_per_iter * iters
  - chieved = total_work / time_total

## Day1 op: vadd
- Kernel: C[i] = A[i] + B[i]
- per-iter model counts:
  - flops_per_iter = N
  - bytes_per_iter = 3 * N * sizeof(dtype)  (read A, read B, write C)

## CSV (minimum)
- Must include: gpu_name, cc, N, iters, time_ms_total, achieved_gbs, achieved_gflops
