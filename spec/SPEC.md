# SPEC.md Day1/Day2 (Colab, CUDA C++)

## Goal
- Build a minimal CUDA C++ benchmark binary with CUDA-event timing.
- Log results to CSV with stable schema.

## Day1 - Vector Add (vadd)
- Kernel: C[i] = A[i] + B[i]
- per-iter model counts:
  - flops_per_iter = N
  - bytes_per_iter = 3 * N * sizeof(dtype)  (read A, read B, write C)

## Global invariants (apply to all ops)
- Timing uses CUDA events to measure TOTAL time across `iters` measured launches.
- Warm-up launches are excluded from `time_ms_total`.
- Work-model counts (`bytes_model`, `flops_model`) are PER-ITERATION.
- Achieved metrics MUST use TOTAL work:
  - bytes_total = bytes_model * iters
  - flops_total = flops_model * iters
  - achieved_Bps   = bytes_total / time_total_sec
  - achieved_FLOPs = flops_total / time_total_sec
- Reported (CSV/log) convenience units (decimal, base-10):
  - achieved_gbs    = achieved_Bps   / 1e9    (GB/s, not GiB/s)
  - achieved_gflops = achieved_FLOPs / 1e9    (GFLOP/s)
- Validation and reference computation MUST be outside the timed region.

## Day2 - Add two kernels (bandwidth-bound + compute-heavy)

### Supported ops (closed set for Day2)
op in { vadd, copy, fma_rpt }

### Data type scope (Day2)
- Day2 uses fp32 only. (No fp16/bf16 in Day2 ops.)
- Let E = sizeof(float) = 4 bytes.
- If the program supports a `--dtype` flag from Day1:
- For op in {copy, fma_rpt}, if dtype != fp32, the program MUST exit with an error (do not silently coerce).

### Kernel definitions and work-models

#### 1) copy (bandwidth-bound)
Operation:
- out[i] = in[i]

Work-model per iter:
- bytes_model = 2 * N * E
- flops_model = 0

Validation:
- copy validation MUST be bitwise (NaN-safe):
  - reinterpret both input and output element as uint32_t and compare equality of bits.
  - (Rationale: fp32 `==` treats NaN != NaN; copy should be validated as bitwise copy.)

#### 2) fma_rpt (compute-heavy, minimal global traffic)
Operation:
- acc = in[i]
- repeat K times: acc = fma(acc, a, b)
- out[i] = acc

Kernel args:
- (const float* in, float* out, int N, int K, float a, float b)

Work-model per iter:
- bytes_model = 2 * N * E
- flops_model = 2 * K * N  (FMA counts as 2 FLOPs)

Validation:
- CPU reference must use fused semantics:
  - acc = in[i]
  - repeat K times: acc = fma(acc, a, b)
- Tolerance: pass if |gpu-ref| <= atol + rtol*|ref| with atol=1e-4, rtol=1e-4.
- NaN/Inf mismatches are failures (handle explicitly).

### Validation coverage (cost-bounded, deterministic)
- Default validation checks a fixed number of indices:
  - VALIDATE_SAMPLES = 1024
- Indices MUST be deterministic (no RNG), including edge cases:
  - checks = min(N, VALIDATE_SAMPLES)
  - If N == 0: validation is skipped (but this case SHOULD be prevented by argument checks).
  - Always include i = 0.
  - If N > 1, always include i = N-1.
  - If checks <= 2, the set is {0} (if N==1) or {0, N-1} (if N>1).
  - Otherwise (checks >= 3):
    - stride = max(1, (N-1) / (checks-1))   // integer division
    - for j in [0 .. checks-1]:
        idx_j = min(N-1, j * stride)
    - Duplicates are allowed but implementations MAY de-duplicate for efficiency; semantics are unchanged.

### Measurement rules (unchanged from Day1; reiterated)
- `time_ms_total` measures TOTAL elapsed time over iters kernel launches.
- Achieved metrics use total work divided by total time.

### CSV logging (schema stabilization from Day2 onward)
- Keep all existing Day1 CSV columns and semantics unchanged.
- Use the existing `op` column if already present in Day1 CSV.
- Append the following columns at the end (in this order):
  - fma_k
  - fma_a
  - fma_b
- For op != fma_rpt, fma_* fields MUST be empty (or NA).

### CSV file policy (avoid header mismatch across days)
- Each run MUST write to a new CSV path (new file). Do not append rows to an existing CSV created with a different header.
- If the implementation supports an output path flag (e.g., --csv), it SHOULD default to a unique filename per run.

### Expected qualitative behavior (for interpretation)
- copy: for sufficiently large N, achieved_GB/s should increase then plateau (bandwidth limit).
- fma_rpt: as K increases, achieved_GFLOPS should increase then plateau; achieved_GB/s remains comparatively lower.
- Small N may be dominated by launch overhead; comparisons should emphasize sufficiently large N regimes.
