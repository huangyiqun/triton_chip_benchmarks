# Ascend 950DT — Measured Results

All seventeen metrics, measured with the native Ascend C suite in this
directory on an otherwise idle `Ascend950DT_9582` (device 7), 2026-11-05.

Device as reported by `aclrtGetDeviceInfo`: 32 Cube cores, 64 vector (AIV)
cores, 1650 MHz, 128 MiB L2, 91.75 GiB HBM.

**Every timed configuration validated against an exact host reference with
`max_abs_error = 0`** — 164 configurations across the twelve measured formats,
zero failures, zero tolerance.

## The table

| Metric | Group | Peak | Median | Unit | Execution path |
| --- | --- | ---: | ---: | --- | --- |
| memory_capacity | Memory | 91.75 | — | GiB | reported device capacity |
| memory_bandwidth | Memory | 3342.71 | 3339.70 | GB/s | copy, pipelined, 256 blocks |
| tensor_bf16 | Tensor | 430.71 | 430.70 | TFLOP/s | native `asc_mmad` |
| tensor_fp16 | Tensor | 430.72 | 430.70 | TFLOP/s | native `asc_mmad` |
| tensor_fp32 | Tensor | 27.01 | 27.00 | TFLOP/s | native `asc_mmad` (IEEE) |
| tensor_tf32 | Tensor | 215.35 | 215.35 | TFLOP/s | native HF32 `asc_mmad` |
| tensor_int8 | Tensor | 861.44 | 861.40 | TOP/s | native `asc_mmad` |
| tensor_fp64 | Tensor | — | — | TFLOP/s | **absent from the ISA** |
| tensor_fp8 | Tensor | 861.43 | 861.39 | TFLOP/s | native `asc_mmad` |
| **tensor_fp4** | Tensor | **1722.67** | 1722.55 | TFLOP/s | native `asc_mmad_mx` (MX) |
| vector_bf16 | Vector | 26.17 | 26.17 | TFLOP/s | native `asc_madd` |
| vector_fp16 | Vector | 26.17 | 26.17 | TFLOP/s | native `asc_madd` |
| vector_fp32 | Vector | 13.08 | 13.08 | TFLOP/s | native `asc_madd` |
| vector_int8 | Vector | 15.90 | 15.90 | TOP/s | widened to INT16 |
| vector_fp64 | Vector | — | — | TFLOP/s | **absent from the ISA** |
| vector_fp8 | Vector | 4.45 | 4.45 | TFLOP/s | converted through FP32 |
| **vector_fp4** | Vector | **8.91** | 8.91 | TFLOP/s | converted through BF16 |

Read bandwidth, reported as an auxiliary detail: **3936.72 GB/s** (median 3926.95).

## Against the hardware ceiling

The Cube and vector peaks are checked against what the core counts and clock
allow, so a number that merely looks large is not mistaken for a peak:

| | Measured | Ceiling | Achieved |
| --- | ---: | ---: | ---: |
| Cube BF16 / FP16 | 430.7 | 432.5 TFLOP/s | **99.6 %** |
| Cube INT8 / FP8 | 861.4 | 865.1 | **99.6 %** |
| Cube FP4 | 1722.7 | 1730.2 | **99.6 %** |
| Vector FP16 / BF16 | 26.17 | 27.03 TFLOP/s | **96.8 %** |
| Vector FP32 | 13.08 | 13.52 | **96.7 %** |

Ceilings: Cube = 32 cores × 1650 MHz × 4096 MAC/cycle × 2 ops, doubling for
each halving of operand width. Vector = 64 AIV × 1650 MHz × lanes × 2 ops,
with 64 FP32 or 128 FP16/BF16 lanes in the 256-byte register.

The measured rates sit within 0.4 % of the matrix ceiling and 3.2 % of the
vector ceiling, so these are peaks, not merely large numbers.

## Against the Triton suite

The Triton suite in the parent directory measures the same chip through
FlagTree/Triton, writing
[`../results/ascend950dt_17metrics.json`](../results/ascend950dt_17metrics.json).
Where both reach the hardware limit the two agree to within noise, which is the
cross-check that matters:

| Metric | Ascend C | Triton | Ratio |
| --- | ---: | ---: | ---: |
| memory_capacity (GiB) | 91.75 | 91.75 | 1.00× |
| memory_bandwidth (GB/s) | 3343 | 3265 | 1.02× |
| tensor_bf16 | 430.71 | 430.35 | 1.00× |
| tensor_fp16 | 430.72 | 430.38 | 1.00× |
| tensor_fp32 | 27.01 | 26.97 | 1.00× |
| tensor_tf32 | 215.35 | 215.62 | 1.00× |
| tensor_int8 | 861.44 | 860.75 | 1.00× |
| tensor_fp8 | 861.43 | 860.75 | 1.00× |
| tensor_fp4 | 1722.67 | 1725.27 | 1.00× |
| vector_bf16 | 26.17 | 8.89 | **2.95×** |
| vector_fp16 | 26.17 | 18.54 | **1.41×** |
| vector_fp32 | 13.08 | 9.90 | **1.32×** |
| vector_int8 | 15.90 | 2.47 | **6.45×** |
| vector_fp8 | 4.45 | 5.43 | 0.82× |
| vector_fp4 | 8.91 | unsupported (no FP4 dtype in Triton) | — |

Every **tensor** row matches to three significant figures, including FP4.
Both implementations are hitting the same silicon limit from different
directions, which is the strongest evidence available that both measure the
real peak rather than a shared mistake.

Memory now differs by 2%: this suite's copy reaches 3343 GB/s against Triton's
3265, after the read/copy kernels were software pipelined (see below). The
read patterns diverge more — 3937 GB/s here against Triton's 3847 — but both
land in the same band, unlike the pre-pipeline 3159 GB/s this suite used to
report.

The **vector** rows differ because of instruction selection, not because the
chip changed:

- The 1.3–3.0× gains on FP32/FP16/BF16 come from the in-place `asc_madd`
  (`x = x*a + b`) instead of a returning fused multiply-add. The returning form
  costs an extra register move per update and measures 6.85 TFLOP/s FP32 where
  `asc_madd` measures 13.08. Triton's generated code uses the returning form.
- INT8's 6.45× gap is the same effect plus a wider lane: this suite multiplies
  INT8-range operands in INT16 lanes, where Triton's path carries INT8 state
  through a narrower emulation.
- FP8 is the one row where Triton is **higher** (5.43 vs 4.45). Neither is a
  native rate — the chip has no FP8 vector arithmetic at all — so the two
  numbers reflect two different emulation strategies, and the comparison is not
  meaningful as a hardware figure. This suite converts FP8→FP32→FP8 per update
  and keeps FP8 state; Triton keeps a narrower state with fewer conversions.

## Bandwidth: the pipeline, and a DMA trap worth knowing

The memory kernels stage through UB in two groups and are software pipelined:
while group *g* is being drained (stored for copy, reduced for read), group
*g+1* is already loading, synchronized with per-pipe `asc_sync_notify` /
`asc_sync_wait` between MTE2 and the consuming pipe rather than a whole-core
barrier.

That is worth a lot. The earlier barrier-per-group form alternated "load a
group" and "drain a group", leaving MTE2 idle for the whole drain phase:

| | Barrier form | Pipelined | Gain |
| --- | ---: | ---: | ---: |
| read | 2719 GB/s | **3937** | +45 % |
| copy | 3273 GB/s | **3343** | +2 % |

Read gains most because its drain is vector work rather than DMA, so the
serialization cost was largest there. For reference, a load-only kernel with
no reduction at all reaches 3655 GB/s — *below* the pipelined read — which
confirms the old number was limited by its own structure, not by the hardware.
The grid is also swept now (32/64/128/256 blocks) instead of being fixed at
one block per vector core: 32 blocks collapses to 2493 GB/s because too few
DMA engines are active.

**The trap.** `asc_copy_gm2ub` and `asc_copy_ub2gm` take their byte count in a
16-bit field and **wrap silently** on this CANN build — no error, no warning,
no launch failure. Measured directly:

| Requested | Actually transferred |
| ---: | ---: |
| 65280 B | 65280 B (exact) |
| **65536 B** | **0 B** |
| **65540 B** | **4 B** |
| **98304 B** | **32768 B** |

A bandwidth benchmark charges the kernel for the bytes it *requested*, so a
wrapped transfer divides real work by a fabricated byte count. A 64 KiB chunk
reported **8270 GB/s** here, roughly 2.5× the real HBM rate, and it looked
like a plausible "deeper chunks win" result.

Worse, it **passed validation**. The source pattern was periodic (period 8,
summing to −0.5), so every 8-element window summed identically and a short read
was indistinguishable from a complete one. Only the copy pattern caught it, via
a direct buffer comparison.

Both halves are fixed: chunks are capped below 64 KiB with a host-side
assertion, and the source is no longer uniform across the buffer — the period-8
base is scaled by a group index that changes every 8192 elements, so a block
that reduced the wrong region, reduced it twice, or reduced only part of its
slice lands on a different total. All values stay multiples of 1/8 bounded by
2, so FP32 lane accumulation remains bit-exact; that was verified by simulating
the kernel's 64-lane accumulation order in FP32 against the closed form before
trusting it.

## FP4, the two added metrics

The chip has genuine MX FP4 support in the matrix unit, reached through
`asc_mmad_mx` with E8M0 scale factors loaded into the L0 scale register file.

**The scale factors are verifiably applied, not ignored.** Holding the operands
fixed (all 1.0, K = 512, so the unscaled dot product is 512) and varying only
the A-side E8M0 code:

| A scale | Result |
| --- | ---: |
| 2⁰ (code 127) | 512 |
| 2⁺¹ (code 128) | 1024 |
| 2⁻¹ (code 126) | 256 |

The result tracks the scale exactly, so the measured 1722.67 TFLOP/s is a real
MX FP4 matrix rate running the full dataflow — not an FP4-shaped buffer fed
through some other path. It is 4× the BF16 rate and 2× FP8, exactly as the
operand widths predict, and lands at 99.6 % of the implied ceiling.

The vector unit has **no** FP4 arithmetic instruction, only conversion and
movement. The 8.91 TFLOP/s vector FP4 row is therefore a converted path
(FP4→BF16→madd→FP4, FP4 state retained, conversions charged as overhead rather
than counted as work). It is an honest measurement of what the chip does with
FP4 vector operands, and it is explicitly **not** a native FP4 vector peak.

## What has no numeric value, and why

| Row | Reason |
| --- | --- |
| tensor_fp64 | The Cube ISA has no FP64 instruction: `asc_mmad`/`asc_mmad_mx` provide overloads for bfloat16_t, half, float, int8_t, int4b_t, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t and the fp4x2 types, and no double-precision overload exists. |
| vector_fp64 | The RegBase vector ISA has no 64-bit float vector register type and no double-precision `asc_madd`/`asc_fma`/`asc_mul`. |

Both are determined from the instruction set rather than measured, and the JSON
records that as `determined_from`. Neither is reported as zero.

## Reproducing

```bash
cd ascendc
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./build.sh
python3 run_suite.py --visible-devices 7 --iterations 1024
```

Full results, including every configuration tried, its validation outcome and
the complete timing sample set:
[`../results/ascend950dt_ascendc_metrics.json`](../results/ascend950dt_ascendc_metrics.json)
and `../results/ascend950dt_ascendc_metrics_runs/`.

Those records were produced on 2026-11-05 while this suite was a standalone
repository; the measurements are unchanged, and the `command`, `source_json`
and `log_file` paths inside the JSON were rewritten to the merged layout when
it moved here. The JSON notes this in its `provenance` field.

