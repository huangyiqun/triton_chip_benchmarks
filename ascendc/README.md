# Ascend 950DT Chip Metric Benchmarks (Ascend C)

Native Ascend C implementation of the chip peak-throughput suite, written
against the SIMD C API and compiled with `bisheng` for `dav-3510` (Ascend 950
series). It reports the **fifteen metrics** defined by the Triton suite in the
parent directory, plus **two FP4 metrics** this chip adds, for **seventeen
rows** total.

This is the second implementation of the same 17-metric table. The parent
directory measures those metrics through Triton/FlagGems on any FlagGems
backend; this directory measures them on one chip through the vendor's own
kernel language. The metric keys, units, and JSON report shape are identical by
construction, so the two tables can be compared row for row — see
[Relation to the Triton suite](#relation-to-the-triton-suite) below and the
cross-implementation table in [`../README.md`](../README.md).

| Group | Metrics |
| --- | --- |
| Memory | capacity (GiB), effective copy bandwidth (GB/s) |
| Tensor / matrix (Cube) | BF16, FP16, FP32, TF32, INT8, FP64, FP8, **FP4** |
| Vector (AIV) | BF16, FP16, FP32, INT8, FP64, FP8, **FP4** |

Floating-point rates are TFLOP/s, integer rates TOP/s. A metric with no
numeric value is reported as `unsupported`, `not_built` or `error` — never as
zero.

## Quick start

Run from this directory. It needs CANN 9.2.x with `bisheng`; the Triton suite
in the parent directory has no such requirement and is unaffected if this
toolchain is absent.

```bash
cd ascendc
source /usr/local/Ascend/ascend-toolkit/set_env.sh

./build.sh                                  # all 16 binaries, into bin/
python3 run_suite.py --visible-devices 0    # the 17-row table
```

`bin/` is a build product and is not tracked in git — `./build.sh` takes about
a minute and the binaries are tied to a specific CANN version and `--npu-arch`,
so building locally is both cheap and safer than shipping them.

Individual metrics:

```bash
./bin/bench_cube_bf16   --iterations 1024   # one tensor format
./bin/bench_vector_fp32                     # one vector format
./bin/bench_memory                          # capacity + bandwidth
python3 run_suite.py --only tensor_fp4 vector_fp4
```

`run_suite.py --quick` runs a smoke configuration of everything. Those rates
are not peak measurements and the JSON marks them `smoke_only`. Note that
`memory_bandwidth` reports `error` under `--quick` by design: peak eligibility
requires a buffer of at least 4x L2 (512 MiB on this chip) and quick mode never
allocates one, so there is no eligible configuration to report. Use a full run
for that row.

## Relation to the Triton suite

The two implementations share the metric contract, not code:

| | `../bench_suite.py` (Triton) | `run_suite.py` (Ascend C) |
| --- | --- | --- |
| Metric keys / units | the same 17 `(key, group, precision, unit)` rows | identical, by construction |
| Kernels | Triton, through FlagGems backend management | Ascend C SIMD C API, `bisheng` |
| Portability | any FlagGems backend | Ascend 950 series only |
| Timing | device events / graph replay | ACL events |
| Extra statuses | `unverified`, `not_implemented` | `measured_widened`, `not_built` |
| Default output | `../results/<chip>_metrics.json` | `../results/ascend950dt_ascendc_metrics.json` |

Both write into the repository-wide `../results/` directory, so a chip
measured both ways keeps its two tables side by side.

The status vocabularies differ because the implementations can know different
things. The Triton suite says `unverified` when it cannot audit the final ISA
for a vendor; the Ascend C suite names the execution path directly
(`measured_widened` plus an `arithmetic_kind`) because it compiled the
instruction itself. Neither ever reports a missing measurement as zero.


## What is measured

### Tensor / matrix peak — `src/bench_cube.asc`

Operands are loaded into L0A/L0B once per launch, then one MMAD tile is issued
`iterations` times accumulating into a resident L0C. No global-memory traffic
enters the timed loop, so the rate reflects the matrix unit's issue rate rather
than a memory-bound GEMM. This mirrors the peak mode of
`triton_chip_benchmarks/bench_tensor.py`.

```
operations = 2 * M * N * K * iterations * blocks     # one multiply + one add per MAC
```

`A` is `[M,K]` K-major and `B` is `[N,K]` K-major, so `C = A @ Bᵀ` and neither
operand needs the transposing L1→L0 path. The sweep covers five tile shapes per
dtype and four occupancies (1, 2, 4, 8 blocks per Cube core).

- **TF32** is FP32 storage with `asc_enable_hf32()`, which switches the Cube to
  its 19-bit HF32 multiply — the TF32-equivalent path on this chip.
- **FP8** uses plain `asc_mmad`: the operands carry their own exponents and no
  scale factors are involved, so it is a dense native FP8 matrix operation.
- **FP4** uses `asc_mmad_mx`, the MX matrix path, with one E8M0 scale factor per
  64 K-elements per operand row loaded into the L0 scale register file by
  `asc_copy_l12l0a_mx`/`asc_copy_l12l0b_mx`. Scales are all 2⁰, so the
  arithmetic value equals the raw FP4 product and the reference stays exact
  while the hardware still runs the full MX dataflow including scale
  application. The host lays FP4 operands out in NZ order itself, because there
  is no ND2NZ path for 4-bit storage.

### Vector peak — `src/bench_vector.asc`

Twelve independent accumulator chains live in vector registers and are updated
with a dependent multiply-add for `iterations` steps. Operands and coefficients
are loaded from UB before the loop, so the timed region issues arithmetic only.
Independent chains hide the multiply-add result latency, so the measured rate
is the issue rate.

```
operations = 2 * 12 chains * lanes * iterations * blocks
```

Two measured facts shaped this kernel:

- **`asc_madd` beats `asc_fma` by ~2×.** The in-place `asc_madd(x, a, b)`
  (`x = x*a + b`) reaches 12.8 TFLOP/s FP32 where the returning
  `asc_fma(x, a, b)` spelling of the identical recurrence reaches 6.85, because
  the returning form costs an extra register move per update. The suite uses
  `asc_madd` wherever it exists.
- **Twelve chains is the register-pressure knee.** The rate climbs from 12.22
  TFLOP/s at 6 chains to 12.80 at 12; 16 or more chains overflow the
  6144-byte VF stack and collapse to roughly a third of that.

### Memory — `src/bench_memory.asc`

Capacity is the runtime-reported device capacity, not a measurement.

Bandwidth streams one HBM-sized buffer through UB, software pipelined across
two groups: while one group is drained (stored for copy, reduced for read) the
next is already loading, so the DMA queue never drains behind the consumer.

```
copy traffic = 2 * buffer bytes          # read + write
read traffic = buffer bytes + per-block FP32 sink
```

Buffers sweep 512 MiB … 4 GiB and the grid sweeps 32/64/128/256 blocks with
2048/4096/8192-element chunks, because neither is safe to assume: 32 blocks
reaches only 2493 GB/s read against 3937 at 256, and the pre-pipeline
barrier-per-group kernel reached just 2719. A size qualifies for the peak only
at **≥ 4× the 128 MiB L2**, so the traffic genuinely reaches HBM; smaller sizes
are reported but excluded. Rates are decimal GB/s. The suite's single bandwidth
figure is the copy pattern, matching the Triton suite; the read rate is
reported alongside as a detail.

Chunks are capped below 64 KiB, and this is a correctness limit rather than
tuning: `asc_copy_gm2ub`/`asc_copy_ub2gm` carry their byte count in a 16-bit
field and wrap **silently** past it (65536 B transfers 0, 98304 B transfers
32768). Since the benchmark charges the kernel for the bytes it requested, a
wrapped transfer fabricates a rate — a 64 KiB chunk reported 8270 GB/s, about
2.5× the real HBM rate. See `RESULTS.md` for why that also passed validation
and what changed so it no longer can.

## Correctness

**Every configuration is validated before it is timed, and a configuration that
fails validation never contributes a peak.** This matters because wrong operand
layouts produce plausible-looking throughput numbers — three real bugs during
development (an aliased reduction accumulator, a mis-strided FP8 chain layout,
and an FP4 nibble mapping) all produced timings that looked reasonable and were
caught only by the output check.

Operands are drawn from small exact values — ternary integers, or ternary times
1/8 — so every product and the whole accumulated sum are exactly representable
in the FP32/INT32 accumulator. **The expected error is therefore identically
zero**, with no tolerance to tune:

- Tensor: every output element is compared against a host `double` reference,
  and every block must match block 0 element for element.
- Vector: every chain is compared against the closed form of its recurrence
  (`a = -1`, so the state has period 2), and every block must agree.
- Memory: the copy's destination must equal its source element for element; the
  read's per-block reduction must equal the host sum exactly.

Outputs are poisoned with `0xFF` before each launch, so a kernel that silently
writes nothing cannot pass.

## Timing

Device-side only, through ACL events — never host wall clock:

- A batch of launches is bracketed by `aclrtRecordEvent` and the elapsed time is
  divided by the actual launch count.
- The batch is grown until the event timer resolves it, so no sample divides by
  a quantized zero.
- **Peak** is the shortest batch average observed; **median** is from the same
  configuration, which is what makes stability visible.
- Compilation, allocation, validation and warmup never enter a sample.

## What this chip does and does not have

Several formats have no native arithmetic instruction on one unit or the other.
The suite measures what the hardware can actually do and labels the execution
path, rather than presenting an emulated rate as a native peak.

| Format | Cube (matrix) | Vector (AIV) |
| --- | --- | --- |
| BF16, FP16 | native `asc_mmad` | native `asc_madd` |
| FP32 | native `asc_mmad` | native `asc_madd` |
| TF32 | native HF32 `asc_mmad` | n/a (not a vector format) |
| INT8 | native `asc_mmad` (INT32 accumulate) | **widened**: `asc_add` exists for `vector_int8_t` but there is **no 8-bit `asc_mul`**, so operands are multiplied and added in INT16 |
| FP8 | native `asc_mmad` | **converted**: no FP8 arithmetic instruction exists, only conversion and movement, so each update converts FP8→FP32, issues a native madd, and converts back, keeping FP8 state |
| FP4 | native `asc_mmad_mx` (MX path) | **converted**: same situation, widening through BF16 |
| FP64 | **absent**: `asc_mmad` has no double overload | **absent**: no 64-bit float vector type, no double-precision madd |

For the widened and converted vector rows, **only the multiply-add's 2
operations per update are counted** — the conversions are charged as overhead,
not as work. Those rows carry `status: measured_widened` and an
`arithmetic_kind` naming the actual path, and the table prints them as
`widened:`. They are genuine measurements of what the chip does with those
operand types, but they are **not** native FP8/FP4/INT8 vector arithmetic peaks,
because no such instruction exists.

FP64 is reported from the instruction set rather than measured; see
`src/bench_fp64.asc` for the reasoning, which is recorded in the JSON as
`determined_from`.

## Build structure

Each compute benchmark is compiled **once per dtype**, selected by
`-DBENCH_CUBE_DTYPE` / `-DBENCH_VECTOR_DTYPE`. Two hard constraints force this:

1. The linker sums static on-chip allocations across every kernel in a module,
   and L0A/L0B hold only 64 KiB each with L0C at 128 KiB — several full-size
   kernels cannot coexist.
2. Aliasing one shared file-scope pool through casts avoids that, but makes
   `bisheng`'s on-chip allocator pathological: the same kernel compiles in ~4 s
   with local declarations and did not finish in 25 minutes with pool casts. The
   shared-pool form also produced nondeterministic results in the vector chains.

Local declarations plus one kernel per binary keeps both the capacity limits and
the compile time. Tile shapes are runtime arguments, so one kernel per dtype
still covers the whole sweep.

## Hardware facts this suite relies on

Queried from `aclrtGetDeviceInfo`, not hardcoded:

| Property | Ascend950DT_9582 |
| --- | --- |
| Cube (AI) cores | 32 |
| Vector (AIV) cores | 64 |
| AI core clock | 1650 MHz |
| Vector register | 256 bytes (64 FP32 lanes) |
| UB per vector core | 216 KiB |
| L1 / L0A / L0B / L0C | 512 / 64 / 64 / 128 KiB |
| L2 | 128 MiB |
| HBM | 91.75 GiB |

Theoretical ceilings these imply, for comparison with the measured peaks:

- Cube BF16: 32 × 1650 MHz × 4096 MAC/cycle × 2 = **432.8 TFLOP/s**
- Vector FP32: 64 × 1650 MHz × 64 lanes × 2 = **13.5 TFLOP/s**

## Output

`run_suite.py` writes the complete table to
`../results/ascend950dt_ascendc_metrics.json`, checkpointed after every metric,
with per-metric JSON and logs under
`../results/ascend950dt_ascendc_metrics_runs/`. Each row keeps its status, the
winning configuration, the full timing sample set, and the validation result.

The `_ascendc_` in the name is what keeps this table distinct from the Triton
suite's `ascend950dt_metrics.json` for the same chip; both live in the shared
`../results/` directory. Pass `--output` to write elsewhere.

## Files

```
src/bench_common.h    device query, ACL event timing, JSON writer, host dtype casts
src/bench_cube.asc    Cube matrix peak; one binary per dtype
src/bench_vector.asc  Vector (AIV) peak; one binary per dtype
src/bench_memory.asc  capacity + copy/read bandwidth
src/bench_fp64.asc    FP64 capability report, per execution unit
build.sh              bisheng invocation and include/library paths
run_suite.py          runs all 17 metrics, emits the table and JSON
RESULTS.md            the measured 17-row table, with ceilings and cross-checks
bin/                  build product, not tracked in git; created by ./build.sh
```
