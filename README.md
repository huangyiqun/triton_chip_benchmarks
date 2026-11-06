README.md # Triton Chip Peak Throughput Benchmarks

The suite attempts device-memory capacity and bandwidth plus Tensor and Vector throughput across the requested numeric formats. It sweeps configurations to find **the highest throughput observed on the current device under the current operating conditions**, while reporting the median for the same configuration to help assess stability. It always produces 17 report rows, but it cannot guarantee 17 numeric peaks: a backend may reject a format, a benchmark path may be missing, or a measurement may fail. Precision, execution units, clock frequency, and access patterns affect peak throughput, so each result retains its dtype, operation, validation, and instruction-audit metadata.

The complete report contains these 17 rows:

| Group | Metrics |
| --- | --- |
| Memory | Capacity (GiB), effective copy bandwidth (GB/s) |
| Tensor/matrix | BF16, FP16, FP32, TF32, INT8, FP64, FP8, FP4 |
| Vector | BF16, FP16, FP32, INT8, FP64, FP8, FP4 |

Floating-point throughput uses TFLOP/s. INT8 uses TOP/s. A missing value is never replaced by zero.

FP4 is the one format with no unscaled `tl.dot` path. It is OCP MX FP4: E2M1 elements packed two per `uint8` container, with one E8M0 scale factor shared by every 32 elements along K, reached through `tl.dot_scaled`. The benchmark supplies unit scales (E8M0 code 127, `2**0`) so the measured arithmetic equals the raw FP4 products and the reference stays exact, while the hardware still runs the full MX dataflow including scale lookup and application. Tile `K` counts E2M1 elements, not container bytes. See [Matrix/Tensor](#1-matrixtensor-repeated-multiply-add-with-resident-operands) for the probes that keep an unscaled or mis-typed path from being reported as an FP4 peak.

### A second implementation: [`ascendc/`](ascendc/README.md)

`ascendc/` holds a native **Ascend C** implementation of the same 17-metric table for the Ascend 950 series, written against the SIMD C API and compiled with `bisheng`. It exists to answer a question the Triton suite cannot answer alone: when a measured rate falls short of the architectural ceiling, is that the chip or the compiler?

The two are independent programs that share the metric contract rather than any code — the same 17 `(key, group, precision, unit)` rows and the same JSON report shape, so their tables diff row for row. The Triton suite stays portable across every FlagGems backend and needs no vendor toolchain; the Ascend C suite runs only on this chip but compiles the instruction itself, so it can name the execution path instead of reporting `unverified`.

On Ascend950DT_9582 every **Tensor** row agrees to three significant figures, including FP4 at ~1723 against ~1725 TFLOP/s — two toolchains reaching the same silicon limit from different directions. The **Vector** rows diverge sharply, and that divergence is the useful result: Ascend C reaches 26.17 TFLOP/s BF16 where Triton reaches 8.88, because the in-place `asc_madd` form costs one fewer register move per update than the returning `asc_fma` spelling. Those gaps are compiler headroom, not hardware limits. See [`ascendc/RESULTS.md`](ascendc/RESULTS.md) for the full side-by-side table and [`ascendc/README.md`](ascendc/README.md) to build and run it.

`ascendc/bin/` is a build product and is not tracked in git; `cd ascendc && ./build.sh` recreates all 16 binaries in about a minute. Both suites write into the shared `results/` directory, the Ascend C tables under `ascend950dt_ascendc_*`.

## Running the Benchmarks

The historical H20 results below used PyTorch 2.11.0+cu130, Triton 3.6.0 (FlagTree), and FlagGems 5.4.0rc1. Install versions of PyTorch, Triton/FlagTree, and FlagGems that support the target chip.

Runtime operations use FlagGems backend management, with no direct calls to `torch.cuda`:

| Function | Shared interface |
| --- | --- |
| Device type and vendor detection | `flag_gems.device`, `flag_gems.vendor_name` |
| Device selection, property/memory queries, and synchronization | `flag_gems.runtime.torch_device_fn` |
| Device events, streams, and graphs | Interfaces such as `Event`, `Stream`, and `graph` exposed by that device module |
| Precision settings for reference computations | `flag_gems.runtime.torch_backend_device`, when the relevant controls are available |
| FP64/BF16 capabilities | Capability information from `flag_gems.runtime.device` |

Device type and vendor are handled separately: multiple vendors may use the `cuda` device name, and NVIDIA-specific options are enabled only when `vendor_name == "nvidia"`. FlagGems detects the hardware automatically by default. To select a vendor explicitly, use its environment variable, for example `GEMS_VENDOR=ascend python bench_vector.py --quick`. The corresponding device backend must already be installed; setting the environment variable does not emulate hardware.

The kernels still require the target Triton backend to support the selected operation and dtype. FP8 Tensor uses E4M3FN and is not automatically replaced with another FP8 format. TF32 requires an explicit precision mode and is not silently replaced with IEEE FP32, BF16, or FP16. [FlagGems backend interface documentation](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/README.md)

### Complete 17-metric suite by chip

Run from the repository root on an idle device. Each command below makes one `bench_suite.py` invocation that attempts all 17 metrics in the table above. Set `--device` to the target device index; for example, use `--device 1` for the second H20. These are full sweeps, not quick-mode smoke tests.

NVIDIA Hopper H20:

```bash
python bench_suite.py --device 0 --timing auto \
  --output results/h20_metrics.json
```

NVIDIA Hopper H100:

```bash
python bench_suite.py --device 0 --timing auto \
  --output results/h100_metrics.json
```

KunlunXin KL3/P800:

```bash
python bench_suite.py --device 0 --timing events \
  --output results/p800_metrics.json
```

T-Head PPU (FlagGems vendor `thead`, e.g. PPU-ZW810E):

```bash
python bench_suite.py --device 0 --timing auto \
  --output results/ppu_zw810e_metrics.json
```

Hygon DCU (FlagGems vendor `hygon`, e.g. BW1100/BW1101):

```bash
python bench_suite.py --device 0 --timing auto \
  --output results/bw1100_metrics.json
```

The scripts default `AMDGCN_USE_BUFFER_OPS` to `0` before importing the device
runtime, so Hygon commands need no shell environment prefix; an explicitly set
value is respected. FlagTree's `hcu` backend lowers masked `tl.load` to buffer
resources through `llvm.amdgcn.make.buffer.rsrc`, which the `clang-18` shipped
in DTK 26.04 rejects with `Intrinsic has incorrect argument type`. Every masked
load in this suite hits it, so the default would fail the whole table at compile
time. Buffer ops are an addressing optimization rather than an arithmetic path:
disabling them changes how operands are fetched, not which matrix or FMA
instruction executes, and the instruction audits still verify the arithmetic
that ran.

Other FlagGems-supported backends (replace the output name with your chip model):

```bash
python bench_suite.py --device 0 --timing auto \
  --output results/other_chip_metrics.json
```

The scripts default `XPU_EVENT_KL3_ENABLE` to `1` before importing the device runtime, so KunlunXin commands need no shell environment prefix. An explicitly set value is respected. A complete 17-row report does not guarantee 17 numeric peaks: an individual metric may be unsupported, not implemented, unverified, or fail at runtime.

The suite runs every metric in a separate child process, continues after unsupported formats or failures, and checkpoints the combined JSON after every row. Per-metric JSON and logs are stored in a directory named after the output stem, such as `results/p800_metrics_runs/`. The console prints the same fixed 17-row table.

Result status has a strict meaning:

- `ok`: a value is available. Compute measurements passed compilation, full correctness validation, device-event timing, and instruction-precision/execution audits. Memory capacity comes from the backend's reported properties.
- `unverified`: compute compilation, numerical validation, and timing succeeded, but native instruction precision or execution is unaudited. The measured value and median remain available; this status does not establish native arithmetic or tensor throughput. Verified IEEE FP32 matrix execution through NVIDIA SIMT remains `ok`, with an explicit SIMT note.
- `unsupported`: the selected PyTorch/FlagGems/backend stack explicitly lacks the requested storage or arithmetic capability, or the compiler/instruction audit rejects the requested native dtype. The numeric value remains `null`; the reason identifies which layer rejected it and does not automatically prove a hardware limit.
- `not_implemented`: the benchmark has no verified native path for the requested metric. This is a benchmark limitation and does not determine whether the chip supports it. The numeric value remains `null`.
- `error`: device discovery, memory, timing, correctness, or another operational failure. An error is not treated as evidence that the chip lacks the dtype.
- `not_measured`: no eligible HBM-sized copy result is available, because the run uses quick mode, buffers are too small, or L2 capacity is unknown. Observed copy/read rates remain in the auxiliary `patterns` details.
- `unavailable`: the backend did not report device-memory capacity.

Use quick mode to check integration only; its rates are not peak results:

```bash
python bench_suite.py --quick --timing events --warmup 2 --rep 5 --rounds 3 \
  --output results/chip_metrics_smoke.json
```

### Individual programs

```bash
# Tensor/matrix: one format per invocation.
python bench_tensor.py --dtype bf16
python bench_tensor.py --dtype fp16
python bench_tensor.py --dtype fp32
python bench_tensor.py --dtype tf32
python bench_tensor.py --dtype int8
python bench_tensor.py --dtype fp64
python bench_tensor.py --dtype fp8
python bench_tensor.py --dtype fp4

# Vector: one format per invocation.
python bench_vector.py --dtype bf16
python bench_vector.py --dtype fp16
python bench_vector.py --dtype fp32
python bench_vector.py --dtype int8
python bench_vector.py --dtype fp64
python bench_vector.py --dtype fp8
python bench_vector.py --dtype fp4

python bench_bandwidth.py --dtype fp32 --output results/peak_bandwidth.json
```

All programs support `--device 0`, `--output file.json`, and `--help`. Add `--quick --warmup 2 --rep 5 --rounds 3` for a smoke test; results from these small workloads are not used to establish peak throughput.

Missing device properties are not filled with specifications from another chip. If available memory is unknown, the preallocation check is skipped. If the compute-unit count is unknown, tensor/vector benchmarks accept `--compute-units count`; if L2 capacity is unknown, supply the actual value with `--l2-mib capacity`. The bandwidth benchmark does not depend on the compute-unit count. Measurements with unknown L2 capacity are still reported, but excluded from the HBM peak summary.

## 1. Matrix/Tensor: Repeated Multiply-Add with Resident Operands

`bench_tensor.py` defaults to `--mode peak`: each program loads its operands from device memory once, repeatedly executes `tl.dot` in a loop with two independent accumulator chains, and writes all results at the end. Increasing the ratio of computation to memory traffic and varying the tile shape and programs per SM helps find the throughput limit of the compute units. FP4 uses `tl.dot_scaled` in the same resident loop, with operands, scale factors, and both accumulators held across iterations.

- BF16, FP16, FP32, TF32, INT8, FP64, FP8 E4M3FN, and MX FP4 E2M1 are tested separately. BF16/FP16/FP32/TF32/FP8/FP4 use FP32 accumulators, INT8 uses INT32, and FP64 uses FP64.
- `--dtype fp32` selects FP32 input storage with IEEE precision. `--dtype tf32` is an alias for FP32 storage with TF32 dot precision; the older `--dtype fp32 --input-precision tf32` spelling remains supported. TF32 is a math mode, not a storage dtype.
- INT8 counts a multiply and add as two integer operations and reports TOP/s. All floating-point modes report TFLOP/s.
- IEEE FP32 may use scalar/vector FMA instructions (SIMT) on NVIDIA. Such results are labeled **IEEE FP32 matrix throughput**, without claiming a tensor-core peak. Backends with native FP32 matrix instructions may use those instead.
- The backend's native accumulation policy is used. FP32 output does not imply that internal accumulation always has full IEEE FP32 precision. On H20, forcing `max_num_imprecise_acc=0` falls back to FP16 instructions, so that path is not labeled as an FP8 peak measurement.
- All outputs are validated numerically. FP32 runs also perform an untimed precision probe using `1 + 2^-12`, `1 + 2^-9`, and `2^20` in `A` with an identity matrix `B`. The first value distinguishes IEEE FP32 from TF32, the second rejects a BF16 fallback, and the third rejects an FP16 fallback. This validates the requested math mode even though the exact-check inputs used for peak timing are representable in all of them. The run stops if compilation rejects the requested mode or this probe disagrees with it.
- The FP64 probe additionally carries a distinct marker value per row, so a backend that computes every product correctly but delivers it to the wrong output row is rejected by name. MetaX has exactly that defect; a probe with uniform rows cannot see it.
- Peak-mode outputs are checked against the exact scaled dot reference with `rtol=0` and an analytic `atol` bounding the accumulated accumulator rounding (`4 × iterations × block_k / 256 × unit_roundoff`). INT8 keeps exact equality. At the 1024-iteration default the FP32-accumulator bound is 128× smaller than the `1/256` lattice step of the exact-fraction operands, so a single dropped iteration, a merged accumulator chain, or an unwritten output is still rejected. Bit-exact equality is not required because some backends round inside the MMA itself: MetaX returns BF16/FP16/TF32 results a few `2^-33` off the exact lattice point. FP4 operands are drawn from the four E2M1 codes for `±0.5` and `±1.0`, so every exact entry is a multiple of `0.25` bounded by `block_k`; its bound is the same expression with that operand magnitude in place of `1/256`.
- Formats whose `tl.dot` the backend cannot lower are reported as `unsupported` from a small compile probe run before the sweep, rather than aborting the metric. Holding a dtype in memory does not imply a dot lowering exists for it: MetaX stores FP8 exactly but fails to lower any FP8 `tl.dot`.
- FP4 gets a stricter pre-sweep probe than the other formats, because compiling proves less for it. Its operands ride in a `uint8` container, so a backend that ignored the E2M1 interpretation and multiplied raw bytes, or that dropped the per-group scales, would still compile and still return plausible numbers. The probe multiplies known E2M1 codes and demands **exact** equality with the FP64 reference, then raises `A`'s scale from `2**0` to `2**1` (E8M0 code 127 → 128) and demands the result be exactly doubled. A backend that fails either check is reported `unsupported` with the specific defect named, never measured. The reference is computed from the FP64 values of the codes that were actually packed into the operands, so a packing mistake cannot cancel itself out between kernel and reference.
- NVIDIA/AMD instruction audits distinguish matrix instructions from SIMT FMA and verify the requested FP16, BF16, FP32, TF32, INT8, FP64, or FP8 operand format. A kernel lowered to another dtype is rejected for that metric. Other vendors retain the validated measurement with suite status `unverified` when instruction precision or execution cannot be audited. Missing register/spill attributes are recorded as `null`.
- Hygon's DCU is audited by the same AMDGCN branch: it is a separate FlagGems vendor with its own ISA dialect, but it compiles through the AMDGCN path and publishes final assembly under the same `amdgcn` key, so its arithmetic is real ISA evidence rather than IR. It spells its matrix instruction `v_mmac_*` where AMD spells it `v_mfma_*`/`v_wmma_*` — on BW1101 the audit reads `v_mmac_f32_16x16x16_f16`, `v_mmac_f32_16x16x16_bf16`, `v_mmac_i32_16x16x32_i8`, `v_mmac_f32_16x16x32_fp8_fp8` and `v_mmac_16x16x8_f32`, and FP64 as SIMT `v_fma_f64`. Hygon also omits the accumulator segment when it matches the operand type (`v_mmac_16x16x8_f32`), which is accepted only when the requested accumulator and operand agree, so the omission cannot match a different accumulator width.
- FP8 uses specially constructed inputs generated at runtime to retain strict numerical validation; structured sparsity instructions are not enabled. The default is 1024 repetitions, and both FP8 and FP4 cap `--iterations` at 4096 so the accumulation stays exactly representable. IEEE FP32 sweeps four smaller tile shapes. Other math modes use three shapes on NVIDIA/AMD and two smaller shapes on other vendors, combined with `1, 2, 4, 8 blocks/SM`. FP4's smallest `K` is 128 rather than the 64 the other narrow formats use: `K` must be a whole number of 32-element MX groups, and the Ascend MX path has no 64-element FP4 fractal, so it rejects `K=64` outright. Its tile list keeps the footprints of the other formats on that vendor branch and extends them to `K=512`, because FP4 packs two elements per byte: a deeper `K` costs the same operand bytes as half that `K` in FP8 while amortizing the MX scale fetch over more MACs. That matters — on Ascend950DT `K=512` measures about 1725 TFLOP/s against 1570 at `K=256`, a 10% difference that a `K=256`-only sweep would have silently reported as the chip's FP4 peak. The T-Head PPU backend instead sweeps four narrow-N shapes (`128×32` through `512×32`, plus `128×64`) with an extended `1..64 blocks/SM` occupancy sweep. Its throughput is governed by the accumulator footprint per warp: shapes holding `block_m × block_n / num_warps == 2048` output elements reach 120+ TFLOP/s FP16, while 4096 exhausts the 256-register file and collapses to about 20 TFLOP/s. Narrow N is the secondary factor (N=32 above N=64, above N=16 and N=128). The generic vendor set measures less than half of the achievable rate. `blocks/SM` is the ratio of grid size to compute-unit count and does not guarantee that all programs are resident simultaneously.

JSON records `input_storage_dtype`, `dot_input_precision`, and `math_mode` separately; FP4 rows additionally record `container_dtype`, `elements_per_container`, `scale_format`, `scale_group_size`, and `dot_operation`, so the `uint8` container is never mistaken for the arithmetic format. Compiled-kernel metadata distinguishes `execution_kind`, `instruction_precision_verified`, and `native_tensor_verified`: a successful SIMT instruction audit does not imply tensor execution. Compare peak results within the same math mode and execution category.

```text
FLOPs = 2 × tile_M × tile_N × tile_K × iterations × chains × blocks
Peak TFLOP/s = FLOPs / (min_ms × 10^9)
Median TFLOP/s = FLOPs / (median_ms × 10^9)
```

`--mode gemm` retains the large-matrix throughput test to measure performance under an application workload:

```bash
python bench_tensor.py --mode gemm --dtype fp16 --sizes 2048 4096 8192
python bench_tensor.py --mode gemm --dtype fp32 --input-precision ieee --sizes 2048 4096 8192
python bench_tensor.py --mode gemm --dtype fp32 --input-precision tf32 --sizes 2048 4096 8192
```

Peak mode supports all eight Tensor/matrix formats. GEMM supports BF16, FP16, FP32, TF32, INT8, and FP64; FP8 and FP4 remain peak-only. GEMM counts `2*M*N*K` floating-point or integer operations, excluding padding. Precision probes and NVIDIA/AMD instruction audits apply to both modes.

## 2. Vector: Typed Independent Arithmetic Chains

`bench_vector.py` supports BF16, FP16, FP32, INT8, FP64, FP8, and FP4 requests. It loads runtime inputs and coefficients, executes eight independent recurrence chains, and stores every result. The default is 4096 iterations per chain. Configurations with reported register spills are excluded from peak selection.

```bash
python bench_vector.py --dtype fp32 --iterations 4096 \
  --blocks-per-sm 8 16 --block-sizes 256 512 \
  --num-warps 4 8 --unroll 1 4 8
```

```text
elements_per_chain = SM/CU count × blocks_per_sm × block_size
floating FLOPs = 2 × 8 × elements_per_chain × iterations
NVIDIA INT8 DP4A operations = 8 × 8 × elements_per_chain × iterations
Other-backend typed INT8 MUL+ADD operations = 2 × 8 × elements_per_chain × iterations
```

Floating-point modes count one multiplication plus one addition as 2 FLOPs. NVIDIA audits the generated PTX for the requested native FMA dtype and rejects tensor instructions or widened FP16/BF16 arithmetic. NVIDIA INT8 uses and audits `dp4a`: four signed INT8 products plus four INT32 additions per update, counted as 8 operations. Other backends try typed INT8 input, coefficients, state, and output with modulo-256 multiply plus add, counted as 2 operations. Integer overflow checks are disabled only for those two intentional modular operations so they remain typed INT8 in Triton IR. The full output is validated exactly; without final ISA evidence of native INT8 arithmetic, the suite retains a numeric rate as `unverified`. An explicit compiler rejection is reported as `unsupported`. Other successfully measured Vector formats without a native instruction audit also receive `unverified`. Floating output validation is exact for BF16/FP16 and bounded against FP64 references for FP32/FP64. FP8 currently returns `not_implemented` because the benchmark has no verified native FP8 vector arithmetic path; FP8 storage or Tensor instructions are not counted as Vector FP8 throughput. FP4 returns `unsupported` for a stronger reason, and the two statuses are deliberately different: FP8 is a gap in this benchmark, whereas FP4 is not an arithmetic element type anywhere in the stack. Triton exposes no FP4 scalar dtype, so there is no FP4 `tl.fma`/`tl.mul` overload to lower and no FP4 multiply-add can be written at all; PyTorch's `float4_e2m1fn_x2` is a two-element container, not a scalar type. The refusal needs no device and records the checked facts in `device.vector_fp4_capability_probe`, so it can be re-verified from the result file. FP4 packing, conversions, or microscaled matrix instructions are not counted as Vector FP4 throughput: where a chip executes FP4 natively on its matrix unit, that rate is the `tensor_fp4` metric.

A backend can carry a dtype through loads, stores, and `*`/`+` while having no fused multiply-add lowering for it. Before sweeping, the benchmark compiles a small `tl.fma` probe for the requested dtype. MetaX fails this probe for BF16: the MACA backend represents BF16 as `i16`, which `llvm.intr.fma` rejects, so `tl.fma` fails in `ConvertTritonGPUToLLVM`.

When `tl.fma` does not lower for BF16 or FP16, the benchmark tries a second probe before giving up on fusion: an FP32-widened FMA that keeps the recurrence state in the narrow dtype. A backend with no native arithmetic for that type computes it by extending to FP32 anyway, so this is the same arithmetic the emulated path performs, with one rounding per update just like `tl.fma`. The loop-invariant coefficients are widened once outside the loop, which the MACA backend does not do on its own. That path records `operation: WIDENED_FMA`, `widened_fma_used: true`, and `roundings_per_update: 1`, and the result file is named `vector_<dtype>_widened_fma`. If this probe also fails, the sweep falls back to source multiply plus add with `roundings_per_update: 2`. `hardware_fusion_verified` stays false on both fallbacks.

On MetaX the benchmark audits the arithmetic width in the final LLVM IR, the last artifact that backend exposes. A BF16 kernel whose only arithmetic is `fmul float`/`fadd float`, or whose FMA intrinsic is `@llvm.mxc.pk.fma.f32`, reached FP32 execution units; that is recorded as `maca_arithmetic_promoted: true` and keeps the suite status at `unverified`. For BF16/FP16 the audit also counts `llvm.mxc.cvt.f32to{bf16,f16}` sites as `maca_narrowing_conversion_sites`: a nonzero count confirms every result was rounded back to the narrow dtype, rather than the state staying in FP32 and converting only once at the end. LLVM IR is not final ISA evidence, so this audit never sets `native_arithmetic_verified`, even when the IR width matches the request.

On Hygon the benchmark reads final AMDGCN assembly, so a match is real ISA evidence and does set `native_arithmetic_verified`. FP16, FP32 and FP64 reach native `v_fma_f16`/`v_fma_f32`/`v_fma_f64` and are reported `ok`. BF16 has no `v_fma_bf16` on this target: the compiler widens it to `v_fma_f32` while the recurrence state stays BF16. That follows the MetaX precedent rather than the NVIDIA one — the measurement is kept, because the arithmetic really ran on BF16 data, but it is recorded as `amdgcn_arithmetic_promoted: true` with the promoted mnemonic named, and the suite status stays `unverified` because it is not a native BF16 execution rate. INT8 has no DP4A here; the typed modulo-256 path lowers to 16-bit integer ops (`v_mul_lo_u16`, `v_mad_legacy_u16`) that carry the declared INT8 values in wider registers, so the rate is retained as `unverified` like the other non-DP4A vendors rather than counted as native INT8 arithmetic.

FlagTree's KunlunXin XPU backend reports clusters as `multiprocessor_count`, uses 64 cores per cluster, and does not use Triton's `num_warps` tuning parameter. The benchmark therefore starts with 1/2 programs per cluster and blocks of 1024/2048 FP32-sized elements, which provide 16/32 elements per core for its 512-bit vectorization path. Typed INT8 uses blocks of 4096/8192 so each core receives 64/128 elements, matching one or two 512-bit INT8 vectors. Floating kernels emit source multiply plus add operations so the XPU vectorizer can process and fuse them. The result remains unverified unless the final textual ISA contains an explicitly matching instruction type; intermediate typed-vector IR is recorded as supporting evidence, but is not treated as final ISA proof. XPU currently supplies placeholder zero register/spill counts, so the JSON records those fields as unknown.

```bash
python bench_vector.py --timing events \
  --output results/kunlunxin_vector.json
```

The legacy option name `--blocks-per-sm` means programs per reported compute unit; on XPU that unit is a cluster. If an older runtime cannot report this value, pass the verified cluster count with `--compute-units`.

## 3. Bandwidth: Streaming Reads and Copies

`bench_bandwidth.py` defaults to `--mode all`, sweeping `read` and `copy` separately. Default source-array sizes are `256, 512, 1024, 2048 MiB`, combined with different block/warp configurations.

- `copy`: executes `dst[i] = src[i]`, counts both source reads and destination writes, and checks equality for every element.
- `read`: reads all elements, reduces them within each block, and writes one FP32 result per block so the compiler cannot remove the loads. All reduction results are checked. Traffic includes source reads and the small result writes; reduction overhead is included in the measured time.
- The default arrays are much larger than H20's L2 cache. A result enters the peak summary for HBM-sized workloads only when the source array is at least 4 times the known L2 capacity and the run is not in smoke mode. Small arrays and results with unknown L2 capacity are excluded from that summary.

```bash
python bench_bandwidth.py --mode copy --sizes-mib 512 1024 2048
python bench_bandwidth.py --mode read --sizes-mib 512 1024 2048
```

```text
copy traffic = 2 × source-array bytes
read traffic = source-array bytes + reduction-result bytes
Peak GB/s = traffic / (min_ms × 10^6)
Median GB/s = traffic / (median_ms × 10^6)
TB/s = GB/s / 1000
```

MiB/GiB use powers of 1024; GB/s/TB/s use powers of 1000. In copy mode, `--sizes-mib 1024` means a 1 GiB source and a 1 GiB destination. Read and copy use different access patterns and should be compared separately. These measurements report effective global-memory bandwidth. Physical DRAM traffic may differ, and host-to-device or device-to-host PCIe/NVLink transfers are not included.

## How Is Time Measured?

Timing is implemented in `common.py`. All timestamps come from **the FlagGems device module's `Event` interface**; Python wall-clock timing is never used as a fallback.

The default `--timing auto` selects a method based on runtime capabilities. When a usable graph API is available, it uses the device module's exported `CUDAGraph`, `NPUGraph`, `MUSAGraph`, `MLUGraph`, or `Graph`. If graphs are absent, capture is explicitly unsupported, or graph replay still produces an unresolved zero timestamp at the maximum batch size, device events bracket a batch of ordinary launches. The selected method and fallback reason are displayed and saved in JSON. An explicit `--timing graph` request requires working graph timing and fails otherwise. Missing device-event timing support also produces an explicit error.

1. Execute the kernel once to complete JIT compilation, then synchronize the device. Data preparation and correctness checks finish before measurement.
2. Start with five ordinary launches to estimate kernel time. If the backend event timer reports zero, increase the calibration batch geometrically until the timer resolves it, up to 2048 launches. After any zero, the measured batch must span at least 20 times the first nonzero reading; this guards against accepting one coarse timer tick as a peak result. Build a batch of R launches, capturing it as a device graph when available, and calibrate R using the actual batch time. The default target is about 10 ms per batch. If 2048 launches still cannot provide a sufficiently resolved timestamp, stop instead of reporting inflated throughput.
3. Warm up the batch for approximately 100 ms by default.
4. On the same device stream, record the start event, execute the batch, and record the end event. Synchronize the FlagGems device API, then read the elapsed device time.
5. Each sample's per-kernel time is `device batch time / R`. By default, at least 10 batches are sampled, with a total timing budget of approximately 200 ms.
6. Calculate the minimum and median across batches. `min_ms` is the fastest **batch-average time**, not the duration of the shortest Python call. Report the highest throughput after sweeping all valid configurations.

Core logic:

```python
from flag_gems.runtime import torch_device_fn

start = torch_device_fn.Event(enable_timing=True)
end = torch_device_fn.Event(enable_timing=True)
start.record()                   # Timestamp on the selected device
submit_batch()                  # Graph replay or R ordinary launches
end.record()                     # Same device stream
torch_device_fn.synchronize()    # Make backend event timestamps visible
one_kernel_ms = start.elapsed_time(end) / R
```

Timing parameters:

```bash
python bench_vector.py --warmup 200 --rep 500 --batch-ms 10 --rounds 20
```

- Compilation, allocation, initialization, validation, graph capture, and warmup are excluded from the reported timing interval.
- Loads, stores, loop control, device scheduling, and amortized event overhead remain included. Graphs can reduce gaps between Python launches; ordinary event-timed batches may still include those gaps. Compare the two timing modes separately, especially for short kernels.
- Neither mode flushes caches between batches. Tensor/vector benchmarks use resident data to increase the share of computation; bandwidth benchmarks use working sets larger than the cache to avoid reporting L2 reuse as an HBM peak.
- JSON records the vendor, device type, FlagGems version, actual timing method, fallback reason, raw batch times, R, per-kernel times, quantiles, and validation results.
- `--timing events` now uses FlagGems device events around ordinary batches and no longer calls the earlier `triton.testing.do_bench` implementation.

For KunlunXin KL3/P800, the benchmark scripts set the FlagGems event switch before importing the device runtime. Run the program directly:

```bash
python bench_tensor.py --dtype fp16 --timing events
```

If a KunlunXin event still reports zero after the adaptive batch reaches 2048 launches, the benchmark stops with a diagnostic that asks you to verify the SDK, driver, and event implementation. It never substitutes a CPU clock or a fabricated epsilon duration.

The "measured peak" is the best result from the current sweep. It does not guarantee that every possible implementation has been tested, and it is not equivalent to the vendor's theoretical peak. Tests use the device's existing clock and power policies. Run them sequentially on an idle GPU and compare the fastest batch with the median. For CUDA event behavior and the definition of effective bandwidth, see [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#performance-metrics).

## Result Files

FlagGems regression records use `results/flaggems_*.json`. The four FP32/TF32 records documented below also use FlagGems. All other files matching `results/h20_peak_*.json` and `results/h20_gpu1_peak_*.json` contain peak measurements from before FlagGems integration. The original `h20_tensor.json`, `h20_vector.json`, and `h20_bandwidth.json` retain sustained-throughput results from the initial version. Interpret results in the context of each version's timing path, workload, and parameters. Historical H20 results do not imply that other chips have been tested.

Tables measured by the native Ascend C suite in `ascendc/` are named `results/ascend950dt_ascendc_metrics.json`, with per-metric records in `results/ascend950dt_ascendc_metrics_runs/`. The `_ascendc_` segment is what distinguishes them from this suite's tables for the same chip; a document's `benchmark` field also states which implementation produced it (`chip_metric_suite` versus `ascendc_chip_metric_suite`). Rows from the Ascend C suite can carry two statuses this suite never emits, `measured_widened` and `not_built`.

Three Ascend 950DT tables coexist and only one is current. `results/ascend950dt_17metrics.json` is this suite's present table and the one documented below. `results/ascend950dt_metrics.json` and `results/950dt_metrics.json` are earlier 15-row runs of this suite on the same chip, taken before the FP4 metrics existed; they are retained as history and are superseded row for row by the 17-row table. Check a document's row count, or whether it contains a `tensor_fp4` row at all, rather than relying on the file name.

### Backend Portability Validation

- CPU-only tests cover NPU/MLU/MUSA graph interfaces, non-NVIDIA vendors using the `cuda` device name, coarse and missing event timing, graph resolution and zero-time fallback, missing device properties, capability reporting, requested Tensor/Vector dtype policies, suite status aggregation, and error handling. Any direct access to `torch.cuda` fails immediately in the runtime tests.
- Separate NVIDIA SM90 offline-compilation tests check generated matrix/vector instruction classes, including the added Tensor INT8/FP64 and Vector BF16/FP16/INT8/FP64 paths. These tests do not execute accelerator kernels and are skipped when the NVIDIA compiler backend is unavailable.
- Historical H20 runs exercised FlagGems graph and ordinary event paths, FP16/FP8/IEEE FP32/TF32 matrix operations, GEMM tail tiles, FP32 vector arithmetic, copy/read modes, and CPU FP64 references. The stored historical results do not validate the newly added Tensor INT8/FP64 or Vector BF16/FP16/INT8/FP64 paths.
- CPU tests, offline compilation, and historical runs do not establish current hardware validation of the complete suite or validation on other vendors. Run the suite on the target device to obtain those measurements and retain its raw results and logs.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python bench_vector.py --quick --timing auto --warmup 2 --rep 5 --rounds 3
python bench_vector.py --quick --timing events --warmup 2 --rep 5 --rounds 3
```

### Ascend 950DT Complete Metric Table on 2026-11-05

Full 17-metric suite on one idle Ascend950DT_9582 of eight (64 compute units, 91.75 GiB, 128 MiB L2) with PyTorch 2.10.0, Triton 3.6.0 (triton-ascend), and FlagGems 5.5.0.dev0+gda6b2b845 (vendor `ascend`), using FlagGems graph execution and device-event timing. No metric ended in `error`; both rows without a value are explicit capability rejections diagnosed before any sweep, not failed measurements.

| Metric | Peak | Median | Unit | Status |
| --- | ---: | ---: | --- | --- |
| memory_capacity | 91.750 | - | GiB | `ok` |
| memory_bandwidth | 3264.695 | 3260.236 | GB/s | `ok` |
| tensor_bf16 | 430.348 | 430.267 | TFLOP/s | `unverified` |
| tensor_fp16 | 430.375 | 430.305 | TFLOP/s | `unverified` |
| tensor_fp32 | 26.965 | 26.959 | TFLOP/s | `unverified` |
| tensor_tf32 | 215.622 | 215.604 | TFLOP/s | `unverified` |
| tensor_int8 | 860.749 | 860.545 | TOP/s | `unverified` |
| tensor_fp64 | - | - | TFLOP/s | `unsupported` |
| tensor_fp8 | 860.751 | 860.618 | TFLOP/s | `unverified` |
| tensor_fp4 | **1725.270** | 1725.185 | TFLOP/s | `unverified` |
| vector_bf16 | 8.883 | 8.882 | TFLOP/s | `unverified` |
| vector_fp16 | 18.532 | 18.530 | TFLOP/s | `unverified` |
| vector_fp32 | 9.896 | 9.895 | TFLOP/s | `unverified` |
| vector_int8 | 2.465 | 2.465 | TOP/s | `unverified` |
| vector_fp64 | - | - | TFLOP/s | `unsupported` |
| vector_fp8 | 5.430 | 5.429 | TFLOP/s | `unverified` |
| vector_fp4 | - | - | TFLOP/s | `unsupported` |

Every compute row is `unverified` because no final-ISA instruction auditor exists for vendor `ascend`; the rates are measured and every configuration passed numerical validation. The copy figure above is the headline bandwidth; the auxiliary read rate on the same run was 3847.693 GB/s peak (3838.059 median).

The Tensor ladder matches the architecture's operand widths: FP4 at 1725 TFLOP/s is 2.0× FP8/INT8 (861), which is 2.0× BF16/FP16 (430), which is 2.0× TF32 (216). IEEE FP32 at 26.97 TFLOP/s sits far below TF32 because it is not a Cube-unit rate. The card's 32 Cube cores at 1650 MHz with 4096 MAC/cycle imply a 432.5 TFLOP/s BF16 ceiling, doubling per halved operand width to 1730.2 at FP4, so every row above is 99.5–99.7% of its architectural limit. FP4's selected configuration was tile `256×128×512` at 8 blocks/SM and 1024 iterations, with zero register spills; all 36 of its configurations validated exactly (`max_abs_error` 0).

`tensor_fp4` is measured through `tl.dot_scaled` on the MX path. Before the sweep, a probe confirmed the backend both computes true E2M1 products (exact against the FP64 reference) and applies the E8M0 group scales (raising `A`'s scale from `2**0` to `2**1` exactly doubled the result). The compiled kernel lowers to `hfusion.matmul_mx` with `lhsFormat`/`rhsFormat` 3, the backend's E2M1 code. FP4 needs `K ≥ 128`: at `K=64` the link fails on an undefined `_mlir_ciface_mma_tile_int8_t_to_float`, since this CANN build ships no 64-element FP4 fractal.

`vector_fp4` is `unsupported` rather than merely unmeasured, and this is a property of the format and toolchain rather than of this benchmark. Triton exposes no FP4 scalar dtype, so no FP4 multiply-add can be expressed at all. The hardware agrees: the Ascend C vector ISA headers give FP4 only conversion, load, store, and permute (`asc_e2m1x22bfloat16`, `asc_bfloat162e2m1x2_*`), while `reg_arith.h` and `reg_fused.h` — the vector arithmetic and FMA headers — list no FP4 type among their operands. The AIV unit can move and convert FP4 but cannot compute on it; the Cube unit is where this chip executes FP4, and that rate is `tensor_fp4`. `vector_fp8` is reported separately and differently: FP8 does keep typed `f8E4M3FN` arithmetic in the Ascend MLIR (64 sites in the selected kernel), so it retains a measured rate at `unverified`.

`tensor_fp64` and `vector_fp64` are `unsupported` because FlagGems reports no FP64 capability for this device.

```bash
python bench_suite.py --device 1 --timing auto \
  --output results/ascend950dt_17metrics.json
```

Raw records: [complete table](results/ascend950dt_17metrics.json), with per-metric JSON and logs in `results/ascend950dt_17metrics_runs/`.

This chip is also measured natively in [`ascendc/`](ascendc/README.md), and the comparison is in [`ascendc/RESULTS.md`](ascendc/RESULTS.md). Every Tensor row above is confirmed to three significant figures by that independent implementation, which is the strongest available evidence that these are real peaks rather than a shared mistake. The Vector rows are not: Ascend C reaches 26.17 TFLOP/s BF16 against the 8.883 above, 15.90 TOP/s INT8 against 2.465, and measures a `vector_fp4` rate of 8.91 TFLOP/s where Triton reports `unsupported`. The `vector_fp4` difference is a toolchain limit on both sides of a real hardware gap — Triton has no FP4 scalar dtype to express the multiply-add, and the AIV has no FP4 arithmetic instruction either, so the Ascend C figure is an explicitly *converted* path through BF16 rather than a native FP4 vector peak. The other Vector gaps are compiler headroom in the Triton path.

### T-Head PPU (PPU-ZW810E) Complete Metric Table on 2026-10-29

The PPU, MetaX and H20 tables below predate the FP4 metrics and were taken when the suite had 15 rows. Their `tensor_fp4` and `vector_fp4` rows were never run on those chips, so no FP4 claim is made for them either way.

Full 15-metric suite on one idle PPU-ZW810E of sixteen (64 compute units, 95.6 GiB, 64 MiB L2) with PyTorch 2.13.0+v0.1.0.ppu2.2.0, Triton 3.7.0+v0.1.0.ppu2.2.0 (T-Head PPU backend), and FlagGems 5.3.0rc2+v0.1.0.ppu2.2.0 (vendor `thead`), with FlagGems graph execution and device-event timing. No metric ended in `error`.

| Metric | Peak | Median | Unit | Status |
| --- | ---: | ---: | --- | --- |
| memory_capacity | 95.625 | - | GiB | `ok` |
| memory_bandwidth | 2140.326 | 2138.425 | GB/s | `ok` |
| tensor_bf16 | 127.197 | 125.710 | TFLOP/s | `unverified` |
| tensor_fp16 | 128.803 | 125.896 | TFLOP/s | `unverified` |
| tensor_fp32 | 27.264 | 26.444 | TFLOP/s | `unverified` |
| tensor_tf32 | 59.031 | 58.688 | TFLOP/s | `unverified` |
| tensor_int8 | 123.825 | 123.724 | TOP/s | `unverified` |
| tensor_fp64 | 6.928 | 6.926 | TFLOP/s | `unverified` |
| tensor_fp8 | - | - | TFLOP/s | `unsupported` |
| vector_bf16 | 54.625 | 54.614 | TFLOP/s | `unverified` |
| vector_fp16 | 55.336 | 55.328 | TFLOP/s | `unverified` |
| vector_fp32 | 27.678 | 27.672 | TFLOP/s | `unverified` |
| vector_int8 | 26.505 | 26.485 | TOP/s | `unverified` |
| vector_fp64 | 6.423 | 6.420 | TFLOP/s | `unverified` |
| vector_fp8 | - | - | TFLOP/s | `not_implemented` |

Every compute row is `unverified` because no instruction auditor exists for vendor `thead`. The compiled LLVM IR shows the `tl.dot` lowerings: BF16/FP16 use `ppu.mma.sync.aligned.m16n16k16...f32.f16.f16.f32`, TF32 uses `ppu.mma.sync.aligned.m16n16k8...f32.tf32.tf32.f32`, and INT8 uses `ppu.mma.sync.aligned.m16n16k32...s32.s8.s8.s32` — all tensor-core (AIU) instructions. IEEE FP32 and FP64 `tl.dot` lower to SIMT `llvm.fmuladd`/`llvm.fma` instead, so those rows are matrix throughput at the SIMT ceiling, not tensor-core peaks. TF32 measures half the FP16 rate because the m16n16k8 instruction carries half the MACs of m16n16k16 at the same instruction rate.

The PPU AIU throughput is governed by the accumulator footprint per warp, not by tile size alone. Every shape holding `block_m × block_n / num_warps == 2048` output elements measures 120+ TFLOP/s FP16 (`128×32×32` at 2 warps, `256×32×32` at 4, `512×32×64` at 8), while 4096 output elements per warp exhaust the 256-register file and collapse to about 20 TFLOP/s. Narrow N is the secondary factor: N=32 measures above N=64, which measures above N=16 and N=128. The generic vendor tile set (32×32 and 64×64 at up to 8 blocks/SM) measures 56 TFLOP/s FP16; the `thead` set reaches 129.2, which is 95% of the 135.3 TFLOP/s that the vendor's acblas library delivers through `torch.matmul` on the same card.

FP64 keeps the generic small-tile set: its wider accumulators spill on the narrow-N shapes, where the measured rate drops from 6.8 to 1.6 TFLOP/s.

```bash
python bench_suite.py --device 0 --timing auto \
  --output results/ppu_zw810e_metrics.json
```

Raw records: [complete table](results/ppu_zw810e_metrics.json), with per-metric JSON and logs in `results/ppu_zw810e_metrics_runs/`. Individual full sweeps: [FP16](results/ppu_peak_tensor_fp16.json), [BF16](results/ppu_peak_tensor_bf16.json), [TF32](results/ppu_peak_tensor_tf32.json), [INT8](results/ppu_peak_tensor_int8.json).

### MetaX C550 BF16 and FP32 Vector Measurements on 2026-10-13

Measured on an idle MetaX C550 (104 compute units, 64 GiB) with PyTorch 2.10.0+metax3.8.1.0, FlagTree 0.6.1+metax3.6 (Triton 3.6.0), FlagGems 5.3.5, and MACA 3.8.1.3. Both sweeps used FlagGems graph execution with device-event timing and 4096 iterations. All 24 BF16 configurations passed numerical validation exactly (`max_abs_error` 0) with zero register spills and no excluded configurations.

| Math mode | Source operation | Peak | Median at that configuration | Suite status |
| --- | --- | ---: | ---: | --- |
| Vector BF16 | FP32-widened FMA, BF16 state | **2.784 TFLOP/s** | 2.784 TFLOP/s | `unverified` |
| Vector FP32 | `tl.fma` | **35.988 TFLOP/s** | 35.981 TFLOP/s | `unverified` |

Selected BF16 configuration: 16 blocks/unit, block 512, unroll 8, 4 warps.

This chip has no native BF16 ALU, so BF16 remains far below FP32. `mxcc` carries `__builtin_mxc_fma_bf16` and `__builtin_mxc_pk_fma_bf16`, but both require target feature `xcore1500` or newer; C550 reports `xcore1002`. The vendor toolchain confirms the consequence: compiling MACA's own `__hfma`/`__hfma2` for `__maca_bfloat16` with `mxcc --offload-arch=xcore1002` emits `@llvm.fma.f32`, while the same code for `__half` emits native `@llvm.fma.f16` and `@llvm.fma.v2f16`.

Every BF16 result therefore has to pass through one `llvm.mxc.cvt.f32tobf16`, and that conversion rather than the FP32 multiply-add is the limit. Keeping the recurrence state in FP32 and converting only once at the end reaches 35.64 TFLOP/s, within 1% of the measured FP32 rate, which shows the FP32 units are not the bottleneck. That variant is FP32 arithmetic with BF16 storage, so it is not reported as a BF16 rate. The 2.784 TFLOP/s figure measures BF16-typed work executed on FP32 units with the BF16 state preserved every iteration (129 narrowing conversion sites in the selected kernel), and is reported as `unverified` rather than `ok`.

Two MACA backend defects were found and worked around while taking this measurement; both are written up in [flagtree_metax_bug.md](flagtree_metax_bug.md). `tl.fma` on BF16 fails to lower at all, and the backend does not hoist the loop-invariant BF16 operand widening out of the loop. Addressing the second in the Triton source raised the measurement from 1.296 to 2.784 TFLOP/s with identical, exactly validated numerics.

```bash
python bench_vector.py --device 0 --dtype bf16 --timing auto \
  --output results/c550_vector_bf16.json
python bench_vector.py --device 1 --dtype fp32 --timing auto \
  --output results/c550_vector_fp32.json
```

Raw records: [BF16 full sweep](results/c550_vector_bf16.json), [FP32 full sweep](results/c550_vector_fp32.json).

### MetaX C550 Complete Metric Table on 2026-10-13

Full 15-metric suite on one idle C550 of eight, same software stack as above, with FlagGems graph execution and device-event timing. No metric ended in `error`; the two without a value are explicit backend rejections, diagnosed by a probe before the sweep rather than by a failed measurement.

| Metric | Peak | Median | Unit | Status |
| --- | ---: | ---: | --- | --- |
| memory_capacity | 63.594 | - | GiB | `ok` |
| memory_bandwidth | 1526.034 | 1525.444 | GB/s | `ok` |
| tensor_bf16 | 121.764 | 121.667 | TFLOP/s | `unverified` |
| tensor_fp16 | 121.755 | 121.690 | TFLOP/s | `unverified` |
| tensor_fp32 | 35.951 | 35.948 | TFLOP/s | `unverified` |
| tensor_tf32 | 100.838 | 100.793 | TFLOP/s | `unverified` |
| tensor_int8 | 101.349 | 101.317 | TOP/s | `unverified` |
| tensor_fp64 | - | - | TFLOP/s | `unsupported` |
| tensor_fp8 | - | - | TFLOP/s | `unsupported` |
| vector_bf16 | 2.784 | 2.784 | TFLOP/s | `unverified` |
| vector_fp16 | 9.081 | 9.077 | TFLOP/s | `unverified` |
| vector_fp32 | 35.990 | 35.982 | TFLOP/s | `unverified` |
| vector_int8 | 9.081 | 9.078 | TOP/s | `unverified` |
| vector_fp64 | 2.264 | 2.263 | TFLOP/s | `unverified` |
| vector_fp8 | - | - | TFLOP/s | `not_implemented` |

Peak values are reproducible to within 0.03% across three independent full-suite runs.

Every Tensor and Vector row is `unverified` because the MACA backend has no final-ISA disassembler here, so native instruction precision and execution unit cannot be audited; the numeric rates are measured and validated. `tensor_fp64` and `tensor_fp8` are `unsupported` because of MACA defects 3 and 4 in [flagtree_metax_bug.md](flagtree_metax_bug.md): FP64 `tl.dot` returns correct products in permuted output rows, and no FP8 `tl.dot` lowers. Vector FP64 is unaffected and measures normally, since it needs no `tl.dot`.

```bash
python bench_suite.py --device 5 --timing auto \
  --output results/metax_c550_metrics.json
```

Raw records: [complete table](results/metax_c550_metrics.json), with per-metric JSON and logs in `results/metax_c550_metrics_runs/`.

### H20 IEEE FP32 and TF32 Measurements on 2026-09-16

GPU 1 was idle before the run and was selected from eight H20 devices. Both full sweeps used FlagGems graph execution with device-event timing, 1024 resident dot iterations, 100 ms of warmup, an approximately 200 ms measurement budget, and at least 10 measured batches. All 28 full-sweep configurations passed the numerical checks, precision probe, and instruction audit with zero register spills. The selected configurations were then repeated independently with 4096 iterations.

| Math mode | Full-sweep peak | Full-sweep median at that configuration | 4096-iteration peak | 4096-iteration median |
| --- | ---: | ---: | ---: | ---: |
| IEEE FP32 matrix (`fma.rn.f32`, SIMT) | 28.79 TFLOP/s | 28.78 TFLOP/s | **28.81 TFLOP/s** | 28.80 TFLOP/s |
| TF32 tensor (`wgmma...tf32`) | 70.39 TFLOP/s | 70.36 TFLOP/s | **70.62 TFLOP/s** | 70.59 TFLOP/s |

The IEEE FP32 path compiled to ordinary FP32 FMA instructions on this H20, so it is matrix throughput rather than a native tensor-core result. Its selected configuration was tile `64×64×32`, `8 blocks/SM`, and 4096 iterations. TF32 compiled to `wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32`; its selected configuration was tile `64×128×32`, `2 blocks/SM`, and 4096 iterations.

```bash
python bench_tensor.py --device 1 --dtype fp32 --input-precision ieee \
  --output results/h20_peak_tensor_fp32_ieee.json
python bench_tensor.py --device 1 --dtype fp32 --input-precision tf32 \
  --output results/h20_peak_tensor_tf32.json

python bench_tensor.py --device 1 --dtype fp32 --input-precision ieee \
  --iterations 4096 --blocks-per-sm 8 \
  --output results/h20_selected_tensor_fp32_ieee.json
python bench_tensor.py --device 1 --dtype fp32 --input-precision tf32 \
  --iterations 4096 --blocks-per-sm 2 \
  --output results/h20_selected_tensor_tf32.json
```

Raw records: [IEEE FP32 full sweep](results/h20_peak_tensor_fp32_ieee.json), [TF32 full sweep](results/h20_peak_tensor_tf32.json), [IEEE FP32 selected validation](results/h20_selected_tensor_fp32_ieee.json), and [TF32 selected validation](results/h20_selected_tensor_tf32.json).

### Historical H20 Measurements on 2026-09-15

These historical records cover the original FP16/BF16/FP8 tensor modes. They predate the IEEE FP32 and TF32 matrix modes and do not include measurements for those additions.

GPU 0 completed a full sweep of 108 configurations: 12 for each of three tensor precisions, 24 for vector throughput, and 48 for bandwidth. All passed validation. GPU 1 ran 18 additional checks of selected configurations, using a longer 4096-iteration loop for tensor measurements. Device indices, parameters, and raw timing for every batch are saved in JSON.

| Metric | GPU 0 full-sweep peak | GPU 1 selected-configuration peak | GPU 1 median throughput for the same configuration |
| --- | ---: | ---: | ---: |
| Tensor FP16 | 130.13 TFLOP/s | 141.12 TFLOP/s | 141.07 TFLOP/s |
| Tensor BF16 | 130.04 TFLOP/s | 140.61 TFLOP/s | 140.37 TFLOP/s |
| Tensor FP8 | 259.95 TFLOP/s | 282.63 TFLOP/s | 282.60 TFLOP/s |
| Vector FP32 | 36.04 TFLOP/s | 39.00 TFLOP/s | 38.99 TFLOP/s |
| Device-memory read | 3.874 TB/s | 3.872 TB/s | 3.871 TB/s |
| Device-memory copy, reads and writes combined | 3.667 TB/s | 3.668 TB/s | 3.667 TB/s |

These values are the highest batch-average throughputs among the tested configurations. SM clock snapshots for the two cards showed 1830 MHz and 1980 MHz, respectively. That observation is consistent with the difference in compute throughput, but the snapshots are not continuous clock measurements over the entire test. Tests used the existing device settings.

Commands to reproduce the GPU 1 configurations:

```bash
python bench_tensor.py --device 1 --dtype fp16 --iterations 4096 --blocks-per-sm 2
python bench_tensor.py --device 1 --dtype bf16 --iterations 4096 --blocks-per-sm 2
python bench_tensor.py --device 1 --dtype fp8 --iterations 4096 --blocks-per-sm 2
python bench_vector.py --device 1 --blocks-per-sm 16 --block-sizes 512 --num-warps 4 --unroll 1
python bench_bandwidth.py --device 1 --sizes-mib 2048 --block-sizes 1024 16384 --num-warps 4 8
```

GPU 1 raw records: [FP16](results/h20_gpu1_peak_tensor_fp16.json), [BF16](results/h20_gpu1_peak_tensor_bf16.json), [FP8](results/h20_gpu1_peak_tensor_fp8.json), [Vector](results/h20_gpu1_peak_vector.json), [Bandwidth](results/h20_gpu1_peak_bandwidth.json).
