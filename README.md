# Triton Chip Peak Throughput Benchmarks

The suite measures device-memory capacity and bandwidth plus Tensor and Vector throughput across the requested numeric formats. It sweeps configurations to find **the highest throughput observed on the current device under the current operating conditions**, while reporting the median for the same configuration to help assess stability. Precision, execution units, clock frequency, and access patterns affect peak throughput, so each result retains its dtype, operation, validation, and instruction-audit metadata.

The complete report contains these 15 rows:

| Group | Metrics |
| --- | --- |
| Memory | Capacity (GiB), effective copy bandwidth (GB/s) |
| Tensor/matrix | BF16, FP16, FP32, TF32, INT8, FP64, FP8 |
| Vector | BF16, FP16, FP32, INT8, FP64, FP8 |

Floating-point throughput uses TFLOP/s. INT8 uses TOP/s. A missing value is never replaced by zero.

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

### Complete table

Run the production sweep on an idle device:

```bash
cd /workspace/triton_chip_benchmarks
python bench_suite.py --device 0 --output results/chip_metrics.json
```

For KunlunXin KL3/P800, enable its event implementation before Python starts:

```bash
XPU_EVENT_KL3_ENABLE=1 python bench_suite.py --device 0 \
  --timing events --output results/chip_metrics.json
```

The suite runs every metric in a separate child process, continues after unsupported formats or failures, and checkpoints the combined JSON after every row. Per-metric JSON and logs are stored in `results/chip_metrics_runs/` by default. The console prints the same fixed 15-row table.

Result status has a strict meaning:

- `ok`: a value is available. Compute measurements passed compilation, full correctness validation, device-event timing, and instruction-precision/execution audits. Memory capacity comes from the backend's reported properties.
- `unverified`: compute compilation, numerical validation, and timing succeeded, but native instruction precision or execution is unaudited. The measured value and median remain available; this status does not establish native arithmetic or tensor throughput. Verified IEEE FP32 matrix execution through NVIDIA SIMT remains `ok`, with an explicit SIMT note.
- `unsupported`: FlagGems reports the capability as absent, the compiler/instruction audit explicitly rejects the requested native dtype, or the benchmark lacks a verified native path. The numeric value remains `null`; the reason distinguishes a hardware/backend rejection from a benchmark-path limitation.
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

# Vector: one format per invocation.
python bench_vector.py --dtype bf16
python bench_vector.py --dtype fp16
python bench_vector.py --dtype fp32
python bench_vector.py --dtype int8
python bench_vector.py --dtype fp64
python bench_vector.py --dtype fp8

python bench_bandwidth.py --dtype fp32 --output results/peak_bandwidth.json
```

All programs support `--device 0`, `--output file.json`, and `--help`. Add `--quick --warmup 2 --rep 5 --rounds 3` for a smoke test; results from these small workloads are not used to establish peak throughput.

Missing device properties are not filled with specifications from another chip. If available memory is unknown, the preallocation check is skipped. If the compute-unit count is unknown, tensor/vector benchmarks accept `--compute-units count`; if L2 capacity is unknown, supply the actual value with `--l2-mib capacity`. The bandwidth benchmark does not depend on the compute-unit count. Measurements with unknown L2 capacity are still reported, but excluded from the HBM peak summary.

## 1. Matrix/Tensor: Repeated Multiply-Add with Resident Operands

`bench_tensor.py` defaults to `--mode peak`: each program loads its operands from device memory once, repeatedly executes `tl.dot` in a loop with two independent accumulator chains, and writes all results at the end. Increasing the ratio of computation to memory traffic and varying the tile shape and programs per SM helps find the throughput limit of the compute units.

- BF16, FP16, FP32, TF32, INT8, FP64, and FP8 E4M3FN are tested separately. BF16/FP16/FP32/TF32/FP8 use FP32 accumulators, INT8 uses INT32, and FP64 uses FP64.
- `--dtype fp32` selects FP32 input storage with IEEE precision. `--dtype tf32` is an alias for FP32 storage with TF32 dot precision; the older `--dtype fp32 --input-precision tf32` spelling remains supported. TF32 is a math mode, not a storage dtype.
- INT8 counts a multiply and add as two integer operations and reports TOP/s. All floating-point modes report TFLOP/s.
- IEEE FP32 may use scalar/vector FMA instructions (SIMT) on NVIDIA. Such results are labeled **IEEE FP32 matrix throughput**, without claiming a tensor-core peak. Backends with native FP32 matrix instructions may use those instead.
- The backend's native accumulation policy is used. FP32 output does not imply that internal accumulation always has full IEEE FP32 precision. On H20, forcing `max_num_imprecise_acc=0` falls back to FP16 instructions, so that path is not labeled as an FP8 peak measurement.
- All outputs are validated numerically. FP32 runs also perform an untimed precision probe using `1 + 2^-12`, `1 + 2^-9`, and `2^20` in `A` with an identity matrix `B`. The first value distinguishes IEEE FP32 from TF32, the second rejects a BF16 fallback, and the third rejects an FP16 fallback. This validates the requested math mode even though the exact-check inputs used for peak timing are representable in all of them. The run stops if compilation rejects the requested mode or this probe disagrees with it.
- NVIDIA/AMD instruction audits distinguish matrix instructions from SIMT FMA and verify the requested FP16, BF16, FP32, TF32, INT8, FP64, or FP8 operand format. A kernel lowered to another dtype is rejected for that metric. Other vendors retain the validated measurement with suite status `unverified` when instruction precision or execution cannot be audited. Missing register/spill attributes are recorded as `null`.
- FP8 uses specially constructed inputs generated at runtime to retain strict numerical validation; structured sparsity instructions are not enabled. The default is 1024 repetitions. IEEE FP32 sweeps four smaller tile shapes. Other math modes use three shapes on NVIDIA/AMD and two smaller shapes on other vendors, combined with `1, 2, 4, 8 blocks/SM`. `blocks/SM` is the ratio of grid size to compute-unit count and does not guarantee that all programs are resident simultaneously.

JSON records `input_storage_dtype`, `dot_input_precision`, and `math_mode` separately. Compiled-kernel metadata distinguishes `execution_kind`, `instruction_precision_verified`, and `native_tensor_verified`: a successful SIMT instruction audit does not imply tensor execution. Compare peak results within the same math mode and execution category.

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

Peak mode supports all seven Tensor/matrix formats. GEMM supports BF16, FP16, FP32, TF32, INT8, and FP64; FP8 remains peak-only. GEMM counts `2*M*N*K` floating-point or integer operations, excluding padding. Precision probes and NVIDIA/AMD instruction audits apply to both modes.

## 2. Vector: Typed Independent Arithmetic Chains

`bench_vector.py` supports BF16, FP16, FP32, INT8, FP64, and FP8 requests. It loads runtime inputs and coefficients, executes eight independent recurrence chains, and stores every result. The default is 4096 iterations per chain. Configurations with reported register spills are excluded from peak selection.

```bash
python bench_vector.py --dtype fp32 --iterations 4096 \
  --blocks-per-sm 8 16 --block-sizes 256 512 \
  --num-warps 4 8 --unroll 1 4 8
```

```text
elements_per_chain = SM/CU count × blocks_per_sm × block_size
floating FLOPs = 2 × 8 × elements_per_chain × iterations
NVIDIA INT8 DP4A operations = 8 × 8 × elements_per_chain × iterations
XPU typed INT8 MUL+ADD operations = 2 × 8 × elements_per_chain × iterations
```

Floating-point modes count one multiplication plus one addition as 2 FLOPs. NVIDIA audits the generated PTX for the requested native FMA dtype and rejects tensor instructions or widened FP16/BF16 arithmetic. NVIDIA INT8 uses and audits `dp4a`: four signed INT8 products plus four INT32 additions per update, counted as 8 operations. XPU INT8 instead keeps input, coefficients, state, and output in INT8 and executes modulo-256 multiply plus add, counted as 2 operations. Integer overflow checks are disabled only for those two intentional modular operations so they remain typed INT8 in Triton IR. The full output is validated exactly; without explicit final XPU ISA evidence, the suite retains the rate as `unverified`. Other non-NVIDIA backends currently return `unsupported` for Vector INT8 because the benchmark has no verified typed path there; this is a benchmark limitation and does not claim that the chip lacks INT8 support. Other successfully measured Vector formats without a native instruction audit also receive `unverified`. Floating output validation is exact for BF16/FP16 and bounded against FP64 references for FP32/FP64. FP8 currently returns `unsupported` because the benchmark has no verified native FP8 vector arithmetic path; FP8 storage or Tensor instructions are not counted as Vector FP8 throughput.

FlagTree's KunlunXin XPU backend reports clusters as `multiprocessor_count`, uses 64 cores per cluster, and does not use Triton's `num_warps` tuning parameter. The benchmark therefore starts with 1/2 programs per cluster and blocks of 1024/2048 FP32-sized elements, which provide 16/32 elements per core for its 512-bit vectorization path. Typed INT8 uses blocks of 4096/8192 so each core receives 64/128 elements, matching one or two 512-bit INT8 vectors. Floating kernels emit source multiply plus add operations so the XPU vectorizer can process and fuse them. The result remains unverified unless the final textual ISA contains an explicitly matching instruction type; intermediate typed-vector IR is recorded as supporting evidence, but is not treated as final ISA proof. XPU currently supplies placeholder zero register/spill counts, so the JSON records those fields as unknown.

```bash
XPU_EVENT_KL3_ENABLE=1 python bench_vector.py --timing events \
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

FlagGems' KunlunXin KL3/P800 backend setup exports the following event switch. Use the same setting before Python imports the device runtime:

```bash
XPU_EVENT_KL3_ENABLE=1 python bench_tensor.py --dtype fp16
```

If a KunlunXin event still reports zero after the adaptive batch reaches 2048 launches, the benchmark stops with a diagnostic that asks you to verify the SDK, driver, and event implementation. It never substitutes a CPU clock or a fabricated epsilon duration.

The "measured peak" is the best result from the current sweep. It does not guarantee that every possible implementation has been tested, and it is not equivalent to the vendor's theoretical peak. Tests use the device's existing clock and power policies. Run them sequentially on an idle GPU and compare the fastest batch with the median. For CUDA event behavior and the definition of effective bandwidth, see [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#performance-metrics).

## Result Files

FlagGems regression records use `results/flaggems_*.json`. The four FP32/TF32 records documented below also use FlagGems. All other files matching `results/h20_peak_*.json` and `results/h20_gpu1_peak_*.json` contain peak measurements from before FlagGems integration. The original `h20_tensor.json`, `h20_vector.json`, and `h20_bandwidth.json` retain sustained-throughput results from the initial version. Interpret results in the context of each version's timing path, workload, and parameters. Historical H20 results do not imply that other chips have been tested.

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
