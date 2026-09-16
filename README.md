# Triton Chip Peak Throughput Benchmarks

Three standalone programs measure matrix/tensor throughput, vector FP32 multiply-add throughput, and HBM/device-memory bandwidth. They sweep configurations to find **the highest throughput observed on the current device under the current operating conditions**, while reporting the median for the same configuration to help assess stability. Precision, execution units, dense or sparse execution, clock frequency, and access patterns affect peak throughput, so these measurements are reported separately.

## Running the Benchmarks

Local environment: NVIDIA H20, PyTorch 2.11.0+cu130, Triton 3.6.0 (FlagTree), and FlagGems 5.4.0rc1. Install versions of PyTorch, Triton/FlagTree, and FlagGems that support the target chip.

Runtime operations use FlagGems backend management, with no direct calls to `torch.cuda`:

| Function | Shared interface |
| --- | --- |
| Device type and vendor detection | `flag_gems.device`, `flag_gems.vendor_name` |
| Device selection, property/memory queries, and synchronization | `flag_gems.runtime.torch_device_fn` |
| Device events, streams, and graphs | Interfaces such as `Event`, `Stream`, and `graph` exposed by that device module |
| Precision settings for reference computations | `flag_gems.runtime.torch_backend_device`, when the relevant controls are available |
| FP64/BF16 capabilities | Capability information from `flag_gems.runtime.device` |

Device type and vendor are handled separately: multiple vendors may use the `cuda` device name, and NVIDIA-specific options are enabled only when `vendor_name == "nvidia"`. FlagGems detects the hardware automatically by default. To select a vendor explicitly, use its environment variable, for example `GEMS_VENDOR=ascend python bench_vector.py --quick`. The corresponding device backend must already be installed; setting the environment variable does not emulate hardware.

The same programs can use the device module detected by FlagGems. The kernels still require the target Triton backend to support operations such as `tl.dot` and `tl.fma`, along with the selected dtype and input-precision mode. FP8 uses E4M3FN and is not automatically replaced with another FP8 format. TF32 requires backend support and is not silently replaced with IEEE FP32 or another math mode. Validation currently includes H20 hardware regression tests and simulated runtime tests without CUDA. Kernels have not yet been validated on other chips. [FlagGems backend interface documentation](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/README.md)

```bash
cd /workspace/triton_chip_benchmarks

# Run sequentially to avoid contention on the same GPU.
python bench_tensor.py --dtype fp16 --output results/peak_tensor_fp16.json
python bench_tensor.py --dtype bf16 --output results/peak_tensor_bf16.json
python bench_tensor.py --dtype fp8  --output results/peak_tensor_fp8.json
python bench_tensor.py --dtype fp32 --input-precision ieee --output results/peak_matrix_fp32_ieee.json
python bench_tensor.py --dtype fp32 --input-precision tf32 --output results/peak_tensor_tf32.json
python bench_vector.py --output results/peak_vector.json
python bench_bandwidth.py --dtype fp32 --output results/peak_bandwidth.json
```

All programs support `--device 0`, `--output file.json`, and `--help`. Add `--quick --warmup 2 --rep 5 --rounds 3` for a smoke test; results from these small workloads are not used to establish peak throughput.

Missing device properties are not filled with specifications from another chip. If available memory is unknown, the preallocation check is skipped. If the compute-unit count is unknown, tensor/vector benchmarks accept `--compute-units count`; if L2 capacity is unknown, supply the actual value with `--l2-mib capacity`. The bandwidth benchmark does not depend on the compute-unit count. Measurements with unknown L2 capacity are still reported, but excluded from the HBM peak summary.

## 1. Matrix/Tensor: Repeated Multiply-Add with Resident Operands

`bench_tensor.py` defaults to `--mode peak`: each program loads its operands from device memory once, repeatedly executes `tl.dot` in a loop with two independent accumulator chains, and writes all results at the end. Increasing the ratio of computation to memory traffic and varying the tile shape and programs per SM helps find the throughput limit of the compute units.

- FP16, BF16, FP8 E4M3FN, IEEE FP32, and TF32 are tested separately; all outputs are FP32.
- `--dtype fp32` selects FP32 input storage and defaults to `--input-precision ieee`. Use `--dtype fp32 --input-precision tf32` for TF32 multiplication with FP32 accumulation. TF32 is a math mode, not a separate storage dtype. `--input-precision` is valid only with `--dtype fp32`.
- TF32 applies only to matrix `tl.dot` operations. The vector benchmark remains IEEE FP32 FMA, and the bandwidth benchmark moves FP32 storage when `--dtype fp32` is selected.
- IEEE FP32 may use scalar/vector FMA instructions (SIMT) on NVIDIA. Such results are labeled **IEEE FP32 matrix throughput**, without claiming a tensor-core peak. Backends with native FP32 matrix instructions may use those instead.
- The backend's native accumulation policy is used. FP32 output does not imply that internal accumulation always has full IEEE FP32 precision. On H20, forcing `max_num_imprecise_acc=0` falls back to FP16 instructions, so that path is not labeled as an FP8 peak measurement.
- All outputs are validated numerically. FP32 runs also perform an untimed precision probe using `1 + 2^-12`, `1 + 2^-9`, and `2^20` in `A` with an identity matrix `B`. The first value distinguishes IEEE FP32 from TF32, the second rejects a BF16 fallback, and the third rejects an FP16 fallback. This validates the requested math mode even though the exact-check inputs used for peak timing are representable in all of them. The run stops if compilation rejects the requested mode or this probe disagrees with it.
- NVIDIA/AMD instruction audits distinguish matrix instructions from SIMT FMA and check the requested FP32 precision. A native TF32 result requires recognized TF32/XF32 matrix instructions; an FP8 kernel lowered to another dtype is not labeled as a native FP8 peak. For other vendors, the FP32 precision probe still runs, but execution units and instruction precision are marked unverified. Missing register/spill attributes are recorded as `null`.
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

Both `--mode peak` and `--mode gemm` support IEEE FP32 and TF32. GEMM also supports FP16/BF16; FP8 remains available only in peak mode. GEMM counts `2*M*N*K` FLOPs, excluding extra operations caused by padding. FP32 precision probes and NVIDIA/AMD instruction audits apply to both modes. GEMM uses a larger numerical tolerance for TF32 to account for its reduced input precision.

## 2. Vector: Independent FP32 FMA Chains

`bench_vector.py` loads independent random initial values and runtime coefficients, executes 8 FP32 FMA chains in registers, and stores all results. The default is 4096 iterations per chain, sweeping block sizes, warp counts, and explicit loop unroll factors across 24 configurations. Configurations with register spills are excluded from peak selection.

```bash
python bench_vector.py --iterations 4096 --blocks-per-sm 8 16 --block-sizes 256 512 \
  --num-warps 4 8 --unroll 1 4 8
```

```text
elements_per_chain = SM/CU count × blocks_per_sm × block_size
FLOPs = 2 × 8 × elements_per_chain × iterations
```

One FMA performs one multiplication and one addition, counting as 2 FLOPs. Each unroll factor must divide the iteration count exactly. Unrolling changes loop-control overhead without increasing the total number of counted iterations. All outputs are checked against an analytic FP64 reference. On NVIDIA, the program also confirms that PTX contains FP32 FMA instructions and no tensor instructions. Static instruction counts cannot be treated directly as dynamic execution counts. When FlagGems reports that the device does not support FP64, vector and read modes first copy bounded chunks to the CPU for FP64 validation. Both the copies and validation are excluded from timing.

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

The default `--timing auto` selects a method based on runtime capabilities. When a usable graph API is available, it uses the device module's exported `CUDAGraph`, `NPUGraph`, `MUSAGraph`, `MLUGraph`, or `Graph`. If graphs are absent or capture is explicitly unsupported, device events bracket a batch of ordinary launches. The selected method and fallback reason are displayed and saved in JSON. An explicit `--timing graph` request requires graph support and fails otherwise. Missing device-event timing support also produces an explicit error.

1. Execute the kernel once to complete JIT compilation, then synchronize the device. Data preparation and correctness checks finish before measurement.
2. Build a batch of R kernel launches, capturing it as a device graph when available. Obtain an initial estimate, then calibrate R using the actual batch time. The default target is about 10 ms per batch, with at most 2048 launches.
3. Warm up the batch for approximately 100 ms by default.
4. On the same device stream, record the start event, execute the batch, and record the end event. Wait for the end event to complete, then read the elapsed device time.
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
end.synchronize()                # Use device synchronize if events lack this method
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

The "measured peak" is the best result from the current sweep. It does not guarantee that every possible implementation has been tested, and it is not equivalent to the vendor's theoretical peak. Tests use the device's existing clock and power policies. Run them sequentially on an idle GPU and compare the fastest batch with the median. For CUDA event behavior and the definition of effective bandwidth, see [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#performance-metrics).

## Result Files

FlagGems regression records use `results/flaggems_*.json`. Files matching `results/h20_peak_*.json` and `results/h20_gpu1_peak_*.json` contain peak measurements from before FlagGems integration. The older `h20_tensor.json`, `h20_vector.json`, and `h20_bandwidth.json` retain sustained-throughput results from the initial version. Interpret results in the context of each version's timing path, workload, and parameters. Historical H20 results do not imply that other chips have been tested.

### Backend Portability Validation

- 30 CPU-only runtime and precision tests cover NPU/MLU/MUSA graph interfaces, non-NVIDIA vendors using the `cuda` device name, event timing without graphs, missing properties and FP64 capabilities, precision-setting restoration, timing normalization, and error handling. They also check FP32 CLI defaults, explicit precision forwarding in both launch paths, precision-probe fallback rejection, and instruction-audit classification and rejection rules. Any access to `torch.cuda` fails immediately in the runtime tests.
- H20 hardware tests cover both FlagGems graph and ordinary event paths, FP16/FP8 tensor operations, GEMM tail tiles, vector operations, copy/read modes, and the branch that forces CPU FP64 references.
- These checks validate the runtime adaptation and NVIDIA regression paths; they do not establish hardware validation of kernels for every vendor.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python bench_vector.py --quick --timing auto --warmup 2 --rep 5 --rounds 3
python bench_vector.py --quick --timing events --warmup 2 --rep 5 --rounds 3
```

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
