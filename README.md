# Triton 芯片峰值测试

三个独立入口：tensor 矩阵单元、vector FP32 乘加、HBM/显存带宽。目标是扫描配置，寻找**当前设备和运行条件下能测到的最高吞吐量**；同时报告相同配置的中位数，便于判断稳定性。精度、稠密/稀疏、频率和访问模式会影响峰值，因此不同口径分别报告。

## 运行

本机环境：NVIDIA H20，PyTorch 2.11.0+cu130，Triton 3.6.0（FlagTree），FlagGems 5.4.0rc1。运行环境需要安装与芯片匹配的 PyTorch、Triton/FlagTree 和 FlagGems。

运行时统一使用 FlagGems 的多后端管理，不直接调用 `torch.cuda`：

| 功能 | 统一入口 |
| --- | --- |
| 设备类型、vendor 识别 | `flag_gems.device`、`flag_gems.vendor_name` |
| 选择设备、查询属性/显存、同步 | `flag_gems.runtime.torch_device_fn` |
| 设备事件、stream、Graph | 上述设备模块实际提供的 `Event`、`Stream`、`graph` 等接口 |
| 参考计算的精度设置 | `flag_gems.runtime.torch_backend_device`，存在相应控制项时使用 |
| FP64/BF16 能力 | `flag_gems.runtime.device` 的能力信息 |

设备类型和厂商分别处理，例如多个厂商都可能使用 `cuda` 设备名，NVIDIA 专用选项只在 `vendor_name == "nvidia"` 时启用。FlagGems 默认自动识别硬件；需要指定厂商时，使用其环境变量，例如 `GEMS_VENDOR=ascend python bench_vector.py --quick`。环境中必须已安装对应的实际设备后端，环境变量不能模拟硬件。

同一套入口可使用 FlagGems 检测到的设备模块。kernel 仍要求目标 Triton 后端支持 `tl.dot`、`tl.fma` 等指令及选定 dtype；FP8 保持 E4M3FN，不自动换成其他 FP8 格式。当前完成了 H20 实机回归及无 CUDA 的模拟运行时测试，其他芯片的 kernel 尚未实机验证。[FlagGems 后端接口说明](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/runtime/backend/README.md)

```bash
cd /workspace/triton_chip_benchmarks

# 依次运行，避免竞争同一张 GPU。
python bench_tensor.py --dtype fp16 --output results/peak_tensor_fp16.json
python bench_tensor.py --dtype bf16 --output results/peak_tensor_bf16.json
python bench_tensor.py --dtype fp8  --output results/peak_tensor_fp8.json
python bench_vector.py --output results/peak_vector.json
python bench_bandwidth.py --output results/peak_bandwidth.json
```

所有入口支持 `--device 0`、`--output 文件.json`、`--help`。快速检查可加 `--quick --warmup 2 --rep 5 --rounds 3`；小负载的结果不用于认定峰值。

设备属性缺失时不会填入其他芯片的规格：未知可用显存时跳过预分配检查；未知计算单元数量时，tensor/vector 可通过 `--compute-units 数量` 指定；未知 L2 时可通过 `--l2-mib 容量` 提供实际值。带宽程序不依赖计算单元数量。未提供 L2 容量的结果仍可测量，但不会进入 HBM 峰值摘要。

## 1. Tensor：驻留操作数的重复矩阵乘加

`bench_tensor.py` 默认 `--mode peak`：每个 program 从显存加载一次操作数，在循环中反复执行 `tl.dot`，维持两条独立累加链，最后写回全部结果。通过增加计算/访存比、改变 tile 和每个 SM 的 program 数，寻找计算单元吞吐量上限。

- FP16、BF16、FP8 E4M3FN 分别测试，输出均为 FP32。
- 使用后端原生累加策略。FP32 输出类型不等于内部累加始终具有完整 IEEE FP32 精度；H20 上强制 `max_num_imprecise_acc=0` 会退回 FP16 指令，因此不将这种路径标为 FP8 峰值。
- 所有输出都进行数值校验；NVIDIA/AMD 另外检查原生 MMA 指令，FP8 降级为其他 dtype 时拒绝标为原生峰值。其他厂商可以运行，但标注 `native execution unverified`，不会误报已完成指令验证。寄存器/spill 属性缺失时记录为 `null`。
- FP8 使用运行时生成的特制输入保持严格数值校验，未启用结构化稀疏指令。默认 1024 次重复，NVIDIA/AMD 扫描三种 tile，其他厂商使用两种较小 tile，搭配 `1、2、4、8 blocks/SM`。`blocks/SM` 表示网格规模与计算单元数量的比例，不保证所有 program 同时驻留。

```text
FLOPs = 2 × tile_M × tile_N × tile_K × iterations × chains × blocks
峰值 TFLOP/s = FLOPs / (min_ms × 10^9)
中位 TFLOP/s = FLOPs / (median_ms × 10^9)
```

`--mode gemm` 保留大矩阵吞吐量测试，用于观察应用负载下的表现：

```bash
python bench_tensor.py --mode gemm --dtype fp16 --sizes 2048 4096 8192
```

GEMM 的计数是 `2*M*N*K`，支持 FP16/BF16；不把 padding 的额外运算计入有效 FLOPs。

## 2. Vector：独立 FP32 FMA 链

`bench_vector.py` 加载独立随机初值和运行时系数，在寄存器中执行 8 条 FP32 FMA 链，最终保存全部结果。默认每条链 4096 次迭代，扫描 block、warp 和显式循环展开倍数，共 24 个配置；寄存器溢出的配置不参与峰值选择。

```bash
python bench_vector.py --iterations 4096 --blocks-per-sm 8 16 --block-sizes 256 512 \
  --num-warps 4 8 --unroll 1 4 8
```

```text
elements_per_chain = SM/CU 数 × blocks_per_sm × block_size
FLOPs = 2 × 8 × elements_per_chain × iterations
```

一次 FMA 是一次乘法和一次加法，计 2 FLOPs。展开因子必须整除迭代数；展开改变循环控制开销，不增加所统计的迭代总数。程序用 FP64 解析参考全量校验输出，在 NVIDIA 上额外确认 PTX 包含 FP32 FMA、不含 tensor 指令。静态指令条数不能直接当作动态执行次数。FlagGems 报告设备不支持 FP64 时，vector 和 read 模式先把有限大小的数据块复制到 CPU，再执行 FP64 校验，校验和复制均不计入耗时。

## 3. 带宽：流式读取与拷贝

`bench_bandwidth.py` 默认 `--mode all`，分别扫描 `read` 和 `copy`；源数组默认 `256、512、1024、2048 MiB`，配合不同 block/warp 配置。

- `copy`：`dst[i] = src[i]`，计入源读取和目标写入，逐元素检查相等。
- `read`：读取全部元素，在每个 block 内归约并写出一个 FP32 结果，确保编译器不能删掉加载；检查所有归约结果。计入源读取和少量结果写入，归约开销包含在耗时中。
- 默认数组远大于 H20 的 L2。只有源数组至少为已知 L2 的 4 倍且非 smoke 模式，才进入 HBM 尺寸的峰值摘要；小数组和未知 L2 容量不会被混入该摘要。

```bash
python bench_bandwidth.py --mode copy --sizes-mib 512 1024 2048
python bench_bandwidth.py --mode read --sizes-mib 512 1024 2048
```

```text
copy 流量 = 2 × 源数组字节数
read 流量 = 源数组字节数 + 归约结果字节数
峰值 GB/s = 流量 / (min_ms × 10^6)
中位 GB/s = 流量 / (median_ms × 10^6)
TB/s = GB/s / 1000
```

MiB/GiB 是 1024 进制，GB/s/TB/s 是 1000 进制。`--sizes-mib 1024` 在拷贝模式表示源、目标各 1 GiB。读与拷贝的访问模式不同，应分别比较。这里测的是有效全局内存带宽；物理 DRAM 事务量可能不同，也不包含主机与设备间的 PCIe/NVLink 传输。

## 时间是怎样记录的？

计时实现位于 `common.py`，所有时间戳都来自 **FlagGems 设备模块的 `Event`**，不使用 Python 墙钟作为替代。

默认 `--timing auto` 按运行时能力选择：有可用的 Graph API 时，使用设备模块导出的 `CUDAGraph`、`NPUGraph`、`MUSAGraph`、`MLUGraph` 或 `Graph`；没有 Graph 或明确不支持捕获时，使用设备事件包围普通批量发射。实际选择和回退原因会显示并写入 JSON。显式 `--timing graph` 要求 Graph 可用，否则报错；缺少设备事件计时能力时也会明确报错。

1. 先执行一次 kernel，完成 JIT 编译，并同步设备。数据准备和正确性检查在测量前完成。
2. 构造包含 R 次 kernel 发射的批次，Graph 可用时捕获为设备 Graph。先粗测，再按实际批次时间校准 R，默认目标每批约 10 ms，最多 2048 次。
3. 预热该批次，默认约 100 ms。
4. 在同一设备 stream 上依次记录开始事件、执行批次、记录结束事件；等待结束事件完成，再读取设备上的时间差。
5. 每次采样的单 kernel 耗时为 `设备批次耗时 / R`。默认至少采样 10 个批次，总计时预算约 200 ms。
6. 对所有批次取最小值和中位数。`min_ms` 对应最快的**批次平均耗时**，不是最短的一次 Python 调用；扫描全部有效配置后取最高吞吐量。

核心逻辑：

```python
from flag_gems.runtime import torch_device_fn

start = torch_device_fn.Event(enable_timing=True)
end = torch_device_fn.Event(enable_timing=True)
start.record()                   # 所选设备的时间戳
submit_batch()                  # Graph 回放或 R 次普通发射
end.record()                     # 同一个设备 stream
end.synchronize()                # 无事件同步方法时使用设备 synchronize
one_kernel_ms = start.elapsed_time(end) / R
```

可调参数：

```bash
python bench_vector.py --warmup 200 --rep 500 --batch-ms 10 --rounds 20
```

- 编译、分配、初始化、校验、graph 捕获和预热不进入报告的计时区间。
- kernel 内部的加载、存储、循环控制、设备调度和摊薄后的事件开销仍包含在内。Graph 可以减少 Python 发射空隙；普通 events 批次仍可能包含这些空隙，所以两个计时模式要分别比较，尤其是短 kernel。
- 两种模式都不在批次之间冲刷缓存：tensor/vector 利用驻留数据提高计算占比；带宽通过大于缓存的工作集避免把 L2 复用当成 HBM 峰值。
- JSON 保存 vendor、设备类型、FlagGems 版本、实际计时方式、回退原因、批次原始时间、R、每 kernel 时间、分位数及验证结果。
- `--timing events` 现在使用 FlagGems 设备事件包围普通批次，已不再调用旧版的 `triton.testing.do_bench`。

“本次实测峰值”是当前扫描得到的最好结果，不保证已穷尽所有实现，也不等同于厂商理论峰值。测试使用设备现有的频率与功耗策略；在空闲 GPU 上顺序执行，并查看最快批次和中位数的差距。CUDA 事件的机制及有效带宽口径参见 [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#performance-metrics)。

## 结果文件

FlagGems 回归记录使用 `results/flaggems_*.json`。`results/h20_peak_*.json` 和 `results/h20_gpu1_peak_*.json` 是接入 FlagGems 前的峰值测试记录；旧文件 `h20_tensor.json`、`h20_vector.json`、`h20_bandwidth.json` 则保留最初版本的持续吞吐量结果。不同版本的计时路径、负载和参数应分别解读，历史 H20 结果不代表其他芯片已经实测。

### 多后端适配验证

- 21 项纯 CPU 运行时测试覆盖 NPU/MLU/MUSA Graph 接口、使用 `cuda` 设备名的非 NVIDIA vendor、无 Graph 时的事件计时、缺失属性和 FP64 能力、精度设置恢复、计时归一化、异常处理。测试中的 `torch.cuda` 访问会直接失败。
- H20 实机验证了 FlagGems Graph/普通事件两种路径，FP16/FP8 tensor、GEMM 尾块、vector、copy/read，并验证了强制使用 CPU FP64 参考的分支。
- 这些验证证明运行时适配和 NVIDIA 回归通过，不等于所有厂商 kernel 已实机验证。

```bash
python -m unittest discover -s tests -p 'test_runtime.py' -v
python bench_vector.py --quick --timing auto --warmup 2 --rep 5 --rounds 3
python bench_vector.py --quick --timing events --warmup 2 --rep 5 --rounds 3
```

### 2026-09-15 H20 实测

GPU 0 完整扫描了 108 个配置（tensor 三种精度各 12 个、vector 24 个、带宽 48 个），全部通过校验。GPU 1 另对选出的配置进行了 18 组验证，tensor 使用更长的 4096 次循环。设备编号、参数及每个批次原始耗时都保存在 JSON 中。

| 指标 | GPU 0 完整扫描峰值 | GPU 1 选定配置峰值 | GPU 1 同配置中位吞吐量 |
| --- | ---: | ---: | ---: |
| Tensor FP16 | 130.13 TFLOP/s | 141.12 TFLOP/s | 141.07 TFLOP/s |
| Tensor BF16 | 130.04 TFLOP/s | 140.61 TFLOP/s | 140.37 TFLOP/s |
| Tensor FP8 | 259.95 TFLOP/s | 282.63 TFLOP/s | 282.60 TFLOP/s |
| Vector FP32 | 36.04 TFLOP/s | 39.00 TFLOP/s | 38.99 TFLOP/s |
| 显存读取 | 3.874 TB/s | 3.872 TB/s | 3.871 TB/s |
| 显存拷贝，读写合计 | 3.667 TB/s | 3.668 TB/s | 3.667 TB/s |

这些数值是已测配置的最高批次平均吞吐量。两张卡的 SM 频率快照曾分别为 1830 MHz 和 1980 MHz；该观察与计算吞吐量差异相符，但快照不等于测试全程的频率采样。测试使用现有设备设置。

GPU 1 对应的复现命令：

```bash
python bench_tensor.py --device 1 --dtype fp16 --iterations 4096 --blocks-per-sm 2
python bench_tensor.py --device 1 --dtype bf16 --iterations 4096 --blocks-per-sm 2
python bench_tensor.py --device 1 --dtype fp8 --iterations 4096 --blocks-per-sm 2
python bench_vector.py --device 1 --blocks-per-sm 16 --block-sizes 512 --num-warps 4 --unroll 1
python bench_bandwidth.py --device 1 --sizes-mib 2048 --block-sizes 1024 16384 --num-warps 4 8
```

GPU 1 原始记录：[FP16](results/h20_gpu1_peak_tensor_fp16.json)、[BF16](results/h20_gpu1_peak_tensor_bf16.json)、[FP8](results/h20_gpu1_peak_tensor_fp8.json)、[Vector](results/h20_gpu1_peak_vector.json)、[带宽](results/h20_gpu1_peak_bandwidth.json)。
