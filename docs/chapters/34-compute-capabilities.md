# 5.1 计算能力

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

---

此页面有帮助吗？

# 5.1. 计算能力

计算设备的通用规格和特性取决于其计算能力（参见[计算能力与流式多处理器版本](../01-introduction/cuda-platform.html#cuda-platform-compute-capability-sm-version)）。

[表 29](#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability)、[表 30](#compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability) 和 [表 31](#compute-capabilities-table-memory-information-per-compute-capability) 展示了当前支持的每种计算能力所关联的特性和技术规格。

所有 NVIDIA GPU 架构均使用小端字节序表示。

## 5.1.1. 获取 GPU 计算能力

[CUDA GPU 计算能力](https://developer.nvidia.com/cuda-gpus)页面提供了从 NVIDIA GPU 型号到其计算能力的完整映射。

或者，可以使用随 [NVIDIA 驱动程序](https://www.nvidia.com/en-us/drivers/) 提供的 [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html) 工具来获取 GPU 的计算能力。例如，以下命令将输出系统中可用的 GPU 名称和计算能力：

```cpp
nvidia-smi --query-gpu=name,compute_cap
```

在运行时，可以使用 CUDA 运行时 API [cudaDeviceGetAttribute()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151)、CUDA 驱动程序 API [cuDeviceGetAttribute()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266) 或 NVML API [nvmlDeviceGetCudaComputeCapability()](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g1f803a2fb4b7dfc0a8183b46b46ab03a) 来获取计算能力：

```cpp
#include <cuda_runtime_api.h>

int computeCapabilityMajor, computeCapabilityMinor;
cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device_id);
cudaDeviceGetAttribute(&computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device_id);
```

```cpp
#include <cuda.h>

int computeCapabilityMajor, computeCapabilityMinor;
cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id);
cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_id);
```

```cpp
#include <nvml.h> // required linking with -lnvidia-ml

int computeCapabilityMajor, computeCapabilityMinor;
nvmlDeviceGetCudaComputeCapability(nvmlDevice, &computeCapabilityMajor, &computeCapabilityMinor);
```

## 5.1.2. 特性可用性

随着计算架构引入的大多数计算特性，都旨在使其在后续的所有架构中可用。这在[表 29](#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability) 中通过某个特性在其引入后的计算能力上标记为“是”来表示。
### 5.1.2.1. 架构特定功能

从计算能力 9.0 的设备开始，随架构引入的专用计算功能可能无法保证在所有后续计算能力中都可用。这些功能被称为*架构特定*功能，旨在加速专用操作，例如 Tensor Core 操作，这些操作并非针对所有类别的计算能力，或者可能在未来的代次中发生重大变化。必须使用架构特定的编译器目标（参见[功能集编译器目标](#compute-capabilities-feature-set-compiler-targets)）编译代码才能启用架构特定功能。使用架构特定编译器目标编译的代码只能在其编译时所针对的精确计算能力上运行。

### 5.1.2.2. 系列特定功能

从计算能力 10.0 的设备开始，一些架构特定功能在多个计算能力的设备中是通用的。包含这些功能的设备属于同一系列，这些功能也可以称为*系列特定*功能。系列特定功能保证在同一系列的所有设备上都可用。启用系列特定功能需要系列特定的编译器目标。参见[第 5.1.2.3 节](#compute-capabilities-feature-set-compiler-targets)。为系列特定目标编译的代码只能在该系列成员的 GPU 上运行。

### 5.1.2.3. 功能集编译器目标

编译器可以针对三组计算功能：

**基线功能集**：引入的主要计算功能集，旨在为后续计算架构提供支持。这些功能及其可用性总结在[表 29](#compute-capabilities-table-features-and-technical-specifications-feature-support-per-compute-capability) 中。

**架构特定功能集**：一小部分高度专业化的功能，称为架构特定功能，旨在加速专用操作，这些功能不保证在后续计算架构中可用或可能发生重大变化。这些功能总结在相应的“计算能力 #.#”小节中。架构特定功能集是系列特定功能集的超集。架构特定编译器目标随计算能力 9.0 设备引入，通过在编译目标中使用 **a** 后缀来选择，例如，指定 `compute_100a` 或 `compute_120a` 作为计算目标。

**系列特定功能集**：一些架构特定功能在多个计算能力的 GPU 中是通用的。这些功能总结在相应的“计算能力 #.#”小节中。除少数例外情况外，具有相同主计算能力的后代设备属于同一系列。[表 28](#compute-capabilities-family-specific-compatibility) 说明了系列特定目标与设备计算能力的兼容性，包括例外情况。系列特定功能集是基线功能集的超集。系列特定编译器目标随计算能力 10.0 设备引入，通过在编译目标中使用 **f** 后缀来选择，例如，指定 `compute_100f` 或 `compute_120f` 作为计算目标。
从计算能力 9.0 开始的所有设备都拥有一组特定于架构的特性。要在特定 GPU 上充分利用这组特性，必须使用带有后缀 **a** 的特定架构编译器目标。此外，从计算能力 10.0 开始，存在一些特性集出现在具有不同次要计算能力的多个设备中。这些指令集被称为特定于系列的（family-specific）特性，共享这些特性的设备被称为属于同一系列。特定于系列的特性是特定于架构的特性集合的一个子集，由该 GPU 系列的所有成员共享。带有后缀 **f** 的特定于系列编译器目标允许编译器生成使用这些架构特定特性的公共子集的代码。

例如：

- `compute_100` 编译目标不允许使用特定于架构的特性。此目标将与所有计算能力 10.0 及更高版本的设备兼容。
- `compute_100f` 特定于系列的编译目标允许使用在该 GPU 系列中通用的特定于架构的特性子集。此目标仅与该 GPU 系列中的设备兼容。在此示例中，它与计算能力 10.0 和计算能力 10.3 的设备兼容。特定于系列的 `compute_100f` 目标中可用的特性是基线 `compute_100` 目标中可用特性的超集。
- `compute_100a` 特定于架构的编译目标允许使用计算能力 10.0 设备中完整的特定于架构的特性集。此目标仅与计算能力 10.0 的设备兼容，不与其他设备兼容。`compute_100a` 目标中可用的特性构成了 `compute_100f` 目标中可用特性的超集。

| 编译目标 | 兼容的计算能力 |  |
| --- | --- | --- |
| compute_100f | 10.0 | 10.3 |
| compute_103f | 10.3 [ 1 ] |  |
| compute_110f | 11.0 [ 1 ] |  |
| compute_120f | 12.0 | 12.1 |
| compute_121f | 12.1 [ 1 ] |  |

[
1
]
(
1
,
2
,
3
)

某些系列在创建时仅包含单个成员。未来可能会扩展以包含更多设备。

## 5.1.3. 特性与技术规格

| 特性支持 | 计算能力 |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| (未列出的特性在所有计算能力上都支持) | 7.x | 8.x | 9.0 | 10.x | 11.0 | 12.x |
| 在共享内存和全局内存中对 128 位整数值进行操作的原子函数 ( Atomic Functions ) | No | Yes |  |  |  |  |
| 在全局内存中对 float2 和 float4 浮点向量进行操作的原子加法 ( atomicAdd() ) | No | Yes |  |  |  |  |
| 线程束归约函数 ( Warp Reduce Functions ) | No | Yes |  |  |  |  |
| Bfloat16 精度浮点运算 | No | Yes |  |  |  |  |
| 128 位精度浮点运算 | No | Yes |  |  |  |  |
| 硬件加速的 memcpy_async（管道） | 否 | 是 |  |  |  |  |
| 硬件加速的分裂到达/等待屏障（异步屏障） | 否 | 是 |  |  |  |  |
| L2 缓存驻留管理（L2 缓存控制） | 否 | 是 |  |  |  |  |
| 用于加速动态规划的 DPX 指令（动态规划扩展 (DPX) 指令） | 多条指令 | 原生 | 多条指令 |  |  |  |
| 分布式共享内存 | 否 | 是 |  |  |  |  |
| 线程块簇（线程块簇） | 否 | 是 |  |  |  |  |
| 张量内存加速器 (TMA) 单元
（使用张量内存加速器 (TMA)） | 否 | 是 |  |  |  |  |

请注意，下表中使用的 KB 和 K 单位分别对应 1024 字节（即 KiB）和 1024。

|  | 计算能力 |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.0 | 10.3 | 11.0 | 12.x |
| FP32 与 FP64 吞吐量之比
[ 2 ] | 32:1 | 2:1 | 64:1 | 2:1 | 64:1 |  |  |  |  |  |
| 每个设备的最大常驻网格数
（并发内核执行） | 128 |  |  |  |  |  |  |  |  |  |
| 网格的最大维度 | 3 |  |  |  |  |  |  |  |  |  |
| 网格的最大 x 维度 | 2 31 -1 |  |  |  |  |  |  |  |  |  |
| 网格的最大 y 或 z 维度 | 65535 |  |  |  |  |  |  |  |  |  |
| 线程块的最大维度 | 3 |  |  |  |  |  |  |  |  |  |
| 线程块的最大 x 或 y 维度 | 1024 |  |  |  |  |  |  |  |  |  |
| 线程块的最大 z 维度 | 64 |  |  |  |  |  |  |  |  |  |
| 每个块的最大线程数 | 1024 |  |  |  |  |  |  |  |  |  |
| 线程束大小 | 32 |  |  |  |  |  |  |  |  |  |
| 每个 SM 的最大常驻块数 | 16 | 32 | 16 | 24 | 32 | 24 |  |  |  |  |
| 每个 SM 的最大常驻线程束数 | 32 | 64 | 48 | 64 | 48 |  |  |  |  |  |
| 每个 SM 的最大常驻线程数 | 1024 | 2048 | 1536 | 2048 | 1536 |  |  |  |  |  |
| 绿色上下文：
useFlags 为 0 时的最小 SM 分区大小 | 2 | 4 | 8 |  |  |  |  |  |  |  |
| 绿色上下文：
useFlags 为 0 时每个分区的 SM 协同调度对齐 | 2 | 8 |  |  |  |  |  |  |  |  |

[
2
]

非张量核心吞吐量。有关吞吐量的更多信息，请参阅 [CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions)

|  | 计算能力 |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.x | 11.0 | 12.x |
| 每个 SM 的 32 位寄存器数量 | 64 K |  |  |  |  |  |  |  |  |
| 每个线程块的最大 32 位寄存器数量 | 64 K |  |  |  |  |  |  |  |  |
| 每个线程的最大 32 位寄存器数量 | 255 |  |  |  |  |  |  |  |  |
| 每个 SM 的最大共享内存量 | 64 KB | 164
KB | 100
KB | 164
KB | 100
KB | 228
KB | 100
KB |  |  |
| 每个 SM 的最大共享
每线程块内存 [ 3 ] | 64 KB | 163
KB | 99 KB | 163
KB | 99 KB | 227
KB | 99 KB |  |  |
| 共享内存
存储体数量 | 32 |  |  |  |  |  |  |  |  |
| 每线程最大本地
内存量 | 512 KB |  |  |  |  |  |  |  |  |
| 常量内存大小 | 64 KB |  |  |  |  |  |  |  |  |
| 每个 SM 用于常量内存的
缓存工作集 | 8 KB |  |  |  |  |  |  |  |  |
| 每个 SM 用于纹理内存的
缓存工作集 | 32 或
64 KB | 28 KB
~ 192
KB | 28 KB
~ 128
KB | 28 KB
~ 192
KB | 28 KB
~ 128
KB | 28 KB
~ 256
KB | 28 KB
~ 128
KB |  |  |

[ 3 ]

依赖每个线程块分配超过 48 KB 共享内存的内核必须使用动态共享内存，并且需要显式选择加入，请参阅 [配置 L1/共享内存平衡](../03-advanced/advanced-kernel-programming.html#advanced-kernel-l1-shared-config)。

| 计算能力 | 统一数据缓存大小 (KB) | 共享内存容量大小 (KB) |
| --- | --- | --- |
| 7.5 | 96 | 32, 64 |
| 8.0 | 192 | 0, 8, 16, 32, 64, 100, 132, 164 |
| 8.6 | 128 | 0, 8, 16, 32, 64, 100 |
| 8.7 | 192 | 0, 8, 16, 32, 64, 100, 132, 164 |
| 8.9 | 128 | 0, 8, 16, 32, 64, 100 |
| 9.0 | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 10.x | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 11.0 | 256 | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 |
| 12.x | 128 | 0, 8, 16, 32, 64, 100 |

[表 33](#compute-capabilities-table-tensor-core-data-types-per-compute-capability) 显示了 Tensor Core 加速支持的输入数据类型。Tensor Core 功能集可通过内联 PTX 在 CUDA 编译工具链中使用。强烈建议应用程序通过 CUDA-X 库（例如 cuDNN、cuBLAS 和 cuFFT）或通过 [CUTLASS](https://docs.nvidia.com/cutlass/index.html) 来使用此功能集。CUTLASS 是一个 CUDA C++ 模板抽象和 Python 领域特定语言 (DSL) 的集合，旨在实现 CUDA 内所有级别的矩阵-矩阵乘法 (GEMM) 及相关计算的高性能。

| 计算能力 | Tensor Core 输入数据类型 |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | FP64 | TF32 | BF16 | FP16 | FP8 | FP6 | FP4 | INT8 | INT4 |
| 7.5 |  | 是 |  | 是 | 是 |  |  |  |  |
| 8.0 | 是 | 是 | 是 | 是 |  | 是 | 是 |  |  |
| 8.6 |  | 是 | 是 | 是 |  | 是 | 是 |  |  |
| 8.7 |  | 是 | 是 | 是 |  | 是 | 是 |  |  |
| 8.9 |  | 是 | 是 | 是 | 是 |  | 是 | 是 |  |
| 9.0 | 是 | 是 | 是 | 是 | 是 |  | 是 |  |  |
| 10.0 | 是 | 是 | 是 | 是 | 是 | 是 | 是 | 是 |  |
| 10.3 |  | 是 | 是 | 是 | 是 | 是 | 是 | 是 |  |
| 11.0 |  | 是 | 是 | 是 | 是 | 是 | 是 | 是 |  |
| 12.x |  | 是 | 是 | 是 | 是 | 是 | 是 | 是 |  |

 本页