# 4.7 延迟加载

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html)

---

此页面是否有帮助？

# 4.7. 延迟加载

## 4.7.1. 简介

延迟加载通过等待 CUDA 模块直到需要时才加载，从而减少程序初始化时间。延迟加载对于仅使用其包含的少量内核的程序特别有效，这在库的使用中很常见。当遵循 CUDA 编程模型时，延迟加载被设计为对用户不可见。[潜在风险](#lazy-loading-potential-hazards)对此进行了详细说明。从 CUDA 12.3 开始，延迟加载在所有平台上默认启用，但可以通过 `CUDA_MODULE_LOADING` 环境变量进行控制。

## 4.7.2. 变更历史

| CUDA 版本 | 变更 |
| --- | --- |
| 12.3 | 延迟加载性能改进。现在在 Windows 上默认启用。 |
| 12.2 | 在 Linux 上默认启用延迟加载。 |
| 11.7 | 首次引入延迟加载，默认禁用。 |

## 4.7.3. 延迟加载的要求

延迟加载是 CUDA 运行时和驱动程序的共同特性。只有在满足运行时和驱动程序版本要求时，延迟加载才可用。

### 4.7.3.1. CUDA 运行时版本要求

延迟加载从 CUDA 运行时版本 11.7 开始可用。由于 CUDA 运行时通常静态链接到程序和库中，只有来自 CUDA 11.7+ 工具包或使用其编译的程序和库才能受益于延迟加载。使用旧版 CUDA 运行时版本编译的库将急切加载所有模块。

### 4.7.3.2. CUDA 驱动程序版本要求

延迟加载需要驱动程序版本 515 或更高。对于早于 515 的驱动程序版本，即使使用 CUDA 工具包 11.7 或更高版本，延迟加载也不可用。

### 4.7.3.3. 编译器要求

延迟加载不需要任何编译器支持。使用 11.7 之前版本编译器编译的 SASS 和 PTX 都可以在启用延迟加载的情况下加载，并将获得该功能的全部优势。但是，如上所述，仍然需要 11.7+ 版本的 CUDA 运行时。

### 4.7.3.4. 内核要求

延迟加载不影响包含托管变量的模块，这些模块仍将被急切加载。

## 4.7.4. 使用方法

### 4.7.4.1. 启用与禁用

通过将 `CUDA_MODULE_LOADING` 环境变量设置为 `LAZY` 来启用延迟加载。通过将 `CUDA_MODULE_LOADING` 环境变量设置为 `EAGER` 可以禁用延迟加载。从 CUDA 12.3 开始，延迟加载在所有平台上默认启用。

### 4.7.4.2. 在运行时检查延迟加载是否启用

CUDA 驱动程序 API 中的 `cuModuleGetLoadingMode` API 可用于确定是否启用了延迟加载。请注意，在运行此函数之前必须初始化 CUDA。示例用法如下面的代码片段所示。

```cuda
#include "<cuda.h>"
#include "<assert.h>"
#include "<iostream>"

int main() {
        CUmoduleLoadingMode mode;

        assert(CUDA_SUCCESS == cuInit(0));
        assert(CUDA_SUCCESS == cuModuleGetLoadingMode(&mode));

        std::cout << "CUDA Module Loading Mode is " << ((mode == CU_MODULE_LAZY_LOADING) ? "lazy" : "eager") << std::endl;

        return 0;
}
```
### 4.7.4.3. 在运行时强制模块立即加载

内核和变量的加载是自动进行的，无需显式加载。即使不执行内核，也可以通过以下方式显式加载它们：

- `cuModuleGetFunction()` 函数将导致模块被加载到设备内存中。
- `cudaFuncGetAttributes()` 函数将导致内核被加载到设备内存中。

!!! note "注意"
    `cuModuleLoad()` 并不保证模块会立即加载。

## 4.7.5. 潜在风险

延迟加载的设计使得应用程序无需任何修改即可使用它。尽管如此，仍有一些注意事项，特别是当应用程序不完全符合 CUDA 编程模型时，如下所述。

### 4.7.5.1. 对并发内核执行的影响

一些程序错误地假设并发内核执行是有保证的。如果需要跨内核同步，但内核执行已被序列化，则可能发生死锁。为了最小化延迟加载对并发内核执行的影响，请执行以下操作之一：

- 在启动所有希望并发执行的内核之前，预先加载它们；或者
- 设置环境变量 `CUDA_MODULE_LOADING = EAGER` 来运行应用程序，以强制立即加载数据，而无需强制每个函数都立即加载。

### 4.7.5.2. 大内存分配

延迟加载将 CUDA 模块的内存分配从程序初始化推迟到更接近执行的时间。如果应用程序在启动时就分配了整个 VRAM，CUDA 可能在运行时无法为模块分配内存。可能的解决方案：

- 使用 `cudaMallocAsync()` 替代在启动时就分配整个 VRAM 的分配器。
- 增加一些缓冲区以补偿内核的延迟加载。
- 在尝试初始化分配器之前，预先加载程序中将要使用的所有内核。

### 4.7.5.3. 对性能测量的影响

延迟加载可能会将 CUDA 模块初始化过程移入测量的执行窗口，从而影响性能测量结果。为避免这种情况：

- 在测量之前至少进行一次预热迭代。
- 在启动待测内核之前预先加载它。

 本页