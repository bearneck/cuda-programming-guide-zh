# 2.1 CUDA C++ 入门

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)

---

此页面是否有帮助？

# 2.1. CUDA C++ 简介

本章通过展示 CUDA 编程模型中的基本概念如何在 C++ 中体现，来介绍这些概念。

本编程指南主要关注 CUDA 运行时 API。CUDA 运行时 API 是在 C++ 中使用 CUDA 最常用的方式，它构建在更低层的 CUDA 驱动程序 API 之上。

[CUDA 运行时 API 与 CUDA 驱动程序 API](../01-introduction/cuda-platform.html#cuda-platform-driver-and-runtime) 讨论了这两个 API 之间的区别，而 [CUDA 驱动程序 API](../03-advanced/driver-api.html#driver-api) 则讨论了混合使用这些 API 编写代码的方法。

本指南假设 CUDA 工具包和 NVIDIA 驱动程序已安装，并且存在受支持的 NVIDIA GPU。有关安装必要 CUDA 组件的说明，请参阅 [CUDA 快速入门指南](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)。

## 2.1.1. 使用 NVCC 编译

用 C++ 编写的 GPU 代码使用 NVIDIA Cuda 编译器 `nvcc` 进行编译。`nvcc` 是一个编译器驱动程序，它简化了编译 C++ 或 PTX 代码的过程：它提供简单且熟悉的命令行选项，并通过调用实现不同编译阶段的工具集合来执行这些选项。

本指南将展示可在任何安装了 CUDA 工具包的 Linux 系统、Windows 命令行或 PowerShell 中，或在安装了 CUDA 工具包的 Windows Subsystem for Linux 中使用的 `nvcc` 命令行。本指南的 [nvcc 章节](nvcc.html#nvcc) 涵盖了 `nvcc` 的常见用例，完整文档由 [nvcc 用户手册](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) 提供。

## 2.1.2. 内核

如 [CUDA 编程模型](../01-introduction/programming-model.html#programming-model) 简介中所述，在 GPU 上执行且可以从主机调用的函数称为内核。内核被编写为由许多并行线程同时运行。

### 2.1.2.1. 指定内核

内核的代码使用 `__global__` 声明说明符来指定。这向编译器表明，此函数将以允许从内核启动调用的方式为 GPU 编译。内核启动是一个启动内核运行的操作，通常从 CPU 发起。内核是返回类型为 `void` 的函数。

```cuda
// 内核定义
__global__ void vecAdd(float* A, float* B, float* C)
{

}
```

### 2.1.2.2. 启动内核

将并行执行内核的线程数量被指定为内核启动的一部分。这称为执行配置。同一内核的不同调用可以使用不同的执行配置，例如不同数量的线程或线程块。

从 CPU 代码启动内核有两种方式：[三重尖括号表示法](#intro-cpp-launching-kernels-triple-chevron) 和 `cudaLaunchKernelEx`。这里介绍最常用的内核启动方式——三重尖括号表示法。使用 `cudaLaunchKernelEx` 启动内核的示例在 [第 3.1.1 节](../03-advanced/advanced-host-programming.html#advanced-host-cudalaunchkernelex) 中展示并详细讨论。
#### 2.1.2.2.1. 三重尖括号表示法

三重尖括号表示法是一种 [CUDA C++ 语言扩展](../05-appendices/cpp-language-extensions.html#execution-configuration)，用于启动内核。它之所以被称为三重尖括号，是因为它使用三个尖括号字符来封装内核启动的执行配置，即 `<<< >>>`。执行配置参数在尖括号内以逗号分隔的列表形式指定，类似于函数调用的参数。下面展示了启动 `vecAdd` 内核的语法。

```cuda
 __global__ void vecAdd(float* A, float* B, float* C)
 {

 }

int main()
{
    ...
    // 内核调用
    vecAdd<<<1, 256>>>(A, B, C);
    ...
}
```

三重尖括号表示法的前两个参数分别是线程网格维度和线程块维度。当使用一维线程块或网格时，可以使用整数来指定维度。

上面的代码启动了一个包含 256 个线程的线程块。每个线程将执行完全相同的内核代码。在 [线程和网格索引内置变量](#intro-cpp-thread-indexing) 中，我们将展示每个线程如何利用其在线程块和网格内的索引来改变其操作的数据。

每个线程块的线程数量是有限制的，因为一个线程块的所有线程都驻留在同一个流式多处理器（SM）上，并且必须共享该 SM 的资源。在当前 GPU 上，一个线程块最多可包含 1024 个线程。如果资源允许，一个 SM 上可以同时调度多个线程块。

内核启动相对于主机线程是异步的。也就是说，内核将在 GPU 上设置好准备执行，但主机代码在继续执行之前，不会等待内核在 GPU 上完成（甚至开始）执行。必须使用某种形式的 GPU 和 CPU 之间的同步来确定内核已完成。最基本的版本是完全同步整个 GPU，如 [同步 CPU 和 GPU](#intro-synchronizing-the-gpu) 所示。更复杂的同步方法在 [异步执行](asynchronous-execution.html#asynchronous-execution) 中介绍。

当使用二维或三维网格或线程块时，CUDA 类型 `dim3` 被用作网格和线程块的维度参数。下面的代码片段展示了一个 `MatAdd` 内核的启动，它使用了一个 16x16 的线程块网格，每个线程块是 8x8。

```cuda
int main()
{
    ...
    dim3 grid(16,16);
    dim3 block(8,8);
    MatAdd<<<grid, block>>>(A, B, C);
    ...
}
```

### 2.1.2.3. 线程和网格索引内置变量

在内核代码中，CUDA 提供了内置变量来访问执行配置的参数以及线程或块的索引。

> threadIdx
> 给出线程在其所属线程块内的索引。线程块中的每个线程将拥有不同的索引。
> blockDim
> 给出线程块的维度，该维度在内核启动的执行配置中指定。
> blockIdx
> 提供线程块在线程网格内的索引。每个线程块将拥有不同的索引。
> gridDim
> 提供线程网格的维度，这些维度在内核启动时的执行配置中指定。

这些内置变量都是具有 `.x`、`.y` 和 `.z` 成员的三分量向量。启动配置中未指定的维度将默认为 1。`threadIdx` 和 `blockIdx` 的索引从 0 开始。也就是说，`threadIdx.x` 的取值范围是从 0 到 `blockDim.x-1`（包含）。`.y` 和 `.z` 在其各自的维度上遵循相同的规则。

类似地，`blockIdx.x` 的取值范围是从 0 到 `gridDim.x-1`（包含），`.y` 和 `.z` 维度也分别遵循相同的规则。

这些变量使得单个线程能够识别它应该执行哪些工作。回到 `vecAdd` 内核，该内核接受三个参数，每个参数都是一个浮点数向量。内核执行 `A` 和 `B` 的逐元素加法，并将结果存储在 `C` 中。内核被并行化，使得每个线程执行一次加法操作。它计算哪个元素由其线程索引和网格索引决定。

```cuda
__global__ void vecAdd(float* A, float* B, float* C)
{
   // 计算此线程负责计算哪个元素
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x

   // 执行计算
   C[workIndex] = A[workIndex] + B[workIndex];
}

int main()
{
    ...
    // A、B 和 C 是包含 1024 个元素的向量
    vecAdd<<<4, 256>>>(A, B, C);
    ...
}
```

在这个例子中，使用了 4 个包含 256 个线程的线程块来对一个包含 1024 个元素的向量进行加法运算。在第一个线程块中，`blockIdx.x` 为 0，因此每个线程的 `workIndex` 就是其 `threadIdx.x`。在第二个线程块中，`blockIdx.x` 为 1，所以 `blockDim.x * blockIdx.x` 等于 `blockDim.x`，在本例中为 256。第二个线程块中每个线程的 `workIndex` 将是其 `threadIdx.x + 256`。在第三个线程块中，`workIndex` 将是 `threadIdx.x + 512`。

这种 `workIndex` 的计算对于一维并行化非常常见。扩展到二维或三维时，通常在各个维度上遵循相同的模式。

#### 2.1.2.3.1. 边界检查

上面给出的例子假设向量的长度是线程块大小（本例中为 256 个线程）的整数倍。为了使内核能够处理任意长度的向量，我们可以添加检查，确保内存访问不会超出数组的边界，如下所示，然后启动一个线程块，其中将包含一些不活跃的线程。

```cuda
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // 计算此线程负责计算哪个元素
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x

     if(workIndex < vectorLength)
     {
         // 执行计算
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}
```

使用上面的内核代码，可以启动比所需更多的线程，而不会导致对数组的越界访问。当 `workIndex` 超过 `vectorLength` 时，线程会退出且不执行任何工作。在一个线程块中启动不执行任何工作的额外线程不会产生很大的开销成本，但是应避免启动其中没有线程执行工作的线程块。现在，这个内核可以处理长度不是线程块大小整数倍的向量。
所需线程块的数量可以通过计算所需线程数（本例中为向量长度）除以每个线程块的线程数，然后向上取整得到。也就是说，将所需线程数除以每个线程块的线程数进行整数除法，然后向上舍入。下面给出了将其表示为单个整数除法的常用方法。通过在整数除法前加上 `threads - 1`，这类似于一个向上取整函数，仅当向量长度不能被每个线程块的线程数整除时，才会增加一个额外的线程块。

```cuda
// vectorLength 是一个存储向量元素数量的整数
int threads = 256;
int blocks = (vectorLength + threads-1)/threads;
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

[CUDA 核心计算库 (CCCL)](https://nvidia.github.io/cccl/) 提供了一个便捷的工具 `cuda::ceil_div`，用于执行这种向上取整除法来计算内核启动所需的线程块数量。通过包含头文件 `<cuda/cmath>` 即可使用此工具。

```cuda
// vectorLength 是一个存储向量元素数量的整数
int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

这里选择每个线程块 256 个线程是任意的，但这通常是一个不错的起始值。

## 2.1.3. GPU 计算中的内存

为了使用上面展示的 `vecAdd` 内核，数组 `A`、`B` 和 `C` 必须位于 GPU 可访问的内存中。有几种不同的方法可以实现这一点，这里将说明其中两种。其他方法将在后面关于[统一内存](understanding-memory.html#memory-unified-memory)的章节中介绍。GPU 上运行的代码可用的内存空间在 [GPU 内存](../01-introduction/programming-model.html#programming-model-memory) 中已介绍，并在 [GPU 设备内存空间](writing-cuda-kernels.html#writing-cuda-kernels-gpu-device-memory-spaces) 中有更详细的说明。

### 2.1.3.1. 统一内存

统一内存是 CUDA 运行时的一项功能，它让 NVIDIA 驱动程序管理主机和设备之间的数据移动。内存可以使用 `cudaMallocManaged` API 分配，或者通过使用 `__managed__` 说明符声明变量来分配。NVIDIA 驱动程序将确保无论 GPU 还是 CPU 尝试访问该内存时，内存都是可访问的。

下面的代码展示了一个启动 `vecAdd` 内核的完整函数，该函数为将在 GPU 上使用的输入和输出向量使用了统一内存。`cudaMallocManaged` 分配可以从 CPU 或 GPU 访问的缓冲区。这些缓冲区使用 `cudaFree` 释放。

```cuda
void unifiedMemExample(int vectorLength)
{
    // 指向内存向量的指针
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // 使用统一内存分配缓冲区
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // 在主机上初始化向量
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // 启动内核。统一内存将确保 A、B 和 C 对 GPU 可访问
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // 等待内核完成执行
    cudaDeviceSynchronize();

    // 在 CPU 上串行执行计算以进行比较
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // 确认 CPU 和 GPU 得到相同的结果
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // 清理
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
```
统一内存在所有 CUDA 支持的操作系统和 GPU 上均受支持，但其底层机制和性能可能因系统架构而异。[统一内存](understanding-memory.html#memory-unified-memory)提供了更多详细信息。在某些 Linux 系统上（例如具备[地址转换服务](understanding-memory.html#memory-unified-address-translation-services)或[异构内存管理](understanding-memory.html#memory-heterogeneous-memory-management)的系统），所有系统内存会自动成为统一内存，无需使用 `cudaMallocManaged` 或 `__managed__` 限定符。

### 2.1.3.2. 显式内存管理

显式管理内存分配和内存空间之间的数据迁移有助于提升应用程序性能，尽管这会使代码更加冗长。以下代码使用 `cudaMalloc` 在 GPU 上显式分配内存。GPU 上的内存使用与前一示例中统一内存相同的 `cudaFree` API 进行释放。

```cuda
void explicitMemExample(int vectorLength)
{
    // 主机内存指针
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    
    // 设备内存指针
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    // 使用 cudaMallocHost API 分配主机内存。当缓冲区将用于 CPU 和 GPU 内存之间的复制时，这是最佳实践
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // 在主机上初始化向量
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // 在 GPU 上分配内存
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // 将数据复制到 GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // 启动内核
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // 等待内核执行完成
    cudaDeviceSynchronize();

    // 将结果复制回主机
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // 在 CPU 上串行执行计算以进行比较
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // 确认 CPU 和 GPU 得到相同的结果
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("显式内存管理：CPU 与 GPU 结果匹配\n");
    }
    else
    {
        printf("显式内存管理：错误 - CPU 与 GPU 结果不匹配\n");
    }

    // 清理
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
```
CUDA API `cudaMemcpy` 用于将数据从驻留在 CPU 上的缓冲区复制到驻留在 GPU 上的缓冲区。除了目标指针、源指针和字节大小外，`cudaMemcpy` 的最后一个参数是 `cudaMemcpyKind_t`。其值可以是 `cudaMemcpyHostToDevice`（用于从 CPU 复制到 GPU）、`cudaMemcpyDeviceToHost`（用于从 GPU 复制到 CPU）或 `cudaMemcpyDeviceToDevice`（用于 GPU 内部或 GPU 之间的复制）。

在此示例中，`cudaMemcpyDefault` 作为最后一个参数传递给 `cudaMemcpy`。这将导致 CUDA 根据源指针和目标指针的值来确定要执行的复制类型。

`cudaMemcpy` API 是同步的。也就是说，在复制完成之前它不会返回。异步复制在 [在 CUDA 流中启动内存传输](asynchronous-execution.html#async-execution-memory-transfers) 中介绍。

代码使用 `cudaMallocHost` 在 CPU 上分配内存。这会在主机上分配[页锁定内存](understanding-memory.html#memory-page-locked-host-memory)，这可以提高复制性能，并且对于[异步](asynchronous-execution.html#async-execution-memory-transfers)内存传输是必需的。通常，对于将在与 GPU 进行数据传输中使用的 CPU 缓冲区，使用页锁定内存是一种良好实践。如果锁定的主机内存过多，某些系统的性能可能会下降。最佳实践是仅锁定将用于向 GPU 发送或从 GPU 接收数据的缓冲区。

### 2.1.3.3. 内存管理与应用程序性能

如上例所示，显式内存管理更为繁琐，需要程序员指定主机和设备之间的复制操作。这正是显式内存管理的优点和缺点：它提供了对数据在主机和设备之间何时复制、内存驻留在何处以及具体在何处分配内存的更多控制。显式内存管理可以通过控制内存传输并将其与其他计算重叠来提供性能优化的机会。

当使用统一内存时，有一些 CUDA API（将在[内存建议与预取](understanding-memory.html#memory-mem-advise-prefetch)中介绍），它们向管理内存的 NVIDIA 驱动程序提供提示，这可以在使用统一内存时实现一些使用显式内存管理的性能优势。

## 2.1.4. 同步 CPU 与 GPU

如[启动内核](#intro-cpp-launching-kernels)中所述，内核启动相对于调用它们的 CPU 线程是异步的。这意味着 CPU 线程的控制流将在内核完成之前（甚至可能在内核启动之前）继续执行。为了保证在内核代码继续执行之前内核已完成执行，需要某种同步机制。

同步 GPU 和主机线程的最简单方法是使用 `cudaDeviceSynchronize`，它会阻塞主机线程，直到 GPU 上所有先前发出的工作都已完成。在本章的示例中，这已经足够，因为 GPU 上只执行单个操作。在更大的应用程序中，可能有多个[流](asynchronous-execution.html#cuda-streams)在 GPU 上执行工作，而 `cudaDeviceSynchronize` 将等待所有流中的工作完成。在这些应用程序中，建议使用[流同步](asynchronous-execution.html#async-execution-stream-synchronization) API 仅与特定流同步，或使用[CUDA 事件](asynchronous-execution.html#cuda-events)。这些将在[异步执行](asynchronous-execution.html#asynchronous-execution)章节中详细介绍。
## 2.1.5. 完整示例

以下清单展示了本章介绍的简单向量加法内核的完整代码，以及用于检查并验证所得答案是否正确的主机代码和工具函数。这些示例默认使用长度为 1024 的向量，但接受不同的向量长度作为可执行文件的命令行参数。

**统一内存**

```cuda
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//unified-memory-begin
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
//unified-memory-end

int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);		
    return 0;
}
```

**显式内存管理**

```cuda
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//explicit-memory-begin
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    
    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
//explicit-memory-end

int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    explicitMemExample(vectorLength);		
    return 0;
}
```
可以使用 nvcc 按如下方式构建和运行这些示例：

```bash
$ nvcc vecAdd_unifiedMemory.cu -o vecAdd_unifiedMemory
$ ./vecAdd_unifiedMemory
Unified Memory: CPU and GPU answers match
$ ./vecAdd_unifiedMemory 4096
Unified Memory: CPU and GPU answers match
```

```bash
$ nvcc vecAdd_explicitMemory.cu -o vecAdd_explicitMemory
$ ./vecAdd_explicitMemory
Explicit Memory: CPU and GPU answers match
$ ./vecAdd_explicitMemory 4096
Explicit Memory: CPU and GPU answers match
```

在这些示例中，所有线程都在执行独立的工作，不需要彼此协调或同步。然而，线程通常需要与其他线程协作和通信以完成其工作。线程块内的线程可以通过[共享内存](writing-cuda-kernels.html#writing-cuda-kernels-shared-memory)共享数据，并通过同步来协调内存访问。

线程块级别最基本的同步机制是 `__syncthreads()` 内部函数，它充当一个屏障，线程块内的所有线程必须在此等待，直到所有线程都到达该点后，才允许任何线程继续执行。[共享内存](writing-cuda-kernels.html#writing-cuda-kernels-shared-memory)部分给出了一个使用共享内存的示例。

为了实现高效协作，共享内存被设计为靠近每个处理器核心的低延迟内存（类似于 L1 缓存），而 `__syncthreads()` 则被设计为轻量级操作。`__syncthreads()` 仅同步单个线程块内的线程。CUDA 编程模型不支持线程块之间的同步。[协作组](../04-special-topics/cooperative-groups.html#cooperative-groups)提供了设置除单个线程块之外的其他同步域的机制。

通常，将同步保持在单个线程块内可以获得最佳性能。线程块仍然可以使用[原子内存函数](writing-cuda-kernels.html#writing-cuda-kernels-atomics)来处理共同的结果，这将在后续章节中介绍。

[第 3.2.4 节](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives)涵盖了 CUDA 同步原语，这些原语提供了非常精细的控制，以最大化性能和资源利用率。

## 2.1.6. 运行时初始化

CUDA 运行时为系统中的每个设备创建一个[CUDA 上下文](../03-advanced/driver-api.html#driver-api-context)。此上下文是该设备的主上下文，并在第一个需要该设备上活动上下文的运行时函数调用时初始化。该上下文由应用程序的所有主机线程共享。作为上下文创建的一部分，设备代码会在必要时进行[即时编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)并加载到设备内存中。这一切都是透明进行的。CUDA 运行时创建的主上下文可以从驱动 API 访问，以实现互操作性，如[运行时与驱动 API 的互操作性](../03-advanced/driver-api.html#driver-api-interop-with-runtime)所述。
自 CUDA 12.0 起，`cudaInitDevice` 和 `cudaSetDevice` 调用会初始化运行时以及与指定设备关联的主[上下文](../03-advanced/driver-api.html#driver-api-context)。如果运行时 API 请求在这些调用之前发生，运行时将隐式使用设备 0 并根据需要自我初始化以处理这些请求。这在计时运行时函数调用以及解释首次调用运行时返回的错误代码时非常重要。在 CUDA 12.0 之前，`cudaSetDevice` 不会初始化运行时。

`cudaDeviceReset` 会销毁当前设备的主上下文。如果在主上下文被销毁后调用 CUDA 运行时 API，将为该设备创建一个新的主上下文。

!!! note "注意"
    CUDA 接口使用全局状态，该状态在主机程序启动期间初始化，并在主机程序终止期间销毁。在程序启动期间或 main 函数之后的终止期间使用这些接口中的任何一个（隐式或显式）将导致未定义行为。自 CUDA 12.0 起，`cudaSetDevice` 在为主机线程更改当前设备后，如果运行时尚未初始化，则会显式初始化运行时。CUDA 的早期版本会延迟在新设备上的运行时初始化，直到在 `cudaSetDevice` 之后进行第一次运行时调用。因此，检查 `cudaSetDevice` 的返回值以获取初始化错误非常重要。参考手册中错误处理和版本管理部分的运行时函数不会初始化运行时。

## 2.1.7. CUDA 中的错误检查

每个 CUDA API 都会返回一个枚举类型 `cudaError_t` 的值。在示例代码中，这些错误通常不会被检查。在生产应用程序中，最佳实践是始终检查并管理每个 CUDA API 调用的返回值。当没有错误时，返回的值为 `cudaSuccess`。许多应用程序选择实现一个实用宏，如下所示：

```cuda
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)
```

此宏使用 `cudaGetErrorString` API，该 API 返回一个描述特定 `cudaError_t` 值含义的人类可读字符串。使用上述宏，应用程序将在 `CUDA_CHECK(expression)` 宏内调用 CUDA 运行时 API，如下所示：

```cuda
    CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));
```
如果这些调用检测到错误，将使用此宏将其打印到 `stderr`。此宏在小型项目中很常见，但在大型应用程序中可以适配到日志系统或其他错误处理机制。

!!! note "注意"
    需要注意的是，任何 CUDA API 调用返回的错误状态也可能指示先前发出的异步操作中的错误。[异步错误处理](#intro-cpp-error-checking-asynchronous) 一节将对此进行更详细的介绍。

### 2.1.7.1. 错误状态

CUDA 运行时为每个主机线程维护一个 `cudaError_t` 状态。该值默认为 `cudaSuccess`，并在发生错误时被覆盖。`cudaGetLastError` 返回当前错误状态，然后将其重置为 `cudaSuccess`。或者，`cudaPeekLastError` 返回错误状态而不重置它。

使用[三重尖括号表示法](#intro-cpp-launching-kernels-triple-chevron)启动的内核不返回 `cudaError_t`。良好的做法是在内核启动后立即检查错误状态，以检测内核启动中的即时错误或内核启动前的[异步错误](#intro-cpp-error-checking-asynchronous)。在内核启动后立即检查错误状态时，值为 `cudaSuccess` 并不意味着内核已成功执行甚至已开始执行。它仅验证传递给运行时的内核启动参数和执行配置没有触发任何错误，并且错误状态不是内核启动之前的先前错误或异步错误。

### 2.1.7.2. 异步错误

CUDA 内核启动和许多运行时 API 是异步的。异步 CUDA 运行时 API 将在[异步执行](asynchronous-execution.html#asynchronous-execution)中详细讨论。每当发生错误时，CUDA 错误状态就会被设置和覆盖。这意味着在异步操作执行期间发生的错误，只有在下次检查错误状态时才会被报告。如前所述，这可能是对 `cudaGetLastError`、`cudaPeekLastError` 的调用，也可能是任何返回 `cudaError_t` 的 CUDA API。

当 CUDA 运行时 API 函数返回错误时，错误状态不会被清除。这意味着来自异步错误（例如内核的无效内存访问）的错误代码，将在每次调用 CUDA 运行时 API 时返回，直到通过调用 `cudaGetLastError` 清除了错误状态。

```cuda
    vecAdd<<<blocks, threads>>>(devA, devB, devC);
    // 在内核启动后检查错误状态
    CUDA_CHECK(cudaGetLastError());
    // 等待内核执行完成
    // CUDA_CHECK 将报告内核执行期间发生的错误
    CUDA_CHECK(cudaDeviceSynchronize());
```

!!! note "注意"
    `cudaStreamQuery` 和 `cudaEventQuery` 可能返回的 `cudaError_t` 值 `cudaErrorNotReady` 不被视为错误，并且不会被 `cudaPeekAtLastError` 或 `cudaGetLastError` 报告。

### 2.1.7.3. CUDA_LOG_FILE

识别 CUDA 错误的另一个好方法是使用 `CUDA_LOG_FILE` 环境变量。设置此环境变量后，CUDA 驱动程序会将遇到的错误消息写入到环境变量中指定的路径文件中。例如，以下错误的 CUDA 代码尝试启动一个大于任何架构支持的最大线程块。

```cuda
__global__ void k()
{ }

int main()
{
        k<<<8192, 4096>>>(); // Invalid block size
        CUDA_CHECK(cudaGetLastError());
        return 0;
}
```

构建并运行此程序后，内核启动后的检查会使用[第 2.1.7 节](#intro-cpp-error-checking)中介绍的宏来检测并报告错误。

```bash
$ nvcc errorLogIllustration.cu -o errlog
$ ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
```

然而，当应用程序在设置了 `CUDA_LOG_FILE` 环境变量指向一个文本文件的情况下运行时，该文件会包含更多关于该错误的详细信息。

```bash
$ env CUDA_LOG_FILE=cudaLog.txt ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
$ cat cudaLog.txt
[12:46:23.854][137216133754880][CUDA][E] One or more of block dimensions of (4096,1,1) exceeds corresponding maximum value of (1024,1024,64)
[12:46:23.854][137216133754880][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel
```

将 `CUDA_LOG_FILE` 设置为 `stdout` 或 `stderr` 将分别打印到标准输出和标准错误。使用 `CUDA_LOG_FILE` 环境变量，可以捕获和识别 CUDA 错误，即使应用程序未对 CUDA 返回值实施适当的错误检查。这种方法对于调试极其有效，但仅凭环境变量无法让应用程序在运行时处理和恢复 CUDA 错误。CUDA 的[错误日志管理](../04-special-topics/error-log-management.html#error-log-management)功能还允许向驱动程序注册一个回调函数，该函数将在检测到错误时被调用。这可用于在运行时捕获和处理错误，并将 CUDA 错误日志无缝集成到应用程序现有的日志系统中。

[第 4.8 节](../04-special-topics/error-log-management.html#error-log-management)展示了更多关于 CUDA 错误日志管理功能的示例。错误日志管理和 `CUDA_LOG_FILE` 在 NVIDIA 驱动程序版本 r570 及更高版本中可用。

## 2.1.8.Device and Host Functions

`__global__` 限定符用于指示内核的入口点。也就是说，该函数将在 GPU 上被调用以进行并行执行。大多数情况下，内核是从主机端启动的，但也可以使用[动态并行](../04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism)从另一个内核内部启动内核。

限定符 `__device__` 表示该函数应编译为 GPU 代码，并可从其他 `__device__` 或 `__global__` 函数中调用。函数（包括类成员函数、函数对象和 lambda 表达式）可以同时指定为 `__device__` 和 `__host__`，如下例所示。

## 2.1.9.Variable Specifiers

[CUDA specifiers](../05-appendices/cpp-language-extensions.html#memory-space-specifiers) can be used on static variable declarations to control placement.
- __device__ 指定变量存储在全局内存中
- __constant__ 指定变量存储在常量内存中
- __managed__ 指定变量存储为统一内存
- __shared__ 指定变量存储在共享内存中

当变量在 `__device__` 或 `__global__` 函数内部声明且没有指定符时，它会在可能的情况下分配到寄存器中，必要时则分配到[本地内存](writing-cuda-kernels.html#writing-cuda-kernels-local-memory)中。任何在 `__device__` 或 `__global__` 函数外部声明且没有指定符的变量都将分配到系统内存中。

### 2.1.9.1. 检测设备编译

当一个函数被指定为 `__host__ __device__` 时，编译器会为该函数同时生成 GPU 和 CPU 代码。在此类函数中，可能需要使用预处理器来指定仅用于 GPU 或 CPU 副本的代码。检查是否定义了 `__CUDA_ARCH_` 是最常见的方法，如下例所示。

## 2.1.10. 线程块集群

从计算能力 9.0 开始，CUDA 编程模型包含一个可选的层次级别，称为线程块集群，它由线程块组成。类似于线程块中的线程保证在流式多处理器（SM）上协同调度，集群中的线程块也保证在 GPU 的 GPU 处理集群（GPC）上协同调度。

与线程块类似，集群也被组织成一维、二维或三维的线程块集群网格，如[图 5](../01-introduction/programming-model.html#figure-thread-block-clusters) 所示。

集群中的线程块数量可以由用户定义，CUDA 中支持的可移植集群大小最多为 8 个线程块。请注意，在 GPU 硬件或 MIG 配置太小而无法支持 8 个多处理器的情况下，最大集群大小将相应减小。识别这些较小的配置，以及支持超过 8 个线程块集群大小的较大配置，是特定于架构的，可以使用 `cudaOccupancyMaxPotentialClusterSize` API 进行查询。

集群中的所有线程块保证在单个 GPU 处理集群（GPC）上协同调度并同时执行，并允许集群中的线程块使用[协作组](../04-special-topics/cooperative-groups.html#cooperative-groups) API `cluster.sync()` 执行硬件支持的同步。集群组还提供了成员函数，分别使用 `num_threads()` 和 `num_blocks()` API 来查询集群组在线程数量或块数量方面的大小。可以使用 `dim_threads()` 和 `dim_blocks()` API 分别查询线程或块在集群组中的秩。

属于集群的线程块可以访问*分布式共享内存*，这是集群中所有线程块的共享内存的组合。集群中的线程块能够对分布式共享内存中的任何地址进行读取、写入和执行原子操作。[分布式共享内存](writing-cuda-kernels.html#writing-cuda-kernels-distributed-shared-memory) 给出了在分布式共享内存中执行直方图的示例。
!!! note "Note"
    在使用集群支持启动的内核中，出于兼容性考虑，`gridDim` 变量仍然表示以线程块数量为单位的尺寸。可以使用协作组（Cooperative Groups）API 来查找线程块在集群中的层级。

### 2.1.10.1. 使用三重尖括号表示法启动集群

可以在内核中通过两种方式启用线程块集群：一种是使用编译时内核属性 `__cluster_dims__(X,Y,Z)`，另一种是使用 CUDA 内核启动 API `cudaLaunchKernelEx`。下面的示例展示了如何使用编译时内核属性启动集群。使用内核属性指定的集群大小在编译时是固定的，然后可以使用经典的 `<<< , >>>` 语法启动内核。如果内核使用了编译时集群大小，则在启动内核时无法修改集群大小。

```c++
// 内核定义
// 编译时集群大小：X 维度为 2，Y 和 Z 维度为 1
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // 使用编译时集群大小调用内核
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // 网格维度不受集群启动的影响，仍然使用块的数量来计数。
    // 网格维度必须是集群大小的整数倍。
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```