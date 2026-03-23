# 4.18 CUDA 动态并行

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)

---

此页面有帮助吗？

# 4.18. CUDA 动态并行

## 4.18.1. 简介

### 4.18.1.1. 概述

CUDA 动态并行（通常缩写为 CDP）是 CUDA 编程模型的一项功能，它允许在 GPU 上运行的代码创建新的 GPU 工作。也就是说，可以通过从已在 GPU 上运行的设备代码启动额外的内核，将新工作添加到 GPU 上。此功能可以减少在主机和设备之间传输执行控制和数据的需求，因为启动配置决策可以由在设备上执行的线程在运行时做出。

数据相关的并行工作可以由内核在运行时生成。在 CDP 被添加到 CUDA 之前，某些算法和编程模式需要进行修改以消除递归、不规则循环结构或其他不适合扁平、单层并行性的构造。使用 CUDA 动态并行可以更自然地表达这些程序结构。

本节记录了更新版本的 CUDA 动态并行，有时称为 CDP2，它是 CUDA 12.0 及更高版本中的默认版本。CDP2 是计算能力 9.0 及更高设备上唯一可用的 CUDA 动态并行版本。对于计算能力低于 9.0 的设备，开发者仍可通过编译器参数 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 选择使用旧版 CUDA 动态并行 CDP1。CDP1 的文档可在 [旧版 CUDA 编程指南](https://developer.nvidia.com/cuda-toolkit-archive) 中找到。CDP1 计划在未来的 CUDA 版本中移除。

## 4.18.2. 执行环境

CUDA 中的动态并行允许 GPU 线程配置、启动和隐式同步新的线程网格。线程网格是内核启动的一个实例，包括线程块和线程块网格的具体形状。内核函数本身与该内核的特定调用（即线程网格）之间的区别在以下部分中需要注意。

### 4.18.2.1. 父网格和子网格

配置和启动新线程网格的设备线程属于父网格。通过调用创建的新线程网格称为子网格。

子网格的调用和完成是正确嵌套的，这意味着在所有由其线程创建的子网格完成之前，父网格不被视为完成，并且运行时保证父网格和子网格之间存在隐式同步。

图 54 父-子启动嵌套 [#](#parent-child-launch-nesting-figure)

### 4.18.2.2. CUDA 原语的作用域

CUDA 动态并行依赖于 [CUDA 设备运行时](../05-appendices/device-callable-apis.html#cuda-device-runtime)，该运行时支持调用一组有限的 API，这些 API 在语法上与 CUDA 运行时 API 相似，但可在设备代码中使用。设备运行时 API 的行为与其主机端对应项类似，但存在一些差异。这些差异记录在 [API 参考](../05-appendices/device-callable-apis.html#device-runtime-api-reference) 部分的表格中。
在主机和设备上，CUDA 运行时都提供了用于启动内核以及通过流和事件来跟踪启动间依赖关系的 API。在设备上，启动的内核和 CUDA 对象对调用网格中的所有线程都是可见的。这意味着，例如，一个流可以由一个线程创建，并由同一网格中的任何其他线程使用。然而，通过设备 API 调用创建的 CUDA 对象（如流和事件）仅在创建它们的网格内有效。

### 4.18.2.3. 流和事件

CUDA *流* 和 *事件* 允许控制内核启动之间的依赖关系：启动到同一流中的内核按顺序执行，事件可用于在流之间创建依赖关系。在设备上创建的流和事件服务于完全相同的目的。

在网格内创建的流和事件存在于网格作用域内，但在创建它们的网格之外使用时，其行为是未定义的。如上所述，当网格退出时，由网格启动的所有工作都会隐式同步；启动到流中的工作也包括在内，所有依赖关系都会得到适当解决。在网格作用域之外被修改的流上执行的操作，其行为是未定义的。

在主机上创建的流和事件在任何内核中使用时，其行为是未定义的，就像父网格创建的流和事件在子网格中使用时其行为是未定义的一样。

### 4.18.2.4. 顺序性和并发性

从设备运行时启动内核的顺序遵循 CUDA 流顺序语义。在一个网格内，所有启动到同一流中的内核（[即发即弃流](../05-appendices/device-callable-apis.html#fire-and-forget-stream) 除外）都是按顺序执行的。当同一网格中的多个线程向同一流启动内核时，流内的顺序取决于网格内的线程调度，这可以通过 `__syncthreads()` 等同步原语来控制。

请注意，虽然命名流由网格内的所有线程共享，但隐式的 *NULL* 流仅由线程块内的所有线程共享。如果线程块中的多个线程向隐式流启动内核，那么这些启动将按顺序执行。如果不同线程块中的线程向隐式流启动内核，这些启动可能会并发执行。如果希望线程块内多个线程的启动能够并发执行，则应使用显式的命名流。

设备运行时在 CUDA 执行模型中并未引入新的并发性保证。也就是说，设备上任意数量的不同线程块之间，都不能保证并发执行。

这种并发性保证的缺失也延伸到父网格及其子网格。当父网格启动子网格时，一旦流依赖关系满足且硬件资源可用，子网格就可能开始执行，但不能保证在父网格到达隐式同步点之前开始执行。
并发性可能因设备配置、应用程序工作负载和运行时调度的不同而变化。因此，依赖不同线程块之间的任何并发性是不安全的。

## 4.18.3. 内存一致性与连贯性

父网格和子网格共享相同的全局内存和常量内存存储，但拥有各自独立的本地内存和共享内存。下表显示了哪些内存空间允许父网格和子网格通过相同的指针进行访问。子网格永远无法访问父网格的本地内存或共享内存，父网格也无法访问子网格的本地内存或共享内存。

| 内存空间 | 父/子使用相同指针？ |
| --- | --- |
| 全局内存 | 是 |
| 映射内存 | 是 |
| 本地内存 | 否 |
| 共享内存 | 否 |
| 纹理内存 | 是（只读） |

### 4.18.3.1. 全局内存

父网格和子网格对全局内存具有连贯的访问权限，但子网格与父网格之间只提供弱一致性保证。在子网格的执行过程中，只有一个时间点其内存视图与父线程完全一致：即当父线程调用子网格的时刻。

在子网格被调用之前，父线程中所有的全局内存操作对子网格都是可见的。随着 `cudaDeviceSynchronize()` 的移除，父网格已无法访问子网格中线程所做的修改。在父网格退出之前，访问子网格中线程所做修改的唯一方式是通过启动到 `cudaStreamTailLaunch` 流中的内核。

在以下示例中，执行 `child_launch` 的子网格仅保证能看到在子网格启动前对 `data` 所做的修改。由于是父线程的线程 0 执行启动，子网格将与父线程的线程 0 所见的内存保持一致。由于第一个 `__syncthreads()` 调用，子网格将看到 `data[0]=0`, `data[1]=1`, ..., `data[255]=255`（如果没有 `__syncthreads()` 调用，则仅保证子网格能看到 `data[0]=0`）。子网格仅保证在隐式同步时返回。这意味着子网格中线程所做的修改永远不保证对父网格可用。为了访问 `child_launch` 所做的修改，一个 `tail_launch` 内核被启动到 `cudaStreamTailLaunch` 流中。

```cpp
__global__ void tail_launch(int *data) {
   data[threadIdx.x] = data[threadIdx.x]+1;
}

__global__ void child_launch(int *data) {
   data[threadIdx.x] = data[threadIdx.x]+1;
}

__global__ void parent_launch(int *data) {
   data[threadIdx.x] = threadIdx.x;

   __syncthreads();

   if (threadIdx.x == 0) {
       child_launch<<< 1, 256 >>>(data);
       tail_launch<<< 1, 256, 0, cudaStreamTailLaunch >>>(data);
   }
}

void host_launch(int *data) {
    parent_launch<<< 1, 256 >>>(data);
}
```

### 4.18.3.2. 映射内存

映射的系统内存具有与全局内存相同的一致性和连贯性保证，并遵循上述详述的语义。内核不能分配或释放映射内存，但可以使用从主机程序传入的指向映射内存的指针。
### 4.18.3.3. 共享内存与本地内存

共享内存和本地内存分别对线程块或线程是私有的，在父级和子级之间不可见且不保持一致性。当这些位置中的对象在其所属作用域之外被引用时，行为是未定义的，并可能导致错误。

如果 NVIDIA 编译器能够检测到指向本地或共享内存的指针正被作为参数传递给内核启动，它将尝试发出警告。在运行时，程序员可以使用 `__isGlobal()` 内部函数来确定指针是否引用全局内存，从而可以安全地传递给子级启动。

调用 `cudaMemcpy*Async()` 或 `cudaMemset*Async()` 可能会在设备上调用新的子内核，以保持流语义。因此，将共享或本地内存指针传递给这些 API 是非法的，并将返回错误。

### 4.18.3.4. 本地内存

本地内存是执行线程的私有存储，在该线程之外不可见。在启动子内核时，将指向本地内存的指针作为启动参数传递是非法的。从子网格解引用此类本地内存指针的结果是未定义的。

例如，以下代码是非法的，如果 `child_launch` 访问了 `x_array`，其行为将是未定义的：

```cpp
int x_array[10];       // 在父级的本地内存中创建 x_array
child_launch<<< 1, 1 >>>(x_array);
```

有时程序员很难意识到变量何时被编译器放入本地内存。作为一般规则，传递给子内核的所有存储都应从全局内存堆中显式分配，可以使用 `cudaMalloc()`、`new()` 或在全局作用域声明 `__device__` 存储。例如：

```cpp
// 正确 - "value" 是全局存储
__device__ int value;
__device__ void x() {
    value = 5;
    child<<< 1, 1 >>>(&value);
}
```

```cpp
// 无效 - "value" 是本地存储
__device__ void y() {
    int value = 5;
    child<<< 1, 1 >>>(&value);
}
```

#### 4.18.3.4.1. 纹理内存

对映射了纹理的全局内存区域进行的写入，相对于纹理访问是不一致的。纹理内存的一致性在子网格调用时以及子网格完成时强制执行。这意味着在子内核启动之前对内存的写入，会反映在子级的纹理内存访问中。与上面的全局内存类似，子级对内存的写入永远不能保证会反映在父级的纹理内存访问中。在父网格退出之前访问子网格中线程所做修改的唯一方法，是通过在 `cudaStreamTailLaunch` 流中启动的内核。父级和子级的并发访问可能导致数据不一致。

## 4.18.4. 编程接口

### 4.18.4.1. 基础

以下示例展示了一个包含动态并行性的简单 *Hello World* 程序：

```cpp
#include <stdio.h>

__global__ void childKernel()
{
    printf("Hello ");
}

__global__ void tailKernel()
{
    printf("World!\n");
}

__global__ void parentKernel()
{
    // 启动子内核
    childKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError()) {
        return;
    }

    // 将尾内核启动到 cudaStreamTailLaunch 流中
    // 隐式同步：等待子内核完成
    tailKernel<<<1,1,0,cudaStreamTailLaunch>>>();

}

int main(int argc, char *argv[])
{
    // 启动父内核
    parentKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError()) {
        return 1;
    }

    // 等待父内核完成
    if (cudaSuccess != cudaDeviceSynchronize()) {
        return 2;
    }

    return 0;
}
```
该程序可以通过命令行单步构建，如下所示：

```cpp
$ nvcc -arch=sm_75 -rdc=true hello_world.cu -o hello -lcudadevrt
```

### 4.18.4.2. 用于 CDP 的 C++ 语言接口

使用 CUDA C++ 进行动态并行化的 CUDA 内核可用的语言接口和 API 称为 [CUDA 设备运行时](../05-appendices/device-callable-apis.html#cuda-device-runtime)。

为了便于可能在主机或设备环境中运行的例程进行代码重用，CUDA 运行时 API 的语法和语义在可能的情况下都得以保留。

与 CUDA C++ 中的所有代码一样，此处概述的 API 和代码是**每线程**代码。这使得每个线程都能就接下来要执行哪个内核或操作做出独特的、动态的决定。线程块内的线程之间执行任何提供的设备运行时 API 都没有同步要求，这使得设备运行时 API 函数可以在任意发散的内核代码中被调用而不会导致死锁。

#### 4.18.4.2.1. 设备端内核启动

可以使用标准的 CUDA `<<< >>>` 语法从设备启动内核：

```cpp
kernel_name<<< Dg, Db, Ns, S >>>([kernel arguments]);
```

- `Dg` 是 `dim3` 类型，指定线程网格的维度和大小。
- `Db` 是 `dim3` 类型，指定每个线程块的维度和大小。
- `Ns` 是 `size_t` 类型，指定除了静态分配的内存外，为此调用每个线程块动态分配的共享内存字节数。`Ns` 是一个可选参数，默认为 0。
- `S` 是 `cudaStream_t` 类型，指定与此调用关联的流。该流必须在进行调用的同一线程网格中分配。`S` 是一个可选参数，默认为 NULL 流。

##### 4.18.4.2.1.1. 启动是异步的

与主机端启动相同，所有设备端内核启动相对于启动线程都是异步的。也就是说，`<<<>>>` 启动命令将立即返回，启动线程将继续执行，直到遇到隐式的启动同步点，例如在启动到 `cudaStreamTailLaunch` 流中的内核时（[尾部启动流](../05-appendices/device-callable-apis.html#tail-launch-stream)）。子网格可以在启动后的任何时间开始执行，但不能保证在启动线程到达隐式启动同步点之前开始执行。

与主机端启动类似，启动到不同流中的工作可能会并发运行，但并不保证实际并发性。依赖于子内核之间并发性的程序不受 CUDA 编程模型支持，并且将具有未定义的行为。

##### 4.18.4.2.1.2. 启动环境配置

所有全局设备配置设置（例如，从 `cudaDeviceGetCacheConfig()` 返回的共享内存和 L1 缓存大小，以及从 `cudaDeviceGetLimit()` 返回的设备限制）都将从父级继承。同样，设备限制（如堆栈大小）将保持配置时的状态。
对于主机启动的内核，从主机设置的每个内核配置将优先于全局设置。当内核从设备启动时，这些配置也将被使用。无法从设备重新配置内核的环境。

#### 4.18.4.2.2. 事件

仅支持 CUDA 事件的流间同步功能。这意味着支持 `cudaStreamWaitEvent()`，但不支持 `cudaEventSynchronize()`、`cudaEventElapsedTime()` 和 `cudaEventQuery()`。由于不支持 `cudaEventElapsedTime()`，必须通过 `cudaEventCreateWithFlags()` 创建 cudaEvent，并传递 `cudaEventDisableTiming` 标志。

与流类似，事件对象可以在创建它们的线程网格内的所有线程之间共享，但它们是该网格本地的，不能传递给其他内核。不保证事件句柄在不同网格之间是唯一的，因此在未创建它的网格内使用事件句柄将导致未定义行为。

#### 4.18.4.2.3. 同步

如果调用线程需要与其他线程调用的子网格同步，则由程序负责执行足够的线程间同步，例如通过 CUDA 事件。

由于无法从父线程显式同步子工作，因此无法保证父网格内的线程能看到子网格中发生的变化。

#### 4.18.4.2.4. 设备管理

只有内核正在运行的设备可以从该内核进行控制。这意味着设备运行时不支持诸如 `cudaSetDevice()` 之类的设备 API。从 GPU 看到的当前活动设备（由 `cudaGetDevice()` 返回）的设备编号将与从主机系统看到的相同。`cudaDeviceGetAttribute()` 调用可以请求关于另一个设备的信息，因为此 API 允许将设备 ID 指定为调用的参数。请注意，设备运行时不提供通用的 `cudaGetDeviceProperties()` API - 必须单独查询属性。

## 4.18.5. 编程指南

### 4.18.5.1. 性能

#### 4.18.5.1.1. 启用动态并行功能的内核开销

在控制动态启动时处于活动状态的系统软件可能会对当时正在运行的任何内核施加开销，无论该内核是否自行调用内核启动。此开销源于设备运行时的执行跟踪和管理软件，并可能导致性能下降。通常，链接到设备运行时库的应用程序会产生此开销。

### 4.18.5.2. 实现限制和局限性

动态并行保证本文档中描述的所有语义，但是，某些硬件和软件资源取决于具体实现，并限制了使用设备运行时的程序的规模、性能和其他属性。

#### 4.18.5.2.1. 运行时

##### 4.18.5.2.1.1. 内存占用

设备运行时系统软件为各种管理目的保留内存，特别是用于跟踪待处理网格启动的保留区。提供配置控制以减少此保留区的大小，但需以某些启动限制为代价。详情请参阅下面的[配置选项](../05-appendices/device-callable-apis.html#device-runtime-configuration-options)。
##### 4.18.5.2.1.2. 待处理的内核启动

当一个内核被启动时，所有相关的配置和参数数据都会被跟踪，直到内核完成。这些数据存储在一个系统管理的启动池中。

固定大小的启动池的大小可以通过在主机端调用 `cudaDeviceSetLimit()` 并指定 `cudaLimitDevRuntimePendingLaunchCount` 来配置。

### 4.18.5.3. 兼容性与互操作性

CDP2 是默认设置。可以通过使用 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 编译函数，以在计算能力低于 9.0 的设备上选择不使用 CDP2。

|  | 使用 CUDA 12.0 及更新版本编译的函数（默认） | 使用 CUDA 12.0 之前版本编译的函数，或使用 CUDA 12.0 及更新版本但指定了 `-DCUDA_FORCE_CDP1_IF_SUPPORTED` 编译的函数 |
| --- | --- | --- |
| 编译 | 如果设备代码引用了 `cudaDeviceSynchronize`，则编译错误。 | 如果代码引用了 `cudaStreamTailLaunch` 或 `cudaStreamFireAndForget`，则编译错误。如果设备代码引用了 `cudaDeviceSynchronize` 且代码是为 sm_90 或更新版本编译的，则编译错误。 |
| 计算能力 < 9.0 | 使用新接口。 | 使用旧接口。 |
| 计算能力 9.0 及更高 | 使用新接口。 | 使用新接口。如果函数在其设备代码中引用了 `cudaDeviceSynchronize`，则函数加载会返回 `cudaErrorSymbolNotFound`（如果代码是为计算能力低于 9.0 的设备编译，但使用 JIT 在计算能力 9.0 或更高的设备上运行时，可能会发生这种情况）。 |

使用 CDP1 和 CDP2 的函数可以在同一上下文中同时加载和运行。CDP1 函数能够使用 CDP1 特有的功能（例如 `cudaDeviceSynchronize`），而 CDP2 函数能够使用 CDP2 特有的功能（例如尾部启动和即发即弃启动）。

使用 CDP1 的函数不能启动使用 CDP2 的函数，反之亦然。如果一个将使用 CDP1 的函数在其调用图中包含一个将使用 CDP2 的函数，或者反之，则在函数加载期间会导致 `cudaErrorCdpVersionMismatch`。

本文档不包含旧版 CDP1 的行为。有关 CDP1 的信息，请参阅 [旧版 CUDA 编程指南](https://developer.nvidia.com/cuda-toolkit-archive)。

## 4.18.6. 从 PTX 进行设备端启动

前面的章节讨论了使用 [CUDA 设备运行时](../05-appendices/device-callable-apis.html#cuda-device-runtime) 来实现动态并行。动态并行也可以从 PTX 执行。对于面向 *并行线程执行*（PTX）并计划在其语言中支持 *动态并行* 的编程语言和编译器实现者，本节提供了与在 PTX 级别支持内核启动相关的底层细节。

### 4.18.6.1. 内核启动 API

设备端内核启动可以使用以下两个可从 PTX 访问的 API 实现：`cudaLaunchDevice()` 和 `cudaGetParameterBuffer()`。`cudaLaunchDevice()` 启动指定的内核，并使用通过调用 `cudaGetParameterBuffer()` 获取并填充了要启动内核参数的参数缓冲区。如果要启动的内核不接受任何参数，则参数缓冲区可以为 NULL，即无需调用 `cudaGetParameterBuffer()`。
#### 4.18.6.1.1.cudaLaunchDevice

在 PTX 层面，`cudaLaunchDevice()` 在使用前需要按以下两种形式之一进行声明。

```cpp
// PTX-level Declaration of cudaLaunchDevice() when .address_size is 64
.extern .func(.param .b32 func_retval0) cudaLaunchDevice
(
  .param .b64 func,
  .param .b64 parameterBuffer,
  .param .align 4 .b8 gridDimension[12],
  .param .align 4 .b8 blockDimension[12],
  .param .b32 sharedMemSize,
  .param .b64 stream
)
;
```

下面的 CUDA 层面声明会映射到上述 PTX 层面声明之一，并可在系统头文件 `cuda_device_runtime_api.h` 中找到。该函数定义在 `cudadevrt` 系统库中，程序必须链接此库才能使用设备端内核启动功能。

```cpp
// CUDA-level declaration of cudaLaunchDevice()
extern "C" __device__
cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer,
                             dim3 gridDimension, dim3 blockDimension,
                             unsigned int sharedMemSize,
                             cudaStream_t stream);
```

第一个参数是指向要启动的内核的指针，第二个参数是保存要启动内核的实际参数的参数缓冲区。参数缓冲区的布局将在下面的[参数缓冲区布局](#parameter-buffer-layout)中解释。其他参数指定启动配置，即线程网格维度、线程块维度、共享内存大小以及与启动关联的流（有关启动配置的详细描述，请参阅[内核配置](../05-appendices/cpp-language-extensions.html#execution-configuration)）。

#### 4.18.6.1.2.cudaGetParameterBuffer

`cudaGetParameterBuffer()` 在使用前需要在 PTX 层面进行声明。PTX 层面的声明必须根据地址大小采用以下两种形式之一：

```cpp
// PTX-level Declaration of cudaGetParameterBuffer() when .address_size is 64
.extern .func(.param .b64 func_retval0) cudaGetParameterBuffer
(
  .param .b64 alignment,
  .param .b64 size
)
;
```

`cudaGetParameterBuffer()` 的以下 CUDA 层面声明映射到上述 PTX 层面声明：

```cpp
// CUDA-level Declaration of cudaGetParameterBuffer()
extern "C" __device__
void *cudaGetParameterBuffer(size_t alignment, size_t size);
```

第一个参数指定参数缓冲区的对齐要求，第二个参数指定以字节为单位的大小要求。在当前实现中，`cudaGetParameterBuffer()` 返回的参数缓冲区始终保证为 64 字节对齐，并且对齐要求参数会被忽略。但是，建议将正确的对齐要求值（即要放入参数缓冲区中的任何参数的最大对齐值）传递给 `cudaGetParameterBuffer()`，以确保未来的可移植性。

### 4.18.6.2.参数缓冲区布局
参数缓冲区中禁止参数重排序，并且要求放置在参数缓冲区中的每个单独参数都进行对齐。也就是说，每个参数必须放置在参数缓冲区的第 *n* 个字节处，其中 *n* 是大于前一个参数所占用的最后一个字节偏移量的参数大小的最小倍数。参数缓冲区的最大大小为 4KB。

有关 CUDA 编译器生成的 PTX 代码的更详细描述，请参阅 PTX-3.5 规范。

 本页