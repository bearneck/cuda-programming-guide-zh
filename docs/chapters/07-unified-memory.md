# 2.4 统一内存与系统内存

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)

---

此页面是否有帮助？

# 2.4. 统一内存与系统内存

异构系统拥有多个可以存储数据的物理内存。主机 CPU 连接着 DRAM，系统中的每个 GPU 也都有其自身连接的 DRAM。当数据驻留在访问它的处理器内存中时，性能最佳。CUDA 提供了用于[显式管理内存放置](intro-to-cuda-cpp.html#intro-cpp-explicit-memory-management)的 API，但这可能使代码冗长并让软件设计复杂化。CUDA 提供了一些特性和功能，旨在简化不同物理内存之间的数据分配、放置和迁移。

本章的目的是介绍和解释这些特性，以及它们对应用程序开发者在功能和性能方面的意义。统一内存有几种不同的表现形式，具体取决于操作系统、驱动版本和所使用的 GPU。本章将展示如何确定适用哪种统一内存范式，以及统一内存的特性在每种范式中的行为表现。后续的[统一内存章节](../04-special-topics/unified-memory.html#um-details-intro)会更详细地解释统一内存。

本章将定义和解释以下概念：

- **统一虚拟地址空间** - CPU 内存和每个 GPU 的内存在一个单一的虚拟地址空间内拥有不同的地址范围。
- **统一内存** - 一种 CUDA 特性，支持托管内存，可以在 CPU 和 GPU 之间自动迁移。
- **受限统一内存** - 一种存在某些限制的统一内存范式。
- **完全统一内存** - 对统一内存特性的完全支持。
- **支持硬件一致性的完全统一内存** - 利用硬件能力对统一内存的完全支持。
- **统一内存提示** - 用于指导特定分配的统一内存行为的 API。
- **页锁定主机内存** - 不可分页的系统内存，某些 CUDA 操作需要此内存。
- **映射内存** - 一种（不同于统一内存的）机制，允许从内核直接访问主机内存。

此外，这里还介绍了讨论统一内存和系统内存时使用的以下术语：

- **异构托管内存** - Linux 内核的一项特性，为完全统一内存提供软件一致性支持。
- **地址转换服务** - 一种硬件特性，当 GPU 通过 NVLink 芯片到芯片互连连接到 CPU 时可用，为完全统一内存提供硬件一致性支持。

## 2.4.1. 统一虚拟地址空间

在单个操作系统进程内，系统中的一个虚拟地址空间用于所有主机内存和所有 GPU 上的所有全局内存。主机和所有设备上的所有内存分配都位于此虚拟地址空间中。无论分配是使用 CUDA API（例如 `cudaMalloc`、`cudaMallocHost`）还是系统分配 API（例如 `new`、`malloc`、`mmap`）进行的，这一点都成立。CPU 和每个 GPU 在统一虚拟地址空间内都有一个唯一的地址范围。

这意味着：

- 任何内存的位置（即，位于 CPU 还是哪个 GPU 的内存中）都可以通过使用 `cudaPointerGetAttributes()` 函数根据指针的值来确定。
- 可以将 cudaMemcpy*() 的 cudaMemcpyKind 参数设置为 cudaMemcpyDefault，以根据指针自动确定复制类型

## 2.4.2. 统一内存

*统一内存* 是 CUDA 的一项内存功能，它允许称为*托管内存* 的内存分配被运行在 CPU 或 GPU 上的代码访问。统一内存已在 [C++ 中的 CUDA 简介](intro-to-cuda-cpp.html#intro-cpp-unified-memory) 中展示。统一内存在所有 CUDA 支持的系统上都可用。

在某些系统上，托管内存必须显式分配。在 CUDA 中，可以通过几种不同的方式显式分配托管内存：

- 使用 CUDA API `cudaMallocManaged`
- 使用 CUDA API `cudaMallocFromPoolAsync`，并配合一个将 `allocType` 设置为 `cudaMemAllocationTypeManaged` 创建的池
- 使用 `__managed__` 限定符的全局变量（参见内存空间限定符）

在具有 [HMM](#memory-heterogeneous-memory-management) 或 [ATS](#memory-unified-address-translation-services) 的系统上，所有系统内存都是隐式的托管内存，无论其分配方式如何。无需特殊分配。

### 2.4.2.1. 统一内存范式

统一内存的功能和行为因操作系统、Linux 内核版本、GPU 硬件以及 GPU-CPU 互连方式而异。可用的统一内存形式可以通过使用 `cudaDeviceGetAttribute` 查询几个属性来确定：

- `cudaDevAttrConcurrentManagedAccess` - 值为 1 表示完全支持统一内存，0 表示有限支持
- `cudaDevAttrPageableMemoryAccess` - 值为 1 表示所有系统内存都是完全支持的统一内存，0 表示只有显式分配为托管内存的内存才是完全支持的统一内存
- `cudaDevAttrPageableMemoryAccessUsesHostPageTables` - 指示 CPU/GPU 一致性的机制：1 表示硬件，0 表示软件。

[图 18](#unified-memory-flow-chart) 直观地说明了如何确定统一内存范式，其后是实现了相同逻辑的 [代码示例](#memory-unified-querying-code)。

统一内存操作有四种范式：

- 完全支持显式托管内存分配
- 完全支持所有分配（软件一致性）
- 完全支持所有分配（硬件一致性）
- 有限的统一内存支持

当完全支持可用时，它可能要求显式分配，或者所有系统内存可能隐式地成为统一内存。当所有内存都是隐式统一内存时，一致性机制可以是软件或硬件。Windows 和一些 Tegra 设备对统一内存的支持有限。

![统一内存范式流程图](../images/unified-memory-explainer.png)

*图 18所有当前的 GPU 都使用统一的虚拟地址空间，并且具有可用的统一内存。当 `cudaDevAttrConcurrentManagedAccess` 为 1 时，完全的统一内存支持可用，否则只有有限支持可用。当完全支持可用时，如果 `cudaDevAttrPageableMemoryAccess` 也为 1，那么所有系统内存都是统一内存。否则，只有使用 CUDA API（如 `cudaMallocManaged`）分配的内存才是统一内存。当所有系统内存都是统一内存时，`cudaDevAttrPageableMemoryAccessUsesHostPageTables` 指示一致性是由硬件（当值为 1 时）还是软件（当值为 0 时）提供的。#*
[表 3](#table-unified-memory-levels) 以表格形式展示了与[图 18](#unified-memory-flow-chart) 相同的信息，并提供了指向本章相关章节以及本指南后续章节中更完整文档的链接。

| 统一内存范式 | 设备属性 | 完整文档 |
| --- | --- | --- |
| 有限的统一内存支持 | cudaDevAttrConcurrentManagedAccess 为 0 | Windows、WSL 和 Tegra 上的统一内存 CUDA for Tegra 内存管理 Tegra 上的统一内存 |
| 对显式管理的内存分配提供完全支持 | cudaDevAttrPageableMemoryAccess 为 0 且 cudaDevAttrConcurrentManagedAccess 为 1 | 仅支持 CUDA 托管内存的设备上的统一内存 |
| 对所有分配提供完全支持（软件一致性） | cudaDevAttrPageableMemoryAccessUsesHostPageTables 为 0 且 cudaDevAttrPageableMemoryAccess 为 1 且 cudaDevAttrConcurrentManagedAccess 为 1 | 具有完整 CUDA 统一内存支持的设备上的统一内存 |
| 对所有分配提供完全支持（硬件一致性） | cudaDevAttrPageableMemoryAccessUsesHostPageTables 为 1 且 cudaDevAttrPageableMemoryAccess 为 1 且 cudaDevAttrConcurrentManagedAccess 为 1 | 具有完整 CUDA 统一内存支持的设备上的统一内存 |

#### 2.4.2.1.1. 统一内存范式：代码示例

以下代码示例演示了如何查询设备属性，并遵循[图 18](#unified-memory-flow-chart) 的逻辑，确定系统中每个 GPU 的统一内存范式。

```cuda
void queryDevices()
{
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    for(int i=0; i<numDevices; i++)
    {
        cudaSetDevice(i);
        cudaInitDevice(0, 0, 0);
        int deviceId = i;

        int concurrentManagedAccess = -1;     
        cudaDeviceGetAttribute (&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, deviceId);    
        int pageableMemoryAccess = -1;
        cudaDeviceGetAttribute (&pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, deviceId);
        int pageableMemoryAccessUsesHostPageTables = -1;
        cudaDeviceGetAttribute (&pageableMemoryAccessUsesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, deviceId);

        printf("Device %d has ", deviceId);
        if(concurrentManagedAccess){
            if(pageableMemoryAccess){
                printf("full unified memory support");
                if( pageableMemoryAccessUsesHostPageTables)
                    { printf(" with hardware coherency\n");  }
                else
                    { printf(" with software coherency\n"); }
            }
            else
                { printf("full unified memory support for CUDA-made managed allocations\n"); }
        }
        else
        {   printf("limited unified memory support: Windows, WSL, or Tegra\n");  }
    }
}
```

### 2.4.2.2. 完整的统一内存功能支持

大多数 Linux 系统都提供完整的统一内存支持。如果设备属性 `cudaDevAttrPageableMemoryAccess` 为 1，那么所有系统内存，无论是通过 CUDA API 还是系统 API 分配的，都将作为统一内存运行，并具备完整的功能支持。这包括使用 `mmap` 创建的文件支持的内存分配。
如果 `cudaDevAttrPageableMemoryAccess` 为 0，则只有由 CUDA 分配为托管内存的内存才表现为统一内存。使用系统 API 分配的内存不受管理，且不一定能从 GPU 内核访问。

通常，对于具有完全支持的统一分配：

- 托管内存通常分配在首次访问它的处理器的内存空间中
- 当托管内存被当前驻留处理器以外的处理器使用时，通常会进行迁移
- 托管内存以内存页（软件一致性）或缓存行（硬件一致性）的粒度进行迁移或访问
- 允许超额订阅：应用程序可以分配比 GPU 上物理可用内存更多的托管内存

分配和迁移行为可能偏离上述情况。程序员可以使用[提示和预取](#memory-mem-advise-prefetch)来影响此行为。关于完全统一内存支持的完整覆盖范围，请参阅[具有完全 CUDA 统一内存支持的设备上的统一内存](../04-special-topics/unified-memory.html#um-pageable-systems)。

#### 2.4.2.2.1. 具有硬件一致性的完全统一内存

在诸如 Grace Hopper 和 Grace Blackwell 等硬件上，其中使用了 NVIDIA CPU 且 CPU 和 GPU 之间的互连是 NVLink 芯片到芯片（C2C），地址转换服务（ATS）可用。当 ATS 可用时，`cudaDevAttrPageableMemoryAccessUsesHostPageTables` 为 1。

使用 ATS 时，除了对所有主机分配提供完全统一内存支持外：

- GPU 分配（例如 `cudaMalloc`）可以从 CPU 访问（`cudaDevAttrDirectManagedMemAccessFromHost` 将为 1）
- CPU 和 GPU 之间的链路支持原生原子操作（`cudaDevAttrHostNativeAtomicSupported` 将为 1）
- 与软件一致性相比，硬件一致性支持可以提高性能

ATS 提供了[HMM](#memory-heterogeneous-memory-management)的所有功能。当 ATS 可用时，HMM 会自动禁用。关于硬件与软件一致性的进一步讨论，请参阅[CPU 和 GPU 页表：硬件一致性与软件一致性](../04-special-topics/unified-memory.html#um-hw-coherency)。

#### 2.4.2.2.2. HMM - 具有软件一致性的完全统一内存

*异构内存管理*（HMM）是 Linux 操作系统（具有适当内核版本）上可用的一项功能，它支持软件一致性的[完全统一内存支持](#memory-unified-memory-full)。异构内存管理为 PCIe 连接的 GPU 带来了 ATS 提供的部分功能和便利性。

在至少具有 Linux 内核 6.1.24、6.2.11 或 6.3 及更高版本的 Linux 系统上，异构内存管理（HMM）可能可用。可以使用以下命令来查找寻址模式是否为 `HMM`。

```c++
$ nvidia-smi -q | grep Addressing
Addressing Mode : HMM
```

当 HMM 可用时，支持[完全统一内存](#memory-unified-memory-full)，并且所有系统分配都是隐式的统一内存。如果系统同时具有[ATS](#memory-unified-address-translation-services)，则 HMM 被禁用并使用 ATS，因为 ATS 提供了 HMM 的所有功能及更多。
### 2.4.2.3. 有限统一内存支持

在 Windows（包括适用于 Linux 的 Windows 子系统 (WSL)）以及某些 Tegra 系统上，仅提供统一内存功能的有限子集。在这些系统上，托管内存可用，但 CPU 和 GPU 之间的迁移行为有所不同。

- 托管内存首先分配在 CPU 的物理内存中
- 托管内存的迁移粒度大于虚拟内存页
- 当 GPU 开始执行时，托管内存会迁移到 GPU
- 在 GPU 处于活动状态时，CPU 不得访问托管内存
- 当 GPU 同步时，托管内存会迁移回 CPU
- 不允许超额订阅 GPU 内存
- 只有由 CUDA 显式分配为托管内存的内存才是统一的

关于此范式的完整说明，请参阅 [Windows、WSL 和 Tegra 上的统一内存](../04-special-topics/unified-memory.html#um-legacy-devices)。

### 2.4.2.4. 内存建议与预取

程序员可以向管理统一内存的 NVIDIA 驱动程序提供提示，以帮助其最大化应用程序性能。CUDA API `cudaMemAdvise` 允许程序员指定分配的属性，这些属性会影响其放置位置以及当从另一个设备访问时内存是否迁移。

`cudaMemPrefetchAsync` 允许程序员建议开始将特定分配异步迁移到不同位置。一个常见的用法是在内核启动之前，开始传输内核将要使用的数据。这使得数据复制可以在其他 GPU 内核执行时进行。

关于 [性能提示](../04-special-topics/unified-memory.html#um-perf-hints) 的部分涵盖了可以传递给 `cudaMemAdvise` 的不同提示，并展示了使用 `cudaMemPrefetchAsync` 的示例。

## 2.4.3. 页锁定主机内存

在 [入门代码示例](intro-to-cuda-cpp.html#intro-cuda-cpp-all-together) 中，使用了 `cudaMallocHost` 在 CPU 上分配内存。这会在主机上分配 *页锁定* 内存（也称为 *固定* 内存）。通过传统分配机制（如 `malloc`、`new` 或 `mmap`）进行的主机分配不是页锁定的，这意味着它们可能被操作系统交换到磁盘或物理上重新定位。

[CPU 和 GPU 之间的异步复制](asynchronous-execution.html#async-execution-memory-transfers) 需要页锁定的主机内存。页锁定的主机内存还能提高同步复制的性能。页锁定内存可以 [映射](#memory-mapped-memory) 到 GPU，以便 GPU 内核直接访问。

CUDA 运行时提供了用于分配页锁定主机内存或锁定现有分配的 API：

- `cudaMallocHost` 分配页锁定的主机内存
- `cudaHostAlloc` 默认行为与 `cudaMallocHost` 相同，但也接受标志来指定其他内存参数
- `cudaFreeHost` 释放由 `cudaMallocHost` 或 `cudaHostAlloc` 分配的内存
- `cudaHostRegister` 对 CUDA API 之外（例如使用 `malloc` 或 `mmap`）分配的现有内存范围进行页锁定
`cudaHostRegister` 使得由第三方库或开发者控制之外的代码所分配的主机内存能够被页锁定，从而可用于异步拷贝或映射。

!!! note "注意"
    页锁定的主机内存可用于系统中所有 GPU 的异步拷贝和映射内存。在非 I/O 一致的 Tegra 设备上，页锁定的主机内存不会被缓存。此外，非 I/O 一致的 Tegra 设备不支持 `cudaHostRegister()`。

### 2.4.3.1. 映射内存

在具有 [HMM](#memory-heterogeneous-memory-management) 或 [ATS](#memory-unified-address-translation-services) 的系统上，所有主机内存都可以使用主机指针直接从 GPU 访问。当 ATS 或 HMM 不可用时，可以通过将内存*映射*到 GPU 的内存空间，使主机分配的内存对 GPU 可访问。映射内存始终是页锁定的。

下面的代码示例将演示直接在映射的主机内存上运行的数组拷贝内核。

```cuda
__global__ void copyKernel(float* a, float* b)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        a[idx] = b[idx];
}
```

虽然映射内存在某些情况下可能有用，例如内核需要访问某些未复制到 GPU 的数据，但在内核中访问映射内存需要通过 CPU-GPU 互连、PCIe 或 NVLink C2C 进行事务传输。与访问设备内存相比，这些操作具有更高的延迟和更低的带宽。对于内核的大部分内存需求，映射内存不应被视为 [统一内存](#memory-unified-memory) 或 [显式内存管理](intro-to-cuda-cpp.html#intro-cpp-explicit-memory-management) 的高性能替代方案。

#### 2.4.3.1.1. cudaMallocHost 和 cudaHostAlloc

使用 `cudaHostMalloc` 或 `cudaHostAlloc` 分配的主机内存会自动映射。这些 API 返回的指针可以直接在内核代码中用于访问主机上的内存。主机内存通过 CPU-GPU 互连进行访问。

**cudaMallocHost**

```cuda
void usingMallocHost() {
  float* a = nullptr;
  float* b = nullptr;
  
  CUDA_CHECK(cudaMallocHost(&a, vLen*sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&b, vLen*sizeof(float)));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(a, b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaMallocHost: ");
  checkAnswer(a,b);
}
```

**cudaAllocHost**

```cuda
void usingCudaHostAlloc() {
  float* a = nullptr;
  float* b = nullptr;

  CUDA_CHECK(cudaHostAlloc(&a, vLen*sizeof(float), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&b, vLen*sizeof(float), cudaHostAllocMapped));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(a, b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaAllocHost: ");
  checkAnswer(a, b);
}
```
#### 2.4.3.1.2. cudaHostRegister

当 ATS 和 HMM 不可用时，仍然可以使用 `cudaHostRegister` 将系统分配器分配的内存映射为可直接从 GPU 内核访问。然而，与使用 CUDA API 创建的内存不同，内核无法使用主机指针访问此内存。必须使用 `cudaHostGetDevicePointer()` 获取设备内存区域中的指针，并且在内核代码中必须使用该指针进行访问。

```cuda
void usingRegister() {
  float* a = nullptr;
  float* b = nullptr;
  float* devA = nullptr;
  float* devB = nullptr;

  a = (float*)malloc(vLen*sizeof(float));
  b = (float*)malloc(vLen*sizeof(float));
  CUDA_CHECK(cudaHostRegister(a, vLen*sizeof(float), 0 ));
  CUDA_CHECK(cudaHostRegister(b, vLen*sizeof(float), 0  ));

  CUDA_CHECK(cudaHostGetDevicePointer((void**)&devA, (void*)a, 0));
  CUDA_CHECK(cudaHostGetDevicePointer((void**)&devB, (void*)b, 0));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(devA, devB);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaHostRegister: ");
  checkAnswer(a, b);
}
```

#### 2.4.3.1.3. 统一内存与映射内存的比较

映射内存使得 GPU 可以访问 CPU 内存，但并不保证所有类型的访问（例如原子操作）在所有系统上都受支持。统一内存则保证支持所有访问类型。

映射内存驻留在 CPU 内存中，这意味着所有 GPU 访问都必须通过 CPU 和 GPU 之间的连接（PCIe 或 NVLink）进行。通过这些链路进行访问的延迟显著高于访问 GPU 内存，并且总可用带宽也更低。因此，将所有内核内存访问都使用映射内存，不太可能充分利用 GPU 计算资源。

统一内存通常会迁移到正在访问它的处理器的物理内存中。首次迁移后，内核对同一内存页或缓存行的重复访问可以利用完整的 GPU 内存带宽。

!!! note "注意"
    映射内存在过去的一些文档中也被称为零拷贝内存。在所有 CUDA 应用程序使用统一虚拟地址空间之前，需要额外的 API 来启用内存映射（使用 `cudaDeviceMapHost` 标志调用 `cudaSetDeviceFlags`）。现在不再需要这些 API。在映射的主机内存上操作的原子函数（参见原子函数），从主机或其他 GPU 的角度来看并不是原子的。CUDA 运行时要求，从设备发起的对主机内存的 1 字节、2 字节、4 字节、8 字节和 16 字节自然对齐的加载和存储操作，从主机和其他设备的角度来看，必须保持为单次访问。在某些平台上，对内存的原子操作可能会被硬件拆分为单独的加载和存储操作。这些组件的加载和存储操作对保持自然对齐访问有相同的要求。CUDA 运行时不支持 PCI Express 总线拓扑中 PCI Express 桥拆分 8 字节自然对齐操作的情况，并且 NVIDIA 不知道有任何拓扑会拆分 16 字节自然对齐的操作。
## 2.4.4. 总结

- 在支持异构内存管理（HMM）或地址转换服务（ATS）的 Linux 平台上，所有系统分配的内存都是托管内存。
- 在不支持 HMM 或 ATS 的 Linux 平台、Tegra 处理器以及所有 Windows 平台上，托管内存必须使用 CUDA 分配：使用 `cudaMallocManaged` 或使用由 `allocType=cudaMemAllocationTypeManaged` 创建的池的 `cudaMallocFromPoolAsync`。使用 `__managed__` 说明符的全局变量。
- 在 Windows 和 Tegra 处理器上，统一内存存在限制。
- 在支持 ATS 的 NVLINK C2C 互连系统上，使用 `cudaMalloc` 分配的设备内存可以直接从 CPU 或其他 GPU 访问。

 本页