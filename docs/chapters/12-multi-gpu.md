# 3.4 多 GPU 系统编程

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)

---

此页面是否有帮助？

# 3.4. 多 GPU 编程系统

多 GPU 编程允许应用程序通过利用多 GPU 系统提供的更大聚合算术性能、内存容量和内存带宽，处理超出单 GPU 能力的问题规模，并达到更高的性能水平。

CUDA 通过主机 API、驱动程序基础设施以及支持的 GPU 硬件技术实现多 GPU 编程：

- 主机线程 CUDA 上下文管理
- 系统中所有处理器的统一内存寻址
- GPU 之间的点对点批量内存传输
- GPU 之间细粒度的点对点加载/存储内存访问
- 更高级别的抽象和支持的系统软件，例如 CUDA 进程间通信、使用 NCCL 的并行归约，以及使用 NVLink 和/或 GPU-Direct RDMA 配合 NVSHMEM 和 MPI 等 API 进行通信

在最基本的层面上，多 GPU 编程要求应用程序同时管理多个活动的 CUDA 上下文，将数据分发到各个 GPU，在 GPU 上启动内核以完成其工作，并进行通信或收集结果，以便应用程序能够处理这些结果。具体实现细节取决于如何将应用程序的算法、可用并行性和现有代码结构最有效地映射到合适的多 GPU 编程方法。一些最常见的多 GPU 编程方法包括：

- 单个主机线程驱动多个 GPU
- 多个主机线程，每个线程驱动自己的 GPU
- 多个单线程主机进程，每个进程驱动自己的 GPU
- 包含多个线程的多个主机进程，每个线程驱动自己的 GPU
- 多节点 NVLink 连接的集群，GPU 由在集群节点间跨多个操作系统实例运行的线程和进程驱动

GPU 可以通过设备内存之间的内存传输和点对点访问进行相互通信，涵盖上述列出的每种多设备工作分发方法。通过查询并启用点对点 GPU 内存访问，并利用 NVLink 实现设备间的高带宽传输和更细粒度的加载/存储操作，可以支持高性能、低延迟的 GPU 通信。

CUDA 统一虚拟寻址允许同一主机进程内的多个 GPU 进行通信，只需最少的额外步骤来查询和启用高性能点对点内存访问和传输（例如，通过 NVLink）。

通过使用进程间通信（IPC）和虚拟内存管理（VMM）API，可以支持由不同主机进程管理的多个 GPU 之间的通信。[进程间通信](../04-special-topics/inter-process-communication.html#interprocess-communication) 部分讨论了高级 IPC 概念和节点内 CUDA IPC API 的介绍。高级虚拟内存管理（VMM）API 支持节点内和多节点 IPC，可在 Linux 和 Windows 操作系统上使用，并允许对内存缓冲区的 IPC 共享进行按分配粒度的控制，如 [虚拟内存管理](../04-special-topics/virtual-memory-management.html#virtual-memory-management) 中所述。
CUDA 本身提供了在一组 GPU（可能包括主机）内实现集体操作所需的 API，但它本身并不提供高级的多 GPU 集体 API。多 GPU 集体操作由更高抽象层次的 CUDA 通信库提供，例如 [NCCL](https://developer.nvidia.com/nccl) 和 [NVSHMEM](https://developer.nvidia.com/nvshmem)。

## 3.4.1. 多设备上下文与执行管理

应用程序要使用多个 GPU，首先需要枚举可用的 GPU 设备，根据其硬件属性、CPU 亲和性以及对等连接性从可用设备中进行适当选择，并为应用程序将使用的每个设备创建 CUDA 上下文。

### 3.4.1.1. 设备枚举

以下代码示例展示了如何查询支持 CUDA 的设备数量、枚举每个设备并查询其属性。

```cpp
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}
```

### 3.4.1.2. 设备选择

主机线程可以随时通过调用 `cudaSetDevice()` 来设置其当前正在操作的设备。设备内存分配和内核启动将在当前设备上进行；流和事件将与当前设置的设备相关联创建。在主机线程调用 `cudaSetDevice()` 之前，当前设备默认为设备 0。

以下代码示例说明了设置当前设备如何影响后续的内存分配和内核执行操作。

```cpp
size_t size = 1024 * sizeof(float);
cudaSetDevice(0);            // 将设备 0 设置为当前设备
float* p0;
cudaMalloc(&p0, size);       // 在设备 0 上分配内存
MyKernel<<<1000, 128>>>(p0); // 在设备 0 上启动内核

cudaSetDevice(1);            // 将设备 1 设置为当前设备
float* p1;
cudaMalloc(&p1, size);       // 在设备 1 上分配内存
MyKernel<<<1000, 128>>>(p1); // 在设备 1 上启动内核
```

### 3.4.1.3. 多设备流、事件和内存复制行为

如果内核启动被提交给一个与当前设备无关的流，则该启动将失败，如下列代码示例所示。

```cpp
cudaSetDevice(0);               // 将设备 0 设置为当前设备
cudaStream_t s0;
cudaStreamCreate(&s0);          // 在设备 0 上创建流 s0
MyKernel<<<100, 64, 0, s0>>>(); // 在设备 0 的流 s0 中启动内核

cudaSetDevice(1);               // 将设备 1 设置为当前设备
cudaStream_t s1;
cudaStreamCreate(&s1);          // 在设备 1 上创建流 s1
MyKernel<<<100, 64, 0, s1>>>(); // 在设备 1 的流 s1 中启动内核

// 此内核启动将失败，因为流 s0 与设备 1 无关：
MyKernel<<<100, 64, 0, s0>>>(); // 在设备 1 的流 s0 中启动内核
```

即使内存复制操作被提交给一个与当前设备无关的流，该操作也会成功。
如果输入事件和输入流关联到不同的设备，`cudaEventRecord()` 将会失败。

如果两个输入事件关联到不同的设备，`cudaEventElapsedTime()` 将会失败。

即使输入事件关联的设备与当前设备不同，`cudaEventSynchronize()` 和 `cudaEventQuery()` 也会成功。

即使输入流和输入事件关联到不同的设备，`cudaStreamWaitEvent()` 也会成功。因此，`cudaStreamWaitEvent()` 可用于在多个设备之间进行同步。

每个设备都有自己的[默认流](../02-basics/asynchronous-execution.html#async-execution-blocking-non-blocking-default-stream)，因此，发往某个设备默认流的命令，与发往任何其他设备默认流的命令相比，可能会乱序执行或并发执行。

## 3.4.2. 多设备点对点传输与内存访问

### 3.4.2.1. 点对点内存传输

CUDA 可以执行设备间的内存传输，并且在点对点内存访问可行时，会利用专用的复制引擎和 NVLink 硬件来最大化性能。

`cudaMemcpy` 可以与复制类型 `cudaMemcpyDeviceToDevice` 或 `cudaMemcpyDefault` 一起使用。

否则，必须使用 `cudaMemcpyPeer()`、`cudaMemcpyPeerAsync()`、`cudaMemcpy3DPeer()` 或 `cudaMemcpy3DPeerAsync()` 执行复制，如下面的代码示例所示。

```cpp
cudaSetDevice(0);                   // 将设备 0 设为当前设备
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // 在设备 0 上分配内存

cudaSetDevice(1);                   // 将设备 1 设为当前设备
float* p1;
cudaMalloc(&p1, size);              // 在设备 1 上分配内存

cudaSetDevice(0);                   // 将设备 0 设为当前设备
MyKernel<<<1000, 128>>>(p0);        // 在设备 0 上启动内核

cudaSetDevice(1);                   // 将设备 1 设为当前设备
cudaMemcpyPeer(p1, 1, p0, 0, size); // 将 p0 复制到 p1
MyKernel<<<1000, 128>>>(p1);        // 在设备 1 上启动内核
```

在两个不同设备内存之间进行的复制（在隐式的 *NULL* 流中）：
- 只有在先前发往任一设备的所有命令都完成后才会开始，并且
- 在复制操作完成后，发往任一设备的任何后续命令（参见异步执行）才能开始。

与流的正常行为一致，两个设备内存之间的异步复制操作，可能与另一个流中的复制或内核操作重叠。

如果两个设备之间启用了点对点访问（例如，如[点对点内存访问](#multi-gpu-peer-to-peer-memory-access)中所述），则这两个设备之间的点对点内存复制不再需要通过主机进行中转，因此速度更快。

### 3.4.2.2. 点对点内存访问

根据系统属性（特别是 PCIe 和/或 NVLink 拓扑结构），设备能够寻址彼此的内存（即，在一个设备上执行的内核可以解引用指向另一个设备内存的指针）。如果 `cudaDeviceCanAccessPeer()` 对指定的设备返回 true，则这两个设备之间支持点对点内存访问。
必须通过调用 `cudaDeviceEnablePeerAccess()` 在两个设备之间启用点对点内存访问，如下方代码示例所示。在未启用 NVSwitch 的系统上，每个设备最多支持八个系统范围内的点对点连接。

两个设备使用统一的虚拟地址空间（参见[统一虚拟地址空间](../02-basics/understanding-memory.html#memory-unified-virtual-address-space)），因此可以使用相同的指针来寻址两个设备上的内存，如下方代码示例所示。

```cpp
cudaSetDevice(0);                   // 将设备 0 设为当前设备
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // 在设备 0 上分配内存
MyKernel<<<1000, 128>>>(p0);        // 在设备 0 上启动内核

cudaSetDevice(1);                   // 将设备 1 设为当前设备
cudaDeviceEnablePeerAccess(0, 0);   // 启用与设备 0 的点对点访问

// 在设备 1 上启动内核
// 此内核启动可以访问设备 0 上地址为 p0 的内存
MyKernel<<<1000, 128>>>(p0);
```

!!! note "注意"
    使用 `cudaDeviceEnablePeerAccess()` 启用点对点内存访问，会对等设备上所有先前及后续的 GPU 内存分配产生全局性影响。
通过 `cudaDeviceEnablePeerAccess()` 启用对某个设备的点对点访问，会增加该对等设备上设备内存分配操作的运行时开销。这是因为需要使这些分配的内存能够立即被当前设备以及任何其他也具有访问权限的对等设备访问，从而增加了与对等设备数量成比例的乘法开销。一个更具可扩展性的替代方案是，不将所有设备内存分配都启用点对点访问，而是利用 CUDA 虚拟内存管理 API，仅在分配时根据需要显式分配可被对等访问的内存区域。
通过在内存分配时显式请求点对点可访问性，对于不可被对等设备访问的内存分配，其分配操作的运行时开销不会受到影响，并且点对点可访问的数据结构被正确地限定范围，从而改进了软件调试和可靠性（参见 ref:: virtual-memory-management）。

### 3.4.2.3. 点对点内存一致性

必须使用同步操作来强制排序并确保分布在多个设备上并发执行的线程网格对内存访问的正确性。跨设备同步的线程在 `thread_scope_system` [同步作用域](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes) 内操作。同样，内存操作也属于 `thread_scope_system` [内存同步域](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-synchronization-domains)。

当只有一个 GPU 访问某个对象时，CUDA ref::atomic-functions 可以在对等设备内存中的该对象上执行读-修改-写操作。点对点原子操作的要求和限制在 CUDA 内存模型的 [原子性要求](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity) 讨论中进行了描述。
### 3.4.2.4. 多设备托管内存

托管内存可用于支持点对点（peer-to-peer）访问的多 GPU 系统。关于并发多设备托管内存访问的详细要求，以及 GPU 独占访问托管内存的 API，在[多 GPU](../04-special-topics/unified-memory.html#um-legacy-multi-gpu) 章节中有详细描述。

### 3.4.2.5. 主机 IOMMU 硬件、PCI 访问控制服务与虚拟机

具体在 Linux 上，CUDA 和显示驱动程序不支持在启用 IOMMU 的裸机 PCIe 点对点内存传输。然而，CUDA 和显示驱动程序确实支持通过虚拟机透传（pass through）使用 IOMMU。在裸机系统上运行 Linux 时，必须禁用 IOMMU，以防止发生静默的设备内存损坏。相反，对于虚拟机，则应启用 IOMMU 并使用 VFIO 驱动程序进行 PCIe 透传。

在 Windows 上，不存在上述 IOMMU 限制。

另请参阅 [在 64 位平台上分配 DMA 缓冲区](https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/README/dma_issues.html)。

此外，可以在支持 IOMMU 的系统上启用 PCI 访问控制服务（ACS）。PCI ACS 功能会通过 CPU 根复合体（root complex）重定向所有 PCI 点对点流量，这可能会由于整体二分带宽的减少而导致显著的性能损失。

 本页