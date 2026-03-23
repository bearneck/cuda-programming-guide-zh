# 4.15 进程间通信

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)

---

本页面是否有帮助？

# 4.15. 进程间通信

通过使用进程间通信（IPC）API 和 IPC 可共享内存缓冲区，可以支持由不同主机进程管理的多个 GPU 之间的通信。这是通过创建进程可移植句柄来实现的，这些句柄随后用于获取指向对等 GPU 上设备内存的进程本地设备指针。

由主机线程创建的任何设备内存指针或事件句柄都可以被同一进程内的任何其他线程直接引用。然而，设备指针或事件句柄在创建它们的进程之外无效，因此不能被属于不同进程的线程直接引用。为了跨进程访问设备内存和 CUDA 事件，应用程序必须使用 CUDA 进程间通信（IPC）或虚拟内存管理 API 来创建进程可移植句柄，这些句柄可以使用标准主机操作系统 IPC 机制（例如，进程间共享内存或文件）与其他进程共享。一旦进程可移植句柄在进程之间交换，就必须使用 CUDA IPC 或 VMM API 从这些句柄获取进程本地设备指针。然后，进程本地设备指针可以像在单个进程内一样使用。

用于单节点和单个操作系统实例内 IPC 的同类可移植句柄方法，也用于多节点 NVLink 连接集群中 GPU 之间的点对点通信。在多节点情况下，通信的 GPU 由每个集群节点上独立操作系统实例内运行的进程管理，这需要在操作系统实例级别之上进行额外的抽象。多节点对等通信是通过在多节点 GPU 对等体之间创建和交换所谓的“结构”句柄，然后在参与进程和对应于多节点等级的相应操作系统实例内获取进程本地设备指针来实现的。

有关用于建立和交换进程可移植以及节点和操作系统实例可移植句柄的具体 API（这些句柄用于获取用于 GPU 通信的进程本地设备指针），请参见下文（单节点 CUDA IPC）和 ref::virtual-memory-management。

!!! note "注意"
    当用于 IPC 时，使用 CUDA IPC API 和虚拟内存管理（VMM）API 各有其优势和限制。CUDA IPC API 目前仅在 Linux 平台上受支持。CUDA 虚拟内存管理 API 允许在内存分配时对每个分配进行对等可访问性和共享控制，但需要使用 CUDA 驱动程序 API。

## 4.15.1. 使用传统进程间通信 API 进行 IPC

为了跨进程共享设备内存指针和事件，应用程序必须使用 CUDA 进程间通信 API，该 API 在参考手册中有详细描述。IPC API 允许应用程序使用 `cudaIpcGetMemHandle()` 获取给定设备内存指针的 IPC 句柄。CUDA IPC 句柄可以使用标准主机操作系统 IPC 机制（例如，进程间共享内存或文件）传递给另一个进程。`cudaIpcOpenMemHandle()` 使用 IPC 句柄来检索可在另一个进程中使用的有效设备指针。事件句柄可以使用类似的入口点进行共享。
使用 IPC API 的一个示例是：单个主进程生成一批输入数据，使多个辅助进程能够访问这些数据，而无需重新生成或复制。

!!! note "注意"
    IPC API 仅在 Linux 上受支持。请注意，cudaMallocManaged 分配不支持 IPC API。使用 CUDA IPC 相互通信的应用程序应使用相同的 CUDA 驱动程序和运行时进行编译、链接和运行。出于性能原因，由 cudaMalloc() 进行的分配可能会从一个更大的内存块中进行子分配。在这种情况下，CUDA IPC API 将共享整个底层内存块，这可能导致其他子分配被共享，从而可能在进程之间造成信息泄露。为防止此行为，建议仅共享具有 2MiB 对齐大小的分配。在计算能力为 7.x 及更高版本的 L4T 和嵌入式 Linux Tegra 设备上，仅支持 IPC 事件共享 API。Tegra 平台不支持 IPC 内存共享 API。

## 4.15.2. 使用虚拟内存管理 API 进行 IPC

CUDA 虚拟内存管理 API 允许创建 IPC 可共享的内存分配，并且通过特定于操作系统的 IPC 句柄数据结构，它支持多个操作系统。

在本页