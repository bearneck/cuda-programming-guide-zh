# 3.5 CUDA 特性全览

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/feature-survey.html)

---

此页面有帮助吗？

# 3.5. CUDA 功能概览

本编程指南的第 1-3 节介绍了 CUDA 和 GPU 编程，涵盖了基础主题的概念和简单代码示例。本指南第 4 部分描述特定 CUDA 功能的章节，假定读者已了解本指南第 1-3 节所涵盖的概念。

CUDA 拥有许多适用于不同问题的功能。并非所有功能都适用于每个用例。本章旨在介绍这些功能，描述其预期用途以及可能有助于解决的问题。功能根据其旨在解决的问题类型进行了粗略分类。某些功能，例如 CUDA 图，可能适用于多个类别。

[第 4 节](../part4.html#cuda-features) 将更详细地介绍这些 CUDA 功能。

## 3.5.1. 提升内核性能

本节概述的功能均旨在帮助内核开发者最大化其内核的性能。

### 3.5.1.1. 异步屏障

[异步屏障](../04-special-topics/async-barriers.html#asynchronous-barriers) 在 [第 3.2.4.2 节](advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers) 中引入，允许对线程间的同步进行更精细的控制。异步屏障将屏障的到达和等待分离开来。这使得应用程序在等待其他线程到达的同时，可以执行不依赖于该屏障的工作。异步屏障可以为不同的 [线程作用域](advanced-kernel-programming.html#advanced-kernels-thread-scopes) 指定。异步屏障的完整细节见 [第 4.9 节](../04-special-topics/async-barriers.html#asynchronous-barriers)。

### 3.5.1.2. 异步数据拷贝与张量内存加速器 (TMA)

在 CUDA 内核代码上下文中，[异步数据拷贝](../04-special-topics/async-copies.html#async-copies) 指的是在执行计算的同时，在共享内存和 GPU DRAM 之间移动数据的能力。这不应与 CPU 和 GPU 之间的异步内存拷贝混淆。此功能利用了异步屏障。[第 4.11 节](../04-special-topics/async-copies.html#async-copies) 详细介绍了异步拷贝的使用。

### 3.5.1.3. 流水线

[流水线](../04-special-topics/pipelines.html#pipelines) 是一种用于暂存工作和协调多缓冲区生产者-消费者模式的机制，通常用于将计算与 [异步数据拷贝](../04-special-topics/async-copies.html#async-copies) 重叠执行。[第 4.10 节](../04-special-topics/pipelines.html#pipelines) 提供了在 CUDA 中使用流水线的详细信息和示例。

### 3.5.1.4. 使用集群启动控制进行工作窃取

工作窃取是一种在不均匀工作负载中保持利用率的技术，已完成工作的“工作者”可以从其他“工作者”那里“窃取”任务。集群启动控制是计算能力 10.0 (Blackwell) 中引入的一项功能，它赋予内核直接控制正在执行中的线程块调度的能力，以便它们能够实时适应不均匀的工作负载。一个线程块可以取消另一个尚未启动的线程块或集群的启动，声明其索引，并立即开始执行被窃取的工作。这种工作窃取流程可以保持 SM 繁忙，并减少在数据不规则或运行时变化下的空闲时间——在不单独依赖硬件调度器的情况下，实现了更细粒度的负载均衡。
[第 4.12 节](../04-special-topics/cluster-launch-control.html#cluster-launch-control) 提供了有关如何使用此功能的详细信息。

## 3.5.2. 降低延迟

本节概述的功能有一个共同的主题，即旨在降低某种类型的延迟，尽管不同功能所解决的延迟类型各不相同。总的来说，它们主要关注内核启动级别或更高级别的延迟。内核内部的 GPU 内存访问延迟不在此处考虑的延迟范围内。

### 3.5.2.1. 绿色上下文

[绿色上下文](../04-special-topics/green-contexts.html#green-contexts)，也称为*执行上下文*，是 CUDA 一项功能的名称，该功能使程序能够创建仅在一个 GPU 的部分 SM 上执行工作的 [CUDA 上下文](driver-api.html#driver-api-context)。默认情况下，内核启动的线程块会被分派到 GPU 内任何能够满足内核资源要求的 SM 上。有许多因素会影响哪些 SM 可以执行线程块，包括但不限于：共享内存使用量、寄存器使用量、集群的使用以及线程块中的线程总数。

执行上下文允许内核在一个特殊创建的上下文中启动，该上下文进一步限制了可用于执行内核的 SM 数量。重要的是，当程序创建一个使用某些 SM 集合的绿色上下文时，GPU 上的其他上下文将不会将线程块调度到分配给该绿色上下文的 SM 上。这包括主上下文，即 CUDA 运行时使用的默认上下文。这使得这些 SM 可以保留给高优先级或对延迟敏感的工作负载。

[第 4.6 节](../04-special-topics/green-contexts.html#green-contexts) 详细介绍了绿色上下文的使用。绿色上下文在 CUDA 13.1 及更高版本的 CUDA 运行时中可用。

### 3.5.2.2. 流序内存分配

[流序内存分配器](../04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocator) 允许程序将 GPU 内存的分配和释放操作排序到 [CUDA 流](../02-basics/asynchronous-execution.html#cuda-streams) 中。与立即执行的 `cudaMalloc` 和 `cudaFree` 不同，`cudaMallocAsync` 和 `cudaFreeAsync` 将内存分配或释放操作插入到 CUDA 流中。[第 4.3 节](../04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocator) 涵盖了这些 API 的所有细节。

### 3.5.2.3. CUDA 图

[CUDA 图](../04-special-topics/cuda-graphs.html#cuda-graphs) 使应用程序能够指定一系列 CUDA 操作（例如内核启动或内存复制）以及这些操作之间的依赖关系，以便它们可以在 GPU 上高效执行。使用 [CUDA 流](../02-basics/asynchronous-execution.html#cuda-streams) 可以获得类似的行为，实际上，创建图的一种机制称为 [流捕获](../04-special-topics/cuda-graphs.html#cuda-graphs-creating-a-graph-using-stream-capture)，它可以将流上的操作记录到 CUDA 图中。也可以使用 [CUDA 图 API](../04-special-topics/cuda-graphs.html#cuda-graphs-creating-a-graph-using-graph-apis) 来创建图。
创建图后，可以多次实例化和执行。这对于指定将重复执行的工作负载非常有用。图在减少与调用 CUDA 操作相关的 CPU 启动开销方面提供了一些性能优势，并且能够实现仅在提前指定整个工作负载时才可用的优化。

[第 4.2 节](../04-special-topics/cuda-graphs.html#cuda-graphs) 描述并演示了如何使用 CUDA 图。

### 3.5.2.4. 编程式依赖启动

[编程式依赖启动](../04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization) 是 CUDA 的一项功能，它允许一个依赖内核（即依赖于先前内核输出的内核）在其依赖的主内核完成之前就开始执行。依赖内核可以执行设置代码和无关工作，直到它需要来自主内核的数据并在那里阻塞。主内核可以在依赖内核所需的数据准备就绪时发出信号，这将释放依赖内核以继续执行。这使得内核之间可以实现一些重叠，有助于保持 GPU 利用率高的同时，最小化关键数据路径的延迟。[第 4.5 节](../04-special-topics/programmatic-dependent-launch.html#programmatic-dependent-launch-and-synchronization) 涵盖了编程式依赖启动。

### 3.5.2.5. 延迟加载

[延迟加载](../04-special-topics/lazy-loading.html#lazy-loading) 是一项允许控制在应用程序启动时 JIT 编译器如何运行的功能。如果应用程序有许多需要从 PTX JIT 编译到 cubin 的内核，并且所有内核都在应用程序启动时作为启动过程的一部分进行 JIT 编译，则可能会经历较长的启动时间。默认行为是模块在需要之前不会被编译。这可以通过使用[环境变量](../05-appendices/environment-variables.html#cuda-environment-variables)来更改，如[第 4.7 节](../04-special-topics/lazy-loading.html#lazy-loading)所述。

## 3.5.3. 功能特性

此处描述的特性有一个共同点，即它们旨在启用额外的能力或功能。

### 3.5.3.1. 扩展 GPU 内存

[扩展 GPU 内存](../04-special-topics/extended-gpu-memory.html#extended-gpu-memory) 是 NVLink-C2C 连接系统中可用的一项功能，它支持从 GPU 内部高效访问系统内的所有内存。EGM 在[第 4.17 节](../04-special-topics/extended-gpu-memory.html#extended-gpu-memory)中有详细介绍。

### 3.5.3.2. 动态并行性

CUDA 应用程序最常见的是从 CPU 上运行的代码启动内核。也可以从 GPU 上运行的内核创建新的内核调用。此功能称为 [CUDA 动态并行性](../04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism)。[第 4.18 节](../04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism) 涵盖了从 GPU 上运行的代码创建新的 GPU 内核启动的详细信息。
## 3.5.4. CUDA 互操作性

### 3.5.4.1. CUDA 与其他 API 的互操作性

除了 CUDA 之外，还有其他在 GPU 上运行代码的机制。GPU 最初是为了加速计算机图形应用而构建的，这些应用使用自己的一套 API，例如 Direct3D 和 Vulkan。应用程序可能希望使用图形 API 之一进行 3D 渲染，同时使用 CUDA 执行计算。CUDA 提供了在 CUDA 上下文和 3D API 使用的 GPU 上下文之间交换存储在 GPU 上数据的机制。例如，应用程序可以使用 CUDA 执行模拟，然后使用 3D API 创建结果的可视化。这是通过使某些缓冲区在 CUDA 和图形 API 中都可读和/或可写来实现的。

用于与图形 API 共享缓冲区的相同机制，也用于与通信机制共享缓冲区，这可以在多节点环境中实现快速、直接的 GPU 到 GPU 通信。

[第 4.19 节](../04-special-topics/graphics-interop.html#cuda-interoperability) 描述了 CUDA 如何与其他 GPU API 互操作，以及如何在 CUDA 与其他 API 之间共享数据，并针对多种不同的 API 提供了具体示例。

### 3.5.4.2. 进程间通信

对于非常大的计算，通常一起使用多个 GPU，以利用更多内存和更多计算资源共同处理一个问题。在单个系统内，或者用集群计算的术语来说，在单个节点内，可以在单个主机进程中使用多个 GPU。这在 [第 3.4 节](multi-gpu-systems.html#multi-gpu-introduction) 中有所描述。

跨单个计算机或多个计算机使用单独的主机进程也很常见。当多个进程协同工作时，它们之间的通信称为进程间通信。CUDA 进程间通信（CUDA IPC）提供了在不同进程之间共享 GPU 缓冲区的机制。[第 4.15 节](../04-special-topics/inter-process-communication.html#interprocess-communication) 解释并演示了如何使用 CUDA IPC 在不同的主机进程之间进行协调和通信。

## 3.5.5. 细粒度控制

### 3.5.5.1. 虚拟内存管理

如 [第 2.4.1 节](../02-basics/understanding-memory.html#memory-unified-virtual-address-space) 所述，系统中的所有 GPU 以及 CPU 内存共享一个统一的虚拟地址空间。大多数应用程序可以使用 CUDA 提供的默认内存管理，而无需更改其行为。然而，[CUDA 驱动程序 API](driver-api.html#driver-api) 为需要它的用户提供了对此虚拟内存空间布局的高级和详细控制。这主要适用于控制缓冲区在单个系统内以及跨多个系统的 GPU 之间共享时的行为。

[第 4.16 节](../04-special-topics/virtual-memory-management.html#virtual-memory-management) 涵盖了 CUDA 驱动程序 API 提供的控制、它们的工作原理以及开发人员何时会发现它们有优势。
### 3.5.5.2. 驱动程序入口点访问

[驱动程序入口点访问](../04-special-topics/driver-entry-point-access.html#driver-entry-point-access) 指的是从 CUDA 11.3 开始，能够获取指向 CUDA 驱动程序和 CUDA 运行时 API 的函数指针的能力。它还允许开发者获取驱动程序函数特定变体的函数指针，并访问比 CUDA 工具包中可用的版本更新的驱动程序中的函数。[第 4.20 节](../04-special-topics/driver-entry-point-access.html#driver-entry-point-access) 涵盖了驱动程序入口点访问。

### 3.5.5.3. 错误日志管理

[错误日志管理](../04-special-topics/error-log-management.html#error-log-management) 提供了用于处理和记录来自 CUDA API 错误的实用程序。设置单个环境变量 `CUDA_LOG_FILE` 即可将 CUDA 错误直接捕获到 stderr、stdout 或文件中。错误日志管理还允许应用程序注册一个回调函数，该回调在 CUDA 遇到错误时触发。[第 4.8 节](../04-special-topics/error-log-management.html#error-log-management) 提供了关于错误日志管理的更多详细信息。

 本页