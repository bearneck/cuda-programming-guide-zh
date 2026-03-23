# 4.14 内存同步域

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html)

---

此页面有帮助吗？

# 4.14. 内存同步域

## 4.14.1. 内存栅栏干扰

某些 CUDA 应用程序可能会遇到性能下降，原因是内存栅栏/刷新操作等待的事务数量超过了 CUDA 内存一致性模型所必需的范围。

| __managed__ int x = 0 ; __device__ cuda :: atomic < int , cuda :: thread_scope_device > a ( 0 ); __managed__ cuda :: atomic < int , cuda :: thread_scope_system > b ( 0 ); |  |  |
| --- | --- | --- |
| 线程 1 (SM) x = 1 ; a = 1 ; | 线程 2 (SM) while ( a != 1 ) ; assert ( x == 1 ); b = 1 ; | 线程 3 (CPU) while ( b != 1 ) ; assert ( x == 1 ); |

考虑上面的例子。CUDA 内存一致性模型保证断言的条件为真，因此线程 1 对 `x` 的写入必须在线程 2 对 `b` 的写入之前对线程 3 可见。

由 `a` 的释放和获取提供的内存排序仅足以使 `x` 对线程 2 可见，而对线程 3 则不足，因为它是一个设备作用域的操作。因此，由 `b` 的释放和获取提供的系统作用域排序，不仅需要确保线程 2 自身发出的写入对线程 3 可见，还需要确保对线程 2 可见的其他线程的写入也对线程 3 可见。这被称为累积性。由于 GPU 在执行时无法知道哪些写入在源代码级别已被保证可见，哪些只是由于偶然的时序才可见，它必须保守地对所有进行中的内存操作撒下一张大网。

这有时会导致干扰：因为 GPU 正在等待那些在源代码级别并不需要等待的内存操作，栅栏/刷新操作可能花费比必要更长的时间。

请注意，栅栏可能作为代码中的显式内建函数或原子操作出现，如示例所示，也可能隐式地用于在任务边界实现 *synchronizes-with* 关系。

一个常见的例子是，当一个内核在本地 GPU 内存中执行计算，而另一个并行内核（例如来自 NCCL）正在与对等节点进行通信。计算完成后，本地内核将隐式刷新其写入，以满足与下游工作之间的任何 *synchronizes-with* 关系。这可能会不必要地、完全或部分地等待来自通信内核的较慢的 nvlink 或 PCIe 写入。

## 4.14.2. 使用域隔离流量

从计算能力 9.0（Hopper 架构）GPU 和 CUDA 12.0 开始，内存同步域特性提供了一种缓解此类干扰的方法。作为对代码显式协助的交换，GPU 可以减少栅栏操作所撒下的网的范围。每个内核启动都被赋予一个域 ID。写入和栅栏操作都带有该 ID 标签，并且栅栏将只对与栅栏域匹配的写入进行排序。在并发计算与通信的例子中，通信内核可以被放置在不同的域中。

使用域时，代码必须遵守以下规则：**在同一 GPU 上不同域之间的排序或同步需要系统作用域的栅栏**。在一个域内，设备作用域的栅栏仍然足够。这对于累积性是必要的，因为一个内核的写入不会被另一个域中的内核发出的栅栏所包含。本质上，通过确保跨域流量提前刷新到系统作用域来满足累积性。
请注意，这修改了 `thread_scope_device` 的定义。然而，由于内核将默认使用域 0（如下所述），因此保持了向后兼容性。

## 4.14.3. 在 CUDA 中使用域

可以通过新的启动属性 `cudaLaunchAttributeMemSyncDomain` 和 `cudaLaunchAttributeMemSyncDomainMap` 访问域。前者在逻辑域 `cudaLaunchMemSyncDomainDefault` 和 `cudaLaunchMemSyncDomainRemote` 之间进行选择，后者提供从逻辑域到物理域的映射。远程域旨在用于执行远程内存访问的内核，以将其内存流量与本地内核隔离开来。但请注意，选择特定域不会影响内核可以合法执行的内存访问。

可以通过设备属性 `cudaDevAttrMemSyncDomainCount` 查询域的数量。计算能力 9.0（Hopper）的设备有 4 个域。为了便于编写可移植代码，所有设备都可以使用域功能，CUDA 将在计算能力 9.0 之前的设备上报告数量为 1。

拥有逻辑域简化了应用程序的组合。在堆栈较低级别（例如来自 NCCL）的单个内核启动，可以选择语义逻辑域，而无需关心周围的应用程序架构。更高级别可以使用映射来引导逻辑域。如果未设置逻辑域，其默认值为默认域，默认映射是将默认域映射到 0，将远程域映射到 1（在具有超过 1 个域的 GPU 上）。特定的库可能会在 CUDA 12.0 及更高版本中使用远程域标记启动；例如，NCCL 2.16 将这样做。总之，这为常见应用程序提供了一个开箱即用的有益使用模式，无需在其他组件、框架或应用程序级别进行代码更改。另一种使用模式，例如在使用 NVSHMEM 的应用程序中或没有明确区分内核类型的情况下，可能是对并行流进行分区。流 A 可以将两个逻辑域都映射到物理域 0，流 B 映射到 1，依此类推。

```cuda
// 使用远程逻辑域启动内核的示例
cudaLaunchAttribute domainAttr;
domainAttr.id = cudaLaunchAttrMemSyncDomain;
domainAttr.val = cudaLaunchMemSyncDomainRemote;
cudaLaunchConfig_t config;
// 填写其他配置字段
config.attrs = &domainAttr;
config.numAttrs = 1;
cudaLaunchKernelEx(&config, myKernel, kernelArg1, kernelArg2...);
```

```cuda
// 为流设置映射的示例
// (此映射是计算能力 9.0 (Hopper) 或更高版本上流的默认映射（如果未显式设置），此处提供用于说明)
cudaLaunchAttributeValue mapAttr;
mapAttr.memSyncDomainMap.default_ = 0;
mapAttr.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(stream, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);
```

```cuda
// 将不同流映射到不同物理域的示例，忽略逻辑域设置
cudaLaunchAttributeValue mapAttr;
mapAttr.memSyncDomainMap.default_ = 0;
mapAttr.memSyncDomainMap.remote = 0;
cudaStreamSetAttribute(streamA, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);
mapAttr.memSyncDomainMap.default_ = 1;
mapAttr.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(streamB, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);
```
与其他启动属性一样，这些属性在 CUDA 流、使用 `cudaLaunchKernelEx` 的单个启动以及 CUDA 图中的内核节点上统一公开。典型的用法是如上所述在流级别设置映射，在启动级别（或包围流使用的一部分）设置逻辑域。

在流捕获期间，这两个属性都会被复制到图节点。图从节点本身获取这两个属性，这本质上是指定物理域的一种间接方式。在启动图的流上设置的与域相关的属性不会在图执行中使用。

 在本页