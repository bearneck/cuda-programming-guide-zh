# 4.5 可编程依赖启动与同步

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

---

此页面是否有帮助？

# 4.5. 编程式依赖启动与同步

*编程式依赖启动*机制允许一个依赖性的*次级*内核，在其所依赖的*主*内核（在同一 CUDA 流中）完成执行之前就启动。此技术从计算能力 9.0 的设备开始可用，当*次级*内核可以完成大量不依赖于*主*内核结果的工作时，它可以提供性能优势。

## 4.5.1. 背景

CUDA 应用程序通过在 GPU 上启动和执行多个内核来利用 GPU。典型的 GPU 活动时间线如[图 39](#gpu-activity)所示。

![GPU 活动时间线](../images/gpu-activity.png)

*图 39 GPU 活动时间线#*

这里，`secondary_kernel` 在 `primary_kernel` 完成其执行后启动。串行化执行通常是必要的，因为 `secondary_kernel` 依赖于 `primary_kernel` 产生的结果数据。如果 `secondary_kernel` 不依赖于 `primary_kernel`，则可以通过使用 [CUDA 流](../02-basics/asynchronous-execution.html#cuda-streams) 来同时启动它们。即使 `secondary_kernel` 依赖于 `primary_kernel`，也存在一些并发执行的潜力。例如，几乎所有的内核都有某种*前导*部分，在此期间执行诸如清零缓冲区或加载常量值等任务。

![``secondary_kernel`` 的前导部分](../images/secondary-kernel-preamble.png)

*图 40 secondary_kernel 的前导部分#*

[图 40](#secondary-kernel-preamble) 展示了 `secondary_kernel` 中可以并发执行而不影响应用程序的部分。请注意，并发启动还允许我们将 `secondary_kernel` 的启动延迟隐藏在 `primary_kernel` 的执行之后。

![``primary_kernel`` 和 ``secondary_kernel`` 的并发执行](../images/preamble-overlap.png)

*图 41 primary_kernel 和 secondary_kernel 的并发执行#*

[图 41](#preamble-overlap) 中所示的 `secondary_kernel` 的并发启动和执行，可以通过使用*编程式依赖启动*来实现。

*编程式依赖启动*引入了对 CUDA 内核启动 API 的更改，如下节所述。这些 API 至少需要计算能力 9.0 才能提供重叠执行。

## 4.5.2. API 描述

在编程式依赖启动中，一个主内核和一个次级内核在同一 CUDA 流中启动。当主内核准备好让次级内核启动时，它应该让所有线程块执行 `cudaTriggerProgrammaticLaunchCompletion`。次级内核必须使用可扩展启动 API 启动，如下所示。

```c++
__global__ void primary_kernel() {
   // 应在启动次级内核前完成的初始工作

   // 触发次级内核
   cudaTriggerProgrammaticLaunchCompletion();

   // 可以与次级内核同时进行的工作
}

__global__ void secondary_kernel()
{
   // 独立工作

   // 将阻塞，直到次级内核所依赖的所有主内核都已完成并将结果刷新到全局内存
   cudaGridDependencySynchronize();

   // 依赖性的工作
}

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;
configSecondary.attrs = attribute;
configSecondary.numAttrs = 1;

primary_kernel<<<grid_dim, block_dim, 0, stream>>>();
cudaLaunchKernelEx(&configSecondary, secondary_kernel);
```
当使用 `cudaLaunchAttributeProgrammaticStreamSerialization` 属性启动次级内核时，CUDA 驱动程序可以安全地提前启动次级内核，而无需等待主内核完成并刷新内存后再启动次级内核。

当所有主线程块都已启动并执行了 `cudaTriggerProgrammaticLaunchCompletion` 后，CUDA 驱动程序即可启动次级内核。如果主内核未执行该触发器，则会在主内核中的所有线程块退出后隐式触发。

无论哪种情况，次级线程块都可能在主内核写入的数据可见之前启动。因此，当次级内核配置为*程序化依赖启动*时，它必须始终使用 `cudaGridDependencySynchronize` 或其他方法来验证主内核的结果数据是否可用。

请注意，这些方法为主内核和次级内核提供了并发执行的机会，但此行为是机会性的，并不保证会导致内核并发执行。依赖这种方式实现并发执行是不安全的，并可能导致死锁。

## 4.5.3. 在 CUDA 图中的使用

程序化依赖启动可以通过[流捕获](cuda-graphs.html#cuda-graphs-creating-a-graph-using-stream-capture)或直接通过[边数据](cuda-graphs.html#cuda-graphs-edge-data)在 [CUDA 图](cuda-graphs.html#cuda-graphs) 中使用。要在具有边数据的 CUDA 图中编程此功能，请在连接两个内核节点的边上使用 `cudaGraphDependencyType` 值为 `cudaGraphDependencyTypeProgrammatic`。此边类型使上游内核对下游内核中的 `cudaGridDependencySynchronize()` 可见。此类型必须与 `cudaGraphKernelNodePortLaunchCompletion` 或 `cudaGraphKernelNodePortProgrammatic` 的出端口一起使用。

流捕获对应的图等效项如下：

```c++
cudaLaunchAttribute attribute;
attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute.val.programmaticStreamSerializationAllowed = 1;
```

```c++
cudaGraphEdgeData edgeData;
edgeData.type = cudaGraphDependencyTypeProgrammatic;
edgeData.from_port = cudaGraphKernelNodePortProgrammatic;
```

```c++
cudaLaunchAttribute attribute;
attribute.id = cudaLaunchAttributeProgrammaticEvent;
attribute.val.programmaticEvent.triggerAtBlockStart = 0;
```

```c++
cudaGraphEdgeData edgeData;
edgeData.type = cudaGraphDependencyTypeProgrammatic;
edgeData.from_port = cudaGraphKernelNodePortProgrammatic;
```

```c++
cudaLaunchAttribute attribute;
attribute.id = cudaLaunchAttributeProgrammaticEvent;
attribute.val.programmaticEvent.triggerAtBlockStart = 1;
```

```c++
cudaGraphEdgeData edgeData;
edgeData.type = cudaGraphDependencyTypeProgrammatic;
edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion;
```

 本页