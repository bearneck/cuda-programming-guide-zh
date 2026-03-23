# 2.3 异步执行

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

---

此页面是否有帮助？

# 2.3. 异步执行

## 2.3.1. 什么是异步并发执行？

CUDA 允许多个任务并发或重叠执行，具体包括：

- 主机上的计算
- 设备上的计算
- 从主机到设备的内存传输
- 从设备到主机的内存传输
- 给定设备内存内部的内存传输
- 设备之间的内存传输

这种并发性通过异步接口来表达，其中调度函数调用或内核启动会立即返回。异步调用通常在调度的操作完成之前就返回，甚至可能在异步操作开始之前就返回。这样，应用程序就可以在最初调度的操作执行的同时，自由地执行其他任务。当需要最初调度操作的最终结果时，应用程序必须执行某种形式的同步，以确保相关操作已经完成。并发执行模式的一个典型例子是主机与设备之间的内存传输与计算重叠，从而减少或消除其开销。

![使用 CUDA 流进行异步并发执行](../images/cuda_streams.png)

*图 17 使用 CUDA 流进行异步并发执行*

通常，异步接口主要提供三种方式来与调度的操作进行同步：

- **阻塞方式**：应用程序调用一个会阻塞或等待的函数，直到操作完成。
- **非阻塞方式（轮询方式）**：应用程序调用一个立即返回的函数，并提供有关操作状态的信息。
- **回调方式**：当操作完成时，执行一个预先注册的函数。

虽然编程接口是异步的，但实际执行各种操作并发的能力取决于 CUDA 版本和所用硬件的计算能力——这些细节将留到本指南后面的章节讨论（参见[计算能力](../05-appendices/compute-capabilities.html#compute-capabilities)）。

在[同步 CPU 和 GPU](intro-to-cuda-cpp.html#intro-synchronizing-the-gpu) 中，介绍了 CUDA 运行时函数 `cudaDeviceSynchronize()`，这是一个阻塞调用，会等待所有先前发出的工作完成。之所以需要调用 `cudaDeviceSynchronize()`，是因为内核启动是异步的并且会立即返回。CUDA 为同步提供了阻塞和非阻塞两种方式的 API，甚至支持使用主机端回调函数。

CUDA 中异步执行的核心 API 组件是 **CUDA 流** 和 **CUDA 事件**。在本节的剩余部分，我们将解释如何使用这些元素来表达 CUDA 中的异步执行。

一个相关的主题是 **CUDA 图**，它允许预先定义一个异步操作图，然后可以以最小的开销重复执行。我们将在 [2.4.9.2 使用流捕获的 CUDA 图简介](#async-execution-cuda-graphs) 一节中非常入门地介绍 CUDA 图，并在 [4.1 CUDA 图](../04-special-topics/cuda-graphs.html#cuda-graphs) 一节中提供更全面的讨论。
## 2.3.2.CUDA 流

在最基本的层面上，CUDA 流是一种抽象，允许程序员表达一系列操作。流就像一个工作队列，程序可以向其中添加操作（例如内存复制或内核启动），这些操作将按顺序执行。对于给定的流，队列前端的操作会被执行，然后出队，使得下一个排队的操作来到前端并等待执行。流中操作的执行顺序是顺序的，操作按照它们被加入流的顺序执行。

应用程序可以同时使用多个流。在这种情况下，运行时将根据 GPU 资源的状态，从有可用工作的流中选择一个任务来执行。可以为流分配优先级，作为影响调度的提示提供给运行时，但这并不保证特定的执行顺序。

在流中操作的 API 函数调用和内核启动相对于主机线程是异步的。应用程序可以通过等待流中任务为空来与流同步，也可以在设备级别进行同步。

CUDA 有一个默认流，没有指定特定流的操作和内核启动会被排队到这个默认流中。未指定流的代码示例隐式地使用了这个默认流。默认流具有一些特定的语义，将在小节 [阻塞和非阻塞流以及默认流](#async-execution-blocking-non-blocking-default-stream) 中讨论。

### 2.3.2.1.创建和销毁 CUDA 流

可以使用 `cudaStreamCreate()` 函数创建 CUDA 流。该函数调用会初始化流句柄，该句柄可用于在后续函数调用中标识该流。

```c
cudaStream_t stream;        // 流句柄
cudaStreamCreate(&stream);  // 创建一个新流

// 基于流的操作 ...

cudaStreamDestroy(stream);  // 销毁流
```

如果应用程序调用 `cudaStreamDestroy()` 时，设备仍在流 `stream` 中执行工作，那么该流将在完成流中的所有工作后才被销毁。

### 2.3.2.2.在 CUDA 流中启动内核

用于启动内核的通常的三尖括号语法也可用于将内核启动到特定的流中。流被指定为内核启动的一个额外参数。在以下示例中，名为 `kernel` 的内核被启动到句柄为 `stream` 的流中，该句柄类型为 `cudaStream_t`，并假定先前已创建：

```c
kernel<<<grid, block, shared_mem_size, stream>>>(...);
```

内核启动是异步的，函数调用会立即返回。假设内核启动成功，内核将在流 `stream` 中执行，并且在内核执行期间，应用程序可以自由地在 CPU 上或在 GPU 的其他流中执行其他任务。
### 2.3.2.3. 在 CUDA 流中启动内存传输

要在流中启动内存传输，我们可以使用函数 `cudaMemcpyAsync()`。此函数与 `cudaMemcpy()` 函数类似，但它需要一个额外的参数来指定用于内存传输的流。下面代码块中的函数调用在流 `stream` 中，将 `src` 指向的主机内存中的 `size` 字节复制到 `dst` 指向的设备内存。

```c
// 在流 `stream` 中，将 `size` 字节从 `src` 复制到 `dst`
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

与其他异步函数调用一样，此函数调用会立即返回，而 `cudaMemcpy()` 函数则会阻塞，直到内存传输完成。为了安全地访问传输结果，应用程序必须使用某种形式的同步来确定操作已完成。

其他 CUDA 内存传输函数（例如 `cudaMemcpy2D()`）也有异步变体。

!!! note "注意"
    为了使涉及 CPU 内存的内存复制能够异步执行，主机缓冲区必须是固定的（pinned）和页锁定的（page-locked）。如果使用未固定和页锁定的主机内存，`cudaMemcpyAsync()` 仍能正常工作，但会退化为同步行为，无法与其他工作重叠。这可能会抑制使用异步内存传输带来的性能优势。建议程序使用 `cudaMallocHost()` 来分配用于向 GPU 发送或从 GPU 接收数据的缓冲区。

### 2.3.2.4. 流同步

与流同步的最简单方法是等待流中的任务清空。这可以通过两种方式实现：使用 `cudaStreamSynchronize()` 函数或 `cudaStreamQuery()` 函数。

`cudaStreamSynchronize()` 函数将阻塞，直到流中的所有工作完成。

```c
// 等待流中的任务清空
cudaStreamSynchronize(stream);

// 此时流已完成
// 我们可以安全地访问流操作的结果
```

如果我们不希望阻塞，而只是需要快速检查流是否为空，可以使用 `cudaStreamQuery()` 函数。

```c
// 查看一下流的状态
// 如果流为空，则返回 cudaSuccess
// 如果流不为空，则返回 cudaErrorNotReady
cudaError_t status = cudaStreamQuery(stream);

switch (status) {
    case cudaSuccess:
        // 流为空
        std::cout << "The stream is empty" << std::endl;
        break;
    case cudaErrorNotReady:
        // 流不为空
        std::cout << "The stream is not empty" << std::endl;
        break;
    default:
        // 发生错误 - 我们应该处理这种情况
        break;
};
```

## 2.3.3. CUDA 事件

CUDA 事件是一种将标记插入 CUDA 流的机制。它们本质上就像示踪粒子，可用于跟踪流中任务的进度。想象一下向一个流中启动两个内核。如果没有这样的跟踪事件，我们只能确定流是空还是非空。如果我们有一个依赖于第一个内核输出的操作，那么在我们知道流为空（此时两个内核都已执行完毕）之前，我们将无法安全地启动该操作。
使用 CUDA 事件我们可以做得更好。通过在第一个内核之后、第二个内核之前，将一个事件直接排入流中，我们可以等待该事件到达流的前端。然后，我们可以安全地启动依赖操作，因为知道第一个内核已经完成，但第二个内核尚未开始。以这种方式使用 CUDA 事件可以在操作和流之间构建依赖关系图。这种图的类比直接对应到后面关于 [CUDA 图](#async-execution-cuda-graphs) 的讨论。

CUDA 流还保存时间信息，可用于计时内核启动和内存传输。

### 2.3.3.1. 创建和销毁 CUDA 事件

可以使用 `cudaEventCreate()` 和 `cudaEventDestroy()` 函数来创建和销毁 CUDA 事件。

```c
cudaEvent_t event;

// 创建事件
cudaEventCreate(&event);

// 执行一些涉及该事件的工作

// 一旦工作完成且不再需要该事件
// 我们可以销毁该事件
cudaEventDestroy(event);
```

应用程序负责在不再需要事件时销毁它们。

### 2.3.3.2. 将事件插入 CUDA 流

可以使用 `cudaEventRecord()` 函数将 CUDA 事件插入流中。

```c
cudaEvent_t event;
cudaStream_t stream;

// 创建事件
cudaEventCreate(&event);

// 将事件插入流中
cudaEventRecord(event, stream);
```

### 2.3.3.3. 在 CUDA 流中计时操作

CUDA 事件可用于计时各种流操作（包括内核）的执行时间。当事件到达流的前端时，它会记录一个时间戳。通过在一个内核前后放置两个事件，我们可以准确测量内核执行的持续时间，如下面的代码片段所示：

```c
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaEvent_t start;
cudaEvent_t stop;

// 创建事件
cudaEventCreate(&start);
cudaEventCreate(&stop);

 // 记录开始事件
cudaEventRecord(start, stream);

// 启动内核
kernel<<<grid, block, 0, stream>>>(...);

// 记录结束事件
cudaEventRecord(stop, stream);

// 等待流完成
// 两个事件都将被触发
cudaStreamSynchronize(stream);

// 获取计时结果
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

// 清理
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```

### 2.3.3.4. 检查 CUDA 事件的状态

与检查流状态的情况类似，我们可以以阻塞或非阻塞的方式检查事件的状态。

`cudaEventSynchronize()` 函数将阻塞直到事件完成。在下面的代码片段中，我们将一个内核启动到流中，然后是一个事件，接着是第二个内核。我们可以使用 `cudaEventSynchronize()` 函数来等待第一个内核之后的事件完成，原则上可以在 kernel2 完成之前立即启动一个依赖任务。

```c
cudaEvent_t event;
cudaStream_t stream;

// create the stream
cudaStreamCreate(&stream);

// create the event
cudaEventCreate(&event);

// launch a kernel into the stream
kernel<<<grid, block, 0, stream>>>(...);

// Record the event
cudaEventRecord(event, stream);

// launch a kernel into the stream
kernel2<<<grid, block, 0, stream>>>(...);

// Wait for the event to complete
// Kernel 1 will be  guaranteed to have completed
// and we can launch the dependent task.
cudaEventSynchronize(event);
dependentCPUtask();

// Wait for the stream to be empty
// Kernel 2 is guaranteed to have completed
cudaStreamSynchronize(stream);

// destroy the event
cudaEventDestroy(event);

// destroy the stream
cudaStreamDestroy(stream);
```

CUDA Events can be checked for completion in a non-blocking way using the `cudaEventQuery()` function. In the example below we launch 2 kernels into a stream. The first kernel, kernel1 generates some data which we would like to copy to the host, however we also have some CPU side work to do. In the code below, we enqueue kernel1 followed by an event (event) and then kernel2 into stream stream1. We then go into a CPU work loop, but occasionally take a peek to see if the event has completed indicating that kernel1 is done. If so, we launch a host to device copy into stream stream2. This approach allows the overlap of the CPU work with the GPU kernel execution and the device to host copy.

```c
cudaEvent_t event;
cudaStream_t stream1;
cudaStream_t stream2;

size_t size = LARGE_NUMBER;
float *d_data;

// Create some data
cudaMalloc(&d_data, size);
float *h_data = (float *)malloc(size);

// create the streams
cudaStreamCreate(&stream1);   // Processing stream
cudaStreamCreate(&stream2);   // Copying stream
bool copyStarted = false;

//  create the event
cudaEventCreate(&event);

// launch kernel1 into the stream
kernel1<<<grid, block, 0, stream1>>>(d_data, size);
// enqueue an event following kernel1
cudaEventRecord(event, stream1);

// launch kernel2 into the stream
kernel2<<<grid, block, 0, stream1>>>();

// while the kernels are running do some work on the CPU
// but check if kernel1 has completed because then we will start
// a device to host copy in stream2
while ( not allCPUWorkDone() || not copyStarted ) {
    doNextChunkOfCPUWork();

    // peek to see if kernel 1 has completed
    // if so enqueue a non-blocking copy into stream2
    if ( not copyStarted ) {
        if( cudaEventQuery(event) == cudaSuccess ) {
            cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2);
            copyStarted = true;
        }
    }
}

// wait for both streams to be done
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

// destroy the event
cudaEventDestroy(event);

// destroy the streams and free the data
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
cudaFree(d_data);
free(h_data);
```

## 2.3.4.Callback Functions from Streams

CUDA provides a mechanism for launching functions on the host from within a stream. There are currently two functions available for this purpose: `cudaLaunchHostFunc()` and `cudaAddCallback()`. However, `cudaAddCallback()` is slated for deprecation, so applications should use `cudaLaunchHostFunc()`.
使用 `cudaLaunchHostFunc()`

`cudaLaunchHostFunc()` 函数的签名如下：

```c
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*func)(void *), void *data);
```

其中

- stream : 要启动回调函数的目标流。
- func : 要启动的回调函数。
- data : 指向要传递给回调函数的数据的指针。

主机函数本身是一个简单的 C 函数，其签名为：

```c
void hostFunction(void *data);
```

其中 `data` 参数指向一个用户定义的数据结构，函数可以对其进行解释。使用此类回调函数时，需要注意一些注意事项。特别是，主机函数不得调用任何 CUDA API。

为了与统一内存配合使用，提供了以下执行保证：
- 在函数执行期间，流被视为空闲。因此，例如，函数始终可以使用附加到其入队所在流的内存。
- 函数开始执行的效果等同于在函数之前立即同步在同一流中记录的事件。因此，它会同步在函数之前已“连接”的流。
- 向任何流添加设备工作不会使流变为活动状态，直到所有先前的主机函数和流回调都已执行。因此，例如，即使工作已添加到另一个流，如果该工作已通过事件排序在函数调用之后，函数仍可能使用全局附加内存。
- 函数的完成不会导致流变为活动状态，除非如上所述。如果函数之后没有设备工作，流将保持空闲状态，并且在连续的主机函数或流回调之间（如果没有设备工作）也将保持空闲。因此，例如，可以通过在流末尾从主机函数发出信号来完成流同步。

### 2.3.4.1. 使用 cudaStreamAddCallback()

!!! note "注意"
    `cudaStreamAddCallback()` 函数已被标记为弃用并将被移除，此处讨论是为了完整性和因为它可能仍出现在现有代码中。应用程序应使用或切换到使用 `cudaLaunchHostFunc()`。

`cudaStreamAddCallback()` 函数的签名如下：

```c
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
```

其中

- stream : 要启动回调函数的目标流。
- callback : 要启动的回调函数。
- userData : 指向要传递给回调函数的数据的指针。
- flags : 目前，为了未来的兼容性，此参数必须为 0。

`callback` 函数的签名与我们使用 `cudaLaunchHostFunc()` 函数时的情况略有不同。在这种情况下，回调函数是一个 C 函数，其签名为：

```c
void callbackFunction(cudaStream_t stream, cudaError_t status, void *userData);
```
其中函数现在接收以下参数：

- stream：启动回调函数的流句柄。
- status：触发回调的流操作状态。
- userData：传递给回调函数的数据指针。

特别需要注意的是，`status` 参数将包含流的当前错误状态，该状态可能由先前的操作设置。与 `cudaLaunchHostFunc()` 函数的情况类似，在主机函数完成之前，流将不会激活并推进任务，并且回调函数内部不得调用任何 CUDA 函数。

### 2.3.4.2. 异步错误处理

在 CUDA 流中，错误可能源自流中的任何操作，包括内核启动和内存传输。这些错误在运行时可能不会传播回用户，直到流被同步，例如通过等待事件或调用 `cudaStreamSynchronize()`。有两种方法可以查明流中可能发生的错误：

- 使用函数 `cudaGetLastError()` —— 此函数返回并清除当前上下文中任何流遇到的最后一个错误。如果在两次调用之间没有发生其他错误，立即第二次调用 `cudaGetLastError()` 将返回 `cudaSuccess`。
- 使用函数 `cudaPeekAtLastError()` —— 此函数返回当前上下文中的最后一个错误，但不清除它。

这两个函数都将错误作为 `cudaError_t` 类型的值返回。可以使用函数 `cudaGetErrorName()` 和 `cudaGetErrorString()` 生成错误的可打印名称。

使用这些函数的示例如下：

清单 1
使用 cudaGetLastError() 和 cudaPeekAtLastError() 的示例
#

```c
// 在流中执行一些工作。
cudaStreamSynchronize(stream);

// 查看最后一个错误但不清除它
cudaError_t err = cudaPeekAtLastError();
if (err != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err));
    printf("Error description: %s\n", cudaGetErrorString(err));
}

// 查看最后一个错误并清除它
cudaError_t err2 = cudaGetLastError();
if (err2 != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err2));
    printf("Error description: %s\n", cudaGetErrorString(err2));
}

if (err2 != err) {
    printf("As expected, cudaPeekAtLastError() did not clear the error\n");
}

// 再次检查
cudaError_t err3 = cudaGetLastError();
if (err3 == cudaSuccess) {
    printf("As expected, cudaGetLastError() cleared the error\n");
}
```

!!! tip "提示"
    当错误在同步时出现，尤其是在包含许多操作的流中，通常很难精确定位错误在流中发生的确切位置。调试这种情况时，一个有用的技巧是设置环境变量 `CUDA_LAUNCH_BLOCKING=1`，然后运行应用程序。此环境变量的效果是在每次内核启动后都进行同步。这有助于追踪是哪个内核或传输导致了错误。
同步操作可能代价高昂；设置此环境变量后，应用程序的运行速度可能会显著降低。

## 2.3.5. CUDA 流排序

既然我们已经讨论了流、事件和回调函数的基本机制，那么考虑流中异步操作的排序语义就非常重要。这些语义旨在让应用程序开发人员能够以安全的方式思考流中操作的顺序。在某些特殊情况下，为了性能优化的目的，这些语义可能会被放宽，例如在*程序化依赖内核启动*场景中，它允许通过使用特殊属性和内核启动机制来重叠两个内核的执行；或者在使用 `cudaMemcpyBatchAsync()` 函数进行批量内存传输时，如果运行时能够并发执行非重叠的批量拷贝。我们将在后面讨论这些优化（*需要链接*）。

最重要的是，CUDA 流被称为**顺序流**。这意味着流中操作的执行顺序与这些操作被加入队列的顺序相同。流中的一个操作不能跨越其他操作。运行时跟踪内存操作（例如拷贝），并且这些操作总是在下一个操作之前完成，以便让依赖的内核能够安全地访问正在传输的数据。

## 2.3.6. 阻塞流、非阻塞流与默认流

在 CUDA 中，有两种类型的流：阻塞流和非阻塞流。这个名字可能有点误导性，因为阻塞和非阻塞语义仅指这些流如何与默认流同步。默认情况下，使用 `cudaStreamCreate()` 创建的流是阻塞流。要创建非阻塞流，必须使用 `cudaStreamCreateWithFlags()` 函数并指定 `cudaStreamNonBlocking` 标志：

```c
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

非阻塞流可以通过通常的方式使用 `cudaStreamDestroy()` 销毁。

### 2.3.6.1. 传统默认流

阻塞流和非阻塞流之间的关键区别在于它们如何与**默认流**同步。CUDA 提供了一个传统默认流（也称为 NULL 流或流 ID 为 0 的流），当在内核启动或阻塞式 `cudaMemcpy()` 调用中没有指定流时，就会使用这个默认流。这个在所有主机线程之间共享的默认流是一个阻塞流。当一个操作被启动到这个默认流中时，它将与所有其他阻塞流同步，换句话说，它将等待所有其他阻塞流完成后才能执行。

```c
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```

默认流的行为意味着，在上面的代码片段中，kernel2 将等待 kernel1 完成，而 kernel3 将等待 kernel2 完成，即使原则上这三个内核可以并发执行。通过创建非阻塞流，我们可以避免这种同步行为。在下面的代码片段中，我们创建了两个非阻塞流。默认流将不再与这些流同步，原则上所有三个内核都可以并发执行。因此，我们不能假设内核的执行有任何顺序，并且应该执行显式同步（例如使用相当重量级的 `cudaDeviceSynchronize()` 调用）以确保内核已完成。

```c
cudaStream_t stream1, stream2;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```

### 2.3.6.2.Per-thread Default Stream

Starting in CUDA-7, CUDA allows for each host thread to have its own independent default stream, rather than the shared legacy default stream. In order to enable this behavior one must either use the nvcc compiler option `--default-stream per-thread` or define the `CUDA_API_PER_THREAD_DEFAULT_STREAM` preprocessor macro. When this behavior is enabled, each host thread will have its own independent default stream which will not synchronize with other streams in the same way the legacy default stream does. In such a situation the [legacy default stream example](#legacy-default-stream-example) will now exhibit the same synchronization behavior as the [non-blocking stream example](#non-blocking-stream-example).

## 2.3.7.Explicit Synchronization

There are various ways to explicitly synchronize streams with each other.

`cudaDeviceSynchronize()` waits until all preceding commands in all streams of all host threads have completed.

`cudaStreamSynchronize()`takes a stream as a parameter and waits until all preceding commands in the given stream have completed. It can be used to synchronize the host with a specific stream, allowing other streams to continue executing on the device.

`cudaStreamWaitEvent()`takes a stream and an event as parameters (see [CUDA Events](#cuda-events) for a description of events)and makes all the commands added to the given stream after the call to `cudaStreamWaitEvent()`delay their execution until the given event has completed.

`cudaStreamQuery()`provides applications with a way to know if all preceding commands in a stream have completed.

## 2.3.8.Implicit Synchronization

Two operations from different streams cannot run concurrently if any CUDA operation on the NULL stream is submitted in-between them, unless the streams are non-blocking streams (created with the `cudaStreamNonBlocking` flag).

Applications should follow these guidelines to improve their potential for concurrent kernel execution:

- All independent operations should be issued before dependent operations,
- Synchronization of any kind should be delayed as long as possible.

## 2.3.9.Miscellaneous and Advanced topics

### 2.3.9.1.Stream Prioritization

As mentioned previously, developers can assign priorities to CUDA streams. Prioritized streams need to be created using the `cudaStreamCreateWithPriority()` function. The function takes two parameters: the stream handle and the priority level. The general scheme is that lower numbers correspond to higher priorities. The given priority range for a given device and context can be queried using the `cudaDeviceGetStreamPriorityRange()` function. The default priority of a stream is 0.

```c
int minPriority, maxPriority;

// Query the priority range for the device
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);

// Create two streams with different priorities
// cudaStreamDefault indicates the stream should be created with default flags
// in other words they will be blocking streams with respect to the legacy default stream
// One could also use the option `cudaStreamNonBlocking` here to create a non-blocking streams
cudaStream_t stream1, stream2;
cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, minPriority);  // Lowest priority
cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, maxPriority);  // Highest priority
```

We should note that a priority of a stream is only a hint to the runtime and generally applies primarily to kernel launches, and may not be respected for memory transfers. Stream priorities will not preempt already executing work, or guarantee any specific execution order.

### 2.3.9.2.Introduction to CUDA Graphs with Stream Capture

CUDA streams allow programs to specify a sequence of operations, kernels or memory copies, in order. Using multiple streams and cross-stream dependencies with `cudaStreamWaitEvent`, an application can specify a full directed acyclic graph (DAG) of operations. Some applications may have a sequence or DAG of operations that needs to be run many times throughout execution.

For this situation, CUDA provides a feature known as CUDA graphs. This section introduces CUDA graphs and one mechanism of creating them called *stream capture*. A more detailed discussion of CUDA graphs is presented in [CUDA Graphs](../04-special-topics/cuda-graphs.html#cuda-graphs). Capturing or creating a graph can help reduce latency and CPU overhead of repeatedly invoking the same chain of API calls from the host thread. Instead, the APIs to specify the graph operations can be called once, and then the resulting graph executed many times.

CUDA Graphs work in the following way:

1. The graph is captured by the application. This step is done once the first time the graph is executed. The graph can also be manually composed using the CUDA graph API.
2. The graph is instantiated . This step is done one time, after the graph is captured. This step can set up all the various runtime structures needed to execute the graph, in order to make launching its components as fast as possible.
3. In the remaining steps, the pre-instantiated graph is executed as many times as required. Since all the runtime structures needed to execute the graph operations are already in place, the CPU overheads of the graph execution are minimized.

Listing 2 
The stages of capturing, instantiating and executing a simple linear graph using CUDA Graphs (from 
CUDA Developer Technical Blog
, A. Gray, 2019)
#

```c
#define N 500000 // tuned such that kernel takes a few microseconds

// A very lightweight kernel
__global__ void shortKernel(float * out_d, float * in_d){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N) out_d[idx]=1.23*in_d[idx];
}

bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;

// The graph will be executed NSTEP times
for(int istep=0; istep<NSTEP; istep++){
    if(!graphCreated){
        // Capture the graph
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // Launch NKERNEL kernels
        for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }

        // End the capture
        cudaStreamEndCapture(stream, &graph);

        // Instantiate the graph
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated=true;
    }

    // Launch the graph
    cudaGraphLaunch(instance, stream);

    // Synchronize the stream
    cudaStreamSynchronize(stream);
}
```
有关 CUDA 图的更多详细信息，请参阅 [CUDA 图](../04-special-topics/cuda-graphs.html#cuda-graphs)。

## 2.3.10. 异步执行总结

本节要点如下：

> 异步 API 允许我们表达任务的并发执行，提供了表达各种操作重叠的方式。实际实现的并发性取决于可用的硬件资源和计算能力。
> CUDA 中用于异步执行的关键抽象是流、事件和回调函数。
> 可以在事件、流和设备级别进行同步。
> 默认流是一个阻塞流，它会与所有其他阻塞流同步，但不会与非阻塞流同步。
> 可以通过编译器选项 `--default-stream per-thread` 或预处理器宏 `CUDA_API_PER_THREAD_DEFAULT_STREAM` 使用每线程默认流来避免默认流的行为。
> 可以创建具有不同优先级的流，这些优先级是对运行时的提示，对于内存传输可能不被遵守。
> CUDA 提供了 API 函数来减少或重叠内核启动和内存传输的开销，例如 CUDA 图、批量内存传输和编程式依赖内核启动。

本页内容