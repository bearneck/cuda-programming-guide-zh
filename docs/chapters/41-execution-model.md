# 5.8 CUDA C++ 执行模型

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html)

---

此页面有帮助吗？

# 5.8. CUDA C++ 执行模型

CUDA C++ 旨在为所有设备执行线程提供[并行前向进展 [intro.progress.9]](https://eel.is/c++draft/intro.progress#9)，以促进使用 CUDA C++ 对现有 C++ 应用程序进行并行化。

[intro.progress]

- [intro.progress.7] : 对于一个提供**并发前向进展保证**的执行线程，只要该线程尚未终止，实现应确保该线程最终会取得进展。[注 5：无论其他执行线程（如果有）是否已经或正在取得进展，此条均适用。最终满足此要求意味着这将在未指定但有限的时间内发生。â 尾注]
- [intro.progress.9] : 对于一个提供**并行前向进展保证**的执行线程，如果该线程尚未执行任何执行步骤，则实现无需确保该线程最终会取得进展；一旦该线程执行了一个步骤，它就提供**并发前向进展保证**。[注 6：这并未规定何时启动此执行线程的要求，这通常由创建此执行线程的实体指定。例如，一个提供并发前向进展保证的执行线程，以任意顺序一个接一个地执行一组任务中的任务，就满足了这些任务的并行前向进展要求。â 尾注]

CUDA C++ 编程语言是 C++ 编程语言的扩展。本节记录了当前 [ISO 国际标准 ISO/IEC 14882 â 编程语言 C++](https://eel.is/c++draft/) 草案中 [[intro.progress]](https://eel.is/c++draft/intro.progress) 部分的修改和扩展。修改的部分被明确标出，其差异以**粗体**显示。所有其他部分均为新增内容。

## 5.8.1. 主机线程

由主机实现创建的用于执行 [main](https://en.cppreference.com/w/cpp/language/main_function)、[std::thread](https://en.cppreference.com/w/cpp/thread/thread) 和 [std::jthread](https://en.cppreference.com/w/cpp/thread/jthread) 的执行线程所提供的前向进展，是主机实现的实现定义行为 [[intro.progress]](https://eel.is/c++draft/intro.progress)。通用主机实现应提供并发前向进展。

如果主机实现提供[并发前向进展 [intro.progress.7]](https://eel.is/c++draft/intro.progress#7)，则 CUDA C++ 为设备线程提供[并行前向进展 [intro.progress.9]](https://eel.is/c++draft/intro.progress#9)。

## 5.8.2. 设备线程

一旦一个设备线程取得进展：

- 如果它是**协作网格**的一部分，则其网格中的所有设备线程最终都应取得进展。
- 否则，其**线程块簇**中的所有设备线程最终都应取得进展。[注：不保证其他线程块簇中的线程最终会取得进展。 - 尾注。] [注：这意味着其线程块内的所有设备线程最终都应取得进展。 - 尾注。]
修改 [[intro.progress.1]](https://eel.is/c++draft/intro.progress#1) 如下（**粗体** 部分为修改内容）：

实现可以假定任何 **主机** 线程最终将执行以下操作之一：

> 终止，
> 调用函数
> std::this_thread::yield
> (
> [thread.thread.this]
> )，
> 调用库 I/O 函数，
> 通过 volatile glvalue 执行访问，
> 执行同步操作或原子操作，或
> 继续执行平凡无限循环 (
> [stmt.iter.general]
> )。

**实现可以假定任何设备线程最终将执行以下操作之一：**

> 终止，
> 调用库 I/O 函数，
> 通过 volatile glvalue 执行访问，除非指定对象具有自动存储期，或
> 执行同步操作或原子读操作，除非指定对象具有自动存储期。
> [注：设备线程相对于主机线程的某些当前限制是我们已知的实现缺陷，我们可能会随着时间的推移进行修复。
> 例如，设备线程最终仅对自动存储期对象执行 volatile 或原子操作所导致的未定义行为。
> 然而，设备线程相对于主机线程的其他限制是经过深思熟虑的选择。它们实现了性能优化，
> 如果设备线程严格遵循 C++ 标准，这些优化将无法实现。
> 例如，为最终仅执行原子写或栅栏操作的程序提供向前进展保证，
> 会降低整体性能，而实际收益甚微。 - 尾注]

由于对 [intro.progress.1] 的修改，主机线程与设备线程之间向前进展保证差异的示例。

以下示例分别使用 "host.threads.<id>" 和 "device.threads.<id>" 来指代上述主机和设备线程实现假设的列举子条款。

```cuda
1// 示例：Execution.Model.Device.0
2// 结果：网格最终根据 device.threads.4 终止，因为原子对象不具有自动存储期。
3__global__ void ex0(cuda::atomic_ref<int, cuda::thread_scope_device> atom) {
4    if (threadIdx.x == 0) {
5        while(atom.load(cuda::memory_order_relaxed) == 0);
6    } else if (threadIdx.x == 1) {
7        atom.store(1, cuda::memory_order_relaxed);
8    }
9}
```

```cuda
1// 示例：Execution.Model.Device.1
2// 允许的结果：没有线程取得进展，因为设备线程不支持 host.threads.2。
3__global__ void ex1() {
4    while(true) cuda::std::this_thread::yield();
5}
```

```cuda
1// 示例：Execution.Model.Device.2
2// 允许的结果：没有线程取得进展，因为设备线程不支持 host.threads.4
3//（对于具有自动存储期的对象，参见 device.threads.3 中的例外情况）。
4__global__ void ex2() {
5    volatile bool True = true;
6    while(True);
7}
```

```cuda
1// 示例：Execution.Model.Device.3
2// 允许的结果：没有线程取得进展，因为设备线程不支持 host.threads.5
3//（对于具有自动存储期的对象，参见 device.threads.4 中的例外情况）。
4__global__ void ex3() {
5    cuda::atomic<bool, cuda::thread_scope_thread> True = true;
6    while(True.load());
7}
```

```cuda
1// Example: Execution.Model.Device.4
2// Allowed outcome: No thread makes progress because device threads don't support host.thread.6.
3__global void ex4() {
4    while(true) { /* empty */ }
5}
```

## 5.8.3.CUDA APIs

A CUDA API call shall eventually either return or ensure at least one device thread makes progress.

CUDA query functions (e.g. [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435), [cudaEventQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7), etc.) shall not consistently return `cudaErrorNotReady` without a device thread making progress.

> [Note: The device thread need not be ârelatedâ to the API call, e.g., an API operating on one stream or process may ensure progress of a device thread on another stream or process. - end note.]
> [Note: A simple but not sufficient method to test a program for CUDA API Forward Progress conformance is to run them with following environment variables set:
> CUDA_DEVICE_MAX_CONNECTIONS=1
> CUDA_LAUNCH_BLOCKING=1
> , and then check that the program still terminates.
> If it does not, the program has a bug.
> This method is not sufficient because it does not catch all Forward Progress bugs, but it does catch many such bugs. - end note.]

Examples of CUDA API forward progress guarantees.

```cuda
 1// Example: Execution.Model.API.1
 2// Outcome: if no other device threads (e.g., from other processes) are making progress,
 3// this program terminates and returns cudaSuccess.
 4// Rationale: CUDA guarantees that if the device is empty:
 5// - `cudaDeviceSynchronize` eventually ensures that at least one device-thread makes progress, which implies that eventually `hello_world` grid and one of its device-threads start.
 6// - All thread-block threads eventually start (due to "if a device thread makes progress, all other threads in its thread-block cluster eventually make progress").
 7// - Once all threads in thread-block arrive at `__syncthreads` barrier, all waiting threads are unblocked.
 8// - Therefore all device threads eventually exit the `hello_world`` grid.
 9// - And `cudaDeviceSynchronize`` eventually unblocks.
10__global__ void hello_world() { __syncthreads(); }
11int main() {
12    hello_world<<<1,2>>>();
13    return (int)cudaDeviceSynchronize();
14}
```

```cuda
 1// Example: Execution.Model.API.2
 2// Allowed outcome: eventually, no thread makes progress.
 3// Rationale: the `cudaDeviceSynchronize` API below is only called if a device thread eventually makes progress and sets the flag.
 4// However, CUDA only guarantees that `producer` device thread eventually starts if the synchronization API is called.
 5// Therefore, the host thread may never be unblocked from the flag spin-loop.
 6cuda::atomic<int, cuda::thread_scope_system> flag = 0;
 7__global__ void producer() { flag.store(1); }
 8int main() {
 9    cudaHostRegister(&flag, sizeof(flag));
10    producer<<<1,1>>>();
11    while (flag.load() == 0);
12    return cudaDeviceSynchronize();
13}
```

```cuda
 1// Example: Execution.Model.API.3
 2// Allowed outcome: eventually, no thread makes progress.
 3// Rationale: same as Example.Model.API.2, with the addition that a single CUDA query API call does not guarantee
 4// the device thread eventually starts, only repeated CUDA query API calls do (see Execution.Model.API.4).
 5cuda::atomic<int, cuda::thread_scope_system> flag = 0;
 6__global__ void producer() { flag.store(1); }
 7int main() {
 8    cudaHostRegister(&flag, sizeof(flag));
 9    producer<<<1,1>>>();
10    (void)cudaStreamQuery(0);
11    while (flag.load() == 0);
12    return cudaDeviceSynchronize();
13}
```

```cuda
 1// Example: Execution.Model.API.4
 2// Outcome: terminates.
 3// Rationale: same as Execution.Model.API.3, but this example repeatedly calls
 4// a CUDA query API in within the flag spin-loop, which guarantees that the device thread
 5// eventually makes progress.
 6cuda::atomic<int, cuda::thread_scope_system> flag = 0;
 7__global__ void producer() { flag.store(1); }
 8int main() {
 9    cudaHostRegister(&flag, sizeof(flag));
10    producer<<<1,1>>>();
11    while (flag.load() == 0) {
12        (void)cudaStreamQuery(0);
13    }
14    return cudaDeviceSynchronize();
15}
```

### 5.8.3.1.Dependencies

A device thread shall not start until all its dependencies have completed.

> [Note: Dependencies that prevent device threads from starting to make progress can be created, for example, via
> CUDA Stream Commands
> .
> These may include dependencies on the completion of, among others,
> CUDA Events
> and
> CUDA Kernels
> . - end note.]

Examples of CUDA API forward progress guarantees due to dependencies

```cuda
 1// Example: Execution.Model.Stream.0
 2// Allowed outcome: eventually, no thread makes progress.
 3// Rationale: while CUDA guarantees that one device thread makes progress, since there
 4// is no dependency between `first` and `second`, it does not guarantee which thread,
 5// and therefore it could always pick the device thread from `second`, which then never
 6// unblocks from the spin-loop.
 7// That is, `second` may starve `first`.
 8cuda::atomic<int, cuda::thread_scope_system> flag = 0;
 9__global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
10__global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
11int main() {
12    cudaHostRegister(&flag, sizeof(flag));
13    cudaStream_t s0, s1;
14    cudaStreamCreate(&s0);
15    cudaStreamCreate(&s1);
16    first<<<1,1,0,s0>>>();
17    second<<<1,1,0,s1>>>();
18    return cudaDeviceSynchronize();
19}
```

```cuda
 1// Example: Execution.Model.Stream.1
 2// Outcome: terminates.
 3// Rationale: same as Execution.Model.Stream.0, but this example has a stream dependency
 4// between first and second, which requires CUDA to run the grids in order.
 5cuda::atomic<int, cuda::thread_scope_system> flag = 0;
 6__global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
 7__global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
 8int main() {
 9    cudaHostRegister(&flag, sizeof(flag));
10    cudaStream_t s0;
11    cudaStreamCreate(&s0);
12    first<<<1,1,0,s0>>>();
13    second<<<1,1,0,s0>>>();
14    return cudaDeviceSynchronize();
15}
```
在本页面