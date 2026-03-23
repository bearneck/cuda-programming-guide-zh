# 5.7 CUDA C++ 内存模型

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)

---

本页面是否有帮助？

# 5.7. CUDA C++ 内存模型

标准 C++ 呈现的视图是：线程同步的成本是统一且较低的。

CUDA C++ 则不同：线程同步的成本随着线程之间的距离增加而增长。在线程块内的线程之间同步成本较低，但在运行于多个 GPU 和 CPU 上的系统中任意线程之间同步成本很高。

为了解释并非总是很低且非均匀的线程同步成本，CUDA C++ 通过**线程作用域**扩展了标准 C++ 内存模型和并发设施（位于 `cuda::` 命名空间中），默认情况下保留了标准 C++ 的语法和语义。

## 5.7.1. 线程作用域

**线程作用域**指定了可以使用同步原语（例如 [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html) 或 [cuda::barrier](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/barrier.html)）相互同步的线程种类。

```cuda
namespace cuda {

enum thread_scope {
  thread_scope_system,
  thread_scope_device,
  thread_scope_block,
  thread_scope_thread
};

}  // namespace cuda
```

### 5.7.1.1. 作用域关系

每个程序线程通过一个或多个线程作用域关系与其他程序线程相关联：

*   系统中的每个线程通过系统线程作用域 `cuda::thread_scope_system` 与系统中的其他每个线程相关联。
*   每个 GPU 线程通过设备线程作用域 `cuda::thread_scope_device` 与同一 CUDA 设备内且在同一内存同步域中的其他每个 GPU 线程相关联。
*   每个 GPU 线程通过块线程作用域 `cuda::thread_scope_block` 与同一 CUDA 线程块中的其他每个 GPU 线程相关联。
*   每个线程通过线程线程作用域 `cuda::thread_scope_thread` 与自身相关联。

## 5.7.2. 同步原语

当使用 `cuda::thread_scope_system` 作用域实例化时，命名空间 `std::` 和 `cuda::std::` 中的类型与命名空间 `cuda::` 中的相应类型具有相同的行为。

## 5.7.3. 原子性

一个原子操作在其指定的作用域下是原子的，如果：

*   它指定的作用域不是 `cuda::thread_scope_system`，或者
*   作用域是 `cuda::thread_scope_system` 并且：它影响系统分配内存中的对象且 `pageableMemoryAccess` 为 1 [0]，或者它影响托管内存中的对象且 `concurrentManagedAccess` 为 1，或者它影响映射内存中的对象且 `hostNativeAtomicSupported` 为 1，或者它是一个影响映射内存中自然对齐且大小为 1、2、4、8 或 16 字节的对象的加载或存储操作 [1]，或者它影响 GPU 内存中的对象，只有 GPU 线程访问它，并且每个访问的源设备 `srcDev` 与对象所在的 GPU `dstDev` 之间的 `cudaDeviceGetP2PAttribute(&val, cudaDevP2PAttrNativeAtomicSupported, srcDev, dstDev)` 返回 1，或者只有来自单个 GPU 的 GPU 线程并发访问它。

!!! note "注意"
    [0] 如果 `PageableMemoryAccessUsesHostPagetables` 为 0，则对内存映射文件或 hugetlbfs 分配的原子操作不是原子的。[1] 如果 `hostNativeAtomicSupported` 为 0，则在系统作用域下影响映射内存中自然对齐且大小为 1、2、4、8 或 16 字节的对象的原子加载或存储操作...
在系统分配的内存或映射内存中，自然对齐的 16 字节宽对象需要系统支持。NVIDIA 尚未发现任何系统缺乏此支持，并且没有可用的 CUDA API 查询来检测此类系统。

有关[系统分配内存](../04-special-topics/unified-memory.html#um-details-intro)、[托管内存](../04-special-topics/unified-memory.html#um-details-intro)、[映射内存](../02-basics/understanding-memory.html#memory-mapped-memory)、CPU 内存和 GPU 内存的更多信息，请参阅本指南的相关章节。

## 5.7.4. 数据竞争

对 ISO/IEC IS 14882（C++ 标准）的 [intro.races 第 21 段](https://eel.is/c++draft/intro.races#21) 修改如下：

> 如果一个程序的执行包含两个可能并发的冲突操作，并且其中至少有一个不是原子操作
> 在包含执行了另一个操作的线程的作用域内
> ，并且两者之间没有"先发生于"关系（除了下面描述的信号处理程序的特殊情况），则该程序的执行包含数据竞争。
> 任何此类数据竞争都会导致未定义行为。[…]

对 ISO/IEC IS 14882（C++ 标准）的 [thread.barrier.class 第 4 段](https://eel.is/c++draft/thread.barrier.class#4) 修改如下：

> 4. 对 `barrier` 成员函数（其析构函数除外）的并发调用不会引入数据竞争
> ，就好像它们是原子操作一样
> 。[…]

对 ISO/IEC IS 14882（C++ 标准）的 [thread.latch.class 第 2 段](https://eel.is/c++draft/thread.latch.class#2) 修改如下：

> 2. 对 `latch` 成员函数（其析构函数除外）的并发调用不会引入数据竞争
> ，就好像它们是原子操作一样
> 。[…]

对 ISO/IEC IS 14882（C++ 标准）的 [thread.sema.cnt 第 3 段](https://eel.is/c++draft/thread.sema.cnt#3) 修改如下：

> 3. 对 `counting_semaphore` 成员函数（其析构函数除外）的并发调用不会引入数据竞争
> ，就好像它们是原子操作一样
> 。

对 ISO/IEC IS 14882（C++ 标准）的 [thread.stoptoken.intro 第 5 段](https://eel.is/c++draft/thread#stoptoken.intro-5) 修改如下：

> 对函数 `request_stop`、`stop_requested` 和 `stop_possible` 的调用不会引入数据竞争
> ，就好像它们是原子操作一样
> 。[…]

对 ISO/IEC IS 14882（C++ 标准）的 [atomics.fences 第 2 至 4 段](https://eel.is/c++draft/atomics.fences#2) 修改如下：

> 如果存在原子操作 X 和 Y，它们都对某个原子对象 M 进行操作，使得释放栅栏 A 在 X 之前被定序，X 修改了 M，Y 在获取栅栏 B 之前被定序，并且 Y 读取了由 X 写入的值，或者读取了如果 X 是一个释放操作，它将引领的假设释放序列中任何副作用写入的值，
> 并且每个操作（A、B、X 和 Y）都指定了一个包含执行了其他每个操作的线程的作用域
> ，则释放栅栏 A 与获取栅栏 B 同步。
> 如果存在原子操作 X，它对原子对象 M 进行操作，使得释放栅栏 A 在 X 之前被定序，X 修改了 M，并且原子操作 B 对 M 执行获取操作，读取了由 X 写入的值，或者读取了如果 X 是一个释放操作，它将引领的假设释放序列中任何副作用写入的值，
> 并且每个操作（A、B 和 X）都指定了一个包含执行了其他每个操作的线程的作用域
> ，则释放栅栏 A 与对原子对象 M 执行获取操作的原子操作 B 同步。
> 存在一个原子操作 X，使得 A 在 X 之前被定序，X 修改 M，并且 B 读取由 X 写入的值，或者读取如果 X 是一个释放操作时，由 X 所引领的假设释放序列中的任何副作用所写入的值，
> 并且每个操作（A、B 和 X）指定的作用域都包含了执行其他每个操作的线程。
>
> 一个作为原子对象 M 上的释放操作的原子操作 A，与一个获取栅栏 B 同步，如果
> 存在 M 上的某个原子操作 X，使得 X 在 B 之前被定序，并且读取由 A 写入的值或由 A 所引领的释放序列中的任何副作用所写入的值，
> 并且每个操作（A、B 和 X）指定的作用域都包含了执行其他每个操作的线程。

## 5.7.5. 示例：消息传递

以下示例通过标志 `f`，将线程块 `0` 中的一个线程存储到变量 `x` 的消息传递给线程块 `1` 中的一个线程：

```cpp
int x = 0, f = 0;
```

```cpp
x = 42;
cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
flag.store(1, memory_order_release);
```

```cpp
cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
while(flag.load(memory_order_acquire) != 1);
assert(x == 42);
```

在以下对前一个示例的变体中，两个线程在没有同步的情况下并发访问 `f` 对象，这导致了**数据竞争**，并表现出**未定义行为**：

```cpp
int x = 0, f = 0;
```

```cpp
x = 42;
cuda::atomic_ref<int, cuda::thread_scope_block> flag(f);
flag.store(1, memory_order_release); // UB: 数据竞争
```

```cpp
cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
while(flag.load(memory_order_acquire) != 1); // UB: 数据竞争
assert(x == 42);
```

虽然对 `f` 的内存操作——存储和加载——是原子的，但存储操作的作用域是“线程块作用域”。由于存储是由线程块 0 的线程 0 执行的，它只包含线程块 0 的所有其他线程。然而，执行加载的线程在线程块 1 中，即它不在线程块 0 中执行的存储操作所包含的作用域内，导致存储和加载不是“原子的”，从而引入了数据竞争。

更多示例请参见 [PTX 内存一致性模型测试用例](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#axioms)。

 本页