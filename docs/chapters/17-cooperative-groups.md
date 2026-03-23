# 4.4 协作组

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)

---

此页面有帮助吗？

# 4.4. 协作组

## 4.4.1. 简介

协作组是 CUDA 编程模型的扩展，用于组织协作线程组。协作组允许开发者控制线程协作的粒度，帮助他们表达更丰富、更高效的并行分解。协作组还提供了常见并行原语的实现，如扫描和并行规约。

从历史上看，CUDA 编程模型为同步协作线程提供了一个单一、简单的构造：跨越线程块所有线程的屏障，通过 `__syncthreads()` 内部函数实现。为了表达更广泛的并行交互模式，许多面向性能的程序员不得不编写自己的、临时的、不安全的原语，用于同步单个线程束内的线程，或同步在单个 GPU 上运行的一组线程块。虽然所实现的性能改进通常很有价值，但这导致了脆弱代码的不断积累，这些代码编写、调优和维护成本高昂，并且难以跨 GPU 代际移植。协作组提供了一种安全且面向未来的机制来编写高性能代码。

完整的协作组 API 可在 [协作组 API](../05-appendices/device-callable-apis.html#cg-api-partition-header) 中找到。

## 4.4.2. 协作组句柄与成员函数

协作组通过协作组句柄进行管理。协作组句柄允许参与的线程了解其在组内的位置、组大小以及其他组信息。下表列出了一些选定的组成员函数。

| 访问器 | 返回值 |
| --- | --- |
| thread_rank() | 调用线程的排名。 |
| num_threads() | 组中的线程总数。 |
| thread_index() | 线程在启动的线程块内的三维索引。 |
| dim_threads() | 启动的线程块的三维尺寸（以线程为单位）。 |

完整的成员函数列表可在 [协作组 API](../05-appendices/device-callable-apis.html#cg-api-common-header) 中找到。

## 4.4.3. 默认行为 / 无组执行

代表网格和线程块的组是基于内核启动配置隐式创建的。这些“隐式”组为开发者提供了一个起点，可以将其显式分解为更细粒度的组。可以使用以下方法访问隐式组：

| 访问器 | 组范围 |
| --- | --- |
| this_thread_block() | 返回包含当前线程块中所有线程的组的句柄。 |
| this_grid() | 返回包含网格中所有线程的组的句柄。 |
| coalesced_threads() [ 1 ] | 返回包含线程束中当前活动线程的组的句柄。 |
| this_cluster() [ 2 ] | 返回包含当前集群中线程的组的句柄。 |

[
1
]

`coalesced_threads()` 操作符返回该时间点上的活动线程集合，并不保证返回哪些线程（只要它们是活动的），也不保证它们在整个执行过程中保持合并状态。
[
2
]

当启动非集群网格时，`this_cluster()` 假定集群大小为 1x1x1。需要计算能力 9.0 或更高。

更多信息请参阅 [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-common-header)。

### 4.4.3.1. 尽早创建隐式组句柄

为了获得最佳性能，建议您预先创建隐式组的句柄（尽可能早，在任何分支发生之前），并在整个内核中使用该句柄。

### 4.4.3.2. 仅通过引用传递组句柄

建议在将组句柄传递给函数时，通过引用传递。组句柄必须在声明时初始化，因为没有默认构造函数。不鼓励复制构造组句柄。

## 4.4.4. 创建协作组

组是通过将父组划分为子组来创建的。当一个组被划分时，会创建一个组句柄来管理生成的子组。开发者可以使用以下划分操作：

| 划分类型 | 描述 |
| --- | --- |
| tiled_partition | 将父组划分为一系列固定大小的子组，这些子组以一维行优先格式排列。 |
| stride_partition | 将父组划分为大小相等的子组，其中线程以循环方式分配给子组。 |
| labeled_partition | 基于条件标签（可以是任何整数类型）将父组划分为一维子组。 |
| binary_partition | 标记划分的专门形式，其中标签只能是“0”或“1”。 |

以下示例展示了如何创建平铺划分：

```cuda
namespace cg = cooperative_groups;
// 获取当前线程的协作组
cg::thread_block my_group = cg::this_thread_block();

// 将协作组划分为大小为 8 的平铺
cg::thread_block_tile<8> my_subgroup = cg::tiled_partition<8>(cta);

// 作为 my_subgroup 执行工作
```

使用哪种最佳划分策略取决于具体上下文。更多信息请参阅 [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-partition-header)。

### 4.4.4.1. 避免组创建风险

划分组是一个集体操作，组中的所有线程都必须参与。如果组是在并非所有线程都能到达的条件分支中创建的，则可能导致死锁或数据损坏。

## 4.4.5. 同步

在引入协作组之前，CUDA 编程模型只允许在内核完成边界处进行线程块之间的同步。协作组允许开发者在不同粒度上同步协作线程组。

### 4.4.5.1. 同步

您可以通过调用集体 `sync()` 函数来同步一个组。与 `__syncthreads()` 类似，`sync()` 函数保证：- 在同步点之前，组中线程进行的所有内存访问（例如，读取和写入）在同步点之后对组中的所有线程可见。- 在允许任何线程继续执行超过同步点之前，组中的所有线程都必须到达该同步点。
以下示例展示了一个等效于 `__syncthreads()` 的 `cooperative_groups::sync()`。

```cuda
namespace cg = cooperative_groups;

cg::thread_block my_group = cg::this_thread_block();

// 同步线程块内的线程
cg::sync(my_group);
```

协作组可用于同步整个线程网格。自 CUDA 13 起，协作组不再能用于多设备同步。详情请参阅[大规模组](#cooperative-groups-large-scale-groups)章节。

关于同步的更多信息，请参见[协作组 API](../05-appendices/device-callable-apis.html#cg-api-sync-header)。

### 4.4.5.2. 屏障

协作组提供了一个类似于 `cuda::barrier` 的屏障 API，可用于更高级的同步。协作组屏障 API 与 `cuda::barrier` 在几个关键方面有所不同：- 协作组屏障会自动初始化 - 组内的所有线程在每个阶段必须到达屏障并等待一次。 - `barrier_arrive` 返回一个 `arrival_token` 对象，该对象必须传递给相应的 `barrier_wait`，在那里它被消耗且不能再次使用。

程序员在使用协作组屏障时必须注意避免风险：- 在调用 `barrier_arrive` 之后和调用 `barrier_wait` 之前，组不能使用任何集体操作。 - `barrier_wait` 仅保证组内的所有线程都已调用 `barrier_arrive`。`barrier_wait` 并不保证所有线程都已调用 `barrier_wait`。

```cuda
namespace cg = cooperative_groups;

cg::thread_block my_group = this_block();

auto token = cluster.barrier_arrive();

// 可选：进行一些本地处理以隐藏同步延迟
local_processing(block);

// 在访问 dsmem 之前，确保集群中的所有其他线程块都在运行且已初始化共享数据
cluster.barrier_wait(std::move(token));
```

## 4.4.6. 集体操作

协作组包含一组可以由一组线程执行的集体操作。这些操作需要指定组内的所有线程参与才能完成操作。

除非[协作组 API](../05-appendices/device-callable-apis.html#cg-api-partition-header) 中明确允许使用不同的值，否则组内的所有线程必须为每个集体调用的相应参数传递相同的值。否则，调用的行为是未定义的。

### 4.4.6.1. 归约

`reduce` 函数用于对指定组内每个线程提供的数据执行并行归约。必须通过提供下表中所示的操作符之一来指定归约的类型。

| 操作符 | 返回结果 |
| --- | --- |
| plus | 组内所有值的总和 |
| less | 最小值 |
| greater | 最大值 |
| bit_and | 按位与归约 |
| bit_or | 按位或归约 |
| bit_xor | 按位异或归约 |

当可用时，归约会使用硬件加速（需要计算能力 8.0 或更高）。对于没有硬件加速的旧硬件，提供了软件回退。只有 4B 类型受硬件加速。
有关归约操作的更多信息，请参阅 [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-reduce-header)。

以下示例展示了如何使用 `cooperative_groups::reduce()` 来执行线程块范围内的求和归约。

```cuda
namespace cg = cooperative_groups;

cg::thread_block my_group = cg::this_thread_block();

int val = data[threadIdx.x];

int sum = cg::reduce(cta, val, cg::plus<int>());

// 存储归约结果
if (my_group.thread_rank() == 0) {
   result[blockIdx.x] = sum;
}
```

### 4.4.6.2. 扫描

Cooperative Groups 包含 `inclusive_scan` 和 `exclusive_scan` 的实现，可用于任意大小的组。这些函数对指定组中每个线程提供的数据执行扫描操作。

程序员可以选择性地指定一个归约运算符，如上方的 [归约运算符表](#cooperative-groups-reduction-operators) 所列。

```cuda
namespace cg = cooperative_groups;

cg::thread_block my_group = cg::this_thread_block();

int val = data[my_group.thread_rank()];

int exclusive_sum = cg::exclusive_scan(my_group, val, cg::plus<int>());

result[my_group.thread_rank()] = exclusive_sum;
```

有关扫描操作的更多信息，请参阅 [Cooperative Groups Scan API](../05-appendices/device-callable-apis.html#cg-api-scan-header)。

### 4.4.6.3. 调用单个线程

Cooperative Groups 提供了一个 `invoke_one` 函数，用于当必须由单个线程代表组执行串行部分工作时。
- `invoke_one` 从调用组中任意选择一个线程，并使用该线程以及提供的参数来调用提供的可调用函数。
- `invoke_one_broadcast` 与 `invoke_one` 相同，只是调用的结果也会广播到组中的所有线程。

线程选择机制不保证是确定性的。

以下示例展示了 `invoke_one` 的基本用法。

```cuda
namespace cg = cooperative_groups;
cg::thread_block my_group = cg::this_thread_block();

// 确保线程块中只有一个线程打印消息
cg::invoke_one(my_group, []() {
   printf("Hello from one thread in the block!");
});

// 同步以确保所有线程等待消息打印完成
cg::sync(my_group);
```

在可调用函数内部，不允许在调用组内进行通信或同步。允许与调用组外的线程进行通信。

## 4.4.7. 异步数据移动

CUDA 中的 Cooperative Groups `memcpy_async` 功能提供了一种在全局内存和共享内存之间执行异步内存拷贝的方法。`memcpy_async` 对于优化内存传输以及重叠计算与数据传输以提高性能特别有用。

`memcpy_async` 函数用于启动从全局内存到共享内存的异步加载。`memcpy_async` 旨在用作一种“预取”操作，在需要数据之前将其加载。
`wait` 函数强制组内的所有线程等待，直到异步内存传输完成。在共享内存中访问数据之前，组内的所有线程都必须调用 `wait`。

以下示例展示了如何使用 `memcpy_async` 和 `wait` 来预取数据。

```cuda
namespace cg = cooperative_groups;

cg::thread_group my_group = cg::this_thread_block();

__shared__ int shared_data[];

// 执行从全局内存到共享内存的异步拷贝
cg::memcpy_async(my_group, shared_data + my_group.rank(), input + my_group.rank(), sizeof(int));

// 在此处执行工作以隐藏延迟。不能使用 shared_data

// 等待异步拷贝完成
cg::wait(my_group);

// 预取的数据现在可用
```

更多信息请参阅 [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-async-header)。

### 4.4.7.1. Memcpy Async 对齐要求

仅当源地址是全局内存、目标地址是共享内存，并且两者都至少 4 字节对齐时，`memcpy_async` 才是异步的。为了获得最佳性能：建议共享内存和全局内存都采用 16 字节对齐。

## 4.4.8. 大规模组

Cooperative Groups 允许创建跨越整个网格的大规模组。前面描述的所有 Cooperative Group 功能都适用于这些大规模组，但有一个显著的例外：同步整个网格需要使用 `cudaLaunchCooperativeKernel` 运行时启动 API。

自 CUDA 13 起，已移除用于 Cooperative Groups 的多设备启动 API 及相关引用。

### 4.4.8.1. 何时使用 cudaLaunchCooperativeKernel

`cudaLaunchCooperativeKernel` 是一个 CUDA 运行时 API 函数，用于启动一个使用协作组的单设备内核，专门设计用于执行需要线程块间同步的内核。此函数确保内核中的所有线程能够在整个网格范围内同步和协作，这是传统的 CUDA 内核（仅允许在单个线程块内同步）无法实现的。`cudaLaunchCooperativeKernel` 确保内核启动是原子的，即如果 API 调用成功，则指定数量的线程块将在指定设备上启动。

最佳实践是首先通过查询设备属性 `cudaDevAttrCooperativeLaunch` 来确保设备支持协作启动：

```cuda
int dev = 0;
int supportsCoopLaunch = 0;
cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
```

如果设备 0 支持该属性，`supportsCoopLaunch` 将被设置为 1。仅支持计算能力 6.0 及更高的设备。此外，您需要运行在以下环境之一：

-   不带 MPS 的 Linux 平台
-   带 MPS 的 Linux 平台，且设备计算能力为 7.0 或更高
-   最新的 Windows 平台

在此页面上