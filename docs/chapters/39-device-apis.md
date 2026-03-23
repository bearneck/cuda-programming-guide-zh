# 5.6 设备端可调用 API

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html)

---

此页面是否有帮助？

# 5.6. 设备可调用 API 和内置函数

本章包含可从 CUDA 内核和设备代码调用的 API 和内置函数的参考材料及 API 文档。

## 5.6.1. 内存屏障原语接口

原语 API 是一个类似 C 的接口，用于访问 `cuda::barrier` 功能。通过包含 `<cuda_awbarrier_primitives.h>` 头文件即可使用这些原语。

### 5.6.1.1. 数据类型

```cpp
typedef /* implementation defined */ __mbarrier_t;
typedef /* implementation defined */ __mbarrier_token_t;
```

### 5.6.1.2. 内存屏障原语 API

```cpp
uint32_t __mbarrier_maximum_count();
void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count);
```

- bar 必须是指向 __shared__ 内存的指针。
- expected_count <= __mbarrier_maximum_count()
- 将 *bar 的当前阶段和下一阶段的预期到达计数初始化为 expected_count。

```cpp
void __mbarrier_inval(__mbarrier_t* bar);
```

- bar 必须是指向驻留在共享内存中的屏障对象的指针。
- 在相应的共享内存可以重新用于其他用途之前，必须使 *bar 失效。

```cpp
__mbarrier_token_t __mbarrier_arrive(__mbarrier_t* bar);
```

- 必须在调用此函数之前完成 *bar 的初始化。
- 待处理计数不能为零。
- 原子地递减屏障当前阶段的待处理计数。
- 返回一个与递减操作之前屏障状态相关联的到达令牌。

```cpp
__mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* bar);
```

- 必须在调用此函数之前完成 *bar 的初始化。
- 待处理计数不能为零。
- 原子地递减屏障当前阶段的待处理计数以及下一阶段的预期计数。
- 返回一个与递减操作之前屏障状态相关联的到达令牌。

```cpp
bool __mbarrier_test_wait(__mbarrier_t* bar, __mbarrier_token_t token);
```

- token 必须与 *bar 的紧邻前一个阶段或当前阶段相关联。
- 如果 token 与 *bar 的紧邻前一个阶段相关联，则返回 true，否则返回 false。

```cpp
bool __mbarrier_test_wait_parity(__mbarrier_t* bar, bool phase_parity);
```

- phase_parity 必须指示 *bar 的当前阶段或紧邻前一个阶段的奇偶性。true 值对应奇数阶段，false 值对应偶数阶段。
- 如果 phase_parity 指示 *bar 的紧邻前一个阶段的整数奇偶性，则返回 true，否则返回 false。

```cpp
bool __mbarrier_try_wait(__mbarrier_t* bar, __mbarrier_token_t token, uint32_t max_sleep_nanosec);
```

- token 必须与 *bar 的紧邻前一个阶段或当前阶段相关联。
- 如果 token 与 *bar 的紧邻前一个阶段相关联，则返回 true。否则，执行线程可能会被挂起。挂起的线程将在指定阶段完成时（返回 true）或在阶段完成前达到系统相关的时间限制后（返回 false）恢复执行。
- max_sleep_nanosec 指定一个以纳秒为单位的时间限制，可用于替代系统相关的时间限制。

```cpp
bool __mbarrier_try_wait_parity(__mbarrier_t* bar, bool phase_parity, uint32_t max_sleep_nanosec);
```

- phase_parity 必须指示 *bar 的当前阶段或紧邻前一阶段的奇偶性。值为 true 对应奇数阶段，值为 false 对应偶数阶段。
- 如果 phase_parity 指示 *bar 的紧邻前一阶段的整数奇偶性，则返回 true。否则，执行线程可能会被挂起。挂起的线程将在指定阶段完成时恢复执行（返回 true）**或**在阶段完成前达到系统相关的时间限制时恢复执行（返回 false）。
- max_sleep_nanosec 指定一个以纳秒为单位的时间限制，可用于替代系统相关的时间限制。

## 5.6.2. 流水线原语接口

流水线原语为 `<cuda/pipeline>` 中可用的功能提供了类似 C 的接口。通过包含 `<cuda_pipeline.h>` 头文件即可使用流水线原语接口。在不兼容 ISO C++ 2011 的情况下编译时，请包含 `<cuda_pipeline_primitives.h>` 头文件。

流水线原语 API 仅支持跟踪从全局内存到共享内存的异步拷贝，并具有特定的尺寸和对齐要求。它提供了与具有 `cuda::thread_scope_thread` 的 `cuda::pipeline` 对象等效的功能。

### 5.6.2.1. memcpy_async 原语

```cpp
void __pipeline_memcpy_async(void* __restrict__ dst_shared,
                             const void* __restrict__ src_global,
                             size_t size_and_align,
                             size_t zfill=0);
```

- 请求提交以下操作进行异步评估：size_t i = 0 ; for (; i < size_and_align - zfill ; ++ i ) (( char * ) dst_shared )[ i ] = (( char * ) src_global )[ i ]; /* 拷贝 */ for (; i < size_and_align ; ++ i ) (( char * ) dst_shared )[ i ] = 0 ; /* 零填充 */
- 要求：dst_shared 必须是指向 memcpy_async 目标的共享内存的指针。src_global 必须是指向 memcpy_async 源的全局内存的指针。size_and_align 必须为 4、8 或 16。zfill <= size_and_align 。size_and_align 必须是 dst_shared 和 src_global 的对齐方式。
- 在等待 memcpy_async 操作完成之前，任何线程修改源内存或观察目标内存都会导致竞态条件。在提交 memcpy_async 操作和等待其完成之间，以下任何操作都会引入竞态条件：从 dst_shared 加载。向 dst_shared 或 src_global 存储。对 dst_shared 或 src_global 应用原子更新。

### 5.6.2.2. 提交原语

```cpp
void __pipeline_commit();
```

- 将已提交的 memcpy_async 作为当前批次提交到流水线。

### 5.6.2.3. 等待原语
```cpp
void __pipeline_wait_prior(size_t N);
```

- Let {0, 1, 2, ..., L} be the sequence of indices associated with invocations of __pipeline_commit() by a given thread.
- Wait for completion of batches at least up to and including L-N .

### 5.6.2.4.Arrive On Barrier Primitive

```cpp
void __pipeline_arrive_on(__mbarrier_t* bar);
```

- bar points to a barrier in shared memory.
- Increments the barrier arrival count by one, when all memcpy_async operations sequenced before this call have completed, the arrival count is decremented by one and hence the net effect on the arrival count is zero. It is userâs responsibility to make sure that the increment on the arrival count does not exceed __mbarrier_maximum_count() .

## 5.6.3.Cooperative Groups API

### 5.6.3.1.cooperative_groups.h

#### 5.6.3.1.1.class thread_block

Any CUDA programmer is already familiar with a certain group of threads: the thread block. The Cooperative Groups extension introduces a new datatype, `thread_block`, to explicitly represent this concept within the kernel.

`class thread_block;`

Constructed via:

```cpp
thread_block g = this_thread_block();
```

**Public Member Functions:**

`static void sync()`: Synchronize the threads named in the group, equivalent to `g.barrier_wait(g.barrier_arrive())`

`thread_block::arrival_token barrier_arrive()`: Arrive on the thread_block barrier, returns a token that needs to be passed into `barrier_wait()`.

`void barrier_wait(thread_block::arrival_token&& t)`: Wait on the `thread_block` barrier, takes arrival token returned from `barrier_arrive()` as an rvalue reference.

`static unsigned int thread_rank()`: Rank of the calling thread within [0, num_threads)

`static dim3 group_index()`: 3-Dimensional index of the block within the launched grid

`static dim3 thread_index()`: 3-Dimensional index of the thread within the launched block

`static dim3 dim_threads()`: Dimensions of the launched block in units of threads

`static unsigned int num_threads()`: Total number of threads in the group

Legacy member functions (aliases):

`static unsigned int size()`: Total number of threads in the group (alias of `num_threads()`)

`static dim3 group_dim()`: Dimensions of the launched block (alias of `dim_threads()`)

**Example:**

```cpp
/// Loading an integer from global into shared memory
__global__ void kernel(int *globalInput) {
    __shared__ int x;
    thread_block g = this_thread_block();
    // Choose a leader in the thread block
    if (g.thread_rank() == 0) {
        // load from global into shared for all threads to work with
        x = (*globalInput);
    }
    // After loading data into shared memory, you want to synchronize
    // if all threads in your thread block need to see it
    g.sync(); // equivalent to __syncthreads();
}
```

**Note:** that all threads in the group must participate in collective operations, or the behavior is undefined.

**Related:** The `thread_block` datatype is derived from the more generic `thread_group` datatype, which can be used to represent a wider class of groups.
#### 5.6.3.1.2.class cluster_group

此群组对象表示在单个集群中启动的所有线程。这些 API 在计算能力 9.0+ 的所有硬件上可用。在这种情况下，当启动一个非集群网格时，API 会假定集群大小为 1x1x1。

`class cluster_group;`

通过以下方式构造：

```cpp
cluster_group g = this_cluster();
```

**公共成员函数：**

`static void sync()`: 同步群组中指定的线程，等同于 `g.barrier_wait(g.barrier_arrive())`

`static cluster_group::arrival_token barrier_arrive()`: 到达集群屏障，返回一个需要传递给 `barrier_wait()` 的令牌。

`static void barrier_wait(cluster_group::arrival_token&& t)`: 在集群屏障上等待，接受从 `barrier_arrive()` 返回的到达令牌作为右值引用。

`static unsigned int thread_rank()`: 调用线程在 [0, num_threads) 范围内的等级

`static unsigned int block_rank()`: 调用线程块在 [0, num_blocks) 范围内的等级

`static unsigned int num_threads()`: 群组中的线程总数

`static unsigned int num_blocks()`: 群组中的线程块总数

`static dim3 dim_threads()`: 以线程为单位的已启动集群的维度

`static dim3 dim_blocks()`: 以线程块为单位的已启动集群的维度

`static dim3 block_index()`: 已启动集群内调用线程块的 3 维索引

`static unsigned int query_shared_rank(const void *addr)`: 获取共享内存地址所属的线程块等级

`static T* map_shared_rank(T *addr, int rank)`: 获取集群中另一个线程块的共享内存变量的地址

旧版成员函数（别名）：

`static unsigned int size()`: 群组中的线程总数（`num_threads()` 的别名）

#### 5.6.3.1.3.class grid_group

此群组对象表示在单个网格中启动的所有线程。除 `sync()` 外的 API 始终可用，但要能够跨网格同步，您需要使用协作启动 API。

`class grid_group;`

通过以下方式构造：

```cpp
grid_group g = this_grid();
```

**公共成员函数：**

`bool is_valid() const`: 返回 grid_group 是否可以同步

`void sync() const`: 同步群组中指定的线程，等同于 `g.barrier_wait(g.barrier_arrive())`

`grid_group::arrival_token barrier_arrive()`: 到达网格屏障，返回一个需要传递给 `barrier_wait()` 的令牌。

`void barrier_wait(grid_group::arrival_token&& t)`: 在网格屏障上等待，接受从 `barrier_arrive()` 返回的到达令牌作为右值引用。

`static unsigned long long thread_rank()`: 调用线程在 [0, num_threads) 范围内的等级

`static unsigned long long block_rank()`: 调用线程块在 [0, num_blocks) 范围内的等级

`static unsigned long long cluster_rank()`: 调用集群在 [0, num_clusters) 范围内的等级

`static unsigned long long num_threads()`: 群组中的线程总数
`static unsigned long long num_blocks()`: 组中线程块的总数

`static unsigned long long num_clusters()`: 组中集群的总数

`static dim3 dim_blocks()`: 以线程块为单位的已启动线程网格的维度

`static dim3 dim_clusters()`: 以集群为单位的已启动线程网格的维度

`static dim3 block_index()`: 线程块在已启动线程网格内的三维索引

`static dim3 cluster_index()`: 集群在已启动线程网格内的三维索引

遗留成员函数（别名）：

`static unsigned long long size()`: 组中线程的总数（`num_threads()` 的别名）

`static dim3 group_dim()`: 已启动线程网格的维度（`dim_blocks()` 的别名）

#### 5.6.3.1.4. class thread_block_tile

瓦片组的模板化版本，其中使用模板参数来指定瓦片的大小——由于在编译时已知，因此有潜力实现更优化的执行。

```cpp
template <unsigned int Size, typename ParentT = void>
class thread_block_tile;
```

通过以下方式构造：

```cpp
template <unsigned int Size, typename ParentT>
_CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)
```

`Size` 必须是 2 的幂且小于或等于 1024。注意事项部分描述了在计算能力 7.5 或更低的硬件上创建大小超过 32 的瓦片所需的额外步骤。

`ParentT` 是此组从中划分出来的父类型。它会自动推断，但值为 `void` 时，此信息将存储在组句柄中，而不是存储在类型中。

**公共成员函数：**

`void sync() const`: 同步组中指定的线程

`unsigned long long num_threads() const`: 组中线程的总数

`unsigned long long thread_rank() const`: 调用线程在组内的排名（范围 [0, num_threads)）

`unsigned long long meta_group_size() const`: 返回父组被划分时创建的组的数量。

`unsigned long long meta_group_rank() const`: 组在从父组划分出的瓦片集合中的线性排名（受 `meta_group_size` 限制）

`T shfl(T var, unsigned int src_rank) const`: 请参阅 [线程束洗牌函数](cpp-language-extensions.html#warp-shuffle-functions)，**注意：对于大小超过 32 的组，组中的所有线程必须指定相同的 `src_rank`，否则行为未定义。**

`T shfl_up(T var, int delta) const`: 请参阅 [线程束洗牌函数](cpp-language-extensions.html#warp-shuffle-functions)，仅适用于大小小于或等于 32 的组。

`T shfl_down(T var, int delta) const`: 请参阅 [线程束洗牌函数](cpp-language-extensions.html#warp-shuffle-functions)，仅适用于大小小于或等于 32 的组。

`T shfl_xor(T var, int delta) const`: 请参阅 [线程束洗牌函数](cpp-language-extensions.html#warp-shuffle-functions)，仅适用于大小小于或等于 32 的组。

`int any(int predicate) const`: 请参阅 [线程束表决函数](index.html#warp-vote-functions)
`int all(int predicate) const`：请参阅[线程束投票函数](index.html#warp-vote-functions)

`unsigned int ballot(int predicate) const`：请参阅[线程束投票函数](index.html#warp-vote-functions)，仅适用于大小小于或等于 32 的线程块。

`unsigned int match_any(T val) const`：请参阅[线程束匹配函数](cpp-language-extensions.html#warp-match-functions)，仅适用于大小小于或等于 32 的线程块。

`unsigned int match_all(T val, int &pred) const`：请参阅[线程束匹配函数](cpp-language-extensions.html#warp-match-functions)，仅适用于大小小于或等于 32 的线程块。

旧版成员函数（别名）：

`unsigned long long size() const`：组内的线程总数（`num_threads()` 的别名）

**注意：**

-   这里使用的是 `thread_block_tile` 模板化数据结构，组的大小是作为模板参数而非函数参数传递给 `tiled_partition` 调用的。
-   当使用 C++11 或更高版本编译时，`shfl`、`shfl_up`、`shfl_down` 和 `shfl_xor` 函数可以接受任何类型的对象。这意味着只要满足以下约束，就可以对非整数类型进行洗牌操作：符合可平凡复制条件，即 `is_trivially_copyable<T>::value == true`；对于大小小于或等于 32 的线程块，`sizeof(T) <= 32`；对于更大的线程块，`sizeof(T) <= 8`。
-   在计算能力 7.5 或更低的硬件上，大小超过 32 的线程块需要为其预留少量内存。这可以通过使用 `cooperative_groups::block_tile_memory` 结构体模板来实现，该模板必须位于共享内存或全局内存中。`template < unsigned int MaxBlockSize = 1024 > struct block_tile_memory ;` `MaxBlockSize` 指定当前线程块的最大线程数。此参数可用于在仅启动较小线程数的内核中，最小化 `block_tile_memory` 的共享内存使用量。然后需要将此 `block_tile_memory` 传递给 `cooperative_groups::this_thread_block`，从而允许将生成的 `thread_block` 划分为大小超过 32 的线程块。接受 `block_tile_memory` 参数的 `this_thread_block` 重载是一个集体操作，必须由线程块中的所有线程调用。在计算能力 8.0 或更高的硬件上也可以使用 `block_tile_memory`，以便能够编写面向多个不同计算能力的单一源代码。在不需要的情况下，当在共享内存中实例化时，它不应消耗任何内存。

**示例：**

```cpp
/// 以下代码将创建两组大小分别为 32 和 4 的线程块组：
/// 后者的来源信息编码在类型中，而前者将其存储在句柄中
thread_block block = this_thread_block();
thread_block_tile<32> tile32 = tiled_partition<32>(block);
thread_block_tile<4, thread_block> tile4 = tiled_partition<4>(block);
```

```cpp
/// 以下代码将在所有计算能力上创建大小为 128 的线程块。
/// 在计算能力 8.0 或更高版本上，可以省略 block_tile_memory。
__global__ void kernel(...) {
    // 为 thread_block_tile 的使用预留共享内存，
    //   指定块大小最多为 256 个线程。
    __shared__ block_tile_memory<256> shared;
    thread_block thb = this_thread_block(shared);

    // 创建包含 128 个线程的线程块。
    auto tile = tiled_partition<128>(thb);

    // ...
}
```
#### 5.6.3.1.5.class coalesced_group

在 CUDA 的 SIMT 架构中，在硬件层面，多处理器以 32 个线程为一组（称为线程束）来执行线程。如果应用程序代码中存在数据相关的条件分支，导致线程束内的线程发生分支，则该线程束会串行执行每个分支，并禁用不在该路径上的线程。在路径上保持活跃的线程被称为**合并的**。协作组（Cooperative Groups）提供了发现和创建包含所有合并线程的组的功能。

通过 `coalesced_threads()` 构造组句柄是机会性的。它返回该时间点活跃的线程集合，并不保证返回哪些线程（只要它们是活跃的），也不保证它们在整个执行过程中保持合并状态（它们会在执行集合操作时重新聚集，但之后可能再次分支）。

`class coalesced_group;`

构造方式：

```cpp
coalesced_group active = coalesced_threads();
```

**公共成员函数：**

`void sync() const`：同步组内指定的线程

`unsigned long long num_threads() const`：组中的线程总数

`unsigned long long thread_rank() const`：调用线程在组内的排名（范围 [0, num_threads)）

`unsigned long long meta_group_size() const`：返回父组被分区时创建的组数量。如果此组是通过查询活跃线程集合创建的（例如 `coalesced_threads()`），则 `meta_group_size()` 的值将为 1。

`unsigned long long meta_group_rank() const`：在从父组分区得到的组集合中，本组的线性排名（受 meta_group_size 限制）。如果此组是通过查询活跃线程集合创建的（例如 `coalesced_threads()`），则 `meta_group_rank()` 的值将始终为 0。

`T shfl(T var, unsigned int src_rank) const`：请参阅 [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)

`T shfl_up(T var, int delta) const`：请参阅 [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)

`T shfl_down(T var, int delta) const`：请参阅 [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)

`int any(int predicate) const`：请参阅 [Warp Vote Functions](index.html#warp-vote-functions)

`int all(int predicate) const`：请参阅 [Warp Vote Functions](index.html#warp-vote-functions)

`unsigned int ballot(int predicate) const`：请参阅 [Warp Vote Functions](index.html#warp-vote-functions)

`unsigned int match_any(T val) const`：请参阅 [Warp Match Functions](cpp-language-extensions.html#warp-match-functions)

`unsigned int match_all(T val, int &pred) const`：请参阅 [Warp Match Functions](cpp-language-extensions.html#warp-match-functions)

旧版成员函数（别名）：

`unsigned long long size() const`：组中的线程总数（`num_threads()` 的别名）

**注意：**

当使用 C++11 或更高版本编译时，`shfl`、`shfl_up` 和 `shfl_down` 函数接受任何类型的对象。这意味着只要满足以下约束，就可以对非整数类型进行洗牌操作：
- 满足可平凡复制条件，即 is_trivially_copyable<T>::value == true
- sizeof(T) <= 32

**示例：**

```cpp
/// 考虑一种情况，代码中存在一个分支，其中每个线程束中只有第 2、4 和 8 个线程是
/// 活跃的。放置在该分支中的 coalesced_threads() 调用将为每个线程束创建一个
/// 包含三个线程（秩为 0-2，包含两端）的组 active。
__global__ void kernel(int *globalInput) {
    // 假设 globalInput 指示线程 2、4、8 应处理数据
    if (threadIdx.x == *globalInput) {
        coalesced_group active = coalesced_threads();
        // active 包含 0-2（包含两端）
        active.sync();
    }
}
```

### 5.6.3.2.cooperative_groups/async.h

#### 5.6.3.2.1.memcpy_async

`memcpy_async` 是一个组范围内的集体内存拷贝操作，它利用硬件加速支持从全局内存到共享内存的非阻塞内存事务。给定组中指定的一组线程，`memcpy_async` 将通过单个流水线阶段移动指定数量的字节或输入类型的元素。此外，为了在使用 `memcpy_async` API 时获得最佳性能，共享内存和全局内存都需要 16 字节对齐。需要注意的是，虽然这通常是一个内存拷贝操作，但仅当源是全局内存、目标是共享内存，并且两者都可以用 16、8 或 4 字节对齐方式寻址时，它才是异步的。异步复制的数据应仅在调用 `wait` 或 `wait_prior` 之后读取，这些调用标志着相应的阶段已完成将数据移动到共享内存。

必须等待所有未完成的请求可能会失去一些灵活性（但获得了简单性）。为了有效地重叠数据传输和执行，重要的是能够在等待并处理请求 **N** 的同时，启动 **N+1** 个 `memcpy_async` 请求。为此，请使用 `memcpy_async` 并基于集体阶段（stage-based）的 `wait_prior` API 来等待它。更多详情请参阅 [wait 和 wait_prior](#cg-api-async-wait)。

用法 1

```cpp
template <typename TyGroup, typename TyElem, typename TyShape>
void memcpy_async(
  const TyGroup &group,
  TyElem *__restrict__ _dst,
  const TyElem *__restrict__ _src,
  const TyShape &shape
);
```

执行 **``shape`` 字节** 的拷贝。

用法 2

```cpp
template <typename TyGroup, typename TyElem, typename TyDstLayout, typename TySrcLayout>
void memcpy_async(
  const TyGroup &group,
  TyElem *__restrict__ dst,
  const TyDstLayout &dstLayout,
  const TyElem *__restrict__ src,
  const TySrcLayout &srcLayout
);
```

执行 **``min(dstLayout, srcLayout)`` 个元素** 的拷贝。如果布局类型为 `cuda::aligned_size_t<N>`，则两者必须指定相同的对齐方式。

**勘误** 在 CUDA 11.1 中引入的 `memcpy_async` API（同时包含源和目标输入布局）期望布局以元素而非字节为单位提供。元素类型从 `TyElem` 推断，其大小为 `sizeof(TyElem)`。如果使用 `cuda::aligned_size_t<N>` 类型作为布局，则指定的元素数量乘以 `sizeof(TyElem)` 必须是 N 的倍数，并且建议使用 `std::byte` 或 `char` 作为元素类型。
如果指定的拷贝形状或布局类型为 `cuda::aligned_size_t<N>`，则保证对齐至少为 `min(16, N)`。在这种情况下，`dst` 和 `src` 指针都需要按 N 字节对齐，并且拷贝的字节数需要是 N 的倍数。

**代码生成要求：** 最低计算能力 5.0，异步操作需要计算能力 8.0，C++11

需要包含 `cooperative_groups/memcpy_async.h` 头文件。

**示例：**

```cpp
/// 此示例将 elementsPerThreadBlock 大小的数据从全局内存流式传输到
/// 一个有限大小的共享内存（elementsInShared）块中进行操作。
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void kernel(int* global_data) {
    cg::thread_block tb = cg::this_thread_block();
    const size_t elementsPerThreadBlock = 16 * 1024;
    const size_t elementsInShared = 128;
    __shared__ int local_smem[elementsInShared];

    size_t copy_count;
    size_t index = 0;
    while (index < elementsPerThreadBlock) {
        cg::memcpy_async(tb, local_smem, elementsInShared, global_data + index, elementsPerThreadBlock - index);
        copy_count = min(elementsInShared, elementsPerThreadBlock - index);
        cg::wait(tb);
        // 对 local_smem 进行操作
        index += copy_count;
    }
}
```

#### 5.6.3.2.2.wait 与 wait_prior

```cpp
template <typename TyGroup>
void wait(TyGroup & group);

template <unsigned int NumStages, typename TyGroup>
void wait_prior(TyGroup & group);
```

`wait` 和 `wait_prior` 集合操作允许等待 memcpy_async 拷贝完成。`wait` 会阻塞调用线程，直到所有先前的拷贝完成。`wait_prior` 允许最新的 NumStages 个拷贝仍未完成，并等待所有更早的请求。因此，对于总共请求的 `N` 次拷贝，它会等待直到前 `N-NumStages` 次拷贝完成，而最后 `NumStages` 次拷贝可能仍在进行中。`wait` 和 `wait_prior` 都会同步指定的组。

**代码生成要求：** 最低计算能力 5.0，异步操作需要计算能力 8.0，C++11

需要包含 `cooperative_groups/memcpy_async.h` 头文件。

**示例：**

```cpp
/// 此示例将 elementsPerThreadBlock 大小的数据从全局内存流式传输到
/// 一个有限大小的共享内存（elementsInShared）块中，以便在多个（两个）阶段中进行操作。
/// 当阶段 N 启动时，我们可以等待并操作阶段 N-1。
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void kernel(int* global_data) {
    cg::thread_block tb = cg::this_thread_block();
    const size_t elementsPerThreadBlock = 16 * 1024 + 64;
    const size_t elementsInShared = 128;
    __align__(16) __shared__ int local_smem[2][elementsInShared];
    int stage = 0;
    // 首先启动一个额外的请求
    size_t copy_count = elementsInShared;
    size_t index = copy_count;
    cg::memcpy_async(tb, local_smem[stage], elementsInShared, global_data, elementsPerThreadBlock - index);
    while (index < elementsPerThreadBlock) {
        // 现在我们启动下一个请求...
        cg::memcpy_async(tb, local_smem[stage ^ 1], elementsInShared, global_data + index, elementsPerThreadBlock - index);
        // ... 但我们等待它之前的那一个
        cg::wait_prior<1>(tb);

        // 现在它已可用，我们可以在此处操作 local_smem[stage]
        // (...)
        //

        // 计算下一次迭代实际拷贝的数据量。
        copy_count = min(elementsInShared, elementsPerThreadBlock - index);
        index += copy_count;

        // 此处可能需要 cg::sync(tb)，具体取决于
        // 对 local_smem[stage] 的操作是否会释放线程使其提前执行
        // 切换到下一个阶段
        stage ^= 1;
    }
    cg::wait(tb);
    // 最后一个 local_smem[stage] 可以在此处处理
}
```
### 5.6.3.3.cooperative_groups/partition.h

#### 5.6.3.3.1.tiled_partition

```cpp
template <unsigned int Size, typename ParentT>
thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g);
```

```cpp
thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);
```

`tiled_partition` 方法是一个集体操作，它将父组划分成一维、行优先的子组平铺。总共将创建 ((size(parent)/tilesz) 个子组，因此父组的大小必须能被 `Size` 整除。允许的父组是 `thread_block` 或 `thread_block_tile`。

该实现可能会导致调用线程等待，直到父组的所有成员都调用了该操作后才恢复执行。功能仅限于原生硬件大小：1/2/4/8/16/32，并且 `cg::size(parent)` 必须大于 `Size` 参数。模板化版本的 `tiled_partition` 也支持 64/128/256/512 的大小，但在计算能力 7.5 或更低的设备上需要一些额外的步骤，详情请参阅 [class thread_block_tile](#cg-api-thread-block-tile)。

**代码生成要求：** 最低计算能力 5.0，大小超过 32 时需要 C++11

#### 5.6.3.3.2.labeled_partition

```cpp
template <typename Label>
coalesced_group labeled_partition(const coalesced_group& g, Label label);
```

```cpp
template <unsigned int Size, typename Label>
coalesced_group labeled_partition(const thread_block_tile<Size>& g, Label label);
```

`labeled_partition` 方法是一个集体操作，它将父组划分成一维子组，子组内的线程是合并的。该实现将评估一个条件标签，并将具有相同标签值的线程分配到同一个组中。

`Label` 可以是任何整数类型。

该实现可能会导致调用线程等待，直到父组的所有成员都调用了该操作后才恢复执行。

**注意：** 此功能仍在评估中，未来可能会有细微变化。

**代码生成要求：** 最低计算能力 7.0，需要 C++11

#### 5.6.3.3.3.binary_partition

```cpp
coalesced_group binary_partition(const coalesced_group& g, bool pred);
```

```cpp
template <unsigned int Size>
coalesced_group binary_partition(const thread_block_tile<Size>& g, bool pred);
```

`binary_partition()` 方法是一个集体操作，它将父组划分成一维子组，子组内的线程是合并的。该实现将评估一个谓词，并将具有相同值的线程分配到同一个组中。这是 `labeled_partition()` 的一种特化形式，其中标签只能是 0 或 1。

该实现可能会导致调用线程等待，直到父组的所有成员都调用了该操作后才恢复执行。

**示例：**

```cpp
/// 此示例将一个大小为 32 的 tile 划分为一个包含奇数的组和一个包含偶数的组
_global__ void oddEven(int *inputArr) {
    auto block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    // inputArr 包含随机整数
    int elem = inputArr[block.thread_rank()];
    // 此后，tile32 被分割成 2 个组，
    // 一个子 tile 中 elem&1 为真，另一个为假
    auto subtile = cg::binary_partition(tile32, (elem & 1));
}
```
### 5.6.3.4.cooperative_groups/reduce.h

#### 5.6.3.4.1.ReduceOperators

以下是可与 `reduce` 一起使用的一些基本操作的函数对象原型。

```cpp
namespace cooperative_groups {
  template <typename Ty>
  struct cg::plus;

  template <typename Ty>
  struct cg::less;

  template <typename Ty>
  struct cg::greater;

  template <typename Ty>
  struct cg::bit_and;

  template <typename Ty>
  struct cg::bit_xor;

  template <typename Ty>
  struct cg::bit_or;
}
```

归约操作受限于编译时实现可用的信息。因此，为了利用 CC 8.0 中引入的硬件内在函数，`cg::` 命名空间暴露了几个反映硬件操作的函数对象。这些对象看起来与 C++ STL 中提供的类似，除了 `less/greater`。与 STL 存在差异的原因是，这些函数对象旨在真实地反映硬件内在函数的操作。

**功能描述：**

- cg::plus：接受两个值并使用 operator+ 返回两者之和。
- cg::less：接受两个值并使用 operator< 返回较小的值。不同之处在于它返回的是较小的值，而不是布尔值。
- cg::greater：接受两个值并使用 operator< 返回较大的值。不同之处在于它返回的是较大的值，而不是布尔值。
- cg::bit_and：接受两个值并返回 operator& 的结果。
- cg::bit_xor：接受两个值并返回 operator^ 的结果。
- cg::bit_or：接受两个值并返回 operator| 的结果。

**示例：**

```cpp
{
    // cg::plus<int> 在 cg::reduce 内部有特化，在 CC 8.0+ 上会调用 __reduce_add_sync(...)
    cg::reduce(tile, (int)val, cg::plus<int>());

    // cg::plus<float> 无法匹配加速器，而是执行基于洗牌的标准归约
    cg::reduce(tile, (float)val, cg::plus<float>());

    // 虽然支持向量的各个分量，但以下情况 reduce 不会使用硬件内在函数
    // 还需要为向量和任何可能使用的自定义类型定义相应的运算符
    int4 vec = {...};
    cg::reduce(tile, vec, cg::plus<int4>())

    // 最后，lambda 表达式和其他函数对象无法被检查以进行分发
    // 而是会使用提供的函数对象执行基于洗牌的归约。
    cg::reduce(tile, (int)val, [](int l, int r) -> int {return l + r;});
}
```

#### 5.6.3.4.2.reduce

```cpp
template <typename TyGroup, typename TyArg, typename TyOp>
auto reduce(const TyGroup& group, TyArg&& val, TyOp&& op) -> decltype(op(val, val));
```

`reduce` 对传入的组中指定的每个线程提供的数据执行归约操作。对于算术加法、最小值或最大值操作以及逻辑 AND、OR 或 XOR 操作，它会利用硬件加速（在计算能力 8.0 及更高的设备上），并在旧一代硬件上提供软件回退。只有 4B 类型受硬件加速。
`group`：有效的组类型为 `coalesced_group` 和 `thread_block_tile`。

`val`：满足以下要求的任何类型：

- 符合可平凡复制条件，即 is_trivially_copyable<TyArg>::value == true
- 对于 `coalesced_group` 和大小小于或等于 32 的 tile，sizeof(T) <= 32；对于更大的 tile，sizeof(T) <= 8
- 对于给定的函数对象，具有合适的算术或比较运算符。

**注意：** 组内的不同线程可以为此参数传递不同的值。

`op`：对于整型能提供硬件加速的有效函数对象是 `plus()`, `less()`, `greater()`, `bit_and()`, `bit_xor()`, `bit_or()`。这些必须被构造，因此需要 TyVal 模板参数，例如 `plus<int>()`。Reduce 也支持 lambda 和其他可以使用 `operator()` 调用的函数对象。

异步归约

```cpp
template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
void reduce_update_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);

template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
void reduce_store_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);

template <typename TyGroup, typename TyArg, typename TyOp>
void reduce_store_async(const TyGroup& group, TyArg* ptr, TyArg&& val, TyOp&& op);
```

API 的 `*_async` 变体是异步计算结果，以便由参与线程之一存储或更新指定的目标，而不是由每个线程返回结果。为了观察这些异步调用的效果，需要同步调用线程组或包含它们的一个更大的组。

- 对于原子存储或更新变体，atomic 参数可以是 CUDA C++ 标准库中可用的 `cuda::atomic` 或 `cuda::atomic_ref` 之一。此 API 变体仅在 CUDA C++ 标准库支持这些类型的平台和设备上可用。归约的结果用于根据指定的 `op` 原子地更新原子变量，例如，在 `cg::plus()` 的情况下，结果会被原子地加到原子变量上。原子变量持有的类型必须与 TyArg 的类型匹配。原子变量的作用域必须包含组中的所有线程，并且如果多个组同时使用同一个原子变量，则其作用域必须包含所有使用它的组中的所有线程。原子更新以宽松的内存顺序执行。
- 对于指针存储变体，归约的结果将被弱存储到 dst 指针。

### 5.6.3.5.cooperative_groups/scan.h

#### 5.6.3.5.1.inclusive_scan 和 exclusive_scan

```cpp
template <typename TyGroup, typename TyVal, typename TyFn>
auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyVal>
TyVal inclusive_scan(const TyGroup& group, TyVal&& val);

template <typename TyGroup, typename TyVal, typename TyFn>
auto exclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyVal>
TyVal exclusive_scan(const TyGroup& group, TyVal&& val);
```
`inclusive_scan` 和 `exclusive_scan` 对传入的组中指定的每个线程提供的数据执行扫描操作。对于 `exclusive_scan`，每个线程的结果是 `thread_rank` 低于该线程的所有线程数据的归约。`inclusive_scan` 的结果在归约中也包含调用线程的数据。

`group`：有效的组类型是 `coalesced_group` 和 `thread_block_tile`。

`val`：满足以下要求的任何类型：

- 符合可平凡复制条件，即 is_trivially_copyable<TyArg>::value == true
- 对于 `coalesced_group` 和大小小于或等于 32 的 tile，sizeof(T) <= 32；对于更大的 tile，sizeof(T) <= 8
- 对于给定的函数对象，具有合适的算术或比较运算符。

**注意：** 组中的不同线程可以为此参数传递不同的值。

`op`：为方便而定义的函数对象是 [cooperative_groups/reduce.h](#cg-api-reduce-header) 中描述的 `plus(), less(), greater(), bit_and(), bit_xor(), bit_or()`。这些必须被构造，因此需要 TyVal 模板参数，例如 `plus<int>()`。`inclusive_scan` 和 `exclusive_scan` 也支持可以使用 `operator()` 调用的 lambda 表达式和其他函数对象。没有此参数的重载使用 `cg::plus<TyVal>()`。

**扫描更新**

```cpp
template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
auto inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyAtomic, typename TyVal>
TyVal inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);

template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
auto exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyAtomic, typename TyVal>
TyVal exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);
```

`*_scan_update` 集合操作接受一个额外的参数 `atomic`，它可以是 [CUDA C++ 标准库](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html) 中可用的 `cuda::atomic` 或 `cuda::atomic_ref`。这些 API 变体仅在 CUDA C++ 标准库支持这些类型的平台和设备上可用。这些变体将根据 `op` 使用组中所有线程的输入值之和来更新 `atomic`。`atomic` 的先前值将与每个线程的扫描结果组合并返回。`atomic` 持有的类型必须与 `TyVal` 的类型匹配。atomic 的作用域必须包含组中的所有线程，如果多个组同时使用同一个 atomic，则其作用域必须包含所有使用它的组中的所有线程。atomic 更新以宽松的内存顺序执行。

以下伪代码说明了扫描的更新变体如何工作：

```cpp
/*
 inclusive_scan_update behaves as the following block,
 except both reduce and inclusive_scan is calculated simultaneously.
auto total = reduce(group, val, op);
TyVal old;
if (group.thread_rank() == selected_thread) {
    atomically {
        old = atomic.load();
        atomic.store(op(old, total));
    }
}
old = group.shfl(old, selected_thread);
return op(inclusive_scan(group, val, op), old);
*/
```

`cooperative_groups/scan.h` header needs to be included.

**Example of stream compaction using exclusive_scan:**

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

// put data from input into output only if it passes test_fn predicate
template<typename Group, typename Data, typename TyFn>
__device__ int stream_compaction(Group &g, Data *input, int count, TyFn&& test_fn, Data *output) {
    int per_thread = count / g.num_threads();
    int thread_start = min(g.thread_rank() * per_thread, count);
    int my_count = min(per_thread, count - thread_start);

    // get all passing items from my part of the input
    //  into a contagious part of the array and count them.
    int i = thread_start;
    while (i < my_count + thread_start) {
        if (test_fn(input[i])) {
            i++;
        }
        else {
            my_count--;
            input[i] = input[my_count + thread_start];
        }
    }

    // scan over counts from each thread to calculate my starting
    //  index in the output
    int my_idx = cg::exclusive_scan(g, my_count);

    for (i = 0; i < my_count; ++i) {
        output[my_idx + i] = input[thread_start + i];
    }
    // return the total number of items in the output
    return g.shfl(my_idx + my_count, g.num_threads() - 1);
}
```

**Example of dynamic buffer space allocation using exclusive_scan_update:**

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

// Buffer partitioning is static to make the example easier to follow,
// but any arbitrary dynamic allocation scheme can be implemented by replacing this function.
__device__ int calculate_buffer_space_needed(cg::thread_block_tile<32>& tile) {
    return tile.thread_rank() % 2 + 1;
}

__device__ int my_thread_data(int i) {
    return i;
}

__global__ void kernel() {
    __shared__ extern int buffer[];
    __shared__ cuda::atomic<int, cuda::thread_scope_block> buffer_used;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    buffer_used = 0;
    block.sync();

    // each thread calculates buffer size it needs
    int buf_needed = calculate_buffer_space_needed(tile);

    // scan over the needs of each thread, result for each thread is an offset
    // of that threadâs part of the buffer. buffer_used is atomically updated with
    // the sum of all thread's inputs, to correctly offset other tileâs allocations
    int buf_offset =
        cg::exclusive_scan_update(tile, buffer_used, buf_needed);

    // each thread fills its own part of the buffer with thread specific data
    for (int i = 0 ; i < buf_needed ; ++i) {
        buffer[buf_offset + i] = my_thread_data(i);
    }

    block.sync();
    // buffer_used now holds total amount of memory allocated
    // buffer is {0, 0, 1, 0, 0, 1 ...};
}
```
### 5.6.3.6.cooperative_groups/sync.h

#### 5.6.3.6.1.barrier_arriveandbarrier_wait

```cpp
T::arrival_token T::barrier_arrive();
void T::barrier_wait(T::arrival_token&&);
```

`barrier_arrive` 和 `barrier_wait` 成员函数提供了一个类似于 `cuda::barrier`[(了解更多)](../04-special-topics/async-barriers.html#asynchronous-barriers) 的同步 API。Cooperative Groups 会自动初始化组屏障，但由于这些操作的集体性质，到达和等待操作有一个额外的限制：组中的所有线程在每个阶段必须到达屏障并等待一次。当使用一个组调用 `barrier_arrive` 时，在该屏障阶段通过 `barrier_wait` 调用观察到完成之前，调用任何集体操作或使用该组进行另一次屏障到达的结果是未定义的。在 `barrier_wait` 上阻塞的线程可能会在其他线程调用 `barrier_wait` 之前从同步中释放，但这只有在组中的所有线程都调用了 `barrier_arrive` 之后才会发生。组类型 `T` 可以是任何[隐式组](../04-special-topics/cooperative-groups.html#cooperative-groups-implicit-groups)。这允许线程在到达之后、等待同步解决之前执行独立的工作，从而可以隐藏部分同步延迟。`barrier_arrive` 返回一个 `arrival_token` 对象，该对象必须传递给相应的 `barrier_wait`。令牌以这种方式被消耗，不能用于另一次 `barrier_wait` 调用。

**使用 barrier_arrive 和 barrier_wait 同步整个集群中共享内存初始化的示例：**

```cpp
#include <cooperative_groups.h>

using namespace cooperative_groups;

void __device__ init_shared_data(const thread_block& block, int *data);
void __device__ local_processing(const thread_block& block);
void __device__ process_shared_data(const thread_block& block, int *data);

__global__ void cluster_kernel() {
    extern __shared__ int array[];
    auto cluster = this_cluster();
    auto block   = this_thread_block();

    // 使用此线程块初始化一些共享状态
    init_shared_data(block, &array[0]);

    auto token = cluster.barrier_arrive(); // 让其他块知道此块正在运行且数据已初始化

    // 执行一些本地处理以隐藏同步延迟
    local_processing(block);

    // 映射集群中下一个块的共享内存中的数据
    int *dsmem = cluster.map_shared_rank(&array[0], (cluster.block_rank() + 1) % cluster.num_blocks());

    // 在访问 dsmem 之前，确保集群中的所有其他块都在运行且已初始化共享数据
    cluster.barrier_wait(std::move(token));

    // 使用分布式共享内存中的数据
    process_shared_data(block, dsmem);
    cluster.sync();
}
```

#### 5.6.3.6.2.sync

```cpp
static void T::sync();

template <typename T>
void sync(T& group);
```

`sync` 同步组中指定的线程。组类型 `T` 可以是任何现有的组类型，因为它们都支持同步。它作为成员函数存在于每个组类型中，或者作为一个以组为参数的独立函数。
##### 5.6.3.6.2.1. 网格同步

在引入协作组（Cooperative Groups）之前，CUDA 编程模型只允许在内核完成边界处进行线程块之间的同步。内核边界伴随着状态的隐式失效，并可能带来性能影响。

例如，在某些用例中，应用程序拥有大量的小型内核，每个内核代表处理流水线中的一个阶段。当前 CUDA 编程模型要求存在这些内核，以确保在一个流水线阶段上操作的线程块，在下一个流水线阶段上操作的线程块准备好消费数据之前，已经产生了数据。在这种情况下，提供全局线程块间同步的能力将允许应用程序被重构为具有持久线程块，这些线程块能够在给定阶段完成时在设备上进行同步。

要在内核内跨整个网格进行同步，只需使用 `grid.sync()` 函数：

```cpp
grid_group grid = this_grid();
grid.sync();
```

并且在启动内核时，需要使用 `cudaLaunchCooperativeKernel` CUDA 运行时启动 API 或其等效的 CUDA 驱动程序 API，而不是 `<<<...>>>` 执行配置语法。

**示例：**

为了保证线程块在 GPU 上的共存性，需要仔细考虑启动的块数量。例如，可以启动与 SM 数量一样多的块，如下所示：

```cpp
int dev = 0;
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
// 初始化，然后启动
cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount, numThreads, args);
```

或者，您可以通过使用占用率计算器计算每个 SM 可以同时容纳多少个块来最大化暴露的并行度，如下所示：

```cpp
/// 这将在默认流上启动一个可以最大程度填充 GPU 的网格，并附带内核参数
int numBlocksPerSm = 0;
 // my_kernel 将启动的线程数
int numThreads = 128;
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, numThreads, 0);
// 启动
void *kernelArgs[] = { /* 添加内核参数 */ };
dim3 dimBlock(numThreads, 1, 1);
dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
cudaLaunchCooperativeKernel((void*)my_kernel, dimGrid, dimBlock, kernelArgs);
```

良好的做法是首先通过查询设备属性 `cudaDevAttrCooperativeLaunch` 来确保设备支持协作启动：

```cpp
int dev = 0;
int supportsCoopLaunch = 0;
cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
```

如果设备 0 支持该属性，这会将 `supportsCoopLaunch` 设置为 1。仅支持计算能力 6.0 及更高的设备。此外，您需要运行在以下任一环境中：
- 不带 MPS 的 Linux 平台
- 带 MPS 且设备计算能力为 7.0 或更高的 Linux 平台
- 最新的 Windows 平台

## 5.6.4. CUDA 设备运行时

CUDA 设备运行时是一个在内核代码中可用的 API，它提供了许多与主机端 CUDA 运行时 API 相同的功能。这些 API 最常用于 [CUDA 动态并行](../04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism) 或 [设备图启动](../04-special-topics/cuda-graphs.html#cuda-graphs-device-graph-launch) 的上下文中。

### 5.6.4.1. 在 CUDA 代码中包含设备运行时 API

与主机端运行时 API 类似，CUDA 设备运行时 API 的原型会在程序编译期间自动包含。无需显式包含 `cuda_device_runtime_api.h`。

### 5.6.4.2. CUDA 设备运行时中的内存

#### 5.6.4.2.1. 配置选项

设备运行时系统软件的资源分配通过主机程序中的 `cudaDeviceSetLimit()` API 控制。必须在启动任何内核之前设置限制，并且在 GPU 正在主动运行程序时不能更改。

可以设置以下命名限制：

| 限制 | 行为 |
| --- | --- |
| cudaLimitDevRuntimePendingLaunchCount | 控制为缓冲尚未开始执行的内核启动和事件预留的内存量，原因可能是未解决的依赖关系或缺乏执行资源。当缓冲区已满时，在设备端内核启动期间尝试分配启动槽位将失败并返回 `cudaErrorLaunchOutOfResources`，而尝试分配事件槽位将失败并返回 `cudaErrorMemoryAllocation`。默认的启动槽位数量是 2048。应用程序可以通过设置 `cudaLimitDevRuntimePendingLaunchCount` 来增加启动和/或事件槽位的数量。分配的事件槽位数量是该限制值的两倍。 |
| cudaLimitStackSize | 控制每个 GPU 线程的栈大小（以字节为单位）。CUDA 驱动程序会根据需要自动增加每次内核启动的每线程栈大小。此大小在每次启动后不会重置回原始值。要将每线程栈大小设置为不同的值，可以调用 `cudaDeviceSetLimit()` 来设置此限制。栈将立即调整大小，并且如有必要，设备将阻塞，直到所有先前请求的任务完成。可以调用 `cudaDeviceGetLimit()` 来获取当前的每线程栈大小。 |

#### 5.6.4.2.2. 分配与生命周期

`cudaMalloc()` 和 `cudaFree()` 在主机环境和设备环境之间具有不同的语义。当从主机调用时，`cudaMalloc()` 从未使用的设备内存中分配一个新区域。当从设备运行时调用时，这些函数映射到设备端的 `malloc()` 和 `free()`。这意味着在设备环境中，可分配的总内存受限于设备 `malloc()` 堆的大小，该大小可能小于可用的未使用设备内存。此外，从主机程序对在设备上通过 `cudaMalloc()` 分配的指针调用 `cudaFree()`，或者反之，都是错误的。
|  | 主机上的 cudaMalloc() | 设备上的 cudaMalloc() |
| --- | --- | --- |
| 主机上的 cudaFree() | 支持 | 不支持 |
| 设备上的 cudaFree() | 不支持 | 支持 |
| 分配限制 | 可用的设备内存 | cudaLimitMallocHeapSize |

##### 5.6.4.2.2.1. 内存声明

###### 5.6.4.2.2.1.1. 设备和常量内存

使用 `__device__` 或 `__constant__` 内存空间说明符在文件作用域声明的内存，在使用设备运行时表现相同。所有内核都可以读取或写入设备变量，无论内核最初是由主机运行时还是设备运行时启动的。同样，所有内核对于在模块作用域声明的 `__constant__` 变量都具有相同的视图。

###### 5.6.4.2.2.1.2. 纹理和表面

> 设备运行时不允许在设备代码内创建或销毁纹理或表面对象。从主机创建的纹理和表面对象可以在设备上自由使用和传递。无论它们是在哪里创建的，动态创建的纹理对象始终有效，并且可以从父内核传递给子内核。

设备运行时不支持在从设备启动的内核中使用旧式的模块作用域（即计算能力 2.0 或 Fermi 风格）的纹理和表面。模块作用域（旧式）纹理可以从主机创建，并像在任何内核中一样在设备代码中使用，但只能由顶级内核（即从主机启动的内核）使用。

###### 5.6.4.2.2.1.3. 共享内存变量声明

在 CUDA C++ 中，共享内存可以声明为静态大小的文件作用域或函数作用域变量，也可以声明为 `extern` 变量，其大小在运行时由内核的调用者通过启动配置参数确定。这两种类型的声明在设备运行时下都是有效的。

```cpp
__global__ void permute(int n, int *data) {
   extern __shared__ int smem[];
   if (n <= 1)
       return;

   smem[threadIdx.x] = data[threadIdx.x];
   __syncthreads();

   permute_data(smem, n);
   __syncthreads();

   // Write back to GMEM since we can't pass SMEM to children.
   data[threadIdx.x] = smem[threadIdx.x];
   __syncthreads();

   if (threadIdx.x == 0) {
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data);
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data+n/2);
   }
}

void host_launch(int *data) {
    permute<<< 1, 256, 256*sizeof(int) >>>(256, data);
}
```

###### 5.6.4.2.2.1.4. 常量内存

常量不能从设备修改。它们只能从主机修改，但是，如果在某个常量的生命周期内，存在一个并发执行的线程网格访问该常量，那么从主机修改该常量的行为是未定义的。

###### 5.6.4.2.2.1.5. 符号地址

设备端符号（即标记为 `__device__` 的符号）可以通过 `&` 运算符在内核中直接引用，因为所有全局作用域的设备变量都在内核的可见地址空间中。这也适用于 `__constant__` 符号，尽管在这种情况下指针将引用只读数据。
由于设备端符号可以直接引用，那些引用符号的 CUDA 运行时 API（例如 `cudaMemcpyToSymbol()` 或 `cudaGetSymbolAddress()`）变得不必要，并且不受设备运行时支持。这意味着，即使是在子内核启动之前，也无法从正在运行的内核内部更改常量数据，因为对 `__constant__` 空间的引用是只读的。

### 5.6.4.3. SM ID 和 Warp ID

请注意，在 PTX 中，`%smid` 和 `%warpid` 被定义为易失值。设备运行时可能会将线程块重新调度到不同的 SM 上，以便更有效地管理资源。因此，依赖 `%smid` 或 `%warpid` 在线程或线程块的整个生命周期内保持不变是不安全的。

### 5.6.4.4. 启动设置 API

[设备端内核启动](../04-special-topics/dynamic-parallelism.html#dynamic-parallelism-device-runtime-kernel-launch) 描述了使用与主机 CUDA 运行时 API 相同的三重尖括号启动符号从设备代码启动内核的语法。

内核启动是通过设备运行时库暴露的系统级机制。它也可以通过 `cudaGetParameterBuffer()` 和 `cudaLaunchDevice()` API 直接从 PTX 使用。允许 CUDA 应用程序自行调用这些 API，其要求与 PTX 相同。在这两种情况下，用户都有责任根据规范以正确的格式正确填充所有必要的数据结构。这些数据结构保证向后兼容。

与主机端启动一样，设备端操作符 `<<<>>>` 映射到底层的内核启动 API。这使得以 PTX 为目标的用户能够执行启动。NVCC 编译器前端将 `<<<>>>` 转换为这些调用。

| 运行时 API 启动函数 | 与主机运行时行为的差异描述（若无描述则行为相同） |
| --- | --- |
| cudaGetParameterBuffer | 由 `<<<>>>` 自动生成。注意与主机对应 API 不同。 |
| cudaLaunchDevice | 由 `<<<>>>` 自动生成。注意与主机对应 API 不同。 |

这些启动函数的 API 与 CUDA 运行时 API 不同，定义如下：

```cpp
extern   device   cudaError_t cudaGetParameterBuffer(void **params);
extern __device__ cudaError_t cudaLaunchDevice(void *kernel,
                                        void *params, dim3 gridDim,
                                        dim3 blockDim,
                                        unsigned int sharedMemSize = 0,
                                        cudaStream_t stream = 0);
```

### 5.6.4.5. 设备管理

设备运行时不支持多 GPU；设备运行时仅能在其当前执行的设备上运行。但是，允许查询系统中任何支持 CUDA 的设备的属性。

### 5.6.4.6. API 参考

设备运行时支持的 CUDA 运行时 API 部分在此详述。主机和设备运行时 API 语法相同；除特别说明外，语义也相同。下表概述了该 API 相对于主机可用版本的情况。
| 运行时 API 函数 | 详细信息 |
| --- | --- |
| cudaDeviceGetCacheConfig |  |
| cudaDeviceGetLimit |  |
| cudaGetLastError | 最后一个错误是每个线程的状态，而不是每个线程块的状态 |
| cudaPeekAtLastError |  |
| cudaGetErrorString |  |
| cudaGetDeviceCount |  |
| cudaDeviceGetAttribute | 将返回任何设备的属性 |
| cudaGetDevice | 始终返回从主机端看到的当前设备 ID |
| cudaStreamCreateWithFlags | 必须传递 cudaStreamNonBlocking 标志 |
| cudaStreamDestroy |  |
| cudaStreamWaitEvent |  |
| cudaEventCreateWithFlags | 必须传递 cudaEventDisableTiming 标志 |
| cudaEventRecord |  |
| cudaEventDestroy |  |
| cudaFuncGetAttributes |  |
| cudaMemcpyAsync | 关于所有 memcpy/memset 函数的说明：仅支持异步 memcpy/set 函数仅允许设备到设备的 memcpy不能传入本地或共享内存指针 |
| cudaMemcpy2DAsync |  |
| cudaMemcpy3DAsync |  |
| cudaMemsetAsync |  |
| cudaMemset2DAsync |  |
| cudaMemset3DAsync |  |
| cudaRuntimeGetVersion |  |
| cudaMalloc | 不能在设备上对主机上创建的指针调用 cudaFree，反之亦然 |
| cudaFree |  |
| cudaOccupancyMaxActiveBlocksPerMultiprocessor |  |
| cudaOccupancyMaxPotentialBlockSize |  |
| cudaOccupancyMaxPotentialBlockSizeVariableSMem |  |

### 5.6.4.7. API 错误与启动失败

与 CUDA 运行时通常情况一样，任何函数都可能返回一个错误代码。返回的最后一个错误代码会被记录下来，并可以通过 `cudaGetLastError()` 调用检索。错误是按线程记录的，因此每个线程都可以识别其最近产生的错误。错误代码的类型是 `cudaError_t`。

与主机端启动类似，设备端启动可能因多种原因（无效参数等）而失败。用户必须调用 `cudaGetLastError()` 来确定启动是否产生了错误，但是，启动后没有错误并不意味着子内核已成功完成。

对于设备端异常，例如访问无效地址，子网格中的错误将返回给主机。

### 5.6.4.8. 设备运行时流

CUDA 设备运行时公开了特殊的命名流，这些流为从设备启动的内核和图提供特定的行为。与设备图启动相关的命名流记录在[设备启动](../04-special-topics/cuda-graphs.html#cuda-graphs-device-graph-device-launch)中。CUDA 设备运行时中可用于内核和 memcpy 操作的另外两个命名流是 `cudaStreamTailLaunch` 和 `cudaStreamTailLaunch`。这些命名流的具体行为在本节中记录。

设备运行时中可以使用命名流和未命名（NULL）流。流句柄不能传递给父网格或子网格。换句话说，流应被视为创建它的网格的私有资源。

设备不支持主机端 NULL 流的跨流屏障语义（详见下文）。为了保持与主机运行时的语义兼容性，所有设备流都必须使用 `cudaStreamCreateWithFlags()` API 创建，并传递 `cudaStreamNonBlocking` 标志。`cudaStreamCreate()` API 在 CUDA 设备运行时中不可用。
由于设备运行时不支持 `cudaStreamSynchronize()` 和 `cudaStreamQuery()`。当应用程序需要知道流启动的子内核已完成时，应改为使用启动到 `cudaStreamTailLaunch` 流中的内核。

#### 5.6.4.8.1. 隐式（NULL）流

在主机程序中，未命名（NULL）流与其他流具有额外的屏障同步语义（详见[阻塞和非阻塞流以及默认流](../02-basics/asynchronous-execution.html#async-execution-blocking-non-blocking-default-stream)）。设备运行时提供了一个在线程块内所有线程之间共享的单一隐式、未命名流，但由于所有命名流都必须使用 `cudaStreamNonBlocking` 标志创建，因此启动到 NULL 流中的工作不会在任何其他流（包括其他线程块的 NULL 流）中的待处理工作上插入隐式依赖。

#### 5.6.4.8.2. 即发即弃流

即发即弃命名流（`cudaStreamFireAndForget`）允许用户以更少的样板代码和无需流跟踪开销的方式启动即发即弃工作。它在功能上与每次启动创建一个新流并启动到该流中相同，但速度更快。

即发即弃启动会立即被调度启动，不依赖于先前启动的网格的完成。除了父网格结束时的隐式同步外，其他网格启动不能依赖于即发即弃启动的完成。因此，尾启动或父网格流中的下一个网格，在父网格的即发即弃工作完成之前不会启动。

```cpp
// 在此示例中，C2 的启动不会等待 C1 完成
__global__ void P( ... ) {
   C1<<< ... , cudaStreamFireAndForget >>>( ... );
   C2<<< ... , cudaStreamFireAndForget >>>( ... );
}
```

即发即弃流不能用于记录事件或等待事件。尝试这样做会导致 `cudaErrorInvalidValue`。当定义了 `CUDA_FORCE_CDP1_IF_SUPPORTED` 进行编译时，不支持即发即弃流。使用即发即弃流要求编译为 64 位模式。

#### 5.6.4.8.3. 尾启动流

尾启动命名流（`cudaStreamTailLaunch`）允许一个网格在其完成后调度一个新网格启动。在大多数情况下，应该可以使用尾启动来实现与 `cudaDeviceSynchronize()` 相同的功能。

每个网格都有自己的尾启动流。在尾流启动之前，网格启动的所有非尾启动工作都会隐式同步。也就是说，父网格的尾启动在父网格以及父网格启动到普通流、每线程流或即发即弃流中的所有工作完成之前不会启动。如果两个网格被启动到同一个网格的尾启动流中，则后一个网格在前一个网格及其所有后代工作完成之前不会启动。

```cpp
// In this example, C2 will only launch after C1 completes.
__global__ void P( ... ) {
   C1<<< ... , cudaStreamTailLaunch >>>( ... );
   C2<<< ... , cudaStreamTailLaunch >>>( ... );
}
```

Grids launched into the tail launch stream will not launch until the completion of all work by the parent grid, including all other grids (and their descendants) launched by the parent in all non-tail launched streams, including work executed or launched after the tail launch.

```cpp
// In this example, C will only launch after all X, F and P complete.
__global__ void P( ... ) {
   C<<< ... , cudaStreamTailLaunch >>>( ... );
   X<<< ... , cudaStreamPerThread >>>( ... );
   F<<< ... , cudaStreamFireAndForget >>>( ... )
}
```

The next grid in the parent gridâs stream will not be launched before a parent gridâs tail launch work has completed. In other words, the tail launch stream behaves as if it were inserted between its parent grid and the next grid in its parent gridâs stream.

```cpp
// In this example, P2 will only launch after C completes.
__global__ void P1( ... ) {
   C<<< ... , cudaStreamTailLaunch >>>( ... );
}

__global__ void P2( ... ) {
}

int main ( ... ) {
   ...
   P1<<< ... >>>( ... );
   P2<<< ... >>>( ... );
   ...
}
```

Each grid only gets one tail launch stream. To tail launch concurrent grids, it can be done like the example below.

```cpp
// In this example,  C1 and C2 will launch concurrently after P's completion
__global__ void T( ... ) {
   C1<<< ... , cudaStreamFireAndForget >>>( ... );
   C2<<< ... , cudaStreamFireAndForget >>>( ... );
}

__global__ void P( ... ) {
   ...
   T<<< ... , cudaStreamTailLaunch >>>( ... );
}
```

The tail launch stream cannot be used to record or wait on events. Attempting to do so results in `cudaErrorInvalidValue`. The tail launch stream is not supported when compiled with `CUDA_FORCE_CDP1_IF_SUPPORTED` defined. Tail launch stream usage requires compilation to be in 64-bit mode.

### 5.6.4.9.ECC Errors

No notification of ECC errors is available to code within a CUDA kernel. ECC errors are reported at the host side once the entire launch tree has completed. Any ECC errors which arise during execution of a nested program will either generate an exception or continue execution (depending upon error and configuration).

 On this page