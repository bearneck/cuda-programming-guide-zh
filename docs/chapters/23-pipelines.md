# 4.10 流水线

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)

---

本页面是否有帮助？

# 4.10. 流水线

流水线在[高级同步原语](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives)中引入，是一种用于暂存工作和协调多缓冲区生产者-消费者模式的机制，通常用于将计算与[异步数据拷贝](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies)重叠执行。

本节主要介绍如何通过 `cuda::pipeline` API 使用流水线（并在适用时提供指向原语的指针）。

## 4.10.1. 初始化

`cuda::pipeline` 可以在不同的线程作用域创建。对于 `cuda::thread_scope_thread` 之外的作用域，需要一个 `cuda::pipeline_shared_state<scope, count>` 对象来协调参与的线程。此状态封装了允许流水线处理最多 `count` 个并发阶段的有限资源。

```cuda
// 在线程作用域创建流水线
constexpr auto scope = cuda::thread_scope_thread;
cuda::pipeline<scope> pipeline = cuda::make_pipeline();
```

```cuda
// 在线程块作用域创建流水线
constexpr auto scope = cuda::thread_scope_block;
constexpr auto stages_count = 2;
__shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
auto pipeline = cuda::make_pipeline(group, &shared_state);
```

流水线可以是*统一的*或*分区的*。在统一流水线中，所有参与的线程既是生产者也是消费者。在分区流水线中，每个参与的线程要么是生产者，要么是消费者，并且其角色在流水线对象的生命周期内不能改变。线程本地流水线不能被分区。要创建分区流水线，我们需要向 `cuda::make_pipeline()` 提供生产者数量或线程的角色。

```cuda
// 在线程块作用域创建分区流水线，其中只有线程 0 是生产者
constexpr auto scope = cuda::thread_scope_block;
constexpr auto stages_count = 2;
__shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
auto thread_role = (group.thread_rank() == 0) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
auto pipeline = cuda::make_pipeline(group, &shared_state, thread_role);
```

为了支持分区，共享的 `cuda::pipeline` 会产生额外的开销，包括每个阶段使用一组共享内存屏障进行同步。即使流水线是统一的并且可以使用 `__syncthreads()` 替代，这些开销也会产生。因此，在可能的情况下，最好使用线程本地流水线，以避免这些开销。

## 4.10.2. 提交工作

将工作提交到流水线阶段涉及以下步骤：

> 生产者线程集合使用 `pipeline.producer_acquire()` **集体获取**流水线**头部**。
> 将异步操作（例如 `memcpy_async`）提交到流水线头部。
> 使用 `pipeline.producer_commit()` **集体提交**（推进）流水线头部。

如果所有资源都在使用中，`pipeline.producer_acquire()` 会阻塞生产者线程，直到消费者线程释放下一个流水线阶段的资源。
## 4.10.3. 消费工作

从先前已提交的阶段消费工作涉及：

> 从一组消费者线程中，**集体等待**阶段完成，例如使用 `pipeline.consumer_wait()` 来等待尾部（最老的）阶段。
> 使用 `pipeline.consumer_release()` **集体释放**该阶段。

对于 `cuda::pipeline<cuda:thread_scope_thread>`，还可以使用友元函数 `cuda::pipeline_consumer_wait_prior<N>()` 来等待除最后 N 个阶段外的所有阶段完成，类似于原语 API 中的 `__pipeline_wait_prior(N)`。

## 4.10.4. 线程束纠缠

流水线机制在同一线程束内的 CUDA 线程间共享。这种共享导致提交的操作序列在同一个线程束内纠缠在一起，在某些情况下可能影响性能。

**提交**。提交操作是合并的，因此对于所有调用提交操作的收敛线程，流水线的序列只递增一次，并且它们提交的操作被批量处理在一起。如果线程束完全收敛，序列递增 1，所有提交的操作将位于流水线的同一阶段；如果线程束完全发散，序列递增 32，所有提交的操作将分散到不同的阶段。

- 令 PB 为线程束共享流水线的实际操作序列。PB = {BP0, BP1, BP2, …, BPL}
- 令 TB 为线程感知的操作序列，如同该序列仅由此线程调用提交操作而递增。TB = {BT0, BT1, BT2, …, BTL}

> `pipeline::producer_commit()` 的返回值来自线程**感知**的批次序列。

- 线程感知序列中的索引总是对齐到实际线程束共享序列中相等或更大的索引。仅当所有提交操作都由完全收敛的线程调用时，两个序列才相等。BTn ≡ BPm，其中 n <= m

例如，当一个线程束完全发散时：

- 线程束共享流水线的实际序列为：PB = {0, 1, 2, 3, ..., 31} ( PL=31 )。
- 该线程束中每个线程的感知序列为：线程 0: TB = {0} ( TL=0 ) 线程 1: TB = {0} ( TL=0 ) … 线程 31: TB = {0} ( TL=0 )

**等待**。CUDA 线程调用 `pipeline::consumer_wait()` 或 `pipeline_consumer_wait_prior<N>()` 来等待**感知**序列 `TB` 中的批次完成。注意，`pipeline::consumer_wait()` 等价于 `pipeline_consumer_wait_prior<N>()`，其中 `N = PL`。

*等待先前* 的变体等待**实际**序列中至少到并包括 `PL-N` 的批次。由于 `TL <= PL`，等待到并包括 `PL-N` 的批次包含了等待批次 `TL-N`。因此，当 `TL < PL` 时，线程将无意中等待额外的、更近的批次。在上面极端的完全发散线程束示例中，每个线程可能等待所有 32 个批次。

!!! note "注意"
    建议通过收敛的线程调用提交操作，以避免过度等待，使线程感知的批次序列与实际序列保持一致。当这些操作之前的代码导致线程发散时，应在调用提交操作之前通过 `__syncwarp` 重新收敛线程束。
## 4.10.5. 提前退出

当一个参与流水线的线程必须提前退出时，该线程必须在退出前使用 `cuda::pipeline::quit()` 显式地退出参与。其余参与的线程可以正常进行后续操作。

## 4.10.6. 跟踪异步内存操作

以下示例演示了如何使用流水线来跟踪复制操作，通过异步内存复制将数据从全局内存集体复制到共享内存。每个线程使用自己的流水线独立提交内存复制，然后等待它们完成并消费数据。有关异步数据复制的更多详细信息，请参阅 [第 3.2.5 节](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies)。

 CUDA C++ 
`cuda::pipeline`

| #include <cuda/pipeline> __global__ void example_kernel ( const float * in ) { constexpr int block_size = 128 ; __shared__ __align__ ( sizeof ( float )) float buffer [ 4 * block_size ]; // 为每个线程创建一个统一的流水线 cuda :: pipeline < cuda :: thread_scope_thread > pipeline = cuda :: make_pipeline (); // 内存复制的第一阶段 pipeline . producer_acquire (); // 每个线程获取第一个块的一个元素 cuda :: memcpy_async ( buffer , in , sizeof ( float ), pipeline ); pipeline . producer_commit (); // 内存复制的第二阶段 pipeline . producer_acquire (); // 每个线程获取第二和第三个块的一个元素 cuda :: memcpy_async ( buffer + block_size , in + block_size , sizeof ( float ), pipeline ); cuda :: memcpy_async ( buffer + 2 * block_size , in + 2 * block_size , sizeof ( float ), pipeline ); pipeline . producer_commit (); // 内存复制的第三阶段 pipeline . producer_acquire (); // 每个线程获取最后一个块的一个元素 cuda :: memcpy_async ( buffer + 3 * block_size , in + 3 * block_size , sizeof ( float ), pipeline ); pipeline . producer_commit (); // 等待最旧的阶段（等待第一阶段） pipeline . consumer_wait (); pipeline . consumer_release (); // __syncthreads(); // 使用第一阶段的数据 // 等待最旧的阶段（等待第二阶段） pipeline . consumer_wait (); pipeline . consumer_release (); // __syncthreads(); // 使用第二阶段的数据 // 等待最旧的阶段（等待第三阶段） pipeline . consumer_wait (); pipeline . consumer_release (); // __syncthreads(); // 使用第三阶段的数据 } |
| --- |

 CUDA C 原语

| #include <cuda_pipeline.h> __global__ void example_kernel ( const float * in ) { constexpr int block_size = 128 ; __shared__ __align__ ( sizeof ( float )) float buffer [ 4 * block_size ]; // 第一批内存复制 // 每个线程获取第一个块的一个元素 __pipeline_memcpy_async ( buffer , in , sizeof ( float )); __pipeline_commit (); // 第二批内存复制 // 每个线程获取第二和第三个块的一个元素 __pipeline_memcpy_async ( buffer + block_size , in + block_size , sizeof ( float )); __pipeline_memcpy_async ( buffer + 2 * block_size , in + 2 * block_size , sizeof ( float )); __pipeline_commit (); // 第三批内存复制 // 每个线程获取最后一个块的一个元素 __pipeline_memcpy_async ( buffer + 3 * block_size , in + 3 * block_size , sizeof ( float )); __pipeline_commit (); // 等待除最后两批之外的所有内存复制（等待第一批） __pipeline_wait_prior ( 2 ); // __syncthreads(); // 使用第一批的数据 // 等待除最后一批之外的所有内存复制（等待第二批） __pipeline_wait_prior ( 1 ); // __syncthreads(); // 使用第二批的数据 // 等待所有批次的内存复制（等待第三批） __pipeline_wait_prior ( 0 ); // __syncthreads(); // 使用最后一批的数据 } |
| --- |

## 4.10.7. 使用流水线的生产者-消费者模式

在[第 4.9.7 节](async-barriers.html#asynchronous-barriers-producer-consumer)中，我们展示了如何对线程块进行空间分区，以使用[异步屏障](async-barriers.html#asynchronous-barriers)实现生产者-消费者模式。使用 `cuda::pipeline`，可以通过一个具有每个数据缓冲区对应一个阶段的单一分区流水线来简化此过程，而无需为每个缓冲区使用两个异步屏障。

 CUDA C++
`cuda::pipeline`

| #include <cuda/pipeline>
#include <cooperative_groups.h>
#pragma nv_diag_suppress static_var_with_dynamic_init
using pipeline = cuda :: pipeline < cuda :: thread_scope_block > ;
__device__ void produce ( pipeline & pipe , int num_stages , int stage , int num_batches , int batch , float * buffer , int buffer_len , float * in , int N ) {
  if ( batch < num_batches ) {
    pipe . producer_acquire ();
    /* 使用异步内存拷贝将数据从 in(batch) 复制到 buffer(stage) */
    pipe . producer_commit ();
  }
}
__device__ void consume ( pipeline & pipe , int num_stages , int stage , int num_batches , int batch , float * buffer , int buffer_len , float * out , int N ) {
  pipe . consumer_wait ();
  /* 消费 buffer(stage) 并更新 out(batch) */
  pipe . consumer_release ();
}
__global__ void producer_consumer_pattern ( float * in , float * out , int N , int buffer_len ) {
  auto block = cooperative_groups :: this_thread_block ();
  /* 下面声明的共享内存缓冲区大小为 2 * buffer_len，以便我们可以在两个缓冲区之间交替工作。
     buffer_0 = buffer 且 buffer_1 = buffer + buffer_len */
  __shared__ extern float buffer [];
  const int num_batches = N / buffer_len ;
  // 创建一个包含 2 个阶段的分区流水线，其中一半线程是生产者，另一半是消费者。
  constexpr auto scope = cuda :: thread_scope_block ;
  constexpr int num_stages = 2 ;
  cuda :: std :: size_t producer_count = block . size () / 2 ;
  __shared__ cuda :: pipeline_shared_state < scope , num_stages > shared_state ;
  pipeline pipe = cuda :: make_pipeline ( block , & shared_state , producer_count );
  // 填充流水线
  if ( block . thread_rank () < producer_count ) {
    for ( int s = 0 ; s < num_stages ; ++ s ) {
      produce ( pipe , num_stages , s , num_batches , s , buffer , buffer_len , in , N );
    }
  }
  // 处理批次
  int stage = 0 ;
  for ( size_t b = 0 ; b < num_batches ; ++ b ) {
    if ( block . thread_rank () < producer_count ) {
      // 预取下一个批次
      produce ( pipe , num_stages , stage , num_batches , b + num_stages , buffer , buffer_len , in , N );
    } else {
      // 消费最旧的批次
      consume ( pipe , num_stages , stage , num_batches , b , buffer , buffer_len , out , N );
    }
    stage = ( stage + 1 ) % num_stages ;
  }
} |
| --- |

在此示例中，我们使用线程块中的一半线程作为生产者，另一半作为消费者。首先，我们需要创建一个 `cuda::pipeline` 对象。由于我们希望一些线程作为生产者，一些作为消费者，我们需要使用一个具有 `cuda::thread_scope_block` 范围的**分区**流水线。分区流水线需要一个 `cuda::pipeline_shared_state` 来协调参与的线程。我们在线程块范围内初始化一个 2 阶段流水线的状态，然后调用 `cuda::make_pipeline()`。接下来，生产者线程通过提交从 `in` 到 `buffer` 的异步拷贝来填充流水线。此时，所有数据拷贝都在进行中。最后，在主循环中，我们遍历所有数据批次，根据线程是生产者还是消费者，我们为未来的批次提交另一个异步拷贝，或者消费当前批次。
在本页面