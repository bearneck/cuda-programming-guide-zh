# 4.11 异步数据拷贝

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)

---

此页面是否有帮助？

# 4.11. 异步数据拷贝

基于[第 3.2.5 节](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies)，本节提供了在 GPU 内存层次结构内进行异步数据移动的详细指导和示例。它涵盖了用于逐元素拷贝的 LDGSTS、用于批量（一维和多维）传输的张量内存加速器（TMA）、用于寄存器到分布式共享内存拷贝的 STAS，并展示了这些机制如何与[异步屏障](async-barriers.html#asynchronous-barriers)和[流水线](pipelines.html#pipelines)集成。

## 4.11.1. 使用 LDGSTS

许多 CUDA 应用程序需要在全局内存和共享内存之间频繁移动数据。这通常涉及拷贝较小的数据元素或执行不规则的内存访问模式。LDGSTS（计算能力 8.0+，参见 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy)）的主要目标是提供一种从全局内存到共享内存的高效异步数据传输机制，用于较小的、逐元素的数据传输，同时通过重叠执行实现计算资源的更好利用。

**维度**。LDGSTS 支持拷贝 4、8 或 16 字节。拷贝 4 或 8 字节总是在所谓的 L1 ACCESS 模式下进行，在这种情况下数据也会缓存在 L1 中；而拷贝 16 字节则启用 L1 BYPASS 模式，在这种情况下 L1 不会被污染。

**源和目标**。使用 LDGSTS 进行异步拷贝操作唯一支持的方向是从全局内存到共享内存。指针需要根据被拷贝数据的大小对齐到 4、8 或 16 字节。当共享内存和全局内存的对齐都是 128 字节时，可以获得最佳性能。

**异步性**。使用 LDGSTS 进行的数据传输是[异步的](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features)，并被建模为异步线程操作（参见[异步线程和异步代理](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)）。这使得发起线程可以在硬件异步拷贝数据的同时继续进行计算。*数据传输在实践中是否异步发生取决于硬件实现，并且未来可能会改变*。

LDGSTS 必须在操作完成时提供一个信号。LDGSTS 可以使用[共享内存屏障](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers)或[流水线](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines)作为提供完成信号的机制。默认情况下，每个线程只等待其自身的 LDGSTS 拷贝。因此，如果你使用 LDGSTS 预取一些将与其他线程共享的数据，在与 LDGSTS 完成机制同步之后，必须使用 `__syncthreads()`。
| 方向 | 异步拷贝（LDGSTS，计算能力 8.0+） |  |  |
| --- | --- | --- | --- | --- |
| 源 | 目标 | 完成机制 | API |
| 全局内存 | 全局内存 |  |  |
| shared::cta | 全局内存 |  |  |
| 全局内存 | shared::cta | 共享内存屏障,
流水线 | cuda::memcpy_async , cooperative_groups::memcpy_async , __pipeline_memcpy_async |
| 全局内存 | shared::cluster |  |  |
| shared::cluster | shared::cta |  |  |
| shared::cta | shared::cta |  |  |

在接下来的章节中，我们将通过示例演示如何使用 LDGSTS，并解释不同 API 之间的区别。

### 4.11.1.1. 在条件代码中批量加载

在这个模板计算示例中，线程块的第一个线程束负责从中心以及左右光晕区域集体加载所有需要的数据。使用同步拷贝时，由于代码的条件性，编译器可能会选择生成一系列从全局内存加载（LDG）到存储到共享内存（STS）的指令，而不是 3 次 LDG 后跟 3 次 STS，而后者才是隐藏全局内存延迟的最佳数据加载方式。

```cpp
__global__ void stencil_kernel(const float *left, const float *center, const float *right)
{
    // Left halo (8 elements) - center (32 elements) - right halo (8 elements)
    __shared__ float buffer[8 + 32 + 8];
    const int tid = threadIdx.x;

    if (tid < 8) {
        buffer[tid] = left[tid]; // Left halo
    } else if (tid >= 32 - 8) {
        buffer[tid + 16] = right[tid]; // Right halo
    }
    if (tid < 32) {
      buffer[tid + 8] = center[tid]; // Center
    }
    __syncthreads();

    // Compute stencil
}
```

为了确保数据以最佳方式加载，我们可以用异步拷贝替换同步内存拷贝，这些异步拷贝直接将数据从全局内存加载到共享内存。这不仅通过将数据直接复制到共享内存来减少寄存器使用，还确保了所有来自全局内存的加载都在进行中。

 CUDA C++ 
`cuda::memcpy_async`

| #include <cooperative_groups.h> #include <cuda/barrier> __global__ void stencil_kernel ( const float * left , const float * center , const float * right ) { auto block = cooperative_groups :: this_thread_block (); auto thread = cooperative_groups :: this_thread (); using barrier_t = cuda :: barrier < cuda :: thread_scope_block > ; __shared__ barrier_t barrier ; __shared__ float buffer [ 8 + 32 + 8 ]; // Initialize synchronization object. if ( block . thread_rank () == 0 ) { init ( & barrier , block . size ()); } __syncthreads (); // Version 1: Issue the copies in individual threads. if ( tid < 8 ) { cuda :: memcpy_async ( buffer + tid , left + tid , cuda :: aligned_size_t < 4 > ( sizeof ( float )), barrier ); // Left halo // or cuda::memcpy_async(thread, buffer + tid, left + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); } else if ( tid >= 32 - 8 ) { cuda :: memcpy_async ( buffer + tid + 16 , right + tid , cuda :: aligned_size_t < 4 > ( sizeof ( float )), barrier ); // Right halo // or cuda::memcpy_async(thread, buffer + tid + 16, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); } if ( tid < 32 ) { cuda :: memcpy_async ( buffer + 40 , right + tid , cuda :: aligned_size_t < 4 > ( sizeof ( float )), barrier ); // Center // or cuda::memcpy_async(thread, buffer + 40, right + tid, cuda::aligned_size_t<4>(sizeof(float)), barrier); } // Version 2: Cooperatively issue the copies across all threads. cuda :: memcpy_async ( block , buffer , left , cuda :: aligned_size_t < 4 > ( 8 * sizeof ( float )), barrier ); // Left halo cuda :: memcpy_async ( block , buffer + 8 , center , cuda :: aligned_size_t < 4 > ( 32 * sizeof ( float )), barrier ); // Center cuda :: memcpy_async ( block , buffer + 40 , right , cuda :: aligned_size_t < 4 > ( 8 * sizeof ( float )), barrier ); // Right halo // Wait for all copies to complete. barrier . arrive_and_wait (); __syncthreads (); // Compute stencil } |
| --- |

 CUDA C++
`cooperative_groups::memcpy_async`

| #include <cooperative_groups.h> #include <cooperative_groups/memcpy_async.h> namespace cg = cooperative_groups ; __global__ void stencil_kernel ( const float * left , const float * center , const float * right ) { cg :: thread_block block = cg :: this_thread_block (); // 左侧光晕 (8 个元素) - 中心 (32 个元素) - 右侧光晕 (8 个元素). __shared__ float buffer [ 8 + 32 + 8 ]; // 跨所有线程协作发起复制操作。 cg :: memcpy_async ( block , buffer , left , 8 * sizeof ( float )); // 左侧光晕 cg :: memcpy_async ( block , buffer + 8 , center , 32 * sizeof ( float )); // 中心 cg :: memcpy_async ( block , buffer + 40 , right , 8 * sizeof ( float )); // 右侧光晕 cg :: wait ( block ); // 等待所有复制操作完成。 __syncthreads (); // 计算模板。 } |
| --- |

 CUDA C 原语

| #include <cuda_pipeline.h> __global__ void stencil_kernel ( const float * left , const float * center , const float * right ) { // 左侧光晕 (8 个元素) - 中心 (32 个元素) - 右侧光晕 (8 个元素). __shared__ float buffer [ 8 + 32 + 8 ]; const int tid = threadIdx . x ; if ( tid < 8 ) { __pipeline_memcpy_async ( buffer + tid , left + tid , sizeof ( float )); // 左侧光晕 } else if ( tid >= 32 - 8 ) { __pipeline_memcpy_async ( buffer + tid + 16 , right + tid , sizeof ( float )); // 右侧光晕 } if ( tid < 32 ) { __pipeline_memcpy_async ( buffer + tid + 8 , center + tid , sizeof ( float )); // 中心 } __pipeline_commit (); __pipeline_wait_prior ( 0 ); __syncthreads (); // 计算模板。 } |
| --- |

用于 `cuda::barrier` 的 `cuda::memcpy_async` 重载支持使用[异步屏障](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers)来同步异步数据传输。此重载通过在执行复制操作时，在创建时递增当前阶段的预期计数，并在复制操作完成时递减它，来执行复制操作，就好像是由绑定到屏障的另一个线程执行的一样。这样，只有当参与屏障的所有线程都已到达，并且绑定到屏障当前阶段的所有 `memcpy_async` 操作都已完成时，`barrier` 的阶段才会推进。我们使用一个块范围的 `barrier`，其中块中的所有线程都参与，并使用 `arrive_and_wait` 合并到达屏障和在屏障上的等待，因为我们在阶段之间不执行任何工作。

请注意，我们可以使用线程级复制（版本 1）或集体复制（版本 2）来达到相同的结果。在版本 2 中，API 会自动处理底层复制是如何完成的。在两个版本中，我们都使用 `cuda::aligned_size_t<4>()` 来通知编译器数据按 4 字节对齐，并且要复制的数据大小是 4 的倍数，以便启用 LDGSTS。请注意，为了与 `cuda::barrier` 互操作，这里使用了来自 `cuda/barrier` 头文件的 `cuda::memcpy_async`。

[cooperative_groups::memcpy_async](../05-appendices/device-callable-apis.html#cg-api-async-memcpy) 实现跨块中的所有线程协作协调内存传输，但使用 `cg::wait(block)` 而不是显式的屏障操作来同步完成。
基于底层原语的实现使用 `__pipeline_memcpy_async()` 来启动逐元素的内存传输，使用 `__pipeline_commit()` 来提交一批复制操作，并使用 `__pipeline_wait_prior(0)` 来等待流水线中的所有操作完成。与更高级的 API 相比，这提供了最直接的控制，代价是代码更冗长。它还确保在底层会使用 LDGSTS，而更高级的 API 则无法保证这一点。

在此示例中，`cooperative_groups::memcpy_async` API 的效率低于其他 API，因为它会在每次启动时立即自动提交每个复制操作，从而阻止了其他 API 所支持的、在单次提交操作之前批量处理多个复制操作的优化。

### 4.11.1.2. 数据预取

在此示例中，我们将演示如何使用异步数据复制将数据从全局内存预取到共享内存。在迭代的复制和计算模式中，这允许用当前迭代的计算来隐藏未来迭代的数据传输延迟，从而可能增加在途字节数。

 CUDA C++
`cuda::memcpy_async`

| #include <cooperative_groups.h>
#include <cuda/pipeline>

template <
  size_t num_stages = 2 /* Pipeline with num_stages stages */
>
__global__ void prefetch_kernel(
  int* global_out,
  int const* global_in,
  size_t size,
  size_t batch_size
) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto thread = cooperative_groups::this_thread();

  assert(size == batch_size * grid.size()); // Assume input size fits batch_size * grid_size

  extern __shared__ int shared[]; // num_stages * block.size() * sizeof(int) bytes

  size_t shared_offset[num_stages];
  for (int s = 0; s < num_stages; ++s) {
    shared_offset[s] = s * block.size();
  }

  cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

  auto block_batch = [&](size_t batch) -> int {
    return block.group_index().x * block.size() + grid.size() * batch;
  };

  // Fill the pipeline with the first ``num_stages`` batches.
  for (int s = 0; s < num_stages; ++s) {
    pipeline.producer_acquire();
    cuda::memcpy_async(
      shared + shared_offset[s] + tid,
      global_in + block_batch(s) + tid,
      cuda::aligned_size_t<4>(sizeof(int)),
      pipeline
    );
    pipeline.producer_commit();
  }

  int stage = 0;
  // compute_batch: next batch to process
  // fetch_batch:   next batch to fetch from global memory
  for (
    size_t compute_batch = 0, fetch_batch = num_stages;
    compute_batch < batch_size;
    ++compute_batch, ++fetch_batch
  ) {
    // Wait for the first requested stage to complete.
    constexpr size_t pending_batches = num_stages - 1;
    cuda::pipeline_consumer_wait_prior<pending_batches>(pipeline);
    __syncthreads(); // Not required if each thread works on the data it copied.

    // Compute on the current batch
    compute(
      global_out + block_batch(compute_batch) + tid,
      shared + shared_offset[stage] + tid
    );

    // Release the current stage.
    pipeline.consumer_release();
    __syncthreads(); // Not required if each thread works on the data it copied.

    // Load future stage ``num_stages`` ahead of current compute batch.
    pipeline.producer_acquire();
    if (fetch_batch < batch_size) {
      cuda::memcpy_async(
        shared + shared_offset[stage] + tid,
        global_in + block_batch(fetch_batch) + tid,
        cuda::aligned_size_t<4>(sizeof(int)),
        pipeline
      );
    }
    pipeline.producer_commit();

    stage = (stage + 1) % num_stages;
  }
} |
| --- |

 CUDA C++
`cooperative_groups::memcpy_async`

| #include <cooperative_groups.h> #include <cooperative_groups/memcpy_async.h> namespace cg = cooperative_groups ; template < size_t num_stages = 2 /* Pipeline with num_stages stages */ > __global__ void prefetch_kernel ( int * global_out , int const * global_in , size_t size , size_t batch_size ) { auto grid = cooperative_groups :: this_grid (); auto block = cooperative_groups :: this_thread_block (); assert ( size == batch_size * grid . size ()); // Assume input size fits batch_size * grid_size extern __shared__ int shared []; // num_stages * block.size() * sizeof(int) bytes size_t shared_offset [ num_stages ]; for ( int s = 0 ; s < num_stages ; ++ s ) shared_offset [ s ] = s * block . size (); cuda :: pipeline < cuda :: thread_scope_thread > pipeline = cuda :: make_pipeline (); auto block_batch = [ & ]( size_t batch ) -> int { return block . group_index (). x * block . size () + grid . size () * batch ; }; // Fill the pipeline with the first ``num_stages`` batches. for ( int s = 0 ; s < num_stages ; ++ s ) { size_t block_batch_idx = block_batch ( s ); cg :: memcpy_async ( block , shared + shared_offset [ s ], global_in + block_batch_idx , cuda :: aligned_size_t < 4 > ( sizeof ( int ))); } int stage = 0 ; // compute_batch: next batch to process // fetch_batch:   next batch to fetch from global memory for ( size_t compute_batch = 0 , fetch_batch = num_stages ; compute_batch < batch_size ; ++ compute_batch , ++ fetch_batch ) { // Wait for the first requested stage to complete. size_t pending_batches = ( fetch_batch < batch_size - num_stages ) ? num_stages - 1 : batch_size - fetch_batch - 1 ; cg :: wait_prior ( pending_batches ); __syncthreads (); // Not required if each thread works on the data it copied. // Compute on the current batch. compute ( global_out + block_batch ( compute_batch ) + tid , shared + shared_offset [ stage ] + tid ); __syncthreads (); // Not required if each thread works on the data it copied. // Load future stage ``num_stages`` ahead of current compute batch. size_t fetch_batch_idx = block_batch ( fetch_batch ); if ( fetch_batch < batch_size ) { cg :: memcpy_async ( block , shared + shared_offset [ stage ], global_in + block_batch ( fetch_batch ), cuda :: aligned_size_t < 4 > ( sizeof ( int )) * block . size ()); } stage = ( stage + 1 ) % num_stages ; } } |
| --- |

 CUDA C 原语

| #include <cooperative_groups.h> #include <cuda_awbarrier_primitives.h> template < size_t num_stages = 2 /* Pipeline with num_stages stages */ > __global__ void prefetch_kernel ( int * global_out , int const * global_in , size_t size , size_t batch_size ) { auto grid = cooperative_groups :: this_grid (); auto block = cooperative_groups :: this_thread_block (); assert ( size == batch_size * grid . size ()); // Assume input size fits batch_size * grid_size extern __shared__ int shared []; // num_stages * block.size() * sizeof(int) bytes size_t shared_offset [ num_stages ]; for ( int s = 0 ; s < num_stages ; ++ s ) shared_offset [ s ] = s * block . size (); auto block_batch = [ & ]( size_t batch ) -> int { return block . group_index (). x * block . size () + grid . size () * batch ; }; // Fill the pipeline with the first ``num_stages`` batches. for ( int s = 0 ; s < num_stages ; ++ s ) { __pipeline_memcpy_async ( shared + shared_offset [ s ] + tid , global_in + block_batch ( s ) + tid , cuda :: aligned_size_t < 4 > ( sizeof ( int ))); __pipeline_commit (); } // compute_batch: next batch to process // fetch_batch:   next batch to fetch from global memory for ( size_t compute_batch = 0 , fetch_batch = num_stages ; compute_batch < batch_size ; ++ compute_batch , ++ fetch_batch ) { // Wait for the first requested stage to complete. constexpr size_t pending_batches = num_stages - 1 ; __pipeline_wait_prior < pending_batches > (); __syncthreads (); // Not required if each thread works on the data it copied. // Compute on the current batch. compute ( global_out + block_batch ( compute_batch ) + tid , shared + shared_offset [ stage ] + tid ); __syncthreads (); // Not required if each thread works on the data it copied. // Load future stage ``num_stages`` ahead of current compute batch. if ( fetch_batch < batch_size ) { __pipeline_memcpy_async ( shared + shared_offset [ stage ] + tid , global_in + block_batch ( fetch_batch ) + tid , cuda :: aligned_size_t < 4 > ( sizeof ( int ))); } __pipeline_commit (); stage = ( stage + 1 ) % num_stages ; } } |
`cuda::memcpy_async` 的实现展示了使用 `cuda::pipeline`（参见[管道](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines)）与 `cuda::memcpy_async` 进行多阶段数据预取。它：

- 初始化一个线程局部的管道。
- 通过调度 num_stages 个 memcpy_async 操作来启动管道。
- 循环遍历所有批次：它阻塞所有线程直到当前批次完成，然后对当前批次执行计算，最后如果还有下一个批次，则调度下一个 memcpy_async。

`cooperative_groups::memcpy_async` 的实现展示了使用 `cooperative_groups::memcpy_async` 进行多阶段数据预取。与之前实现的主要区别在于我们不使用管道对象，而是依赖 `cooperative_groups::memcpy_async` 在底层分阶段调度内存传输。

CUDA C 原语的实现展示了使用低级原语进行多阶段数据预取，其方式与第一个示例非常相似。

在此示例中，实现高效代码生成的一个重要细节是，即使没有更多批次需要获取，也要在管道中保持 `num_stages` 个批次。这是通过即使没有更多批次需要获取也提交到管道（`pipeline.producer_commit()` 或 `__pipeline_commit()`）来实现的。请注意，这对于协作组 API 是不可能的，因为我们无法访问内部管道。

### 4.11.1.3. 通过线程束专用化实现生产者-消费者模式

在此示例中，我们将演示如何实现生产者-消费者模式，其中一个线程束被专用化为生产者，执行从全局内存到共享内存的异步数据拷贝，而其余线程束则从共享内存消费数据并执行计算。为了实现生产者和消费者线程之间的并发，我们在共享内存中使用双缓冲。当消费者线程束处理一个缓冲区中的数据时，生产者线程束将下一批数据异步获取到另一个缓冲区中。

 CUDA C++ 
`cuda::memcpy_async`

```cpp
#include <cooperative_groups.h>
#include <cuda/pipeline>
#pragma nv_diag_suppress static_var_with_dynamic_init
using pipeline = cuda :: pipeline < cuda :: thread_scope_block > ;
__device__ void produce ( pipeline & pipe , int num_stages , int stage , int num_batches , int batch , float * buffer , int buffer_len , float * in , int N ) {
  if ( batch < num_batches ) {
    pipe . producer_acquire ();
    /* copy data from in(batch) to buffer(stage) using asynchronous memory copies */
    cuda :: memcpy_async ( buffer + stage * buffer_len + threadIdx . x ,
                           in + batch * buffer_len + threadIdx . x ,
                           cuda :: aligned_size_t < 4 > ( sizeof ( float )),
                           pipe );
    pipe . producer_commit ();
  }
}
__device__ void consume ( pipeline & pipe , int num_stages , int stage , int num_batches , int batch , float * buffer , int buffer_len , float * out , int N ) {
  pipe . consumer_wait ();
  /* consume buffer(stage) and update out(batch) */
  pipe . consumer_release ();
}
__global__ void producer_consumer_pattern ( float * in , float * out , int N , int buffer_len ) {
  auto block = cooperative_groups :: this_thread_block ();
  constexpr int warpSize = 32 ;
  /* Shared memory buffer declared below is of size 2 * buffer_len so that we can alternatively work between two buffers.
     buffer_0 = buffer and buffer_1 = buffer + buffer_len */
  __shared__ extern float buffer [];
  const int num_batches = N / buffer_len ;
  // Create a partitioned pipeline with 2 stages where the first warp is the producer and the other warps are consumers.
  constexpr auto scope = cuda :: thread_scope_block ;
  constexpr int num_stages = 2 ;
  cuda :: std :: size_t producer_count = warpSize ;
  __shared__ cuda :: pipeline_shared_state < scope , num_stages > shared_state ;
  pipeline pipe = cuda :: make_pipeline ( block , & shared_state , producer_count );
  // Producer fills the pipeline
  if ( block . thread_rank () < producer_count )
    for ( int s = 0 ; s < num_stages ; ++ s )
      produce ( pipe , num_stages , s , num_batches , s , buffer , buffer_len , in , N );
  // Process the batches
  int stage = 0 ;
  for ( size_t b = 0 ; b < num_batches ; ++ b ) {
    if ( block . thread_rank () < producer_count ) {
      // Producers prefetch the next batch
      produce ( pipe , num_stages , stage , num_batches , b + num_stages , buffer , buffer_len , in , N );
    } else {
      // Consumers consume the oldest batch
      consume ( pipe , num_stages , stage , num_batches , b , buffer , buffer_len , out , N );
    }
    stage = ( stage + 1 ) % num_stages ;
  }
}
```
| --- |

 CUDA C 原语

| #include <cooperative_groups.h> #include <cuda_awbarrier_primitives.h> __device__ void produce ( __mbarrier_t ready [], __mbarrier_t filled [], float * buffer , int buffer_len , float * in , int N ) { for ( int i = 0 ; i < N / buffer_len ; ++ i ) { __mbarrier_token_t token = __mbarrier_arrive ( & ready [ i % 2 ]); /* 等待 buffer_(i%2) 准备好被填充 */ while ( ! __mbarrier_try_wait ( & ready [ i % 2 ], token , 1000 )) {} /* 生产，即填充 buffer_(i%2) */ __pipeline_memcpy_async ( buffer + i * buffer_len + threadIdx . x , in + i * buffer_len + threadIdx . x , cuda :: aligned_size_t < 4 > ( sizeof ( float ))); __pipeline_arrive_on ( filled [ i % 2 ]); __mbarrier_arrive ( filled [ i % 2 ]); /* buffer_(i%2) 已填充 */ } } __device__ void consume ( __mbarrier_t ready [], __mbarrier_t filled [], float * buffer , int buffer_len , float * out , int N ) { __mbarrier_arrive ( & ready [ 0 ]); /* buffer_0 准备好进行初始填充 */ __mbarrier_arrive ( & ready [ 1 ]); /* buffer_1 准备好进行初始填充 */ for ( int i = 0 ; i < N / buffer_len ; ++ i ) { __mbarrier_token_t token = __mbarrier_arrive ( & filled [ i % 2 ]); while ( ! __mbarrier_try_wait ( & filled [ i % 2 ], token , 1000 )) {} /* 消费 buffer_(i%2) */ __mbarrier_arrive ( & ready [ i % 2 ]); /* buffer_(i%2) 准备好被重新填充 */ } } __global__ void producer_consumer_pattern ( int N , float * in , float * out , int buffer_len ) { /* 下面声明的共享内存缓冲区大小为 2 * buffer_len，以便我们可以在两个缓冲区之间交替工作。 buffer_0 = buffer 且 buffer_1 = buffer + buffer_len */ __shared__ extern float buffer []; /* bar[0] 和 bar[1] 跟踪缓冲区 buffer_0 和 buffer_1 是否准备好被填充，而 bar[2] 和 bar[3] 分别跟踪缓冲区 buffer_0 和 buffer_1 是否已被填充 */ __shared__ __mbarrier_t bar [ 4 ]; // 初始化屏障 auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () < 4 ) __mbarrier_init ( bar + block . thread_rank (), block . size ()); __syncthreads (); if ( block . thread_rank () < warpSize ) produce ( bar , bar + 2 , buffer , buffer_len , in , N ); else consume ( bar , bar + 2 , buffer , buffer_len , out , N ); } |
| --- |

`cuda::memcpy_async` 的实现展示了具有最高抽象级别的 API，使用了 `cuda::memcpy_async` 和一个具有 2 个阶段的 `cuda::pipeline`。它使用了一个分区流水线（参见[流水线](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-pipelines)），其中第一个线程束作为生产者，其余线程束作为消费者。生产者最初填充两个流水线阶段。然后，在主处理循环中，当消费者处理当前批次时，生产者获取未来批次的数据，从而维持稳定的工作流。

基于原语的 CUDA C 原语实现将 `__pipeline_memcpy_async()` 与[共享内存屏障](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers)作为完成机制相结合，以协调异步内存传输。`__pipeline_arrive_on()` 函数将内存复制与屏障关联起来。它将屏障到达计数加一，并且当所有在其之前排序的异步操作完成后，到达计数会自动减一，因此对到达计数的净影响为零。因此，我们还需要使用 `__mbarrier_arrive()` 显式地等待屏障。
## 4.11.2. 使用张量内存加速器 (TMA)

许多应用需要在全局内存中来回移动大量数据。通常，这些数据在全局内存中以多维数组的形式布局，并具有非顺序的数据访问模式。为了减少全局内存访问，在计算使用之前，此类数组的子图块会被复制到共享内存中。加载和存储操作涉及地址计算，这些计算容易出错且具有重复性。为了卸载这些计算，计算能力 9.0 (Hopper) 及更高版本（参见 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk)）引入了*张量内存加速器* (TMA)。TMA 的主要目标是为多维数组提供一种从全局内存到共享内存的高效数据传输机制。

**命名**。张量内存加速器 (TMA) 是一个广义术语，用于指代本节描述的特性。为了向前兼容并减少与 PTX ISA 的差异，本节文本根据所使用的具体复制类型，将 TMA 操作称为*批量异步复制*或*批量张量异步复制*。术语“批量”用于将这些操作与上一节描述的异步内存操作区分开来。

**维度**。TMA 支持复制一维和多维数组（最多 5 维）。一维连续数组的批量异步复制的编程模型与多维数组的批量张量异步复制的编程模型不同。要执行多维数组的批量张量异步复制，硬件需要一个[张量映射](https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html#structCUtensorMap)。此对象描述了多维数组在全局内存和共享内存中的布局。张量映射通常在主机上使用 [cuTensorMapEncode API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY) 创建，然后作为带有 `__grid_constant__` 注解的 `const` 内核参数从主机传输到设备（参见 [__grid_constant__ 参数](../05-appendices/cpp-language-extensions.html#grid-constant)）。张量映射作为带有 `__grid_constant__` 注解的 `const` 内核参数从主机传输到设备，并可在设备上用于在共享内存和全局内存之间复制数据图块。相比之下，执行连续一维数组的批量异步复制则不需要张量映射：它可以在设备上使用指针和大小参数来执行。

**源地址和目的地址**。TMA 操作的源地址和目的地址可以位于共享内存或全局内存中。这些操作可以从全局内存读取数据到共享内存，从共享内存写入数据到全局内存，也可以从共享内存复制到同一集群中另一个线程块的[分布式共享内存](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-distributed-shared-memory)。此外，在集群中时，批量异步张量操作可以指定为*多播*。在这种情况下，数据可以从全局内存传输到集群内多个线程块的共享内存。多播特性针对目标架构 `sm_90a` 进行了优化，在其他目标上可能[性能显著降低](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)。因此，建议与[计算架构](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list) `sm_90a` 一起使用。
**异步性**。使用 TMA 的数据传输是[异步的](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features)，并被建模为异步代理操作（参见[异步线程和异步代理](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)）。这使得发起线程可以在硬件异步复制数据的同时继续进行计算。*数据传输在实践中是否异步发生取决于硬件实现，并且未来可能会改变*。批量异步操作可以使用几种[完成机制](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms)来发出已完成信号。当操作从全局内存读取数据到共享内存时，线程块中的任何线程都可以通过等待[共享内存屏障](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers)来等待数据在共享内存中可读。当批量异步操作将数据从共享内存写入全局内存或分布式共享内存时，只有发起线程可以等待操作完成。这是通过使用基于*批量异步组*的完成机制来实现的。描述完成机制的表格如下所示，也可在[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)中找到。

| 方向 | 异步复制 (TMA, CC 9.0+) |  |
| --- | --- | --- |
| 源 | 目标 | 完成机制 |
| global | global |  |
| shared::cta | global | bulk async-group |
| global | shared::cta | shared memory barrier |
| global | shared::cluster | shared memory barrier (multicast) |
| shared::cta | shared::cluster | shared memory barrier |
| shared::cta | shared::cta |  |

### 4.11.2.1. 使用 TMA 传输一维数组

下表总结了批量异步 TMA 可能的源和目标内存空间、完成机制以及公开其功能的 API。

| 方向 | 批量异步复制 (TMA, CC9.0+) |  |  |
| --- | --- | --- | --- |
| 源 | 目标 | 完成机制 | API |
| global | global |  |  |
| shared::cta | global | bulk async-group | cuda::ptx::cp_async_bulk |
| global | shared::cta | shared memory barrier | cuda::memcpy_async , cuda::device::memcpy_async_tx , cuda::ptx::cp_async_bulk |
| global | shared::cluster | shared memory barrier | cuda::ptx::cp_async_bulk |
| shared::cta | shared::cluster | shared memory barrier | cuda::ptx::cp_async_bulk |
| shared::cta | shared::cta |  |  |

某些功能需要内联 PTX，目前通过 [CUDA 标准 C++](https://nvidia.github.io/cccl/libcudacxx/ptx_api.html) 库中的 `cuda::ptx` 命名空间提供。这些包装器的可用性可以通过以下代码检查：

```cpp
#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Device code is being compiled with older architectures that are incompatible with TMA.");
#endif // __CUDA_MINIMUM_ARCH__
```

Note that `cuda::memcpy_async` uses TMA if the source and destination addresses are 16-byte aligned and the size is a multiple of 16 bytes, otherwise it falls back to synchronous copies. On the other hand, `cuda::device::memcpy_async_tx` and `cuda::ptx::cp_async_bulk` always use TMA and will result in undefined behavior if the requirements are not met.

In the following, we demonstrate how to use bulk-asynchronous copies through an example. The example read-modify-writes a one-dimensional array. The kernel goes through the following steps:

1. Initialize a shared memory barrier as a completion mechanism for the bulk-asynchronous copy from global to shared memory.
2. Initiate the copy of a block of memory from global to shared memory.
3. Arrive and wait on the shared memory barrier for completion of the copy.
4. Increment the shared memory buffer values.
5. Use a proxy fence to ensure shared memory writes (generic proxy) become visible to the subsequent bulk-asynchronous copy (async proxy).
6. Initiate a bulk-asynchronous copy of the buffer in shared memory to global memory.
7. Wait for the bulk-asynchronous copy to have finished reading shared memory.

```cpp
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;

__device__ inline bool is_elected()
{
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
}

__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operation should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory from a single thread.
  if (is_elected()) {
    // Launch the async copy and communicate how many bytes are expected to come in (the transaction count).
    
    // Version 1: cuda::memcpy_async
    cuda::memcpy_async(
        smem_data, data + offset, 
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar);
    
    // Version 2: cuda::device::memcpy_async_tx
    // cuda::device::memcpy_async_tx(
    //   smem_data, data + offset, 
    //   cuda::aligned_size_t<16>(sizeof(smem_data)),
    //   bar);
    // cuda::device::barrier_expect_tx(
    //     cuda::device::barrier_native_handle(bar),
    //     sizeof(smem_data));

    // Version 3: cuda::ptx::cp_async_bulk
    // ptx::cp_async_bulk(
    //     ptx::space_shared, ptx::space_global,
    //     smem_data, data + offset, 
    //     sizeof(smem_data), 
    //     cuda::device::barrier_native_handle(bar));
    // cuda::device::barrier_expect_tx(
    //     cuda::device::barrier_native_handle(bar),
    //     sizeof(smem_data));
  }
  
  // 3a. All threads arrive on the barrier.
  barrier::arrival_token token = bar.arrive();
  
  // 3b. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory.
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory.
  if (is_elected()) {
    ptx::cp_async_bulk(
        ptx::space_global, ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}
```
**屏障初始化**。屏障使用参与线程块的线程数进行初始化。因此，只有当所有线程都到达此屏障时，屏障才会翻转。共享内存屏障在[共享内存屏障](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers)中有更详细的描述。

**TMA 读取**。批量异步复制指令指示硬件将一大块数据复制到共享内存中，并在完成读取后更新共享内存屏障的[事务计数](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations)。通常，发出尽可能少、尺寸尽可能大的批量复制能获得最佳性能。因为复制可以由硬件异步执行，所以没有必要将复制拆分成更小的块。

发起批量异步复制操作的线程还会告知屏障预期有多少事务（tx）会到达。在本例中，事务以字节为单位计数。`cuda::memcpy_async` 会自动执行此操作，但 `cuda::device::memcpy_async_tx` 和 `cuda::ptx::cp_async_bulk` 不会，使用后者后我们需要显式调用 `cuda::ptx::mbarrier_expect_tx`。如果多个线程更新事务计数，预期事务数将是这些更新的总和。屏障只有在所有线程都已到达 **并且** 所有字节都已到达后才会翻转。一旦屏障翻转，从共享内存读取字节就是安全的，无论是线程还是后续的批量异步复制操作都可以安全读取。关于屏障事务计数的更多信息，请参阅[跟踪异步内存操作](async-barriers.html#asynchronous-barriers-tracking)。

**屏障等待**。等待屏障翻转是使用令牌通过 `bar.wait()` 完成的。使用屏障的显式阶段跟踪可能更高效（参见[显式阶段跟踪](async-barriers.html#asynchronous-barriers-explicit-phase)）。

**共享内存写入与同步**。缓冲区值的递增会读取和写入共享内存。为了使后续的批量异步复制操作能看到这些写入，需要使用 `cuda::ptx::fence_proxy_async` 函数。这确保了在后续通过异步代理进行读取的批量异步复制操作之前，对共享内存的写入操作被排序。因此，每个线程首先通过 `cuda::ptx::fence_proxy_async` 对异步代理中共享内存对象的写入操作进行排序，然后所有线程的这些操作在使用 `__syncthreads()` 的线程 0 执行的异步操作之前被排序。

**TMA 写入与同步**。从共享内存到全局内存的写入再次由单个线程发起。写入的完成不由共享内存屏障跟踪。相反，使用了一种线程本地机制。多个写入可以批处理到一个所谓的*批量异步组*中。之后，线程可以等待该组中的所有操作完成从共享内存的读取（如上述代码所示）或完成向全局内存的写入，从而使发起线程能看到这些写入。更多信息，请参阅 [cp.async.bulk.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group) 的 PTX ISA 文档。请注意，批量异步和非批量异步复制指令有不同的异步组：存在 `cp.async.wait_group` 和 `cp.async.bulk.wait_group` 两种指令。
建议由线程块中的单个线程发起 TMA 操作。虽然使用 `if (threadIdx.x == 0)` 看起来足够，但编译器无法验证确实只有一个线程在发起复制，可能会插入一个在所有活动线程上执行的剥离循环，这会导致线程束序列化和性能下降。为了防止这种情况，我们定义了 `is_elected()` 辅助函数，它使用 `cuda::ptx::elect_sync` 从 warp 0 中选择一个线程——编译器知道这一点——来执行复制，从而允许编译器生成更高效的代码。或者，也可以使用 [cooperative_groups::invoke_one](cooperative-groups.html#cooperative-groups-invoke-one) 达到相同的效果。

批量异步指令对其源地址和目标地址有特定的对齐要求。更多信息见下表。

| 地址 / 大小 | 对齐要求 |
| --- | --- |
| 全局内存地址 | 必须 16 字节对齐。 |
| 共享内存地址 | 必须 16 字节对齐。 |
| 共享内存屏障地址 | 必须 8 字节对齐（这由 cuda::barrier 保证）。 |
| 传输大小 | 必须是 16 字节的倍数。 |

#### 4.11.2.1.1. 数据预取

在这个例子中，我们将演示如何使用 TMA 将数据从全局内存预取到共享内存。在迭代的复制和计算模式中，这允许用当前迭代的计算来隐藏未来迭代的数据传输延迟，潜在地增加了传输中的字节数。

 CUDA C++ 
`cuda::device::memcpy_async_tx`

| #include <cooperative_groups.h> #include <cuda/barrier> #include <cuda/ptx> namespace ptx = cuda :: ptx ; namespace cg = cooperative_groups ; __device__ inline bool is_elected () { unsigned int tid = threadIdx . x ; unsigned int warp_id = tid / 32 ; unsigned int uniform_warp_id = __shfl_sync ( 0xFFFFFFFF , warp_id , 0 ); // Broadcast from lane 0. return ( uniform_warp_id == 0 && ptx :: elect_sync ( 0xFFFFFFFF )); // Elect a leader thread among warp 0. } template < int block_size , int num_stages > __global__ void prefetch_kernel ( int * global_out , int const * global_in , size_t size , size_t batch_size ) { auto grid = cg :: this_grid (); auto block = cg :: this_thread_block (); const int tid = threadIdx . x ; assert ( size == batch_size * grid . size ()); // Assume input size fits batch_size * grid_size // 1. Initialization Phase __shared__ int shared [ num_stages * block_size ]; size_t shared_offset [ num_stages ]; for ( int s = 0 ; s < num_stages ; ++ s ) shared_offset [ s ] = s * block . size (); auto block_batch = [ & ]( size_t batch ) -> int { return block . group_index (). x * block . size () + grid . size () * batch ; }; // Initialize shared memory barrier with the number of threads participating in the barrier. // We will use explicit phase tracking for the barrier, which allows us to have only one // thread arrive on the barrier to set the transaction count and other threads wait for // a parity-based phase flip. #pragma nv_diag_suppress static_var_with_dynamic_init __shared__ cuda :: barrier < cuda :: thread_scope_block > bar [ num_stages ]; if ( tid == 0 ) { #pragma unroll num_stages for ( int i = 0 ; i < num_stages ; i ++ ) { init ( & bar [ i ], 1 ); } } __syncthreads (); // Fill the pipeline with the first ``num_stages`` batches. if ( is_elected ()) { size_t num_bytes = block_size * sizeof ( int ); #pragma unroll num_stages for ( int s = 0 ; s < num_stages ; ++ s ) { cuda :: device :: memcpy_async_tx ( & shared [ shared_offset [ s ]], & global_in [ block_batch ( s )], cuda :: aligned_size_t < 16 > ( num_bytes ), bar [ s ]); ( void ) cuda :: device :: barrier_arrive_tx ( bar [ s ], 1 , num_bytes ); } } // 2. Main Processing Loop. // compute_batch: next batch to process. // fetch_batch:   next batch to fetch from global memory. int stage = 0 ; // current stage in the shared memory buffer. uint32_t parity = 0 ; // barrierparity for ( size_t compute_batch = 0 , fetch_batch = num_stages ; compute_batch < batch_size ; ++ compute_batch , ++ fetch_batch ) { // (a) Wait on current batch. while ( ! ptx :: mbarrier_try_wait_parity ( ptx :: sem_acquire , ptx :: scope_cta , cuda :: device :: barrier_native_handle ( bar [ stage ]), parity )) {} // (b) Compute on the current batch. compute ( global_out + block_batch ( compute_batch ) + tid , shared + shared_offset [ stage ] + tid ); __syncthreads (); // (c) Load next stage ``num_stages`` ahead of current compute batch. if ( is_elected () && fetch_batch < batch_size ) { size_t num_bytes = block_size * sizeof ( int ); cuda :: device :: memcpy_async_tx ( & shared [ shared_offset [ stage ]], & global_in [ block_batch ( fetch_batch )], cuda :: aligned_size_t < 16 > ( num_bytes ), bar [ stage ]); ( void ) cuda :: device :: barrier_arrive_tx ( bar [ stage ], 1 , num_bytes ); } // (d) Stage management. stage ++ ; if ( stage == num_stages ) { stage = 0 ; parity ^= 1 ; } } } |
| --- |

此示例使用 `cuda::device::memcpy_async_tx` 实现针对 TMA 拷贝的*多级数据预取*，并采用带有显式阶段跟踪的共享内存屏障来同步拷贝操作。

1.  **初始化阶段**：设置共享内存屏障（每个阶段一个），并将第一批 `num_stages` 批次的数据预加载到不同的共享内存区域。
2.  **主处理循环**：
    *   **等待**：使用 `mbarrier_try_wait_parity()` 等待当前批次完成拷贝。
    *   **计算**：处理当前批次的数据。
    *   **预取**：为未来的数据（始终保持领先 `num_stages` 个批次）调度下一个 `memcpy_async_tx` 操作。
    *   **阶段管理**：使用循环缓冲区方法在各个阶段间循环，并跟踪屏障奇偶性。

### 4.11.2.2. 使用 TMA 传输多维数组

在本节中，我们将重点介绍多维 TMA 拷贝。一维和多维情况的主要区别在于，必须在主机上创建张量映射并将其传递给 CUDA 内核。

下表总结了批量张量异步 TMA 的可能源和目标内存空间、完成机制，以及在设备代码中公开此功能的 API。

| 方向 | 批量张量异步拷贝 (TMA, CC9.0+) |  |  |
| --- | --- | --- | --- |
| 源 | 目标 | 完成机制 | API |
| global | global |  |  |
| shared::cta | global | bulk async-group | cuda::ptx::cp_async_bulk_tensor |
| global | shared::cta | shared memory barrier | cuda::ptx::cp_async_bulk_tensor |
| global | shared::cluster | shared memory barrier | cuda::ptx::cp_async_bulk_tensor |
| shared::cta | shared::cluster | shared memory barrier | cuda::ptx::cp_async_bulk_tensor |
| shared::cta | shared::cta |  |  |

所有功能都需要内联 PTX，目前可通过 [CUDA Standard C++](https://nvidia.github.io/cccl/libcudacxx/ptx_api.html) 库中的 `cuda::ptx` 命名空间获得。

接下来，我们将描述如何使用 CUDA 驱动程序 API 创建张量映射，如何将其传递给设备，以及如何在设备上使用它。

**驱动程序 API**。张量映射是使用 [cuTensorMapEncodeTiled](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html) 驱动程序 API 创建的。可以通过直接链接驱动程序 (`-lcuda`) 或使用 [cudaGetDriverEntryPointByVersion](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER__ENTRY__POINT.html) API 来访问此 API。下面，我们展示如何获取指向 `cuTensorMapEncodeTiled` API 的指针。更多信息，请参阅 [驱动程序入口点访问](driver-entry-point-access.html#driver-entry-point-access)。

```cpp
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
  assert(driver_status == cudaDriverEntryPointSuccess);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}
```
**创建**。创建张量映射需要许多参数。其中包括指向全局内存中数组的基指针、数组的大小（以元素数量计）、从一行到下一行的步幅（以字节为单位）、共享内存缓冲区的大小（以元素数量计）。以下代码创建了一个张量映射，用于描述一个大小为 `GMEM_HEIGHT x GMEM_WIDTH` 的二维行主序数组。请注意参数的顺序：变化最快的维度排在前面。

```cpp
  CUtensorMap tensor_map{};
  // rank 是数组的维度数量。
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
  // stride 是从一行的第一个元素到下一行需要遍历的字节数。
  // 它必须是 16 的倍数。
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
  // box_size 是用作 TMA 传输目标的共享内存缓冲区的大小。
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
  // 元素之间的距离，以 sizeof(element) 为单位。例如，步幅为 2
  // 可用于仅加载复数值张量的实部。
  uint32_t elem_stride[rank] = {1, 1};

  // 获取指向 cuTensorMapEncodeTiled 驱动程序 API 的函数指针。
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // 创建张量描述符。
  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
    rank,                       // cuuint32_t tensorRank,
    tensor_ptr,                 // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,
    // 交错模式可用于加速加载长度小于 4 字节的值。
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // 混洗可用于避免共享内存体冲突。
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    // L2 提升可用于将缓存策略的效果扩展到更广泛的 L2 缓存行集合。
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // TMA 传输会将任何超出边界的元素设置为零。
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
```

**主机到设备传输**。有三种方法可以使设备代码访问张量映射。推荐的方法是将张量映射作为 `const __grid_constant__` 参数传递给内核。其他可能性包括使用 `cudaMemcpyToSymbol` 将张量映射复制到设备 `__constant__` 内存中，或通过全局内存访问它。当将张量映射作为参数传递时，某些版本的 GCC C++ 编译器会发出警告“GCC 4.6 中传递具有 64 字节对齐参数的 ABI 已更改”。可以忽略此警告。

```cpp
#include <cuda.h>

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map)
{
   // Use tensor_map here.
}
int main() {
  CUtensorMap map;
  // [ ..Initialize map.. ]
  kernel<<<1, 1>>>(map);
}
```

As an alternative to the `__grid_constant__` kernel parameter, a global `__constant__` variable can be used. An example is included below.

```cpp
#include <cuda.h>

__constant__ CUtensorMap global_tensor_map;
__global__ void kernel()
{
  // Use global_tensor_map here.
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..Initialize map.. ]
  cudaMemcpyToSymbol(global_tensor_map, &local_tensor_map, sizeof(CUtensorMap));
  kernel<<<1, 1>>>();
}
```

Finally, it is possible to copy the tensor map to global memory. Using a pointer to a tensor map in global device memory requires a fence in each thread block before any thread in the block uses the updated tensor map. Further uses of the tensor map by that thread block do not need to be fenced unless the tensor map is modified again. Note that this mechanism may be slower than the two mechanisms described above.

```cpp
#include <cuda.h>
#include <cuda/ptx>
namespace ptx = cuda::ptx;

__device__ CUtensorMap global_tensor_map;
__global__ void kernel(CUtensorMap *tensor_map)
{
  // Fence acquire tensor map:
  ptx::n32_t<128> size_bytes;
  // Since the tensor map was modified from the host using cudaMemcpy,
  // the scope should be .sys.
  ptx::fence_proxy_tensormap_generic(
     ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes
 );
 // Safe to use tensor_map after fence inside this thread.
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..Initialize map.. ]
  cudaMemcpy(&global_tensor_map, &local_tensor_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  kernel<<<1, 1>>>(global_tensor_map);
}
```

**Use**. The kernel below loads a 2D tile of size `SMEM_HEIGHT x SMEM_WIDTH` from a larger 2D array. The top-left corner of the tile is indicated by the indices `x` and `y`. The tile is loaded into shared memory, modified, and written back to global memory.

```cpp
#include <cuda.h>         // CUtensormap
#include <cuda/barrier>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

__device__ inline bool is_elected()
{
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
}

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (is_elected()) {
    // Initiate bulk tensor copy.
    int32_t tensor_coords[2] = { x, y };
    ptx::cp_async_bulk_tensor(
      ptx::space_shared, ptx::space_global,
      &smem_buffer, &tensor_map, tensor_coords,
      cuda::device::barrier_native_handle(bar));
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  // Symbolically modify a value in shared memory.
  smem_buffer[0][threadIdx.x] += threadIdx.x;

  // Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (is_elected()) {
    int32_t tensor_coords[2] = { x, y };
    ptx::cp_async_bulk_tensor(
      ptx::space_global, ptx::space_shared,
      &tensor_map, tensor_coords, &smem_buffer);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}
```
**负索引和越界情况**。当从全局内存读取到共享内存的图块部分越界时，对应越界区域的共享内存会被零填充。图块的左上角索引也可能为负。当从共享内存写入到全局内存时，图块的部分区域可能越界，但左上角不能有任何负索引。

**大小和步长**。张量的大小是指沿某一维度的元素数量。所有大小必须大于一。步长是指同一维度上相邻元素之间的字节数。例如，一个 4 x 4 的整数矩阵，其大小为 4 和 4。由于每个元素占 4 字节，其步长分别为 4 和 16 字节。由于对齐要求，一个 4 x 3 的行主序整数矩阵也必须具有 4 和 16 字节的步长。每一行都填充了额外的 4 个字节，以确保下一行的起始地址对齐到 16 字节。更多关于对齐要求的信息可以在下表中找到。

| 地址 / 大小 | 对齐要求 |
| --- | --- |
| 全局内存地址 | 必须 16 字节对齐。 |
| 全局内存大小 | 必须大于或等于一。不必是 16 字节的倍数。 |
| 全局内存步长 | 必须是 16 字节的倍数。 |
| 共享内存地址 | 必须 128 字节对齐。 |
| 共享内存屏障地址 | 必须 8 字节对齐（这由 `cuda::barrier` 保证）。 |
| 传输大小 | 必须是 16 字节的倍数。 |

#### 4.11.2.2.1. 在设备上编码张量映射

前面的章节已经描述了如何使用 CUDA 驱动程序 API 在主机上创建张量映射。

本节解释如何在设备上编码平铺类型的张量映射。这在某些情况下很有用，例如，当不希望使用典型的张量映射传输方式（使用 `const __grid_constant__` 内核参数）时，比如在单个内核启动中处理一批不同大小的张量。

推荐模式如下：

1.  在主机上使用驱动程序 API 创建一个张量映射"模板"，`template_tensor_map`。
2.  在设备内核中，复制 `template_tensor_map`，修改副本，存储在全局内存中，并进行适当的栅栏操作。
3.  在另一个内核中使用该张量映射，并进行适当的栅栏操作。

高级代码结构如下：

```cpp
// 初始化设备上下文：
CUDA_CHECK(cudaDeviceSynchronize());

// 使用 cuTensorMapEncodeTiled 驱动程序函数创建张量映射模板
CUtensorMap template_tensor_map = make_tensormap_template();

// 在全局内存中分配张量映射和张量
CUtensorMap* global_tensor_map;
CUDA_CHECK(cudaMalloc(&global_tensor_map, sizeof(CUtensorMap)));
char* global_buf;
CUDA_CHECK(cudaMalloc(&global_buf, 8 * 256));

// 用数据填充全局缓冲区。
fill_global_buf<<<1, 1>>>(global_buf);

// 定义将在设备上创建的张量映射的参数。
tensormap_params p{};
p.global_address    = global_buf;
p.rank              = 2;
p.box_dim[0]        = 128; // 共享内存中的框宽度为完整缓冲区的一半
p.box_dim[1]        = 4;   // 共享内存中的框高度为完整缓冲区的一半
p.global_dim[0]     = 256; //
p.global_dim[1]     = 8;   //
p.global_stride[0]  = 256; //
p.element_stride[0] = 1;   //
p.element_stride[1] = 1;   //

// 在设备上编码 global_tensor_map：
encode_tensor_map<<<1, 32>>>(template_tensor_map, p, global_tensor_map);

// 在另一个内核中使用它：
consume_tensor_map<<<1, 1>>>(global_tensor_map);

// 检查错误：
CUDA_CHECK(cudaDeviceSynchronize());
```
以下部分描述了高级步骤。在示例中，以下 `tensormap_params` 结构体包含要更新的字段的新值。此处包含它是为了在阅读示例时参考。

```cpp
struct tensormap_params {
  void* global_address;
  int rank;
  uint32_t box_dim[5];
  uint64_t global_dim[5];
  size_t global_stride[4];
  uint32_t element_stride[5];
};
```

#### 4.11.2.2.2. 设备端的张量映射编码与修改

在全局内存中编码张量映射的推荐过程如下。

1.  将一个现有的张量映射 `template_tensor_map` 传递给内核。与在 `cp.async.bulk.tensor` 指令中使用张量映射的内核不同，这可以通过任何方式完成：指向全局内存的指针、内核参数、`__const__` 变量等。
2.  使用 `template_tensor_map` 的值在共享内存中复制初始化一个张量映射。
3.  使用 `cuda::ptx::tensormap_replace` 函数修改共享内存中的张量映射。这些函数封装了 `tensormap.replace` PTX 指令，该指令可用于修改平铺类型张量映射的任何字段，包括基地址、大小、步长等。
4.  使用 `cuda::ptx::tensormap_copy_fenceproxy` 函数，将修改后的张量映射从共享内存复制到全局内存，并执行任何必要的栅栏操作。

以下代码包含一个遵循这些步骤的内核。为了完整起见，它修改了张量映射的所有字段。通常，内核只会修改少数几个字段。

在此内核中，`template_tensor_map` 作为内核参数传递。这是将 `template_tensor_map` 从主机移动到设备的首选方式。如果内核旨在更新设备内存中的现有张量映射，它可以接收指向要修改的现有张量映射的指针。

张量映射的格式可能会随时间变化。因此，[cuda::ptx::tensormap_replace](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_replace.html) 函数和相应的 [tensormap.replace.tile](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace) PTX 指令被标记为特定于 sm_90a。要使用它们，请使用 `nvcc -arch sm_90a ....` 进行编译。

在 sm_90a 上，共享内存中零初始化的缓冲区也可以用作初始张量映射值。这使得可以完全在设备上编码张量映射，而无需使用驱动程序 API 来编码 `template_tensor_map` 值。

设备端修改仅支持平铺类型的张量映射；其他类型的张量映射无法在设备上修改。有关张量映射类型的更多信息，请参阅[驱动程序 API 参考](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY)。

```cpp
#include <cuda/ptx>

namespace ptx = cuda::ptx;

// launch with 1 warp.
__launch_bounds__(32)
__global__ void encode_tensor_map(const __grid_constant__ CUtensorMap template_tensor_map, tensormap_params p, CUtensorMap* out) {
   __shared__ alignas(128) CUtensorMap smem_tmap;
   if (threadIdx.x == 0) {
      // Copy template to shared memory:
      smem_tmap = template_tensor_map;

      const auto space_shared = ptx::space_shared;
      ptx::tensormap_replace_global_address(space_shared, &smem_tmap, p.global_address);
      // For field .rank, the operand new_val must be ones less than the desired
      // tensor rank as this field uses zero-based numbering.
      ptx::tensormap_replace_rank(space_shared, &smem_tmap, p.rank - 1);

      // Set box dimensions:
      if (0 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.box_dim[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.box_dim[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.box_dim[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.box_dim[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.box_dim[4]); }
      // Set global dimensions:
      if (0 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, (uint32_t) p.global_dim[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, (uint32_t) p.global_dim[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, (uint32_t) p.global_dim[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, (uint32_t) p.global_dim[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, (uint32_t) p.global_dim[4]); }
      // Set global stride:
      if (1 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.global_stride[0]); }
      if (2 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.global_stride[1]); }
      if (3 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.global_stride[2]); }
      if (4 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.global_stride[3]); }
      // Set element stride:
      if (0 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.element_stride[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.element_stride[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.element_stride[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.element_stride[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.element_stride[4]); }

      // These constants are documented in this table:
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensormap-new-val-validity
      auto u8_elem_type = ptx::n32_t<0>{};
      ptx::tensormap_replace_elemtype(space_shared, &smem_tmap, u8_elem_type);
      auto no_interleave = ptx::n32_t<0>{};
      ptx::tensormap_replace_interleave_layout(space_shared, &smem_tmap, no_interleave);
      auto no_swizzle = ptx::n32_t<0>{};
      ptx::tensormap_replace_swizzle_mode(space_shared, &smem_tmap, no_swizzle);
      auto zero_fill = ptx::n32_t<0>{};
      ptx::tensormap_replace_fill_mode(space_shared, &smem_tmap, zero_fill);
   }
   // Synchronize the modifications with other threads in warp
   __syncwarp();
   // Copy the tensor map to global memory collectively with threads in the warp.
   // In addition: make the updated tensor map visible to other threads on device that
   // for use with cp.async.bulk.
   ptx::n32_t<128> bytes_128;
   ptx::tensormap_cp_fenceproxy(ptx::sem_release, ptx::scope_gpu, out, &smem_tmap, bytes_128);
}
```
#### 4.11.2.2.3. 修改后的张量映射的使用

与使用作为 `const __grid_constant__` 内核参数传递的张量映射不同，使用全局内存中的张量映射需要在修改张量映射的线程与使用它的线程之间，在张量映射代理中显式建立释放-获取模式。

该模式的释放部分已在上一节展示。它是通过使用 [cuda::ptx::tensormap.cp_fenceproxy](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap_cp_fenceproxy.html) 函数完成的。

获取部分是通过使用包装了 [fence.proxy.tensormap::generic.acquire](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence) 指令的 [cuda::ptx::fence_proxy_tensormap_generic](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/fence.html) 函数完成的。如果参与释放-获取模式的两个线程在同一设备上，使用 `.gpu` 作用域就足够了。如果线程在不同的设备上，则必须使用 `.sys` 作用域。一旦一个线程获取了张量映射，在足够的同步（例如使用 `__syncthreads()`）之后，线程块中的其他线程就可以使用它。使用张量映射的线程和执行 fence 的线程必须在同一个线程块中。也就是说，如果线程位于，例如，同一集群、同一线程网格或不同内核的两个不同线程块中，像 `cooperative_groups::cluster` 或 `grid_group::sync()` 或流顺序同步这样的同步 API 不足以建立张量映射更新的顺序，也就是说，这些其他线程块中的线程在使用更新后的张量映射之前，仍然需要在正确的作用域获取张量映射代理。如果没有中间修改，则不必在每次 `cp.async.bulk.tensor` 指令之前重复执行 fence。

以下示例展示了 `fence` 以及随后对张量映射的使用。

```cpp
// 全局内存中张量映射的消费者：
__global__ void consume_tensor_map(CUtensorMap* tensor_map) {
  // Fence 获取张量映射：
  ptx::n32_t<128> size_bytes;
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
  // Fence 之后可以安全使用 tensor_map。

  __shared__ uint64_t bar;
  __shared__ alignas(128) char smem_buf[4][128];

  if (threadIdx.x == 0) {
    // 初始化屏障
    ptx::mbarrier_init(&bar, 1);
    // 发出 TMA 请求
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem_buf, tensor_map, {0, 0}, &bar);
    // 到达屏障。期望 4 * 128 字节。
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buf));
  }
  const int parity = 0;
  // 等待加载完成
  while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}

  // 打印项目：
  printf("Got:\n\n");
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 128; ++i) {
      printf("%3d ", smem_buf[j][i]);
      if (i % 32 == 31) { printf("\n"); };
    }
    printf("\n");
  }
}
```
#### 4.11.2.2.4. 使用驱动 API 创建模板张量映射值

以下代码创建了一个最小化的平铺类型张量映射，该映射随后可以在设备上进行修改。

```cpp
CUtensorMap make_tensormap_template() {
  CUtensorMap template_tensor_map{};
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  uint32_t dims_32         = 16;
  uint64_t dims_strides_64 = 16;
  uint32_t elem_strides    = 1;

  // 创建张量描述符。
  CUresult res = cuTensorMapEncodeTiled(
    &template_tensor_map, // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    1,                // cuuint32_t tensorRank,
    nullptr,          // void *globalAddress,
    &dims_strides_64, // const cuuint64_t *globalDim,
    &dims_strides_64, // const cuuint64_t *globalStrides,
    &dims_32,         // const cuuint32_t *boxDim,
    &elem_strides,    // const cuuint32_t *elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CU_CHECK(res);
  return template_tensor_map;
}
```

#### 4.11.2.2.5. 共享内存存储体交错

默认情况下，TMA 引擎将数据加载到共享内存的顺序与其在全局内存中的布局顺序相同。然而，这种布局对于某些共享内存访问模式可能不是最优的，因为它可能导致共享内存存储体冲突。为了提高性能并减少存储体冲突，我们可以通过应用“交错模式”来改变共享内存的布局。

共享内存有 32 个存储体，其组织方式是连续的 32 位字映射到连续的存储体。每个存储体每个时钟周期的带宽为 32 位。在加载和存储共享内存时，如果在一个事务中多次使用同一个存储体，就会产生存储体冲突，导致带宽降低。请参阅[共享内存访问模式](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory-access-patterns)。

为了确保数据在共享内存中的布局方式能够使用户代码避免共享内存存储体冲突，可以指示 TMA 引擎在将数据存储到共享内存之前对其进行“交错”，并在将数据从共享内存复制回全局内存时进行“解交错”。张量映射编码了“交错模式”，指示使用哪种交错模式。

示例：矩阵转置

一个例子是矩阵转置，其中数据从按行优先访问映射到按列优先访问。数据在全局内存中按行主序存储，但我们希望在共享内存中也按列访问它，这会导致存储体冲突。然而，通过使用 128 字节的“交错”模式和新的共享内存索引，这些冲突可以被消除。

在此示例中，我们加载一个 8x8 的 `int4` 类型矩阵，该矩阵在全局内存中按行主序存储到共享内存。然后，每组八个线程从共享内存缓冲区加载一行，并将其存储到单独的转置共享内存缓冲区中的一列。这导致存储时产生八路存储体冲突。最后，将转置缓冲区写回全局内存。
为避免存储体冲突，可以使用 `CU_TENSOR_MAP_SWIZZLE_128B` 布局。此布局匹配 128 字节的行长度，并以一种方式改变共享内存布局，使得按列和按行访问都不需要在每次事务中访问相同的存储体。

下面的两个表格，[图 48](#figure-swizzle-example1) 和 [图 49](#figure-swizzle-example2)，展示了 `int4` 类型的 8x8 矩阵及其转置矩阵的正常和经过置换的共享内存布局。颜色表示矩阵元素映射到八个四存储体组中的哪一个，边距行和边距列列出了全局内存的行和列索引。条目显示了 16 字节矩阵元素的共享内存索引。

图 48：在没有置换的共享内存数据布局中，共享内存索引等同于全局内存索引。
每次加载指令读取一行并存储在转置缓冲区的一列中。由于转置中该列的所有矩阵元素都落在同一个存储体中，存储操作必须串行化，导致八次存储事务，从而在存储每列时产生八路存储体冲突。[#](#id2)

图 49：使用 CU_TENSOR_MAP_SWIZZLE_128B 置换的共享内存数据布局。一行存储在一列中，对于行和列，每个矩阵元素都来自不同的存储体，因此没有任何存储体冲突。[#](#id3)

```cpp
__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
   // 批量张量操作的目标共享内存缓冲区
   // 使用 128 字节置换模式，应对齐到 1024 字节。
   __shared__ alignas(1024) int4 smem_buffer[8][8];
   __shared__ alignas(1024) int4 smem_buffer_tr[8][8];

   // 初始化共享内存屏障
   #pragma nv_diag_suppress static_var_with_dynamic_init
   __shared__ barrier bar;

   if (threadIdx.x == 0) {
     init(&bar, blockDim.x);
   }
   __syncthreads();

   barrier::arrival_token token;
   if (is_elected()) {
     // 发起从全局内存到共享内存的批量张量复制，
     // 方式与不使用置换时相同。
     int32_t tensor_coords[2] = { 0, 0 };
     ptx::cp_async_bulk_tensor(
       ptx::space_shared, ptx::space_global,
       &smem_buffer, &tensor_map, tensor_coords,
       cuda::device::barrier_native_handle(bar));
     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
   } else {
     token = bar.arrive();
   }

   bar.wait(std::move(token));

   /* 矩阵转置
    *  当使用正常的共享内存布局时，存储到转置矩阵时会发生八次
    *  八路共享内存存储体冲突。
    *  当启用 128 字节置换模式并使用相应的访问模式时，
    *  加载和存储的冲突都被消除了。 */
   for(int sidx_j =threadIdx.x; sidx_j < 8; sidx_j+= blockDim.x){
      for(int sidx_i = 0; sidx_i < 8; ++sidx_i){
         const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
         const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
         smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
      }
   }

   // 等待共享内存写入对 TMA 引擎可见。
   ptx::fence_proxy_async(ptx::space_shared);
   __syncthreads();

   /* 发起 TMA 传输，将转置后的共享内存缓冲区复制回全局内存，
    *  它将"去置换"数据。 */
   if (is_elected()) {
       int32_t tensor_coords[2] = { x, y };
       ptx::cp_async_bulk_tensor(
         ptx::space_global, ptx::space_shared,
         &tensor_map, tensor_coords, &smem_buffer_tr);
      ptx::cp_async_bulk_commit_group();
      ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
   }

   // 销毁屏障
   if (threadIdx.x == 0) {
     (&bar)->~barrier();
   }
}

// --------------------------------- main ----------------------------------------

int main(){

...
   void* tensor_ptr = d_data;

   CUtensorMap tensor_map{};
   // rank 是数组的维度数。
   constexpr uint32_t rank = 2;
   // 全局内存大小
   uint64_t size[rank] = {4*8, 8};
   // 全局内存步幅，必须是 16 的倍数。
   uint64_t stride[rank - 1] = {8 * sizeof(int4)};
   // 内部共享内存框维度（字节），等于置换跨度。
   uint32_t box_size[rank] = {4*8, 8};

   uint32_t elem_stride[rank] = {1, 1};

   // 创建张量描述符。
   CUresult res = cuTensorMapEncodeTiled(
       &tensor_map,                // CUtensorMap *tensorMap,
       CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
       rank,                       // cuuint32_t tensorRank,
       tensor_ptr,                 // void *globalAddress,
       size,                       // const cuuint64_t *globalDim,
       stride,                     // const cuuint64_t *globalStrides,
       box_size,                   // const cuuint32_t *boxDim,
       elem_stride,                // const cuuint32_t *elementStrides,
       CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
       // 使用 128 字节的置换模式。
       CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
       CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
       CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
   );

   kernel_tma<<<1, 8>>>(tensor_map);
 ...
}
```
**备注**。此示例旨在展示 swizzle 的使用，其“原样”形式性能不佳，且无法扩展到给定维度之外。

**解释**。在数据传输期间，TMA 引擎会根据 swizzle 模式对数据进行重排，如下表所述。这些 swizzle 模式定义了沿 swizzle 宽度的 16 字节块到四个存储体子组的映射。其类型为 `CUtensorMapSwizzle`，有四个选项：无、32 字节、64 字节和 128 字节。请注意，共享内存框的内部维度必须小于或等于 swizzle 模式的跨度。

Swizzle 模式

如前所述，有四种 swizzle 模式。下表展示了不同的 swizzle 模式，包括新的共享内存索引的关系。这些表定义了沿 128 字节的 16 字节块到八个由四个存储体组成的子组的映射。

图 50 TMA Swizzle 模式概述[#](#id4)

**注意事项**。应用 TMA swizzle 模式时，必须遵守特定的内存要求：

- 全局内存对齐：全局内存必须对齐到 128 字节。
- 共享内存对齐：为简化起见，共享内存应根据 swizzle 模式重复的字节数进行对齐。当共享内存缓冲区未按 swizzle 模式自身重复的字节数对齐时，swizzle 模式与共享内存之间会存在偏移。请参阅下面的注释。
- 内部维度：共享内存块的内部维度必须满足表 25 中规定的大小要求。如果未满足这些要求，则该指令被视为无效。此外，如果 swizzle 宽度超过内部维度，请确保分配的共享内存能够容纳完整的 swizzle 宽度。
- 粒度：swizzle 映射的粒度固定为 16 字节。这意味着数据以 16 字节块为单位组织和访问，在规划内存布局和访问模式时必须考虑这一点。

**Swizzle 模式指针偏移计算**。这里，我们描述了当共享内存缓冲区未按 swizzle 模式自身重复的字节数对齐时，如何确定 swizzle 模式与共享内存之间的偏移。使用 TMA 时，要求共享内存对齐到 128 字节。要找出共享内存缓冲区相对于 swizzle 模式偏移了多少次，请应用相应的偏移公式。

| Swizzle 模式 | 偏移公式 | 索引关系 |
| --- | --- | --- |
| CU_TENSOR_MAP_SWIZZLE_128B | (reinterpret_cast <uintptr_t>(smem_ptr)/128)%8 | smem[y][x] <-> smem[y][((y+offset)%8)^x] |
| CU_TENSOR_MAP_SWIZZLE_64B | (reinterpret_cast <uintptr_t>(smem_ptr)/128)%4 | smem[y][x] <-> smem[y][((y+offset)%4)^x] |
| CU_TENSOR_MAP_SWIZZLE_32B | (reinterpret_cast <uintptr_t>(smem_ptr)/128)%2 | smem[y][x] <-> smem[y][((y+offset)%2)^x] |

在[图 50](#figure-swizzle-overview)中，此偏移表示初始行偏移，因此，在 swizzle 索引计算中，它被加到行索引 `y` 上。以下代码片段展示了如何在 `CU_TENSOR_MAP_SWIZZLE_128B` 模式下访问经过 swizzle 处理的共享内存。

```cpp
data_t* smem_ptr = &smem[0][0];
int offset = (reinterpret_cast<uintptr_t>(smem_ptr)/128)%8;
smem[y][((y+offset)%8)^x] = ...
```

**Summary.** The following [Table 25](#table-swizzle-pattern-properties-and-requirements) summarizes the requirements and properties of the different swizzle patterns for Compute Capability 9.

| Pattern | Swizzle width | Shared boxâs
inner dimension | Repeats
after | Shared
memory
alignment | Global
memory
alignment |
| --- | --- | --- | --- | --- | --- |
| CU_TENSOR_MAP_SWIZZLE_128B | 128 bytes | <=128 bytes | 1024 bytes | 128 bytes | 128 bytes |
| CU_TENSOR_MAP_SWIZZLE_64B | 64 bytes | <=64 bytes | 512 bytes | 128 bytes | 128 bytes |
| CU_TENSOR_MAP_SWIZZLE_32B | 32 bytes | <=32 bytes | 256 bytes | 128 bytes | 128 bytes |
| CU_TENSOR_MAP_SWIZZLE_NONE
(default) |  |  |  | 128 bytes | 16 bytes |

## 4.11.3.Using STAS

CUDA applications using [thread block clusters](../02-basics/intro-to-cuda-cpp.html#thread-block-clusters) may need to move small data elements between thread blocks within the cluster. STAS instructions (CC 9.0+, see [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async)) enable asynchronous data copies directly from registers to distributed shared memory. STAS is only exposed through a lower-level `cuda::ptx::st_async` API available in the [libcu++](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/st_async.html?highlight=st_async#) library.

**Dimensions**. STAS supports copying 4, 8 or 16 bytes.

**Source and destination**. The only direction supported for asynchronous copy operations with STAS is from registers to distributed shared memory. The destination pointer needs to be aligned to 4, 8, or 16 bytes depending on the size of the data being copied.

**Asynchronicity**. Data transfers using STAS are [asynchronous](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features) and are modeled as async thread operations (see [Async Thread and Async Proxy](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)). This allows the initiating thread to continue computing while the hardware asynchronously copies the data. *Whether the data transfer occurs asynchronously in practice is up to the hardware implementation and may change in the future*. The completion mechanisms that STAS operations can use to signal that they have completed are [shared memory barriers](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives-barriers).

In the following example, we show how to use STAS to implement a producer-consumer pattern within a thread-block cluster. This kernel creates a circular communication pipeline where 8 thread blocks are arranged in a ring, and each block simultaneously:

- Produces data for the next block in the sequence.
- Consumes data from the previous block in the sequence.
为实现此模式，每个线程块需要两个共享内存屏障：一个用于通知消费者块数据已复制到共享内存缓冲区（`filled`），另一个用于通知生产者块消费者上的缓冲区已准备好接收数据（`ready`）。

 CUDA C++ 
`cuda::ptx`

| #include <cooperative_groups.h> #include <cuda/barrier> #include <cuda/ptx> __global__ __cluster_dims__ ( 8 , 1 , 1 ) void producer_consumer_kernel () { using namespace cooperative_groups ; using namespace cuda :: device ; using namespace cuda :: ptx ; using barrier_t = cuda :: barrier < cuda :: thread_scope_block > ; auto cluster = this_cluster (); #pragma nv_diag_suppress static_var_with_dynamic_init __shared__ int buffer [ BLOCK_SIZE ]; __shared__ barrier_t filled ; __shared__ barrier_t ready ; // Initialize shared memory barriers. if ( threadIdx . x == 0 ) { init ( & filled , 1 ); init ( & ready , BLOCK_SIZE ); } // Sync cluster to ensure remote barriers are initialized. cluster . sync (); // Define my own and my neighbor's ranks. int rk = cluster . block_rank (); int rk_next = ( rk + 1 ) % 8 ; int rk_prev = ( rk + 7 ) % 8 ; // Get addresses of remote buffer we are writing to and remote barriers of previous and next blocks. auto buffer_next = cluster . map_shared_rank ( buffer , rk_next ); auto bar_next = cluster . map_shared_rank ( barrier_native_handle ( filled ), rk_next ); auto bar_prev = cluster . map_shared_rank ( barrier_native_handle ( ready ), rk_prev ); int phase = 0 ; for ( int it = 0 ; it < 1000 ; ++ it ) { // As producers, send data to our right neighbor. st_async ( & buffer_next [ threadIdx . x ], rk , bar_next ); if ( threadIdx . x == 0 ) { // Thread 0 arrives on local barrier and indicates it expects to receive a certain number of bytes. mbarrier_arrive_expect_tx ( sem_release , scope_cluster , space_shared , barrier_native_handle ( filled ), sizeof ( buffer )); } // As consumers, wait on local barrier for data from left neighbor to arrive. while ( ! mbarrier_try_wait_parity ( barrier_native_handle ( filled ), phase , 1000 )) {} // At this point, the data has been copied to our local buffer. int r = buffer [ threadIdx . x ]; // Use the data to do something. // As consumers, notify our left neighbor that we are done with the data. mbarrier_arrive ( sem_release , scope_cluster , space_cluster , bar_prev ); // As producers, wait on local barrier until the right neighbor is ready to receive new data. while ( ! mbarrier_try_wait_parity ( barrier_native_handle ( ready ), phase , 1000 )) {} phase ^= 1 ; } } |
| --- |

- 共享内存屏障由每个块的第一个线程初始化。屏障 `filled` 初始化为 1，屏障 `ready` 初始化为块中的线程数。
- 执行集群范围的同步，以确保在任何线程开始通信之前所有屏障都已初始化。
- 每个线程确定其邻居的排名，并使用它们来映射远程共享内存屏障和远程共享内存缓冲区以写入数据。
- 在每次迭代中：作为生产者，每个线程向其右侧邻居发送数据。作为消费者，线程 0 到达本地已填充屏障，并表明它期望接收一定数量的字节。作为消费者，每个线程在本地已填充屏障上等待来自左侧邻居的数据到达。作为消费者，每个线程使用数据执行某些操作。作为消费者，每个线程通知左侧邻居它已完成对数据的处理。作为生产者，每个线程在本地就绪屏障上等待，直到右侧邻居准备好接收新数据。

请注意，对于每个屏障，我们需要使用正确的空间。对于映射的远程屏障，我们需要使用 `space_cluster` 空间，而对于本地屏障，我们需要使用 `space_shared` 空间。

 在本页