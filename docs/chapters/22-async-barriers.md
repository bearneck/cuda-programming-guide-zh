# 4.9 异步屏障

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)

---

此页面有帮助吗？

# 4.9. 异步屏障

异步屏障在[高级同步原语](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives)中引入，将 CUDA 同步扩展到 `__syncthreads()` 和 `__syncwarp()` 之外，实现了细粒度、非阻塞的协调，以及通信与计算之间更好的重叠。

本节主要介绍如何通过 `cuda::barrier` API（并在适用时指向 `cuda::ptx` 和原语）来使用异步屏障。

## 4.9.1. 初始化

初始化必须在任何线程开始参与屏障之前完成。

 CUDA C++
`cuda::barrier`

| #include <cuda/barrier> #include <cooperative_groups.h> __global__ void init_barrier () { __shared__ cuda :: barrier < cuda :: thread_scope_block > bar ; auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () == 0 ) { // A single thread initializes the total expected arrival count. init ( & bar , block . size ()); } block . sync (); } |
| --- |

 CUDA C++
`cuda::ptx`

| #include <cuda/ptx> #include <cooperative_groups.h> __global__ void init_barrier () { __shared__ uint64_t bar ; auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () == 0 ) { // A single thread initializes the total expected arrival count. cuda :: ptx :: mbarrier_init ( & bar , block . size ()); } block . sync (); } |
| --- |

 CUDA C 原语

| #include <cuda_awbarrier_primitives.h> #include <cooperative_groups.h> __global__ void init_barrier () { __shared__ uint64_t bar ; auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () == 0 ) { // A single thread initializes the total expected arrival count. __mbarrier_init ( & bar , block . size ()); } block . sync (); } |
| --- |

在任何线程可以参与屏障之前，必须使用 `cuda::barrier::init()` 友元函数初始化屏障。这必须在任何线程到达屏障之前发生。这带来了一个引导挑战：线程在参与屏障之前必须同步，但线程创建屏障的目的正是为了同步。在此示例中，将要参与的线程是协作组的一部分，并使用 `block.sync()` 来引导初始化。由于整个线程块都参与屏障，也可以使用 `__syncthreads()`。

`init()` 的第二个参数是*预期到达计数*，即在参与线程从其 `bar.wait(std::move(token))` 调用中被解除阻塞之前，参与线程将调用 `bar.arrive()` 的次数。在此示例和之前的示例中，屏障被初始化为线程块中的线程数，即 `cooperative_groups::this_thread_block().size()`，以便线程块内的所有线程都可以参与屏障。

异步屏障在指定线程*如何*参与（分离到达/等待）以及*哪些*线程参与方面非常灵活。相比之下，`this_thread_block.sync()` 或 `__syncthreads()` 适用于整个线程块，而 `__syncwarp(mask)` 适用于线程束的指定子集。尽管如此，如果用户的意图是同步整个线程块或整个线程束，我们建议分别使用 `__syncthreads()` 和 `__syncwarp()` 以获得更好的性能。
## 4.9.2.A 屏障的阶段：到达、倒计时、完成与重置

异步屏障从预期的到达次数倒数至零，参与线程通过调用 `bar.arrive()` 来实现。当倒计时达到零时，屏障在当前阶段完成。当最后一次调用 `bar.arrive()` 导致倒计时达到零时，倒计时会自动且原子性地重置。重置会将倒计时值设为预期的到达次数，并将屏障推进到下一个阶段。

`token` 对象属于 `cuda::barrier::arrival_token` 类，由 `token=bar.arrive()` 返回，它与屏障的当前阶段相关联。调用 `bar.wait(std::move(token))` 会阻塞调用线程，只要屏障仍处于当前阶段，即与该令牌关联的阶段与屏障的阶段匹配。如果在调用 `bar.wait(std::move(token))` 之前阶段已经推进（因为倒计时已达到零），那么线程不会阻塞；如果在线程被 `bar.wait(std::move(token))` 阻塞期间阶段推进了，线程将被解除阻塞。

**了解重置何时可能或不可能发生至关重要，尤其是在非平凡的到达/等待同步模式中。**

*   线程对 `token=bar.arrive()` 和 `bar.wait(std::move(token))` 的调用必须按顺序进行，使得 `token=bar.arrive()` 发生在屏障的当前阶段，而 `bar.wait(std::move(token))` 发生在同一阶段或下一阶段。
*   线程对 `bar.arrive()` 的调用必须发生在屏障计数器非零时。在屏障初始化之后，如果线程对 `bar.arrive()` 的调用导致倒计时达到零，那么必须在屏障可以被重用进行后续的 `bar.arrive()` 调用之前，发生一次对 `bar.wait(std::move(token))` 的调用。
*   `bar.wait()` 只能使用当前阶段或紧邻前一阶段的令牌对象来调用。对于令牌对象的任何其他值，其行为是未定义的。

对于简单的到达/等待同步模式，遵守这些使用规则是直截了当的。

### 4.9.2.1. 线程束纠缠

线程束发散会影响到达操作更新屏障的次数。如果调用线程束完全收敛，则屏障更新一次。如果调用线程束完全发散，则会对屏障应用 32 次单独的更新。

建议由收敛的线程使用 `arrive-on(bar)` 调用来最小化对屏障对象的更新。当这些操作之前的代码导致线程发散时，应在调用到达操作之前，通过 `__syncwarp` 使线程束重新收敛。

## 4.9.3. 显式阶段跟踪

异步屏障可以有多个阶段，具体取决于它被用于同步线程和内存操作的次数。我们可以不使用令牌来跟踪屏障阶段的翻转，而是直接使用通过 `cuda::ptx` 和原语 API 提供的 `mbarrier_try_wait_parity()` 系列函数来跟踪阶段。
在最简单的形式下，`cuda::ptx::mbarrier_try_wait_parity(uint64_t* bar, const uint32_t& phaseParity)` 函数等待一个具有特定奇偶性的阶段。`phaseParity` 操作数是屏障对象当前阶段或紧邻前一阶段的整数奇偶性。偶数阶段的整数奇偶性为 0，奇数阶段的整数奇偶性为 1。当我们初始化一个屏障时，其阶段的奇偶性为 0。因此，`phaseParity` 的有效值是 0 和 1。显式阶段跟踪在跟踪[异步内存操作](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies)时可能很有用，因为它允许只有一个线程到达屏障并设置事务计数，而其他线程只等待基于奇偶性的阶段翻转。这可能比让所有线程都到达屏障并使用令牌更高效。此功能仅适用于线程块和集群范围内的共享内存屏障。

 CUDA C++ 
`cuda::barrier`

| #include <cuda/ptx> #include <cooperative_groups.h> __device__ void compute ( float * data , int iteration ); __global__ void split_arrive_wait ( int iteration_count , float * data ) { using barrier_t = cuda :: barrier < cuda :: thread_scope_block > ; __shared__ barrier_t bar ; int parity = 0 ; // 初始阶段奇偶性为 0。 auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () == 0 ) { // 使用期望到达计数初始化屏障。 init ( & bar , block . size ()); } block . sync (); for ( int i = 0 ; i < iteration_count ; ++ i ) { /* 到达前的代码 */ // 此线程到达。到达不会阻塞线程。 // 获取原生屏障的句柄以与 cuda::ptx API 一起使用。 ( void ) cuda :: ptx :: mbarrier_arrive ( cuda :: device :: barrier_native_handle ( bar )); compute ( data , i ); // 等待参与屏障的所有线程完成 mbarrier_arrive()。 // 获取原生屏障的句柄以与 cuda::ptx API 一起使用。 while ( ! cuda :: ptx :: mbarrier_try_wait_parity ( cuda :: device :: barrier_native_handle ( bar ), parity )) {} // 翻转奇偶性。 parity ^= 1 ; /* 等待后的代码 */ } } |
| --- |

 CUDA C++ 
`cuda::ptx`

| #include <cuda/ptx> #include <cooperative_groups.h> __device__ void compute ( float * data , int iteration ); __global__ void split_arrive_wait ( int iteration_count , float * data ) { __shared__ uint64_t bar ; int parity = 0 ; // 初始阶段奇偶性为 0。 auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () == 0 ) { // 使用期望到达计数初始化屏障。 cuda :: ptx :: mbarrier_init ( & bar , block . size ()); } block . sync (); for ( int i = 0 ; i < iteration_count ; ++ i ) { /* 到达前的代码 */ // 此线程到达。到达不会阻塞线程。 ( void ) cuda :: ptx :: mbarrier_arrive ( & bar ); compute ( data , i ); // 等待参与屏障的所有线程完成 mbarrier_arrive()。 while ( ! cuda :: ptx :: mbarrier_try_wait_parity ( & bar , parity )) {} // 翻转奇偶性。 parity ^= 1 ; /* 等待后的代码 */ } } |
| --- |

 CUDA C 原语

| #include <cuda_awbarrier_primitives.h> #include <cooperative_groups.h> __device__ void compute ( float * data , int iteration ); __global__ void split_arrive_wait ( int iteration_count , float * data ) { __shared__ __mbarrier_t bar ; bool parity = false ; // 初始阶段的奇偶性为 false。 auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () == 0 ) { // 使用预期的到达计数初始化屏障。 __mbarrier_init ( & bar , block . size ()); } block . sync (); for ( int i = 0 ; i < iteration_count ; ++ i ) { /* 到达前的代码 */ // 此线程到达。到达不会阻塞线程。 ( void ) __mbarrier_arrive ( & bar ); compute ( data , i ); // 等待参与屏障的所有线程完成 __mbarrier_arrive()。 while ( ! __mbarrier_try_wait_parity ( & bar , parity , 1000 )) {} parity ^= 1 ; /* 等待后的代码 */ } } |
| --- |

## 4.9.4. 提前退出

当一个参与一系列同步操作的线程必须提前退出该序列时，该线程必须在退出前明确退出参与。剩余的参与线程可以正常进行后续的到达和等待操作。

 CUDA C++ 
`cuda::barrier`

| #include <cuda/barrier> #include <cooperative_groups.h> __device__ bool condition_check (); __global__ void early_exit_kernel ( int N ) { __shared__ cuda :: barrier < cuda :: thread_scope_block > bar ; auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () == 0 ) { init ( & bar , block . size ()); } block . sync (); for ( int i = 0 ; i < N ; ++ i ) { if ( condition_check ()) { bar . arrive_and_drop (); return ; } // 其他线程可以正常进行。 auto token = bar . arrive (); /* 到达和等待之间的代码 */ // 等待所有线程到达。 bar . wait ( std :: move ( token )); /* 等待后的代码 */ } } |
| --- |

 CUDA C 原语

| #include <cuda_awbarrier_primitives.h> #include <cooperative_groups.h> __device__ bool condition_check (); __global__ void early_exit_kernel ( int N ) { __shared__ __mbarrier_t bar ; auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () == 0 ) { __mbarrier_init ( & bar , block . size ()); } block . sync (); for ( int i = 0 ; i < N ; ++ i ) { if ( condition_check ()) { __mbarrier_token_t token = __mbarrier_arrive_and_drop ( & bar ); return ; } // 其他线程可以正常进行。 __mbarrier_token_t token = __mbarrier_arrive ( & bar ); /* 到达和等待之间的代码 */ // 等待所有线程到达。 while ( ! __mbarrier_try_wait ( & bar , token , 1000 )) {} /* 等待后的代码 */ } } |
| --- |

`bar.arrive_and_drop()` 操作在屏障上到达，以履行参与线程在**当前**阶段到达的义务，然后递减**下一**阶段的预期到达计数，使得该线程不再被期望到达屏障。

## 4.9.5. 完成函数

`cuda::barrier` API 支持一个可选的完成函数。`cuda::barrier<Scope, CompletionFunction>` 的 `CompletionFunction` 在每个阶段执行一次，在最后一个线程*到达*之后，在任何线程从 `wait` 解除阻塞之前执行。在该阶段到达 `barrier` 的线程执行的内存操作对于执行 `CompletionFunction` 的线程是可见的，并且在 `CompletionFunction` 内执行的所有内存操作对于所有在 `barrier` 处等待的线程，一旦它们从 `wait` 解除阻塞，都是可见的。
CUDA C++
`cuda::barrier`

| #include <cuda/barrier> #include <cooperative_groups.h> #include <functional> namespace cg = cooperative_groups ; __device__ int divergent_compute ( int * , int ); __device__ int independent_computation ( int * , int ); __global__ void psum ( int * data , int n , int * acc ) { auto block = cg :: this_thread_block (); constexpr int BlockSize = 128 ; __shared__ int smem [ BlockSize ]; assert ( BlockSize == block . size ()); assert ( n % BlockSize == 0 ); auto completion_fn = [ & ] { int sum = 0 ; for ( int i = 0 ; i < BlockSize ; ++ i ) { sum += smem [ i ]; } * acc += sum ; }; /* 屏障存储。注意：由于捕获，completion_fn 不是默认可构造的，因此屏障也不是默认可构造的。 */ using completion_fn_t = decltype ( completion_fn ); using barrier_t = cuda :: barrier < cuda :: thread_scope_block , completion_fn_t > ; __shared__ std :: aligned_storage < sizeof ( barrier_t ), alignof ( barrier_t ) > bar_storage ; // 初始化屏障。 barrier_t * bar = ( barrier_t * ) & bar_storage ; if ( block . thread_rank () == 0 ) { assert ( * acc == 0 ); assert ( blockDim . x == blockDim . y == blockDim . y == 1 ); new ( bar ) barrier_t { block . size (), completion_fn }; /* 等价于：init(bar, block.size(), completion_fn); */ } block . sync (); // 主循环。 for ( int i = 0 ; i < n ; i += block . size ()) { smem [ block . thread_rank ()] = data [ i ] + * acc ; auto token = bar -> arrive (); // 我们可以在这里进行独立计算。 bar -> wait ( std :: move ( token )); // 共享内存在下一次迭代中可以安全地重用， // 因为所有线程（包括执行归约的那个线程）都已使用完毕。 } } |
| --- |

## 4.9.6. 跟踪异步内存操作

异步屏障可用于跟踪[异步内存拷贝](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies)。当异步拷贝操作绑定到屏障时，该拷贝操作在启动时会自动递增当前屏障阶段的预期计数，并在完成时递减该计数。此机制确保屏障的 `wait()` 操作将阻塞，直到所有关联的异步内存拷贝都已完成，从而为同步多个并发内存操作提供了一种便捷的方法。

从计算能力 9.0 开始，具有线程块或集群作用域的共享内存中的异步屏障可以**显式地**跟踪异步内存操作。我们将这些屏障称为*异步事务屏障*。除了预期的到达计数外，屏障对象还可以接受一个**事务计数**，该计数可用于跟踪异步事务的完成情况。事务计数跟踪尚未完成的异步事务数量，其单位由异步内存操作指定（通常为字节）。当前阶段要跟踪的事务计数可以在到达时通过 `cuda::device::barrier_arrive_tx()` 设置，或直接通过 `cuda::device::barrier_expect_tx()` 设置。当屏障使用事务计数时，它会在等待操作处阻塞线程，直到所有生产者线程都执行了到达操作*并且*所有事务计数的总和达到预期值。
CUDA C++
`cuda::barrier`

| #include <cuda/barrier> #include <cooperative_groups.h> __global__ void track_kernel () { __shared__ cuda :: barrier < cuda :: thread_scope_block > bar ; auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () == 0 ) { init ( & bar , block . size ()); } block . sync (); auto token = cuda :: device :: barrier_arrive_tx ( bar , 1 , 0 ); bar . wait ( cuda :: std :: move ( token )); } |
| --- |

CUDA C++
`cuda::ptx`

| #include <cuda/ptx> #include <cooperative_groups.h> __global__ void track_kernel () { __shared__ uint64_t bar ; auto block = cooperative_groups :: this_thread_block (); if ( block . thread_rank () == 0 ) { cuda :: ptx :: mbarrier_init ( & bar , block . size ()); } block . sync (); uint64_t token = cuda :: ptx :: mbarrier_arrive_expect_tx ( cuda :: ptx :: sem_release , cuda :: ptx :: scope_cluster , cuda :: ptx :: space_shared , & bar , 1 , 0 ); while ( ! cuda :: ptx :: mbarrier_try_wait ( & bar , token )) {} } |
| --- |

在此示例中，`cuda::device::barrier_arrive_tx()` 操作构造了一个与当前阶段的阶段同步点相关联的到达令牌对象。然后，将到达计数减 1，并将预期事务计数增加 0。由于事务计数更新为 0，因此屏障不跟踪任何事务。后续关于[使用张量内存加速器 (TMA)](async-copies.html#async-copies-tma) 的部分包含了跟踪异步内存操作的示例。

## 4.9.7. 使用屏障的生产者-消费者模式

线程块可以在空间上进行分区，以允许不同的线程执行独立的操作。这通常是通过将线程块内不同线程束（warp）的线程分配给特定任务来实现的。这种技术被称为*线程束专业化*。

本节展示了一个生产者-消费者模式中空间分区的示例，其中一个线程子集生产数据，同时由另一个（不相交的）线程子集消费数据。生产者-消费者空间分区模式需要两次单向同步来管理生产者和消费者之间的数据缓冲区。

| 生产者 | 消费者 |
| --- | --- |
| 等待缓冲区准备好被填充 | 发出缓冲区准备好被填充的信号 |
| 生产数据并填充缓冲区 |  |
| 发出缓冲区已填充的信号 | 等待缓冲区被填充 |
|  | 消费已填充缓冲区中的数据 |

生产者线程等待消费者线程发出缓冲区准备好被填充的信号；然而，消费者线程不等待此信号。消费者线程等待生产者线程发出缓冲区已填充的信号；然而，生产者线程不等待此信号。为了实现完全的生产者/消费者并发，此模式需要（至少）双缓冲，其中每个缓冲区需要两个屏障。

CUDA C++
`cuda::barrier`

| #include <cuda/barrier> using barrier_t = cuda :: barrier < cuda :: thread_scope_block > ; __device__ void produce ( barrier_t ready [], barrier_t filled [], float * buffer , int buffer_len , float * in , int N ) { for ( int i = 0 ; i < N / buffer_len ; ++ i ) { ready [ i % 2 ]. arrive_and_wait (); /* 等待 buffer_(i%2) 准备好被填充 */ /* 生产，即填充 buffer_(i%2)  */ barrier_t :: arrival_token token = filled [ i % 2 ]. arrive (); /* buffer_(i%2) 已填充 */ } } __device__ void consume ( barrier_t ready [], barrier_t filled [], float * buffer , int buffer_len , float * out , int N ) { barrier_t :: arrival_token token1 = ready [ 0 ]. arrive (); /* buffer_0 准备好进行初始填充 */ barrier_t :: arrival_token token2 = ready [ 1 ]. arrive (); /* buffer_1 准备好进行初始填充 */ for ( int i = 0 ; i < N / buffer_len ; ++ i ) { filled [ i % 2 ]. arrive_and_wait (); /* 等待 buffer_(i%2) 被填充 */ /* 消费 buffer_(i%2) */ barrier_t :: arrival_token token3 = ready [ i % 2 ]. arrive (); /* buffer_(i%2) 准备好被重新填充 */ } } __global__ void producer_consumer_pattern ( int N , float * in , float * out , int buffer_len ) { constexpr int warpSize = 32 ; /* 下面声明的共享内存缓冲区大小为 2 * buffer_len，以便我们可以在两个缓冲区之间交替工作。 buffer_0 = buffer 且 buffer_1 = buffer + buffer_len */ __shared__ extern float buffer []; /* bar[0] 和 bar[1] 跟踪缓冲区 buffer_0 和 buffer_1 是否准备好被填充，而 bar[2] 和 bar[3] 分别跟踪缓冲区 buffer_0 和 buffer_1 是否已被填充 */ #pragma nv_diag_suppress static_var_with_dynamic_init __shared__ barrier_t bar [ 4 ]; if ( threadIdx . x < 4 ) { init ( bar + threadIdx . x , blockDim . x ); } __syncthreads (); if ( threadIdx . x < warpSize ) { produce ( bar , bar + 2 , buffer , buffer_len , in , N ); } else { consume ( bar , bar + 2 , buffer , buffer_len , out , N ); } } |
| --- |

 CUDA C++
`cuda::ptx`

| #include <cuda/ptx> __device__ void produce ( barrier ready [], barrier filled [], float * buffer , int buffer_len , float * in , int N ) { for ( int i = 0 ; i < N / buffer_len ; ++ i ) { uint64_t token1 = cuda :: ptx :: mbarrier_arrive ( ready [ i % 2 ]); while ( ! cuda :: ptx :: mbarrier_try_wait ( & ready [ i % 2 ], token1 )) {} /* 等待 buffer_(i%2) 准备好被填充 */ /* 生产，即填充 buffer_(i%2)  */ uint64_t token2 = cuda :: ptx :: mbarrier_arrive ( & filled [ i % 2 ]); /* buffer_(i%2) 已填充 */ } } __device__ void consume ( barrier ready [], barrier filled [], float * buffer , buffer_len , float * out , int N ) { uint64_t token1 = cuda :: ptx :: mbarrier_arrive ( & ready [ 0 ]); /* buffer_0 准备好进行初始填充 */ uint64_t token2 = cuda :: ptx :: mbarrier_arrive ( & ready [ 1 ]); /* buffer_1 准备好进行初始填充 */ for ( int i = 0 ; i < N / buffer_len ; ++ i ) { uint64_t token3 = cuda :: ptx :: mbarrier_arrive ( & filled [ i % 2 ]); while ( ! cuda :: ptx :: mbarrier_try_wait ( & filled [ i % 2 ], token3x )) {} /* 等待 buffer_(i%2) 被填充 */ /* 消费 buffer_(i%2) */ uint64_t token4 = cuda :: ptx :: mbarrier_arrive ( & ready [ i % 2 ]); /* buffer_(i%2) 准备好被重新填充 */ } } __global__ void producer_consumer_pattern ( int N , float * in , float * out , int buffer_len ) { constexpr int warpSize = 32 ; /* 下面声明的共享内存缓冲区大小为 2 * buffer_len，以便我们可以在两个缓冲区之间交替工作。 buffer_0 = buffer 且 buffer_1 = buffer + buffer_len */ __shared__ extern float buffer []; /* bar[0] 和 bar[1] 跟踪缓冲区 buffer_0 和 buffer_1 是否准备好被填充，而 bar[2] 和 bar[3] 分别跟踪缓冲区 buffer_0 和 buffer_1 是否已被填充 */ #pragma nv_diag_suppress static_var_with_dynamic_init __shared__ uint64_t bar [ 4 ]; if ( threadIdx . x < 4 ) { cuda :: ptx :: mbarrier_init ( bar + block . thread_rank (), block . size ()); } __syncthreads (); if ( threadIdx . x < warpSize ) { produce ( bar , bar + 2 , buffer , buffer_len , in , N ); } else { consume ( bar , bar + 2 , buffer , buffer_len , out , N ); } } |
| --- |

 CUDA C 原语

| #include <cuda_awbarrier_primitives.h> __device__ void produce ( __mbarrier_t ready [], __mbarrier_t filled [], float * buffer , int buffer_len , float * in , int N ) { for ( int i = 0 ; i < N / buffer_len ; ++ i ) { __mbarrier_token_t token1 = __mbarrier_arrive ( & ready [ i % 2 ]); /* 等待 buffer_(i%2) 准备好被填充 */ while ( ! __mbarrier_try_wait ( & ready [ i % 2 ], token1 , 1000 )) {} /* 生产，即填充 buffer_(i%2)  */ __mbarrier_token_t token2 = __mbarrier_arrive ( filled [ i % 2 ]); /* buffer_(i%2) 已填充 */ } } __device__ void consume ( __mbarrier_t ready [], __mbarrier_t filled [], float * buffer , int buffer_len , float * out , int N ) { __mbarrier_token_t token1 = __mbarrier_arrive ( & ready [ 0 ]); /* buffer_0 准备好进行初始填充 */ __mbarrier_token_t token2 = __mbarrier_arrive ( & ready [ 1 ]); /* buffer_1 准备好进行初始填充 */ for ( int i = 0 ; i < N / buffer_len ; ++ i ) { __mbarrier_token_t token3 = __mbarrier_arrive ( & filled [ i % 2 ]); while ( ! __mbarrier_try_wait ( & filled [ i % 2 ], token3 , 1000 )) {} /* 消费 buffer_(i%2) */ __mbarrier_token_t token4 = __mbarrier_arrive ( & ready [ i % 2 ]); /* buffer_(i%2) 准备好被重新填充 */ } } __global__ void producer_consumer_pattern ( int N , float * in , float * out , int buffer_len ) { constexpr int warpSize = 32 ; /* 下面声明的共享内存缓冲区大小为 2 * buffer_len，以便我们可以在两个缓冲区之间交替工作。 buffer_0 = buffer 且 buffer_1 = buffer + buffer_len */ __shared__ extern float buffer []; /* bar[0] 和 bar[1] 跟踪缓冲区 buffer_0 和 buffer_1 是否准备好被填充，而 bar[2] 和 bar[3] 分别跟踪缓冲区 buffer_0 和 buffer_1 是否已被填充 */ #pragma nv_diag_suppress static_var_with_dynamic_init __shared__ __mbarrier_t bar [ 4 ]; if ( threadIdx . x < 4 ) { __mbarrier_init ( bar + threadIdx . x , blockDim . x ); } __syncthreads (); if ( threadIdx . x < warpSize ) { produce ( bar , bar + 2 , buffer , buffer_len , in , N ); } else { consume ( bar , bar + 2 , buffer , buffer_len , out , N ); } } |
在此示例中，第一个线程束被专门用作生产者，其余线程束被专门用作消费者。所有生产者和消费者线程都参与（调用 `bar.arrive()` 或 `bar.arrive_and_wait()`）四个屏障中的每一个，因此预期的到达计数等于 `block.size()`。

生产者线程等待消费者线程发出可以填充共享内存缓冲区的信号。为了等待屏障，生产者线程必须首先到达 `ready[i%2].arrive()` 以获取令牌，然后使用该令牌执行 `ready[i%2].wait(token)`。为简化起见，`ready[i%2].arrive_and_wait()` 合并了这些操作。

```cpp
bar.arrive_and_wait();
/* is equivalent to */
bar.wait(bar.arrive());
```

生产者线程计算并填充就绪缓冲区，然后通过到达已填充屏障 `filled[i%2].arrive()` 来发出缓冲区已填充的信号。生产者线程此时并不等待，而是等待下一次迭代的缓冲区（双缓冲）准备好被填充。

消费者线程首先发出两个缓冲区都已准备好被填充的信号。消费者线程此时并不等待，而是等待本次迭代的缓冲区被填充，即 `filled[i%2].arrive_and_wait()`。在消费者线程消费完缓冲区后，它们发出缓冲区可以再次被填充的信号 `ready[i%2].arrive()`，然后等待下一次迭代的缓冲区被填充。

 在此页面上