# 4.12 集群启动控制

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)

---

此页面有帮助吗？

# 4.12. 使用集群启动控制进行工作窃取

在开发 CUDA 应用程序时，处理可变数据和计算规模的问题是至关重要的。传统上，CUDA 开发者使用两种主要方法来确定要启动的内核线程块数量：*每个线程块的固定工作量*和*固定数量的线程块*。这两种方法各有优缺点。

**每个线程块的固定工作量：** 在这种方法中，线程块的数量由问题规模决定，而每个线程块完成的工作量保持不变。

这种方法的主要优点：

-   **SM 之间的负载均衡** 当线程块运行时间表现出可变性，和/或当线程块数量远大于 GPU 可以同时执行的数量（导致低尾效应）时，这种方法允许 GPU 调度器在某些 SM 上运行比其他 SM 更多的线程块。
-   **抢占** GPU 调度器可以开始执行更高优先级的内核，即使它是在较低优先级内核已经开始执行后才启动的，其方式是在较低优先级内核的线程块完成时调度其线程块。一旦更高优先级的内核执行完毕，它就可以恢复执行较低优先级的内核。

**固定数量的线程块：** 这种方法通常实现为块步长或网格步长循环，线程块的数量不依赖于问题规模。相反，每个线程块完成的工作量是问题规模的函数。通常，线程块的数量基于执行内核的 GPU 上的 SM 数量以及期望的占用率。

这种方法的主要优点：

-   **减少线程块开销** 这种方法不仅减少了分摊的线程块启动延迟，还最小化了与所有线程块之间共享操作相关的计算开销。这些开销可能显著高于启动延迟开销。例如，在卷积内核中，用于计算卷积系数的序言（与线程块索引无关）由于线程块数量固定，可以减少计算次数，从而减少冗余计算。

**集群启动控制** 是 NVIDIA Blackwell GPU 架构（计算能力 10.0）中引入的一项功能，旨在结合前两种方法的优点。它通过允许开发者取消线程块或线程块集群，为开发者提供了对线程块调度的更多控制。这种机制实现了工作窃取。工作窃取是并行计算中的一种动态负载均衡技术，空闲的处理器会主动从繁忙处理器的工作队列中“窃取”任务，而不是等待分配工作。

图 51 集群启动控制流程[#](#cluster-launch-control-diagram)

通过集群启动控制，一个线程块会尝试取消另一个尚未开始执行的线程块的启动。如果取消请求成功，它会通过使用被取消线程块的索引来执行任务，从而“窃取”其工作。如果没有更多可用的线程块索引，或者由于其他原因（例如调度了更高优先级的内核），取消将失败。在后一种情况下，如果一个线程块在取消失败后退出，调度器可以开始执行更高优先级的内核，之后将继续调度当前内核的剩余线程块执行。上面的[图表](#cluster-launch-control-diagram)展示了此过程的执行流程。
下表总结了三种方法的优缺点：

|  | 固定每个线程块的工作量 | 固定线程块数量 | 集群启动控制 |
| --- | --- | --- | --- |
| 减少开销 | \(\textcolor{red}{\textbf{X}}\) | \(\textcolor{lime}{\textbf{V}}\) | \(\textcolor{lime}{\textbf{V}}\) |
| 抢占 | \(\textcolor{lime}{\textbf{V}}\) | \(\textcolor{red}{\textbf{X}}\) | \(\textcolor{lime}{\textbf{V}}\) |
| 负载均衡 | \(\textcolor{lime}{\textbf{V}}\) | \(\textcolor{red}{\textbf{X}}\) | \(\textcolor{lime}{\textbf{V}}\) |

## 4.12.1.API 详情

通过集群启动控制 API 取消线程块是异步完成的，并使用共享内存屏障进行同步，遵循类似于[异步数据拷贝](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies)的编程模式。

该 API 通过 [libcu++](https://nvidia.github.io/cccl/libcudacxx/ptx_api.html) 提供，包含：

- 一条请求指令，将编码的取消结果写入一个 __shared__ 变量。
- 解码指令，用于提取成功/失败状态和被取消的线程块索引。

请注意，集群启动控制操作被建模为异步代理操作（参见[异步线程和异步代理](../03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)）。

### 4.12.1.1. 线程块取消

使用集群启动控制的首选方式是从单个线程进行，即一次一个请求。

取消过程涉及五个步骤：

- 设置阶段（步骤 1-2）：声明并初始化取消结果和同步变量。
- 工作窃取循环（步骤 3-5）：重复执行以请求、同步和处理取消结果。

1.  声明用于线程块取消的变量：
    __shared__ uint4 result ; // 请求结果。
    __shared__ uint64_t bar ; // 同步屏障。
    int phase = 0 ; // 同步屏障阶段。
2.  使用单个到达计数初始化共享内存屏障：
    if ( cg :: thread_block :: thread_rank () == 0 )
        ptx :: mbarrier_init ( & bar , 1 );
    __syncthreads ();
3.  由单个线程提交异步取消请求并设置事务计数：
    if ( cg :: thread_block :: thread_rank () == 0 ) {
        cg :: invoke_one ( cg :: coalesced_threads (), [ & ](){
            ptx :: clusterlaunchcontrol_try_cancel ( & result , & bar );
        });
        ptx :: mbarrier_arrive_expect_tx ( ptx :: sem_relaxed , ptx :: scope_cta , ptx :: space_shared , & bar , sizeof ( uint4 ));
    }
    由于线程块取消是一条统一指令，建议在 invoke_one 线程选择器内部提交它。这允许编译器优化掉剥离循环。
4.  同步（完成）异步取消请求：
    while ( ! ptx :: mbarrier_try_wait_parity ( & bar , phase )) {}
    phase ^= 1 ;
5.  检索取消状态和被取消的线程块索引：
    bool success = ptx :: clusterlaunchcontrol_query_cancel_is_canceled ( result );
    if ( success ) {
        // 对于 1D/2D 线程块，不需要全部三个：
        int bx = ptx :: clusterlaunchcontrol_query_cancel_get_first_ctaid_x ( result );
        int by = ptx :: clusterlaunchcontrol_query_cancel_get_first_ctaid_y ( result );
        int bz = ptx :: clusterlaunchcontrol_query_cancel_get_first_ctaid_z ( result );
    }
6. 确保异步代理与通用代理之间共享内存操作的可见性，并防止工作窃取循环迭代之间的数据竞争。

### 4.12.1.2. 线程块取消的约束

这些约束与失败的取消请求相关：

-   在观察到先前失败的请求后提交另一个取消请求是未定义行为。在下面的两个代码示例中，假设第一个取消请求失败，只有第一个示例表现出未定义行为。第二个示例是正确的，因为在取消请求之间没有观察操作：
    **无效代码：**
    ```cpp
    // 第一个请求：
    ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);
    // 第一个请求查询：
    [同步 bar0 的代码在此。]
    bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
    assert(!success0); // 观察到失败；第二次取消将无效。
    // 第二个请求 - 下一行是未定义行为：
    ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);
    ```
    **有效代码：**
    ```cpp
    // 第一个请求：
    ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);
    // 第二个请求：
    ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);
    // 第一个请求查询：
    [同步 bar0 的代码在此。]
    bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
    assert(!success0); // 观察到失败；第二次取消是有效的。
    ```
-   检索失败取消请求的线程块索引是未定义行为。
-   不建议从多个线程提交取消请求。这会导致多个线程块被取消，并且需要仔细处理，例如：
    *   每个提交线程必须提供一个唯一的 `__shared__` 结果指针以避免数据竞争。
    *   如果使用相同的屏障进行同步，则必须相应地调整到达计数和事务计数。

## 4.12.2. 示例：向量标量乘法

在以下小节中，我们通过集群启动控制演示工作窃取，使用一个向量标量乘法内核。我们展示同一问题的两个变体：一个使用线程块，另一个使用线程块集群。

### 4.12.2.1. 用例：线程块

下面的三个内核演示了向量标量乘法 \(\overline{v} := \alpha \overline{v}\) 的*每个线程块固定工作量*、*固定线程块数量*和*集群启动控制*方法。

-   **每个线程块固定工作量：**
    ```cpp
    __global__ void kernel_fixed_work(float* data, int n) {
        // 前导部分：
        float alpha = compute_scalar();
        // 计算部分：
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] *= alpha;
    }
    // 启动：
    kernel_fixed_work<<<1024, (n + 1023) / 1024>>>(data, n);
    ```
-   **固定线程块数量：**
    ```cpp
    __global__ void kernel_fixed_blocks(float* data, int n) {
        // 前导部分：
        float alpha = compute_scalar();
        // 计算部分：
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < n) {
            data[i] *= alpha;
            i += gridDim.x * blockDim.x;
        }
    }
    // 启动：
    kernel_fixed_blocks<<<1024, SM_COUNT>>>(data, n);
    ```
- **集群启动控制**：
```cpp
#include <cooperative_groups.h>
#include <cuda/ptx>
namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__global__ void kernel_cluster_launch_control(float* data, int n) {
    // 集群启动控制初始化：
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;
    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);

    // 序言：
    float alpha = compute_scalar(); // 此代码片段中未显示的设备函数。

    // 工作窃取循环：
    int bx = blockIdx.x; // 假设为一维 x 轴线程块。
    while (true) {
        // 保护 result 在下次迭代中不被覆盖，
        // （同时确保在第一次迭代时完成屏障初始化）：
        __syncthreads();

        // 取消请求：
        if (cg::thread_block::thread_rank() == 0) {
            // 在异步代理中获取 result 的写入权限：
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
            cg::invoke_one(cg::coalesced_threads(), [&](){
                ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
            });
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
        }

        // 计算：
        int i = bx * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] *= alpha;

        // 取消请求同步：
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
        phase ^= 1;

        // 取消请求解码：
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success)
            break;
        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);

        // 向异步代理释放 result 的读取权限：
        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}

// 启动：
kernel_cluster_launch_control<<<1024, (n + 1023) / 1024>>>(data, n);
```

### 4.12.2.2. 使用案例：线程块集群

在[线程块集群](../02-basics/intro-to-cuda-cpp.html#thread-block-clusters)的情况下，线程块取消步骤与非集群设置中的步骤相同，仅需进行微调。与非集群情况一样，不建议从**集群内**的多个线程提交取消请求，因为这将尝试取消多个集群。

-   取消请求由单个集群线程提交。
-   每个集群的线程块的共享内存 `result` 将接收到相同的（编码后的）被取消线程块索引值（即，`result` 值被多播）。所有线程块接收到的 `result` 对应于集群内的本地块索引 {0, 0, 0}。因此，集群内的线程块需要加上本地块索引。
-   同步由每个集群的线程块使用本地 `__shared__` 内存屏障执行。屏障操作必须使用 `ptx::scope_cluster` 作用域执行。
-   在集群情况下取消需要所有线程块都存在。
用户可以通过使用同步 API 中的 `cg::cluster_group::sync()` 来保证所有线程块都在运行。

下面的内核演示了使用线程块集群的集群启动控制方法。

```cuda
#include <cooperative_groups.h>
#include <cuda/ptx>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__global__ __cluster_dims__(2, 1, 1)
void kernel_cluster_launch_control (float* data, int n)
{
    // Cluster launch control initialization:
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    if (cg::thread_block::thread_rank() == 0) {
        ptx::mbarrier_init(&bar, 1);
        ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster); // CGA-level fence.
    }

    // Prologue:
    float alpha = compute_scalar(); // Device function not shown in this code snippet.

    // Work-stealing loop:
    int bx = blockIdx.x; // Assuming 1D x-axis thread blocks.

    while (true) {
        // Protect result from overwrite in the next iteration,
        // (also ensure all thread blocks have started at 1st iteration):
        cg::cluster_group::sync();

        // Cancellation request by a single cluster thread:
        if (cg::cluster_group::thread_rank() == 0) {
            // Acquire write of result in the async proxy:
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

            cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel_multicast(&result, &bar);});
        }

        // Cancellation completion tracked by each thread block:
        if (cg::thread_block::thread_rank() == 0)
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cluster, ptx::space_shared, &bar, sizeof(uint4));

        // Computation:
        int i = bx * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] *= alpha;

        // Cancellation request synchronization:
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cluster, &bar, phase))
        {}
        phase ^= 1;

        // Cancellation request decoding:
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success)
            break;

        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
        bx += cg::cluster_group::block_index().x; // Add local offset.

        // Release read of result to the async proxy:
        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}

// Launch: kernel_cluster_launch_control<<<1024, (n + 1023) / 1024>>>(data, n);
```

 在本页