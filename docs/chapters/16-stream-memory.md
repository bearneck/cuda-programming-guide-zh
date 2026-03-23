# 4.3 流有序内存分配

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)

---

本页面是否有帮助？

# 4.3. 流序内存分配器

## 4.3.1. 简介

使用 `cudaMalloc` 和 `cudaFree` 管理内存分配会导致 GPU 在所有正在执行的 CUDA 流之间进行同步。流序内存分配器使应用程序能够将内存分配和释放操作与提交到 CUDA 流中的其他工作（例如内核启动和异步拷贝）进行排序。这通过利用流排序语义来重用内存分配，从而改善了应用程序的内存使用。该分配器还允许应用程序控制分配器的内存缓存行为。当设置了适当的释放阈值后，缓存行为允许分配器在应用程序表明其愿意接受更大的内存占用时，避免昂贵的操作系统调用。该分配器还支持进程间简单且安全的分配共享。

流序内存分配器：

> 减少了对自定义内存管理抽象的需求，并使需要高性能自定义内存管理的应用程序更容易实现。
> 使多个库能够共享由驱动程序管理的公共内存池。这可以减少过度的内存消耗。
> 允许驱动程序基于其对分配器和其他流管理 API 的了解来执行优化。

!!! note "注意"
    自 CUDA 11.3 起，Nsight Compute 和下一代 CUDA 调试器已支持该分配器。

## 4.3.2. 内存管理

`cudaMallocAsync` 和 `cudaFreeAsync` 是实现流序内存管理的 API。`cudaMallocAsync` 返回一个分配，`cudaFreeAsync` 释放一个分配。这两个 API 都接受流参数，以定义分配何时变为可用以及何时停止可用。这些函数允许内存操作与特定的 CUDA 流绑定，使它们能够在不阻塞主机或其他流的情况下发生。通过避免 `cudaMalloc` 和 `cudaFree` 可能带来的昂贵同步，可以提高应用程序性能。

这些 API 可以通过内存池用于进一步的性能优化，内存池管理并重用大块内存，以实现更高效的分配和释放。内存池有助于减少开销并防止碎片化，从而在频繁进行内存分配操作的场景中提高性能。

### 4.3.2.1. 分配内存

`cudaMallocAsync` 函数在 GPU 上触发与特定 CUDA 流关联的异步内存分配。`cudaMallocAsync` 允许内存分配在不阻碍主机或其他流的情况下进行，从而消除了昂贵的同步需求。

!!! note "注意"
    `cudaMallocAsync` 在确定分配位置时会忽略当前的设备/上下文。相反，`cudaMallocAsync` 根据指定的内存池或提供的流来确定合适的设备。

下面的代码清单展示了一个基本的使用模式：内存被分配、使用，然后释放回同一个流。

```c++
void *ptr;
size_t size = 512;
cudaMallocAsync(&ptr, size, cudaStreamPerThread);
// do work using the allocation
kernel<<<..., cudaStreamPerThread>>>(ptr, ...);
// An asynchronous free can be specified without synchronizing the cpu and GPU
cudaFreeAsync(ptr, cudaStreamPerThread);
```

!!! note "Note"
    When accessing allocation from a stream other than the stream that made the allocation, the
user must guarantee that the access occurs after the allocation
operation, otherwise the behavior is undefined.

### 4.3.2.2.Freeing Memory

`cudaFreeAsync()` asynchronously frees device memory in a stream-ordered fashion, meaning the memory deallocation is assigned to a specific CUDA stream and does not block the host or other streams.

The user must guarantee that the free operation happens after the allocation operation and any uses of the allocation. Any use of the allocation after the free operation starts results in undefined behavior.

Events and/or stream synchronizing operations should be used to guarantee any access to the allocation from other streams is complete before the free operation begins, as illustrated in the following example.

```c++
cudaMallocAsync(&ptr, size, stream1);
cudaEventRecord(event1, stream1);
//stream2 must wait for the allocation to be ready before accessing
cudaStreamWaitEvent(stream2, event1);
kernel<<<..., stream2>>>(ptr, ...);
cudaEventRecord(event2, stream2);
// stream3 must wait for stream2 to finish accessing the allocation before
// freeing the allocation
cudaStreamWaitEvent(stream3, event2);
cudaFreeAsync(ptr, stream3);
```

Memory allocated with `cudaMalloc()` can be freed with with `cudaFreeAsync()`. As above, all accesses to the memory must be complete before the free operation begins.

```c++
cudaMalloc(&ptr, size);
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);
```

Likewise, memory allocated with `cudaMallocAsync` can be freed with `cudaFree()`. When freeing such allocations through the `cudaFree()` API, the driver assumes that all accesses to the allocation are complete and performs no further synchronization. The user can use `cudaStreamQuery` / `cudaStreamSynchronize` / `cudaEventQuery` / `cudaEventSynchronize` / `cudaDeviceSynchronize` to guarantee that the appropriate asynchronous work is complete and that the GPU will not try to access the allocation.

```c++
cudaMallocAsync(&ptr, size,stream);
kernel<<<..., stream>>>(ptr, ...);
// synchronize is needed to avoid prematurely freeing the memory
cudaStreamSynchronize(stream);
cudaFree(ptr);
```

## 4.3.3.Memory Pools

Memory pools encapsulate virtual address and physical memory resources that are allocated and managed according to the pools attributes and properties. The primary aspect of a memory pool is the kind and location of memory it manages.

All calls to `cudaMallocAsync` use resources from memory pool. If a memory pool is not specified, `cudaMallocAsync` uses the current memory pool of the supplied streamâs device. The current memory pool for a device may be set with `cudaDeviceSetMempool` and queried with `cudaDeviceGetMempool`. Each device has a default memory pool, which is active if `cudaDeviceSetMempool` has not been called.
API `cudaMallocFromPoolAsync` 和 [c++ overloads of
cudaMallocAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb) 允许用户指定用于分配的内存池，而无需将其设置为当前池。API `cudaDeviceGetDefaultMempool` 和 `cudaMemPoolCreate` 返回内存池的句柄。`cudaMemPoolSetAttribute` 和 `cudaMemPoolGetAttribute` 控制内存池的属性。

!!! note "注意"
    设备的当前内存池将是该设备本地的。因此，在不指定内存池的情况下进行分配，将始终产生流设备本地的分配。

### 4.3.3.1. 默认/隐式池

可以通过调用 `cudaDeviceGetDefaultMempool` 来获取设备的默认内存池。从设备默认内存池进行的分配是位于该设备上的不可迁移设备分配。这些分配始终可以从该设备访问。默认内存池的可访问性可以通过 `cudaMemPoolSetAccess` 修改，并通过 `cudaMemPoolGetAccess` 查询。由于默认池不需要显式创建，它们有时被称为隐式池。设备的默认内存池不支持 IPC。

### 4.3.3.2. 显式池

`cudaMemPoolCreate` 创建一个显式池。这允许应用程序为其分配请求超出默认/隐式池所提供的属性。这些属性包括 IPC 能力、最大池大小、在支持的平台上驻留在特定 CPU NUMA 节点上的分配等。

```c++
// 创建一个类似于设备 0 上隐式池的池
int device = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = device;
poolProps.location.type = cudaMemLocationTypeDevice;

cudaMemPoolCreate(&memPool, &poolProps));
```

以下代码片段演示了在有效的 CPU NUMA 节点上创建支持 IPC 的内存池的示例。

```c++
// 创建一个驻留在 CPU NUMA 节点上、能够进行 IPC 共享（通过文件描述符）的池。
int cpu_numa_id = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = cpu_numa_id;
poolProps.location.type = cudaMemLocationTypeHostNuma;
poolProps.handleType = cudaMemHandleTypePosixFileDescriptor;

cudaMemPoolCreate(&ipcMemPool, &poolProps));
```

### 4.3.3.3. 多 GPU 支持的设备可访问性

与通过虚拟内存管理 API 控制的分配可访问性类似，内存池分配的可访问性不遵循 `cudaDeviceEnablePeerAccess` 或 `cuCtxEnablePeerAccess`。对于内存池，API `cudaMemPoolSetAccess` 修改哪些设备可以访问池中的分配。默认情况下，分配只能从分配所在的设备访问。此访问权限不能被撤销。要启用来自其他设备的访问，访问设备必须与内存池的设备具备对等能力。这可以通过 `cudaDeviceCanAccessPeer` 来验证。如果未检查对等能力，设置访问可能会失败并返回 `cudaErrorInvalidDevice`。但是，如果尚未从池中进行任何分配，即使设备不具备对等能力，`cudaMemPoolSetAccess` 调用也可能成功。在这种情况下，下一次从池中进行分配将会失败。
值得注意的是，`cudaMemPoolSetAccess` 会影响内存池中的所有分配，而不仅仅是未来的分配。同样，`cudaMemPoolGetAccess` 报告的可访问性也适用于池中的所有分配，而不仅仅是未来的分配。不建议频繁更改内存池对特定 GPU 的可访问性设置。也就是说，一旦内存池被设置为可从特定 GPU 访问，在该内存池的整个生命周期内，它都应保持对该 GPU 的可访问性。

```c++
// 展示 cudaMemPoolSetAccess 用法的代码片段：
cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice,
              int accessingDevice) {
    cudaMemAccessDesc accessDesc = {};
    accessDesc.location.type = cudaMemLocationTypeDevice;
    accessDesc.location.id = accessingDevice;
    accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

    int canAccess = 0;
    cudaError_t error = cudaDeviceCanAccessPeer(&canAccess, accessingDevice,
              residentDevice);
    if (error != cudaSuccess) {
        return error;
    } else if (canAccess == 0) {
        return cudaErrorPeerAccessUnsupported;
    }

    // 使地址可访问
    return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
}
```

### 4.3.3.4. 为 IPC 启用内存池

可以为进程间通信（IPC）启用内存池，以便在进程之间轻松、高效且安全地共享 GPU 内存。CUDA 的 IPC 内存池提供了与 CUDA [虚拟内存管理 API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-memory-management) 相同的安全优势。

使用内存池在进程间共享内存需要两个步骤：进程首先需要共享对内存池的访问权限，然后共享该池中的特定分配。第一步建立并强制执行安全性。第二步协调每个进程中使用的虚拟地址，以及映射需要在导入进程中何时生效。

#### 4.3.3.4.1. 创建和共享 IPC 内存池

共享对内存池的访问权限涉及使用 `cudaMemPoolExportToShareableHandle()` 获取内存池的 OS 原生句柄，使用 OS 原生 IPC 机制将句柄传输到导入进程，然后使用 `cudaMemPoolImportFromShareableHandle()` API 创建导入的内存池。要使 `cudaMemPoolExportToShareableHandle` 成功，内存池必须是在池属性结构中指定了所请求句柄类型的情况下创建的。

请参考[示例](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocationIPC)了解在进程间传输 OS 原生句柄的适当 IPC 机制。其余步骤可在以下代码片段中找到。

```c++
// 在导出进程中
// 在设备 0 上创建一个可导出、支持 IPC 的池
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = 0;
poolProps.location.type = cudaMemLocationTypeDevice;

// 将 handleTypes 设置为非零值将使池可导出（支持 IPC）
poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

cudaMemPoolCreate(&memPool, &poolProps));

// 基于文件描述符（FD）的句柄是整数类型
int fdHandle = 0;

// 获取内存池的 OS 原生句柄。
// 注意，这里传入的是指向句柄内存的指针。
cudaMemPoolExportToShareableHandle(&fdHandle,
             memPool,
             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
             0);

// 必须使用适当的操作系统特定 API 将句柄发送到导入进程。
```

```c++
// in importing process
 int fdHandle;
// The handle needs to be retrieved from the exporting process with the
// appropriate OS-specific APIs.
// Create an imported pool from the shareable handle.
// Note that the handle is passed by value here.
cudaMemPoolImportFromShareableHandle(&importedMemPool,
          (void*)fdHandle,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
          0);
```

#### 4.3.3.4.2.Set Access in the Importing Process

Imported memory pools are initially only accessible from their resident device. The imported memory pool does not inherit any accessibility set by the exporting process. The importing process needs to enable access with `cudaMemPoolSetAccess` from any GPU it plans to access the memory from.

If the imported memory pool belongs to a device that is not visible to importing process, the user must use the `cudaMemPoolSetAccess` API to enable access from the GPUs the allocations will be used on. (See [Device Accessibility for Multi-GPU Support](#stream-ordered-deviceaccessibility))

#### 4.3.3.4.3.Creating and Sharing Allocations from an Exported Pool

Once the pool has been shared, allocations made with `cudaMallocAsync()` from the pool in the exporting process can be shared with processes that have imported the pool. Since the poolâs security policy is established and verified at the pool level, the OS does not need extra bookkeeping to provide security for specific pool allocations. In other words, the opaque `cudaMemPoolPtrExportData` required to import a pool allocation may be sent to the importing process using any mechanism.

While allocations may be exported and imported without synchronizing with the allocating stream in any way, the importing process must follow the same rules as the exporting process when accessing the allocation. Specifically, access to the allocation must happen after the allocation operation in the allocating stream executes. The two following code snippets show `cudaMemPoolExportPointer()` and `cudaMemPoolImportPointer()` sharing the allocation with an IPC event used to guarantee that the allocation isnât accessed in the importing process before the allocation is ready.

```c++
// preparing an allocation in the exporting process
cudaMemPoolPtrExportData exportData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t readyIpcEventHandle;

// ipc event for coordinating between processes
// cudaEventInterprocess flag makes the event an ipc event
// cudaEventDisableTiming  is set for performance reasons

cudaEventCreate(&readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess)

// allocate from the exporting mem pool
cudaMallocAsync(&ptr, size,exportMemPool, stream);

// event for sharing when the allocation is ready.
cudaEventRecord(readyIpcEvent, stream);
cudaMemPoolExportPointer(&exportData, ptr);
cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

// Share IPC event and pointer export data with the importing process using
//  any mechanism. Here we copy the data into shared memory
shmem->ptrData = exportData;
shmem->readyIpcEventHandle = readyIpcEventHandle;
// signal consumers data is ready
```

```c++
// Importing an allocation
cudaMemPoolPtrExportData *importData = &shmem->prtData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t *readyIpcEventHandle = &shmem->readyIpcEventHandle;

// Need to retrieve the ipc event handle and the export data from the
// exporting process using any mechanism.  Here we are using shmem and just
// need synchronization to make sure the shared memory is filled in.

cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);

// import the allocation. The operation does not block on the allocation being ready.
cudaMemPoolImportPointer(&ptr, importedMemPool, importData);

// Wait for the prior stream operations in the allocating stream to complete before
// using the allocation in the importing process.
cudaStreamWaitEvent(stream, readyIpcEvent);
kernel<<<..., stream>>>(ptr, ...);
```

When freeing the allocation, the allocation must be freed in the importing process before it is freed in the exporting process. The following code snippet demonstrates the use of CUDA IPC events to provide the required synchronization between the `cudaFreeAsync` operations in both processes. Access to the allocation from the importing process is obviously restricted by the free operation in the importing process side. It is worth noting that `cudaFree` can be used to free the allocation in both processes and that other stream synchronization APIs may be used instead of CUDA IPC events.

```c++
// The free must happen in importing process before the exporting process
kernel<<<..., stream>>>(ptr, ...);

// Last access in importing process
cudaFreeAsync(ptr, stream);

// Access not allowed in the importing process after the free
cudaIpcEventRecord(finishedIpcEvent, stream);
```

```c++
// Exporting process
// The exporting process needs to coordinate its free with the stream order
// of the importing processâs free.
cudaStreamWaitEvent(stream, finishedIpcEvent);
kernel<<<..., stream>>>(ptrInExportingProcess, ...);

// The free in the importing process doesnât stop the exporting process
// from using the allocation.
cudFreeAsync(ptrInExportingProcess,stream);
```

#### 4.3.3.4.4.IPC Export Pool Limitations

IPC pools currently do not support releasing physical blocks back to the OS. As a result the `cudaMemPoolTrimTo` API has no effect and the `cudaMemPoolAttrReleaseThreshold` is effectively ignored. This behavior is controlled by the driver, not the runtime and may change in a future driver update.

#### 4.3.3.4.5.IPC Import Pool Limitations

Allocating from an import pool is not allowed; specifically, import pools cannot be set current and cannot be used in the `cudaMallocFromPoolAsync` API. As such, the allocation reuse policy attributes do not have meaning for these pools.

IPC Import pools, like IPC export pools, currently do not support releasing physical blocks back to the OS.

The resource usage stat attribute queries only reflect the allocations imported into the process and the associated physical memory.

## 4.3.4.Best Practices and Tuning
### 4.3.4.1. 查询支持情况

应用程序可以通过调用 `cudaDeviceGetAttribute()`（参见[开发者博客](https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/)）并传入设备属性 `cudaDevAttrMemoryPoolsSupported`，来确定设备是否支持流序内存分配器。

IPC 内存池支持可以通过 `cudaDevAttrMemoryPoolSupportedHandleTypes` 设备属性来查询。此属性在 CUDA 11.3 中添加，在较旧的驱动程序上查询此属性将返回 `cudaErrorInvalidValue`。

```c++
int driverVersion = 0;
int deviceSupportsMemoryPools = 0;
int poolSupportedHandleTypes = 0;
cudaDriverGetVersion(&driverVersion);
if (driverVersion >= 11020) {
    cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                           cudaDevAttrMemoryPoolsSupported, device);
}
if (deviceSupportsMemoryPools != 0) {
    // `device` 支持流序内存分配器
}

if (driverVersion >= 11030) {
    cudaDeviceGetAttribute(&poolSupportedHandleTypes,
              cudaDevAttrMemoryPoolSupportedHandleTypes, device);
}
if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
   // 可以在指定设备上创建基于 POSIX 文件描述符的 IPC 内存池
}
```

在查询之前检查驱动程序版本，可以避免在尚未定义该属性的驱动程序上遇到 `cudaErrorInvalidValue` 错误。也可以使用 `cudaGetLastError` 来清除错误，而不是避免它。

### 4.3.4.2. 物理页缓存行为

默认情况下，分配器会尝试最小化内存池占用的物理内存。为了最小化操作系统分配和释放物理内存的调用，应用程序必须为每个内存池配置一个内存占用上限。应用程序可以通过释放阈值属性（`cudaMemPoolAttrReleaseThreshold`）来实现这一点。

释放阈值是内存池在尝试将内存释放回操作系统之前应保留的内存量（以字节为单位）。当内存池持有的内存超过释放阈值时，分配器将在下一次调用流同步、事件同步或设备同步时，尝试将内存释放回操作系统。将释放阈值设置为 `UINT64_MAX` 将阻止驱动程序在每次同步后尝试收缩内存池。

```c++
Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
```

将 `cudaMemPoolAttrReleaseThreshold` 设置得足够高以有效禁用内存池收缩的应用程序，可能希望显式地收缩内存池的内存占用。`cudaMemPoolTrimTo` 允许应用程序这样做。在修剪内存池的占用空间时，`minBytesToKeep` 参数允许应用程序保留指定数量的内存，例如它在后续执行阶段预期需要的内存量。

```c++
Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

// 应用程序阶段需要从流序分配器获取大量内存
for (i=0; i<10; i++) {
    for (j=0; j<10; j++) {
        cudaMallocAsync(&ptrs[j],size[j], stream);
    }
    kernel<<<...,stream>>>(ptrs,...);
    for (j=0; j<10; j++) {
        cudaFreeAsync(ptrs[j], stream);
    }
}

// 进程在下一阶段不需要那么多内存。
// 进行同步，以便修剪操作知道分配的内存不再使用。
cudaStreamSynchronize(stream);
cudaMemPoolTrimTo(mempool, 0);

// 其他进程/分配机制现在可以使用修剪操作释放的物理内存。
```
### 4.3.4.3. 资源使用统计

查询内存池的 `cudaMemPoolAttrReservedMemCurrent` 属性，会报告该池当前消耗的 GPU 物理内存总量。查询内存池的 `cudaMemPoolAttrUsedMemCurrent` 属性，会返回从该池中已分配且不可复用的所有内存的总大小。

`cudaMemPoolAttr*MemHigh` 属性是记录自上次重置以来，其对应的 `cudaMemPoolAttr*MemCurrent` 属性所达到的最大值的水位标记。可以通过使用 `cudaMemPoolSetAttribute` API 将它们重置为当前值。

```c++
// 用于批量获取使用统计信息的示例辅助函数
struct usageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};

void getUsageStatistics(cudaMemoryPool_t memPool, struct usageStatistics *statistics)
{
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, statistics->reserved);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, statistics->reservedHigh);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, statistics->used);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, statistics->usedHigh);
}

// 重置水位标记将使它们取当前值。
void resetStatistics(cudaMemoryPool_t memPool)
{
    cuuint64_t value = 0;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
}
```

### 4.3.4.4. 内存复用策略

为了满足分配请求，驱动程序会尝试复用先前通过 `cudaFreeAsync()` 释放的内存，然后再尝试从操作系统分配更多内存。例如，在流中释放的内存可以立即在同一流的后续分配请求中复用。当一个流与 CPU 同步后，先前在该流中释放的内存就可以在任何流的分配中复用了。复用策略可以应用于默认内存池和显式内存池。

流序分配器有几个可控的分配策略。内存池属性 `cudaMemPoolReuseFollowEventDependencies`、`cudaMemPoolReuseAllowOpportunistic` 和 `cudaMemPoolReuseAllowInternalDependencies` 控制这些策略，详情如下。这些策略可以通过调用 `cudaMemPoolSetAttribute` 来启用或禁用。升级到更新的 CUDA 驱动程序可能会更改、增强、扩充和/或重新排序复用策略的枚举。

#### 4.3.4.4.1. cudaMemPoolReuseFollowEventDependencies

在分配更多 GPU 物理内存之前，分配器会检查由 CUDA 事件建立的依赖信息，并尝试从另一个流中释放的内存进行分配。

```c++
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);
cudaEventRecord(event,originalStream);

// 在另一个流中等待捕获了释放操作的事件，
// 当启用 cudaMemPoolReuseFollowEventDependencies 时，
// 允许分配器复用该内存来满足另一个流中的新分配请求。
cudaStreamWaitEvent(otherStream, event);
cudaMallocAsync(&ptr2, size, otherStream);
```
#### 4.3.4.4.2. cudaMemPoolReuseAllowOpportunistic

当启用 `cudaMemPoolReuseAllowOpportunistic` 策略时，分配器会检查已释放的分配，以查看释放操作的流顺序语义是否已满足，例如流是否已通过释放操作指示的执行点。当此策略被禁用时，分配器仍将重用当流与 CPU 同步时变得可用的内存。禁用此策略不会阻止 `cudaMemPoolReuseFollowEventDependencies` 策略的应用。

```c++
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// after some time, the kernel finishes running
wait(10);

// When cudaMemPoolReuseAllowOpportunistic is enabled this allocation request
// can be fulfilled with the prior allocation based on the progress of originalStream.
cudaMallocAsync(&ptr2, size, otherStream);
```

#### 4.3.4.4.3. cudaMemPoolReuseAllowInternalDependencies

当无法从操作系统分配和映射更多物理内存时，驱动程序将寻找其可用性依赖于另一个流待处理进度的内存。如果找到此类内存，驱动程序会将所需的依赖项插入到分配流中并重用该内存。

```c++
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// When cudaMemPoolReuseAllowInternalDependencies is enabled
// and the driver fails to allocate more physical memory, the driver may
// effectively perform a cudaStreamWaitEvent in the allocating stream
// to make sure that future work in âotherStreamâ happens after the work
// in the original stream that would be allowed to access the original allocation.
cudaMallocAsync(&ptr2, size, otherStream);
```

#### 4.3.4.4.4. 禁用重用策略

虽然可控的重用策略提高了内存重用率，但用户可能希望禁用它们。允许机会性重用（例如 `cudaMemPoolReuseAllowOpportunistic`）会基于 CPU 和 GPU 执行的交错，在分配模式中引入运行间的差异。内部依赖项插入（例如 `cudaMemPoolReuseAllowInternalDependencies`）可能会以意外且可能非确定性的方式序列化工作，而用户可能更希望在分配失败时显式同步事件或流。

### 4.3.4.5. 同步 API 操作

分配器作为 CUDA 驱动程序一部分所带来的优化之一是与同步 API 的集成。当用户请求 CUDA 驱动程序同步时，驱动程序会等待异步工作完成。在返回之前，驱动程序将确定哪些释放操作已由同步保证完成。无论指定的流或禁用的分配策略如何，这些分配都将变为可用于分配。驱动程序还会在此处检查 `cudaMemPoolAttrReleaseThreshold` 并释放任何可以释放的过量物理内存。
## 4.3.5. 附录

### 4.3.5.1. cudaMemcpyAsync 对当前上下文/设备的敏感性

在当前 CUDA 驱动程序中，任何涉及来自 `cudaMallocAsync` 的内存的异步 `memcpy` 操作，都应使用指定流的上下文作为调用线程的当前上下文来完成。这对于 `cudaMemcpyPeerAsync` 来说不是必需的，因为该 API 引用的是指定的设备主上下文，而不是当前上下文。

### 4.3.5.2. cudaPointerGetAttributes 查询

在对一个分配调用 `cudaFreeAsync` 之后，再对其调用 `cudaPointerGetAttributes` 会导致未定义行为。具体来说，无论该分配是否仍可从给定流访问，行为都是未定义的。

### 4.3.5.3. cudaGraphAddMemsetNode

`cudaGraphAddMemsetNode` 不适用于通过流序分配器分配的内存。但是，可以通过流捕获来对这些分配执行 memset 操作。

### 4.3.5.4. 指针属性

`cudaPointerGetAttributes` 查询适用于流序分配。由于流序分配不与上下文关联，查询 `CU_POINTER_ATTRIBUTE_CONTEXT` 会成功，但会在 `*data` 中返回 NULL。属性 `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` 可用于确定分配的位置：在使用 `cudaMemcpyPeerAsync` 进行 p2h2p 复制时，这有助于选择上下文。属性 `CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE` 在 CUDA 11.3 中添加，可用于调试以及在执行 IPC 之前确认分配来自哪个池。

### 4.3.5.5. CPU 虚拟内存

使用 CUDA 流序内存分配器 API 时，请避免使用 "ulimit -v" 设置 VRAM 限制，因为这不被支持。