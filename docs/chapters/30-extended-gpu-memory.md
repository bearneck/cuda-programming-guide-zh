# 4.17 扩展 GPU 内存

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html)

---

此页面有帮助吗？

# 4.17. 扩展 GPU 内存

扩展 GPU 内存（EGM）功能利用高带宽的 NVLink-C2C，促进了 GPU 在单节点和多节点系统中高效访问所有系统内存。EGM 适用于集成了 CPU-GPU 的 NVIDIA 系统，它允许分配可以从该设置内的任何 GPU 线程访问的物理内存。EGM 确保所有 GPU 都能以 GPU-GPU NVLink 或 NVLink-C2C 的速度访问其资源。

在此设置中，内存访问通过本地高带宽的 NVLink-C2C 进行。对于远程内存访问，则使用 GPU NVLink，在某些情况下也使用 NVLink-C2C。借助 EGM，GPU 线程获得了通过 NVSwitch 架构访问所有可用内存资源（包括 CPU 附加内存和 HBM3）的能力。

## 4.17.1. 预备知识

在深入探讨 EGM 功能的 API 变更之前，我们将先介绍当前支持的拓扑结构、标识符分配、虚拟内存管理的先决条件以及用于 EGM 的 CUDA 类型。

### 4.17.1.1. EGM 平台：系统拓扑

目前，EGM 可以在以下几种平台中启用：**(1) 单节点，单 GPU**：由一个基于 Arm 的 CPU、CPU 附加内存和一个 GPU 组成。CPU 和 GPU 之间通过高带宽的 C2C（芯片到芯片）互连连接。**(2) 单节点，多 GPU**：由多个基于 ARM 的 CPU（每个都有附加内存）和通过基于 NVLink 的网络连接的多个 GPU 组成。**(3) 多节点，多 GPU**：两个或多个单节点系统，每个系统如上述 (1) 或 (2) 所述，通过基于 NVLink 的网络连接。

使用 `cgroups` 来限制可用设备会阻塞通过 EGM 的路由并导致性能问题。请改用 `CUDA_VISIBLE_DEVICES`。

### 4.17.1.2. 插槽标识符：它们是什么？如何访问它们？

NUMA（非统一内存访问）是一种用于多处理器计算机系统的内存架构，它将内存划分为多个节点。每个节点都有自己的处理器和内存。在这样的系统中，NUMA 将系统划分为节点，并为每个节点分配一个唯一的标识符（numaID）。

EGM 使用操作系统分配的 NUMA 节点标识符。请注意，此标识符与设备的序号不同，并且它与最近的主机节点相关联。除了现有方法外，用户可以通过调用 [cuDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266) 并指定 `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` 属性类型来获取主机节点的标识符（numaID），如下所示：

```cpp
int numaId;
cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, deviceOrdinal);
```

### 4.17.1.3. 分配器与 EGM 支持

将系统内存映射为 EGM 不会引起任何性能问题。实际上，访问映射为 EGM 的远程插槽系统内存将会更快。因为，EGM 流量保证通过 NVLinks 进行路由。目前，`cuMemCreate` 和 `cudaMemPoolCreate` 分配器支持适当的位置类型和 NUMA 标识符。
### 4.17.1.4. 对现有 API 的内存管理扩展

目前，EGM 内存可以通过虚拟内存 (`cuMemCreate`) 或流序内存 (`cudaMemPoolCreate`) 分配器进行映射。用户负责在所有插槽上分配物理内存并将其映射到虚拟内存地址空间。

多节点、多 GPU 平台需要进程间通信。因此，我们建议读者参阅 [第 4.15 章](inter-process-communication.html#interprocess-communication)。

我们建议读者阅读 CUDA 编程指南的 [第 4.16 章](virtual-memory-management.html#virtual-memory-management) 和 [第 4.3 章](stream-ordered-memory-allocation.html#stream-ordered-memory-allocator) 以获得更好的理解。

新的 CUDA 属性类型已添加到 API 中，以允许这些方法使用类似 NUMA 的节点标识符来理解分配位置：

| CUDA 类型 | 与以下一起使用 |
| --- | --- |
| CU_MEM_LOCATION_TYPE_HOST_NUMA | 用于 cuMemCreate 的 CUmemAllocationProp |
| cudaMemLocationTypeHostNuma | 用于 cudaMemPoolCreate 的 cudaMemPoolProps |

请参阅 [CUDA 驱动程序 API](https://www.google.com/url?q=https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html&sa=D&source=editors&ust=1696873412599124&usg=AOvVaw0Ru93Acs_FpJG0gl02BLMX) 和 [CUDA 运行时数据类型](https://www.google.com/url?q=https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html%23group__CUDART__TYPES_1gg2279aa08666f329f3ba4afe397fa60f024dc63fb938dee27b41e3842da35d2d0&sa=D&source=editors&ust=1696873412599344&usg=AOvVaw2O-SyvDt1G37IjcpFzc-4C) 以了解更多关于 NUMA 特定的 CUDA 类型。

## 4.17.2. 使用 EGM 接口

### 4.17.2.1. 单节点，单 GPU

任何现有的 CUDA 主机分配器以及系统分配的内存都可以用来受益于高带宽的 C2C。对用户而言，本地访问就是当前主机分配的方式。

有关内存分配器和页面大小的更多信息，请参阅调优指南。

### 4.17.2.2. 单节点，多 GPU

在多 GPU 系统中，用户必须为主机放置提供信息。如前所述，表达该信息的一种自然方式是使用 NUMA 节点 ID，EGM 遵循这种方法。因此，使用 `cuDeviceGetAttribute` 函数，用户应该能够获知最近的 NUMA 节点 ID。（参见 [插槽标识符：它们是什么？如何访问它们？](#socket-identifiers-what-are-they-how-to-access-them)）。然后，用户可以使用 VMM（虚拟内存管理）API 或 CUDA 内存池来分配和管理 EGM 内存。

#### 4.17.2.2.1. 使用 VMM API

使用虚拟内存管理 API 分配内存的第一步是创建一个物理内存块，该块将为分配提供支持。更多详情请参阅 CUDA 编程指南的 [虚拟内存管理部分](#virtual-memory-management)。在 EGM 分配中，用户必须显式提供 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 作为位置类型，并提供 numaID 作为位置标识符。同样，在 EGM 中，分配必须与平台的适当粒度对齐。以下代码片段展示了使用 `cuMemCreate` 分配物理内存：

```cpp
CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop, MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
CUmemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);
```

After physical memory allocation, we have to reserve an address space and map it to a pointer. These procedures do not have EGM-specific changes:

```cpp
CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);
```

Finally, the user has to explicitly protect mapped virtual address ranges. Otherwise access to the mapped space would result in a crash. Similar to the memory allocation, the user has to provide `CU_MEM_LOCATION_TYPE_HOST_NUMA` as the location type and numaId as the location identifier. Following code snippet create an access descriptors for the host node and the GPU to give read and write access for the mapped memory to both of them:

```cpp
CUmemAccessDesc accessDesc[2]{{}};
accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
accessDesc[0].location.id = numaId;
accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
accessDesc[1].location.id = currentDev;
accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(dptr, size, accessDesc, 2);
```

#### 4.17.2.2.2.Using CUDA Memory Pool

To define EGM, the user can create a memory pool on a node and give access to peers. In this case, the user has to explicitly define `cudaMemLocationTypeHostNuma` as the location type and numaId as the location identifier. The following code snippet shows creating a memory pool `cudaMemPoolCreate`:

```cpp
cudaSetDevice(homeDevice);
cudaMemPoolProps props{};
props.allocType = cudaMemAllocationTypePinned;
props.location.type = cudaMemLocationTypeHostNuma;
props.location.id = numaId;
cudaMemPoolCreate(&memPool, &props);
```

Additionally, for direct connect peer access, it is also possible to use the existing peer access API, `cudaMemPoolSetAccess`. An example for an accessingDevice is shown in the following code snippet:

```cpp
cudaMemAccessDesc desc{};
desc.flags = cudaMemAccessFlagsProtReadWrite;
desc.location.type = cudaMemLocationTypeDevice;
desc.location.id = accessingDevice;
cudaMemPoolSetAccess(memPool, &desc, 1);
```

When the memory pool is created, and accesses are given, the user can set created memory pool to the residentDevice and start allocating memory using `cudaMallocAsync`:

```cpp
cudaDeviceSetMemPool(residentDevice, memPool);
cudaMallocAsync(&ptr, size, memPool, stream);
```

EGM is mapped with 2MB pages. Therefore, users may encounter more TLB misses when accessing very large allocations.

### 4.17.2.3.Multi-Node, Multi-GPU

Beyond memory allocation, remote peer access does not have EGM-specific modification and it follows CUDA inter process (IPC) protocol. See [CUDA Programming Guide](https://www.google.com/url?q=https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%23allocating-physical-memory&sa=D&source=editors&ust=1696873412606850&usg=AOvVaw0IF8bdtDWgRlAiW3tIoyXg) for more details in IPC.
用户应使用 `cuMemCreate` 分配内存，并且同样需要显式地将 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 指定为位置类型，并将 numaID 指定为位置标识符。此外，应将 `CU_MEM_HANDLE_TYPE_FABRIC` 定义为请求的句柄类型。以下代码片段展示了在节点 A 上分配物理内存：

```cpp
CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop,
                              MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
size_t page_size = ...;
assert(padded_size % page_size == 0);
CUmemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);
```

使用 `cuMemCreate` 创建分配句柄后，用户可以通过调用 `cuMemExportToShareableHandle` 将该句柄导出到另一个节点，即节点 B：

```cpp
cuMemExportToShareableHandle(&fabricHandle, allocHandle,
                             CU_MEM_HANDLE_TYPE_FABRIC, 0);
// 此时，fabricHandle 应通过 TCP/IP 发送到节点 B。
```

在节点 B 上，可以使用 `cuMemImportFromShareableHandle` 导入该句柄，并将其视为任何其他 fabric 句柄：

```cpp
// 此时，fabricHandle 应通过 TCP/IP 从节点 A 接收。
CUmemGenericAllocationHandle allocHandle;
cuMemImportFromShareableHandle(&allocHandle, &fabricHandle,
                               CU_MEM_HANDLE_TYPE_FABRIC);
```

当句柄在节点 B 导入后，用户可以以常规方式保留地址空间并在本地进行映射：

```cpp
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop,
                              MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
size_t page_size = ...;
assert(padded_size % page_size == 0);
CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);
```

作为最后一步，用户应为节点 B 上的每个本地 GPU 授予适当的访问权限。以下是一个授予八个本地 GPU 读写权限的示例代码片段：

```cpp
// 授予所有 8 个本地 GPU 对位于节点 A 的导出 EGM 内存的访问权限。                                                               |
CUmemAccessDesc accessDesc[8];
for (int i = 0; i < 8; i++) {
   accessDesc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
   accessDesc[i].location.id = i;
   accessDesc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}
cuMemSetAccess(dptr, size, accessDesc, 8);
```

 在本页