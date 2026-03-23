# 4.13 L2 缓存控制

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)

---

此页面是否有帮助？

# 4.13. L2 缓存控制

当 CUDA 内核重复访问全局内存中的某个数据区域时，此类数据访问可视为**持久化**访问。另一方面，如果数据仅被访问一次，则此类数据访问可视为**流式**访问。

计算能力 8.0 及以上的设备能够影响数据在 L2 缓存中的持久性，从而可能为全局内存访问提供更高的带宽和更低的延迟。

此功能主要通过两个 API 提供：

- CUDA 运行时 API（从 CUDA 11.0 开始）提供了对 L2 缓存持久性的编程控制。
- libcu++ 库中的 `cuda::annotated_ptr` API（从 CUDA 11.5 开始）通过内存访问属性注解 CUDA 内核中的指针，以达到类似效果。

以下章节重点介绍 CUDA 运行时 API。有关 `cuda::annotated_ptr` 方法的详细信息，请参阅 [libcu++ 文档](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/annotated_ptr.html)。

## 4.13.1. 为持久化访问预留 L2 缓存

可以预留一部分 L2 缓存，用于全局内存的持久化数据访问。持久化访问优先使用 L2 缓存的这部分预留区域，而对全局内存的普通或流式访问，仅当持久化访问未使用该部分时才能利用它。

用于持久化访问的 L2 缓存预留大小可以在一定范围内调整：

```cuda
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/
```

当 GPU 配置为多实例 GPU (MIG) 模式时，L2 缓存预留功能将被禁用。

使用多进程服务 (MPS) 时，无法通过 `cudaDeviceSetLimit` 更改 L2 缓存预留大小。预留大小只能通过环境变量 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` 在 MPS 服务器启动时指定。

## 4.13.2. 持久化访问的 L2 策略

访问策略窗口指定了全局内存的一个连续区域，以及该区域内访问在 L2 缓存中的持久性属性。

以下代码示例展示了如何使用 CUDA 流设置 L2 持久化访问窗口。

**CUDA 流示例**

```cuda
cudaStreamAttrValue stream_attribute;                                         // 流级别属性数据结构
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // 全局内存数据指针
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // 持久化访问的字节数。
                                                                              // (必须小于 cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // 缓存命中率的提示
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // 缓存命中时的访问属性类型
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // 缓存未命中时的访问属性类型

// 将属性设置到类型为 cudaStream_t 的 CUDA 流
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```
当内核随后在 CUDA `stream` 中执行时，对全局内存范围 `[ptr..ptr+num_bytes)` 的内存访问比访问其他全局内存位置更有可能持久保留在 L2 缓存中。

L2 持久性也可以为 CUDA 图内核节点设置，如下例所示：

**CUDA GraphKernelNode 示例**

```cuda
cudaKernelNodeAttrValue node_attribute;                                     // 内核级别属性数据结构
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // 全局内存数据指针
node_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // 持久性访问的字节数。
                                                                            // (必须小于 cudaDeviceProp::accessPolicyMaxWindowSize)
node_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // 缓存命中率的提示
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // 缓存命中时的访问属性类型
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // 缓存未命中时的访问属性类型

// 将属性设置为类型为 cudaGraphNode_t 的 CUDA 图内核节点
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
```

`hitRatio` 参数可用于指定接收 `hitProp` 属性的访问比例。在上述两个示例中，全局内存区域 `[ptr..ptr+num_bytes)` 中 60% 的内存访问具有持久性属性，40% 的内存访问具有流式属性。哪些特定的内存访问被归类为持久性（`hitProp`）是随机的，概率约为 `hitRatio`；概率分布取决于硬件架构和内存范围。

例如，如果 L2 预留缓存大小为 16KB，且 `accessPolicyWindow` 中的 `num_bytes` 为 32KB：

- 当 hitRatio 为 0.5 时，硬件将随机选择 32KB 窗口中的 16KB 指定为持久性，并缓存在预留的 L2 缓存区域中。
- 当 hitRatio 为 1.0 时，硬件将尝试将整个 32KB 窗口缓存在预留的 L2 缓存区域中。由于预留区域小于窗口，缓存行将被逐出，以将 32KB 数据中最近使用的 16KB 保留在 L2 缓存的预留部分中。

因此，`hitRatio` 可用于避免缓存行的抖动，并总体上减少移入和移出 L2 缓存的数据量。

低于 1.0 的 `hitRatio` 值可用于手动控制来自并发 CUDA 流的不同 `accessPolicyWindow` 可以在 L2 中缓存的数据量。例如，假设 L2 预留缓存大小为 16KB；两个不同 CUDA 流中的两个并发内核，每个都有一个 16KB 的 `accessPolicyWindow`，并且两者的 `hitRatio` 值都为 1.0，在竞争共享的 L2 资源时，可能会相互逐出对方的缓存行。但是，如果两个 `accessPolicyWindow` 的 hitRatio 值都为 0.5，它们逐出自己或对方持久性缓存行的可能性就会降低。
## 4.13.3. L2 访问属性

为不同的全局内存数据访问定义了三种访问属性：

1.  **cudaAccessPropertyStreaming**：具有流式属性的内存访问不太可能在 L2 缓存中持久保留，因为这些访问会被优先逐出。
2.  **cudaAccessPropertyPersisting**：具有持久属性的内存访问更可能在 L2 缓存中持久保留，因为这些访问会被优先保留在 L2 缓存的预留部分。
3.  **cudaAccessPropertyNormal**：此访问属性强制将先前应用的持久访问属性重置为正常状态。来自先前 CUDA 内核的、具有持久属性的内存访问，可能会在其预期使用之后很长时间内仍保留在 L2 缓存中。这种使用后的持久性会减少可供后续不使用持久属性的内核使用的 L2 缓存量。使用 `cudaAccessPropertyNormal` 属性重置访问属性窗口，将移除先前访问的持久（优先保留）状态，就好像先前的访问没有设置访问属性一样。

## 4.13.4. L2 持久性示例

以下示例展示了如何为持久访问预留 L2 缓存，如何通过 CUDA 流在 CUDA 内核中使用预留的 L2 缓存，然后重置 L2 缓存。

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);                                                                  // 创建 CUDA 流

cudaDeviceProp prop;                                                                        // CUDA 设备属性变量
cudaGetDeviceProperties( &prop, device_id);                                                 // 查询 GPU 属性
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                                  // 为持久访问预留 3/4 的 L2 缓存或允许的最大值

size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // 选择用户定义的 num_bytes 和最大窗口大小的较小值。

cudaStreamAttrValue stream_attribute;                                                       // 流级别属性数据结构
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);               // 全局内存数据指针
stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // 持久访问的字节数
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                                        // 缓存命中率的提示
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // 持久属性
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // 缓存未命中时的访问属性类型

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // 将属性设置到 CUDA 流

for(int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);                                 // 此 data1 被内核多次使用
}                                                                                           // [data1 + num_bytes) 受益于 L2 持久性
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);                                     // 同一流中的不同内核也可以受益于 data1 的持久性

stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // 将窗口大小设置为 0 以禁用它
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // 覆盖 CUDA 流的访问策略属性
cudaCtxResetPersistingL2Cache();                                                            // 移除 L2 中的任何持久行

cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                     // data2 现在可以在正常模式下受益于完整的 L2 缓存
```
## 4.13.5. 将 L2 访问重置为普通模式

先前 CUDA 内核的持久化 L2 缓存行可能在它被使用后很长时间内仍然驻留在 L2 中。因此，将 L2 缓存重置为普通模式对于流式或普通内存访问以正常优先级利用 L2 缓存非常重要。有三种方法可以将持久化访问重置为普通状态。

1.  使用访问属性 `cudaAccessPropertyNormal` 重置先前的持久化内存区域。
2.  通过调用 `cudaCtxResetPersistingL2Cache()` 将所有持久化 L2 缓存行重置为普通模式。
3.  最终，未被触及的缓存行会自动重置为普通模式。**强烈不建议**依赖自动重置，因为自动重置发生所需的时间长度是不确定的。

## 4.13.6. 管理预留 L2 缓存的利用率

在不同 CUDA 流中并发执行的多个 CUDA 内核可能为其流分配了不同的访问策略窗口。然而，预留的 L2 缓存部分是在所有这些并发 CUDA 内核之间共享的。因此，这部分预留缓存的净利用率是所有并发内核各自使用量的总和。当持久化访问的数量超过预留的 L2 缓存容量时，将内存访问指定为持久化的好处就会减少。

为了管理预留 L2 缓存部分的利用率，应用程序必须考虑以下因素：

-   预留 L2 缓存的大小。
-   可能并发执行的 CUDA 内核。
-   所有可能并发执行的 CUDA 内核的访问策略窗口。
-   何时以及如何需要重置 L2，以允许普通或流式访问以同等优先级利用先前预留的 L2 缓存。

## 4.13.7. 查询 L2 缓存属性

与 L2 缓存相关的属性是 `cudaDeviceProp` 结构体的一部分，可以使用 CUDA 运行时 API `cudaGetDeviceProperties` 进行查询。

CUDA 设备属性包括：

-   `l2CacheSize`：GPU 上可用的 L2 缓存总量。
-   `persistingL2CacheMaxSize`：可以为持久化内存访问预留的最大 L2 缓存量。
-   `accessPolicyMaxWindowSize`：访问策略窗口的最大大小。

## 4.13.8. 控制用于持久化内存访问的 L2 缓存预留大小

用于持久化内存访问的 L2 预留缓存大小使用 CUDA 运行时 API `cudaDeviceGetLimit` 进行查询，并使用 CUDA 运行时 API `cudaDeviceSetLimit` 作为 `cudaLimit` 进行设置。设置此限制的最大值是 `cudaDeviceProp::persistingL2CacheMaxSize`。

```cuda
enum cudaLimit {
    /* 其他字段未显示 */
    cudaLimitPersistingL2CacheSize
};
```