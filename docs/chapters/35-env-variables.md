# 5.2 CUDA 环境变量

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)

---

本页面是否有帮助？

# 5.2. CUDA 环境变量

以下部分列出了 CUDA 环境变量。与多进程服务（MPS）相关的变量记录在 [GPU 部署和管理指南](https://docs.nvidia.com/deploy/mps/index.html#environment-variables) 中。

## 5.2.1. 设备枚举与属性

### 5.2.1.1. CUDA_VISIBLE_DEVICES

此环境变量控制哪些 GPU 设备对 CUDA 应用程序可见，以及它们的枚举顺序。

- 如果未设置此变量，则所有 GPU 设备都可见。
- 如果将此变量设置为空字符串，则没有 GPU 设备可见。

**可能的值**：一个以逗号分隔的 GPU 标识符序列。

GPU 标识符提供方式如下：

- **整数索引**：这些索引对应于系统中 GPU 的序号（由 `nvidia-smi` 确定），从 0 开始。例如，设置 `CUDA_VISIBLE_DEVICES=2,1` 将使设备 0 不可见，并在枚举时先列出设备 2，然后是设备 1。如果遇到无效索引，则只有列表中出现在该无效索引之前的索引对应的设备可见。例如，设置 `CUDA_VISIBLE_DEVICES=0,2,-1,1` 将使设备 0 和 2 可见，而设备 1 不可见，因为它出现在无效索引 `-1` 之后。
- **GPU UUID 字符串**：这些字符串应遵循与 `nvidia-smi -L` 给出的相同格式，例如 `GPU-8932f937-d72c-4106-c12f-20bd9faed9f6`。但是，为了方便起见，允许使用缩写形式；只需指定 GPU UUID 开头足够多的数字，以在目标系统中唯一标识该 GPU。例如，假设系统中没有其他 GPU 共享此前缀，则 `CUDA_VISIBLE_DEVICES=GPU-8932f937` 可能是引用上述 GPU UUID 的有效方式。
- **多实例 GPU（MIG）支持**：`MIG-<GPU-UUID>/<GPU 实例 ID>/<计算实例 ID>`。例如，`MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2`。仅支持单个 MIG 实例枚举。

`cudaGetDeviceCount()` API 返回的设备计数仅包括可见设备，因此使用整数设备标识符的 CUDA API 仅支持范围在 `[0, 可见设备数量 - 1]` 内的序号。GPU 设备的枚举顺序决定了序号值。例如，使用 `CUDA_VISIBLE_DEVICES=2,1` 时，调用 `cudaSetDevice(0)` 将把设备 2 设置为当前设备，因为它首先被枚举并分配序号 0。之后调用 `cudaGetDevice(&device_ordinal)` 也会将 `device_ordinal` 设置为 0，这对应于设备 2。

**示例**：

```bash
nvidia-smi -L # 获取 GPU UUID 列表
CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=GPU-8932f937-d72c-4106-c12f-20bd9faed9f6
CUDA_VISIBLE_DEVICES=MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2
```

---

### 5.2.1.2. CUDA_DEVICE_ORDER

此环境变量控制 CUDA 枚举可用设备的顺序。

**可能的值**：

- `FASTEST_FIRST`：使用简单的启发式方法从最快到最慢枚举可用设备（默认值）。
-   **PCI_BUS_ID**：可用设备按 PCI 总线 ID 升序枚举。PCI 总线 ID 可通过 `nvidia-smi --query-gpu=name,pci.bus_id` 命令获取。

**示例**：

```bash
CUDA_DEVICE_ORDER=FASTEST_FIRST
CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=name,pci.bus_id # 获取 PCI 总线 ID 列表
```

---

### 5.2.1.3. CUDA_MANAGED_FORCE_DEVOCE_ALLOC

此环境变量会改变[统一内存](../02-basics/understanding-memory.html#memory-unified-memory)在多 GPU 系统中的物理存储方式。

**可能取值**：数值，零或非零。

-   **非零值**：强制驱动程序使用设备内存进行物理存储。进程中使用的所有支持托管内存的设备必须支持点对点访问。否则，将返回 `cudaErrorInvalidDevice`。
-   **0**：默认行为。

**示例**：

```bash
CUDA_MANAGED_FORCE_DEVICE_ALLOC=0
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 # 强制使用设备内存
```

---

## 5.2.2. JIT 编译

### 5.2.2.1. CUDA_CACHE_DISABLE

此环境变量控制磁盘上[即时 (JIT) 编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)缓存的行为。禁用 JIT 缓存会强制 CUDA 应用程序在每次执行时都将 PTX 编译为 CUBIN，除非在二进制文件中找到了适用于当前运行架构的 CUBIN 代码。

禁用 JIT 缓存会增加应用程序首次执行时的加载时间。但是，它对于减少应用程序的磁盘空间以及诊断不同驱动程序版本或构建标志之间的差异很有用。

**可能取值**：

-   **1**：禁用 PTX JIT 缓存。
-   **0**：启用 PTX JIT 缓存（默认）。

**示例**：

```bash
CUDA_CACHE_DISABLE=1 # 禁用缓存
CUDA_CACHE_DISABLE=0 # 启用缓存
```

---

### 5.2.2.2. CUDA_CACHE_PATH

此环境变量指定[即时 (JIT) 编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)缓存的目录路径。

**可能取值**：缓存目录的绝对路径（需具有适当的访问权限）。默认值为：

-   在 Windows 上：`%APPDATA%\NVIDIA\ComputeCache`
-   在 Linux 上：`~/.nv/ComputeCache`

**示例**：

```bash
CUDA_CACHE_PATH=~/tmp
```

---

### 5.2.2.3. CUDA_CACHE_MAXSIZE

此环境变量以字节为单位指定[即时 (JIT) 编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)缓存的大小。超过此大小的二进制文件将不会被缓存。如果需要，较旧的二进制文件将从缓存中逐出，以便为新文件腾出空间。

**可能取值**：字节数。默认值为：

-   在桌面/服务器平台上：`1073741824` (1 GiB)
-   在嵌入式平台上：`268435456` (256 MiB)

`4294967296` (4 GiB) 是最大尺寸。

**示例**：

```bash
CUDA_CACHE_MAXSIZE=268435456 # 256 MiB
```

---

### 5.2.2.4. CUDA_FORCE_PTX_JIT 和 CUDA_FORCE_JIT

这些环境变量指示 CUDA 驱动程序忽略应用程序中嵌入的任何 CUBIN，并改为对嵌入的 PTX 代码执行[即时 (JIT) 编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)。
强制进行 JIT 编译会增加应用程序在初始执行时的加载时间。然而，它可以用来验证 PTX 代码是否已嵌入应用程序，以及其即时编译功能是否正常工作。这确保了与未来架构的[前向兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/)。

`CUDA_FORCE_PTX_JIT` 会覆盖 `CUDA_FORCE_JIT`。

**可能的值**：

- 1 : 强制进行 PTX JIT 编译。
- 0 : 默认行为。

**示例**：

```bash
CUDA_FORCE_PTX_JIT=1
```

---

### 5.2.2.5. CUDA_DISABLE_PTX_JIT 和 CUDA_DISABLE_JIT

这些环境变量禁用嵌入式 PTX 代码的[即时 (JIT) 编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)，并使用应用程序中嵌入的兼容 CUBIN。

如果内核没有嵌入二进制代码，或者嵌入的二进制代码是为不兼容的架构编译的，则内核将无法加载。这些环境变量可用于验证应用程序是否为每个内核生成了兼容的 CUBIN 代码。更多详情请参阅[二进制兼容性](../01-introduction/cuda-platform.html#cuda-platform-compute-binary-compatibility)部分。

`CUDA_DISABLE_PTX_JIT` 会覆盖 `CUDA_DISABLE_JIT`。

**可能的值**：

- 1 : 禁用 PTX JIT 编译。
- 0 : 默认行为。

**示例**：

```bash
CUDA_DISABLE_PTX_JIT=1
```

---

### 5.2.2.6. CUDA_FORCE_PRELOAD_LIBRARIES

此环境变量影响 [NVVM](https://docs.nvidia.com/cuda/nvvm-ir-spec/) 和[即时 (JIT) 编译](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation)所需库的预加载。

**可能的值**：

- 1 : 强制驱动程序在初始化期间预加载 NVVM 和即时 (JIT) 编译所需的库。这会增加内存占用和 CUDA 驱动程序初始化所需的时间。设置此环境变量对于避免涉及多线程的某些死锁情况是必要的。
- 0 : 默认行为。

**示例**：

```bash
CUDA_FORCE_PRELOAD_LIBRARIES=1
```

---

## 5.2.3. 执行

### 5.2.3.1. CUDA_LAUNCH_BLOCKING

此环境变量指定是禁用还是启用异步内核启动。

禁用异步执行会导致执行速度变慢，但有助于调试。它强制 GPU 工作从 CPU 的角度同步运行。这使得 CUDA API 错误可以在触发它们的精确 API 调用处被观察到，而不是在稍后的执行过程中。同步执行对调试目的很有用。

**可能的值**：

- 1 : 禁用异步执行。
- 0 : 异步执行（默认）。

**示例**：

```bash
CUDA_LAUNCH_BLOCKING=1
```

---

### 5.2.3.2. CUDA_DEVICE_MAX_CONNECTIONS

此环境变量控制并发计算和复制引擎连接（工作队列）的数量，将两者都设置为指定值。如果独立的 GPU 任务（即从不同 CUDA 流启动的内核或复制操作）映射到同一个工作队列，就会产生虚假依赖，这可能导致 GPU 工作串行化，因为使用了相同的基础资源。为了减少此类虚假依赖的可能性，建议通过此环境变量控制的工作队列数量大于或等于每个上下文中活跃 CUDA 流的数量。
设置此环境变量也会修改复制连接的数量，除非它们已通过 `CUDA_DEVICE_MAX_COPY_CONNECTIONS` 环境变量显式设置。

**可能取值**：`1` 到 `32` 个连接，默认值为 `8`（假设未启用 MPS）

**示例**：

```bash
CUDA_DEVICE_MAX_CONNECTIONS=16
```

---

### 5.2.3.3. CUDA_DEVICE_MAX_COPY_CONNECTIONS

此环境变量控制参与复制操作的并发复制连接（工作队列）的数量。它仅影响[计算能力](compute-capabilities.html#compute-capabilities) 8.0 及以上的设备。

如果同时设置了 `CUDA_DEVICE_MAX_COPY_CONNECTIONS` 和 `CUDA_DEVICE_MAX_CONNECTIONS`，则 `CUDA_DEVICE_MAX_COPY_CONNECTIONS` 会覆盖通过 `CUDA_DEVICE_MAX_CONNECTIONS` 设置的复制连接值。

**可能取值**：`1` 到 `32` 个连接，默认值为 `8`（假设未启用 MPS）

**示例**：

```bash
CUDA_DEVICE_MAX_COPY_CONNECTIONS=16
```

---

### 5.2.3.4. CUDA_SCALE_LAUNCH_QUEUES

此环境变量指定用于启动工作（命令缓冲区）的队列大小的缩放因子，即可以在设备上排队等待的待处理内核或主机/设备复制操作的总数。

**可能取值**：`0.25x`, `0.5x`, `2x`, `4x`

- 任何非 0.25x、0.5x、2x 或 4x 的值都将被解释为 1x。

**示例**：

```bash
CUDA_SCALE_LAUNCH_QUEUES=2x
```

---

### 5.2.3.5. CUDA_GRAPHS_USE_NODE_PRIORITY

此环境变量控制 CUDA 图相对于其从启动它的流中继承的流优先级的执行优先级。

`CUDA_GRAPHS_USE_NODE_PRIORITY` 会覆盖图实例化时的 [cudaGraphInstantiateFlagUseNodePriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1gd4d586536547040944c05249ee26bc62) 标志。

**可能取值**：

- 0 : 继承图被启动到的流的优先级（默认）。
- 1 : 遵循每个节点的启动优先级。CUDA 运行时将节点级优先级视为就绪可运行图节点的调度提示。

**示例**：

```bash
CUDA_GRAPHS_USE_NODE_PRIORITY=1
```

---

### 5.2.3.6. CUDA_DEVICE_WAITS_ON_EXCEPTION

此环境变量控制 CUDA 应用程序在发生异常（错误）时的行为。

启用后，当发生设备端异常时，CUDA 应用程序将暂停并等待，允许调试器（例如 [cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html)）附加以在进程退出或继续之前检查实时的 GPU 状态。

**可能取值**：

- 0 : 默认行为。
- 1 : 当发生设备异常时暂停。

**示例**：

```bash
CUDA_DEVICE_WAITS_ON_EXCEPTION=1
```

---

### 5.2.3.7. CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT

此环境变量控制为[持久访问](../04-special-topics/l2-cache-control.html#l2-set-aside)预留的 GPU L2 缓存的默认"预留"部分，表示为 L2 大小的百分比。

它适用于支持持久 L2 缓存的 GPU，特别是使用 [CUDA 多进程服务 (MPS)](https://docs.nvidia.com/deploy/mps/index.html) 时，计算能力为 8.0 或更高的设备。必须在启动 CUDA MPS 控制守护进程（即在运行 `nvidia-cuda-mps-control -d` 命令之前）设置此环境变量。
**可能取值**：介于 0 到 100 之间的百分比值，默认值为 0。

**示例**：

```bash
CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=25 # 25%
```

---

### 5.2.3.8. CUDA_DISABLE_PERF_BOOST

在 Linux 主机上，将此环境变量设置为 1 可防止提升设备性能状态（pstate），而是可以根据各种启发式方法隐式选择 pstate。此选项可能用于降低功耗，但由于动态性能状态选择，在某些情况下可能导致更高的延迟。

**示例**：

```bash
CUDA_DISABLE_PERF_BOOST=1 # 禁用性能提升，仅限 Linux。
CUDA_DISABLE_PERF_BOOST=0 # 默认行为
```

### 5.2.3.9. CUDA_AUTO_BOOST[[已弃用]]

此环境变量影响 GPU 时钟的“自动提升”行为，即动态时钟提升。它会覆盖 `nvidia-smi` 工具的“自动提升”选项，即 `nvidia-smi --auto-boost-default=0`。

!!! note "注意"
    此环境变量已弃用。强烈建议使用 `nvidia-smi --applications-clocks=<memory,graphics>` 或 NVML API，而不是 `CUDA_AUTO_BOOST` 环境变量。

---

## 5.2.4. 模块加载

### 5.2.4.1. CUDA_MODULE_LOADING

此环境变量影响 CUDA 运行时加载模块的方式，特别是其初始化设备代码的方式。

**可能取值**：

- DEFAULT：默认行为，等同于 LAZY。
- LAZY：特定内核的加载会延迟，直到使用 `cuModuleGetFunction()` 或 `cuKernelGetFunction()` API 调用提取到 CUDA 函数句柄 `CUfunc`。在这种情况下，CUBIN 中的数据会在加载 CUBIN 中的第一个内核或访问 CUBIN 中的第一个变量时加载。驱动程序在首次调用内核时加载所需代码；后续调用不会产生额外开销。这减少了启动时间和 GPU 内存占用。
- EAGER：在程序初始化时完全加载 CUDA 模块和内核。来自 CUBIN、FATBIN 或 PTX 文件的所有内核和数据都会在相应的 `cuModuleLoad*` 和 `cuLibraryLoad*` 驱动程序 API 调用时完全加载。启动时间和 GPU 内存占用更高。内核启动开销是可预测的。

**示例**：

```bash
CUDA_MODULE_LOADING=EAGER
CUDA_MODULE_LOADING=LAZY
```

---

### 5.2.4.2. CUDA_MODULE_DATA_LOADING

此环境变量影响 CUDA 运行时加载与模块关联的数据的方式。

这是对 `CUDA_MODULE_LOADING` 中专注于内核的设置的补充。此环境变量不影响内核的 `LAZY` 或 `EAGER` 加载。如果未设置此环境变量，数据加载行为将从 `CUDA_MODULE_LOADING` 继承。

**可能取值**：

- DEFAULT：默认行为，等同于 LAZY。
- LAZY：模块数据的加载会延迟，直到需要 CUDA 函数句柄 `CUfunc`。在这种情况下，CUBIN 中的数据会在加载 CUBIN 中的第一个内核或访问 CUBIN 中的第一个变量时加载。延迟数据加载可能需要上下文同步，这可能会减慢并发执行速度。
- EAGER：来自 CUBIN、FATBIN 或 PTX 文件的所有数据在相应的 cuModuleLoad* 和 cuLibraryLoad* API 调用时完全加载。

**示例**：

```bash
CUDA_MODULE_DATA_LOADING=EAGER
```

### 5.2.4.3.CUDA_BINARY_LOADER_THREAD_COUNT

设置加载设备二进制文件时要使用的 CPU 线程数。当设置为 0 时，使用的 CPU 线程数将设置为默认值 1。

**可能的值**：

> 要使用的线程整数数量。默认为 0，即使用 1 个线程。

**示例**：

```bash
CUDA_BINARY_LOADER_THREAD_COUNT=4
```

---

## 5.2.5.CUDA 错误日志管理

### 5.2.5.1.CUDA_LOG_FILE

此环境变量指定一个位置，当支持的 CUDA API 调用返回错误时，描述性的错误日志消息将打印到该位置。

例如，如果尝试使用无效的网格配置启动内核，例如 `kernel<<<1, dim3(1,1,128)>>>(...)`，该内核将无法启动，并且 `cudaGetLastError()` 将返回一个通用的 `invalid configuration argument` 错误。如果设置了 `CUDA_LOG_FILE` 环境变量，用户可以在日志中看到以下描述性错误消息：`[CUDA][E] Block Dimensions (1,1,128) include one or more values that exceed the device limit of (1024,1024,64)`，从而轻松确定指定的块 z 维度是无效的。更多详细信息，请参阅[错误日志管理](../04-special-topics/error-log-management.html#error-log-management)。

**可能的值**：`stdout`、`stderr` 或有效的文件路径（具有适当的访问权限）

**示例**：

```bash
CUDA_LOG_FILE=stdout
CUDA_LOG_FILE=/tmp/dbg_cuda_log
```

 在此页面上