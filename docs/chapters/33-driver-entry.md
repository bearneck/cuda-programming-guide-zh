# 4.20 驱动入口点访问

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html)

---

此页面是否有帮助？

# 4.20. 驱动程序入口点访问

## 4.20.1. 简介

`驱动程序入口点访问 API` 提供了一种检索 CUDA 驱动程序函数地址的方法。从 CUDA 11.3 开始，用户可以使用从这些 API 获取的函数指针来调用可用的 CUDA 驱动程序 API。

这些 API 提供的功能类似于其在 POSIX 平台上的对应物 `dlsym` 和在 Windows 上的 `GetProcAddress`。提供的 API 将允许用户：

*   使用 CUDA 驱动程序 API 检索驱动程序函数的地址。
*   使用 CUDA 运行时 API 检索驱动程序函数的地址。
*   请求 CUDA 驱动程序函数的每线程默认流版本。更多详情，请参阅 [检索每线程默认流版本](#)。
*   在较旧的工具包上访问新的 CUDA 功能，但需要较新的驱动程序。

## 4.20.2. 驱动程序函数类型定义

为了帮助检索 CUDA 驱动程序 API 入口点，CUDA 工具包提供了对包含所有 CUDA 驱动程序 API 函数指针定义的头文件的访问。这些头文件随 CUDA 工具包一起安装，并可在工具包的 `include/` 目录中找到。下表总结了包含每个 CUDA API 头文件 `typedefs` 的头文件。

| API 头文件 | API 类型定义头文件 |
| --- | --- |
| cuda.h | cudaTypedefs.h |
| cudaGL.h | cudaGLTypedefs.h |
| cudaProfiler.h | cudaProfilerTypedefs.h |
| cudaVDPAU.h | cudaVDPAUTypedefs.h |
| cudaEGL.h | cudaEGLTypedefs.h |
| cudaD3D9.h | cudaD3D9Typedefs.h |
| cudaD3D10.h | cudaD3D10Typedefs.h |
| cudaD3D11.h | cudaD3D11Typedefs.h |

上述头文件本身并不定义实际的函数指针；它们定义了函数指针的类型定义。例如，`cudaTypedefs.h` 为驱动程序 API `cuMemAlloc` 定义了以下类型定义：

```c++
typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v3020)(CUdeviceptr_v2 *dptr, size_t bytesize);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v2000)(CUdeviceptr_v1 *dptr, unsigned int bytesize);
```

CUDA 驱动程序符号有一个基于版本的命名方案，在其名称中带有 `_v*` 扩展名（第一个版本除外）。当特定 CUDA 驱动程序 API 的签名或语义发生变化时，我们会增加相应驱动程序符号的版本号。以 `cuMemAlloc` 驱动程序 API 为例，第一个驱动程序符号名称是 `cuMemAlloc`，下一个符号名称是 `cuMemAlloc_v2`。在 CUDA 2.0 (2000) 中引入的第一个版本的类型定义是 `PFN_cuMemAlloc_v2000`。在 CUDA 3.2 (3020) 中引入的下一个版本的类型定义是 `PFN_cuMemAlloc_v3020`。

这些 `typedefs` 可用于在代码中更轻松地定义适当类型的函数指针：

```c++
PFN_cuMemAlloc_v3020 pfn_cuMemAlloc_v2;
PFN_cuMemAlloc_v2000 pfn_cuMemAlloc_v1;
```

如果用户对 API 的特定版本感兴趣，上述方法是更可取的。此外，头文件为已安装的 CUDA 工具包发布时可用的所有驱动程序符号的最新版本预定义了宏；这些类型定义没有 `_v*` 后缀。对于 CUDA 11.3 工具包，`cuMemAlloc_v2` 是最新版本，因此我们也可以如下定义其函数指针：

```c++
PFN_cuMemAlloc pfn_cuMemAlloc;
```

## 4.20.3.Driver Function Retrieval

Using the Driver Entry Point Access APIs and the appropriate typedef, we can get the function pointer to any CUDA driver API.

### 4.20.3.1.Using the Driver API

The driver API requires CUDA version as an argument to get the ABI compatible version for the requested driver symbol. CUDA Driver APIs have a per-function ABI denoted with a `_v*` extension. For example, consider the versions of `cuStreamBeginCapture` and their corresponding `typedefs` from `cudaTypedefs.h`:

```c++
// cuda.h
CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream);
CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode);

// cudaTypedefs.h
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10000)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10010)(CUstream hStream, CUstreamCaptureMode mode);
```

From the above `typedefs` in the code snippet, version suffixes `_v10000` and `_v10010` indicate that the above APIs were introduced in CUDA 10.0 and CUDA 10.1 respectively.

```c++
#include <cudaTypedefs.h>

// Declare the entry points for cuStreamBeginCapture
PFN_cuStreamBeginCapture_v10000 pfn_cuStreamBeginCapture_v1;
PFN_cuStreamBeginCapture_v10010 pfn_cuStreamBeginCapture_v2;

// Get the function pointer to the cuStreamBeginCapture driver symbol
cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v1, 10000, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
// Get the function pointer to the cuStreamBeginCapture_v2 driver symbol
cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v2, 10010, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
```

Referring to the code snippet above, to retrieve the address to the `_v1` version of the driver API `cuStreamBeginCapture`, the CUDA version argument should be exactly 10.0 (10000). Similarly, the CUDA version for retrieving the address to the `_v2` version of the API should be 10.1 (10010). Specifying a higher CUDA version for retrieving a specific version of a driver API might not always be portable. For example, using 11030 here would still return the `_v2` symbol, but if a hypothetical `_v3` version is released in CUDA 11.3, the `cuGetProcAddress` API would start returning the newer `_v3` symbol instead when paired with a CUDA 11.3 driver. Since the ABI and function signatures of the `_v2` and `_v3` symbols might differ, calling the `_v3` function using the `_v10010` typedef intended for the `_v2` symbol would exhibit undefined behavior.

To retrieve the latest version of a driver API for a given CUDA Toolkit, we can also specify CUDA_VERSION as the `version` argument and use the unversioned typedef to define the function pointer. Since `_v2` is the latest version of the driver API `cuStreamBeginCapture` in CUDA 11.3, the below code snippet shows a different method to retrieve it.

```c++
// Assuming we are using CUDA 11.3 Toolkit

#include <cudaTypedefs.h>

// Declare the entry point
PFN_cuStreamBeginCapture pfn_cuStreamBeginCapture_latest;

// Initialize the entry point. Specifying CUDA_VERSION will give the function pointer to the
// cuStreamBeginCapture_v2 symbol since it is latest version on CUDA 11.3.
cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_latest, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
```
请注意，使用无效的 CUDA 版本请求驱动程序 API 将返回错误 `CUDA_ERROR_NOT_FOUND`。在上述代码示例中，传入小于 10000（CUDA 10.0）的版本将是无效的。

### 4.20.3.2. 使用运行时 API

运行时 API `cudaGetDriverEntryPoint` 使用 CUDA 运行时版本来获取所请求驱动程序符号的 ABI 兼容版本。在下面的代码片段中，所需的最低 CUDA 运行时版本为 CUDA 11.2，因为 `cuMemAllocAsync` 是在该版本中引入的。

```c++
#include <cudaTypedefs.h>

// 声明入口点
PFN_cuMemAllocAsync pfn_cuMemAllocAsync;

// 初始化入口点。假设 CUDA 运行时版本 >= 11.2
cudaGetDriverEntryPoint("cuMemAllocAsync", &pfn_cuMemAllocAsync, cudaEnableDefault, &driverStatus);

// 调用入口点
if(driverStatus == cudaDriverEntryPointSuccess && pfn_cuMemAllocAsync) {
    pfn_cuMemAllocAsync(...);
}
```

运行时 API `cudaGetDriverEntryPointByVersion` 使用用户提供的 CUDA 版本来获取所请求驱动程序符号的 ABI 兼容版本。这允许对请求的 ABI 版本进行更精细的控制。

### 4.20.3.3. 检索每线程默认流版本

某些 CUDA 驱动程序 API 可以配置为具有*默认流*或*每线程默认流*语义。具有*每线程默认流*语义的驱动程序 API 在其名称后缀为 *_ptsz* 或 *_ptds*。例如，`cuLaunchKernel` 有一个名为 `cuLaunchKernel_ptsz` 的*每线程默认流*变体。通过驱动程序入口点访问 API，用户可以请求驱动程序 API `cuLaunchKernel` 的*每线程默认流*版本，而不是*默认流*版本。为 CUDA 驱动程序 API 配置*默认流*或*每线程默认流*语义会影响同步行为。更多详细信息可参见[此处](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream)。

可以通过以下方式之一获取驱动程序 API 的*默认流*或*每线程默认流*版本：

- 使用编译标志 `--default-stream per-thread` 或定义宏 `CUDA_API_PER_THREAD_DEFAULT_STREAM` 以获得每线程默认流行为。
- 分别使用标志 `CU_GET_PROC_ADDRESS_LEGACY_STREAM`/`cudaEnableLegacyStream` 或 `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM`/`cudaEnablePerThreadDefaultStream` 强制使用默认流或每线程默认流行为。

### 4.20.3.4. 访问新的 CUDA 功能

始终建议安装最新的 CUDA 工具包以访问新的 CUDA 驱动程序功能，但如果由于某些原因，用户不想更新或无法访问最新工具包，可以使用此 API 仅通过更新 CUDA 驱动程序来访问新的 CUDA 功能。为了讨论，假设用户正在使用 CUDA 11.3，并希望使用 CUDA 12.0 驱动程序中提供的新驱动程序 API `cuFoo`。以下代码片段说明了此用例：

```c++
int main()
{
    // 假设我们安装了 CUDA 12.0 驱动程序。

    // 手动定义原型，因为 CUDA 11.3 中的 cudaTypedefs.h 没有 cuFoo 的 typedef
    typedef CUresult (CUDAAPI *PFN_cuFoo)(...);
    PFN_cuFoo pfn_cuFoo = NULL;
    CUdriverProcAddressQueryResult driverStatus;

    // 使用 cuGetProcAddress 获取 cuFoo API 的地址。将 CUDA 版本指定为
    // 12000，因为 cuFoo 是在该版本引入的，或者使用 cuDriverGetVersion
    // 动态获取驱动程序版本
    int driverVersion;
    cuDriverGetVersion(&driverVersion);
    CUresult status = cuGetProcAddress("cuFoo", &pfn_cuFoo, driverVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);

    if (status == CUDA_SUCCESS && pfn_cuFoo) {
        pfn_cuFoo(...);
    }
    else {
        printf("Cannot retrieve the address to cuFoo - driverStatus = %d. Check if the latest driver for CUDA 12.0 is installed.\n", driverStatus);
        assert(0);
    }

    // 其余代码在此处

}
```
## 4.20.4. cuGetProcAddress 的潜在影响

以下是一组关于 `cuGetProcAddress` 和 `cudaGetDriverEntryPoint` 潜在问题的具体和理论示例。

### 4.20.4.1. cuGetProcAddress 与隐式链接的影响

`cuDeviceGetUuid` 在 CUDA 9.2 中引入。该 API 在 CUDA 11.4 中引入了一个较新的修订版（`cuDeviceGetUuid_v2`）。为了保持次要版本兼容性，在 CUDA 12.0 之前，`cuDeviceGetUuid` 在 cuda.h 中不会被版本提升为 `cuDeviceGetUuid_v2`。这意味着通过 `cuGetProcAddress` 获取其函数指针并调用它可能会产生不同的行为。直接使用 API 的示例：

```c++
#include <cuda.h>

CUuuid uuid;
CUdevice dev;
CUresult status;

status = cuDeviceGet(&dev, 0); // 获取设备 0
// 处理 status

status = cuDeviceGetUuid(&uuid, dev) // 获取设备 0 的 uuid
```

在此示例中，假设用户使用 CUDA 11.4 进行编译。请注意，这将执行 `cuDeviceGetUuid` 的行为，而不是 _v2 版本。现在是一个使用 `cuGetProcAddress` 的示例：

```c++
#include <cudaTypedefs.h>

CUuuid uuid;
CUdevice dev;
CUresult status;
CUdriverProcAddressQueryResult driverStatus;

status = cuDeviceGet(&dev, 0); // 获取设备 0
// 处理 status

PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid;
status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
    // pfn_cuDeviceGetUuid 指向 ???
}
```

在此示例中，假设用户使用 CUDA 11.4 进行编译。这将获取 `cuDeviceGetUuid_v2` 的函数指针。调用该函数指针将调用新的 _v2 函数，而不是与上一个示例中相同的 `cuDeviceGetUuid`。

### 4.20.4.2. cuGetProcAddress 中编译时版本与运行时版本的使用

让我们以相同的问题为例，做一个小调整。上一个示例使用了编译时常量 `CUDA_VERSION` 来确定要获取哪个函数指针。如果用户使用 `cuDriverGetVersion` 或 `cudaDriverGetVersion` 动态查询驱动程序版本并传递给 `cuGetProcAddress`，则会出现更复杂的情况。示例：

```c++
#include <cudaTypedefs.h>

CUuuid uuid;
CUdevice dev;
CUresult status;
int cudaVersion;
CUdriverProcAddressQueryResult driverStatus;

status = cuDeviceGet(&dev, 0); // 获取设备 0
// 处理 status

status = cuDriverGetVersion(&cudaVersion);
// 处理 status

PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid;
status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
    // pfn_cuDeviceGetUuid 指向 ???
}
```

在此示例中，假设用户使用 CUDA 11.3 进行编译。用户将使用获取 `cuDeviceGetUuid`（而非 _v2 版本）的已知行为来调试、测试和部署此应用程序。由于 CUDA 保证了次要版本之间的 ABI 兼容性，预计同一应用程序在驱动程序升级到 CUDA 11.4 后（无需更新工具包和运行时）无需重新编译即可运行。但这将导致未定义行为，因为 `PFN_cuDeviceGetUuid` 的 typedef 仍将是原始版本的签名，但由于 `cudaVersion` 现在将是 11040（CUDA 11.4），`cuGetProcAddress` 将返回指向 _v2 版本的函数指针，这意味着调用它可能会产生未定义行为。
注意，在这种情况下，原始（非 _v2 版本）的 typedef 看起来像：

```c++
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v9020)(CUuuid *uuid, CUdevice_v1 dev);
```

但 _v2 版本的 typedef 看起来像：

```c++
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v11040)(CUuuid *uuid, CUdevice_v1 dev);
```

因此，在这种情况下，API/ABI 将是相同的，运行时 API 调用可能不会引起问题——仅存在返回未知 uuid 的潜在风险。在 [API/ABI 的影响](#implications-to-api-abi) 部分，我们将讨论一个更成问题的 API/ABI 兼容性案例。

### 4.20.4.3. 带有显式版本检查的 API 版本升级

上面是一个具体的示例。现在，让我们使用一个理论上的示例，该示例在跨驱动程序版本时仍然存在兼容性问题。例如：

```c++
CUresult cuFoo(int bar); // 在 CUDA 11.4 中引入
CUresult cuFoo_v2(int bar); // 在 CUDA 11.5 中引入
CUresult cuFoo_v3(int bar, void* jazz); // 在 CUDA 11.6 中引入

typedef CUresult (CUDAAPI *PFN_cuFoo_v11040)(int bar);
typedef CUresult (CUDAAPI *PFN_cuFoo_v11050)(int bar);
typedef CUresult (CUDAAPI *PFN_cuFoo_v11060)(int bar, void* jazz);
```

请注意，自 CUDA 11.4 中最初创建以来，API 已被修改了两次，而 CUDA 11.6 中的最新版本也修改了函数的 API/ABI 接口。针对 CUDA 11.5 编译的用户代码中的用法是：

```c++
#include <cuda.h>
#include <cudaTypedefs.h>

CUresult status;
int cudaVersion;
CUdriverProcAddressQueryResult driverStatus;

status = cuDriverGetVersion(&cudaVersion);
// 处理状态

PFN_cuFoo_v11040 pfn_cuFoo_v11040;
PFN_cuFoo_v11050 pfn_cuFoo_v11050;
if(cudaVersion < 11050 ) {
    // 我们知道要获取 CUDA 11.4 版本
    status = cuGetProcAddress("cuFoo", &pfn_cuFoo_v11040, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    // 处理状态并验证 pfn_cuFoo_v11040
}
else {
    // 假设 >= CUDA 11.5 版本，我们可以使用第二个版本
    status = cuGetProcAddress("cuFoo", &pfn_cuFoo_v11050, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    // 处理状态并验证 pfn_cuFoo_v11050
}
```

在这个示例中，如果没有为 CUDA 11.6 中的新 typedef 进行更新，并使用这些新的 typedef 和情况处理重新编译应用程序，应用程序将获得 cuFoo_v3 函数指针的返回，并且任何使用该函数的行为都将导致未定义行为。这个示例的目的是说明，即使对 `cuGetProcAddress` 进行显式版本检查，也可能无法安全地覆盖 CUDA 主要版本内的次要版本升级。

### 4.20.4.4. 运行时 API 使用的问题

上述示例主要关注在获取驱动程序 API 函数指针时，驱动程序 API 使用中存在的问题。现在我们将讨论 `cudaApiGetDriverEntryPoint` 的运行时 API 使用中存在的潜在问题。

我们将从使用类似于上述的运行时 API 开始。

```c++
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

CUresult status;
cudaError_t error;
int driverVersion, runtimeVersion;
CUdriverProcAddressQueryResult driverStatus;

// 向运行时请求函数
PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidRuntime;
error = cudaGetDriverEntryPoint ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidRuntime, cudaEnableDefault, &driverStatus);
if(cudaSuccess == error && pfn_cuDeviceGetUuidRuntime) {
    // pfn_cuDeviceGetUuid 指向 ???
}
```
本例中的函数指针比前述仅使用驱动程序的示例更为复杂，因为无法控制获取哪个版本的函数；它总是获取当前 CUDA Runtime 版本的 API。更多信息请参见下表：

|  | 静态运行时版本链接 |  |
| --- | --- | --- |
| 已安装的驱动程序版本 | V11.3 | V11.4 |
| V11.3 | v1 | v1x |
| V11.4 | v1 | v2 |

```text
V11.3 => 11.3 CUDA Runtime 和 Toolkit（包含头文件 cuda.h 和 cudaTypedefs.h）
V11.4 => 11.4 CUDA Runtime 和 Toolkit（包含头文件 cuda.h 和 cudaTypedefs.h）
v1 => cuDeviceGetUuid
v2 => cuDeviceGetUuid_v2

x => 表示 typedef 函数指针与返回的函数指针不匹配。
     在这些情况下，编译时使用 CUDA 11.4 运行时的 typedef 会匹配 _v2 版本，
     但返回的函数指针将是原始（非 _v2）函数。
```

上表中标为 v1x 的情况存在问题，它出现在较新的 CUDA 11.4 Runtime 和 Toolkit 与较旧的驱动程序（CUDA 11.3）组合时。这种组合会导致驱动程序返回指向较旧函数（非 _v2）的指针，但应用程序中使用的 typedef 却是针对新函数指针的。

### 4.20.4.5. 运行时 API 与动态版本控制的问题

当我们考虑应用程序编译所用的 CUDA 版本、CUDA 运行时版本以及应用程序动态链接的 CUDA 驱动程序版本的不同组合时，会出现更多复杂情况。

```c++
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

CUresult status;
cudaError_t error;
int driverVersion, runtimeVersion;
CUdriverProcAddressQueryResult driverStatus;
enum cudaDriverEntryPointQueryResult runtimeStatus;

PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidDriver;
status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuidDriver, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuidDriver) {
    // pfn_cuDeviceGetUuidDriver 指向 ???
}

// 向运行时请求函数
PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidRuntime;
error = cudaGetDriverEntryPoint ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidRuntime, cudaEnableDefault, &runtimeStatus);
if(cudaSuccess == error && pfn_cuDeviceGetUuidRuntime) {
    // pfn_cuDeviceGetUuidRuntime 指向 ???
}

// 根据驱动程序版本（通过运行时获取）向驱动程序请求函数
error = cudaDriverGetVersion(&driverVersion);
PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidDriverDriverVer;
status = cuGetProcAddress ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidDriverDriverVer, driverVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuidDriverDriverVer) {
    // pfn_cuDeviceGetUuidDriverDriverVer 指向 ???
}
```

预期的函数指针矩阵如下：

| 函数指针 | 应用程序编译/运行时动态链接版本/驱动程序版本 |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (3 => CUDA 11.3 且 4 => CUDA 11.4) |  |  |  |  |  |  |  |  |
| 3/3/3 | 3/3/4 | 3/4/3 | 3/4/4 | 4/3/3 | 4/3/4 | 4/4/3 | 4/4/4 |  |
| pfn_cuDeviceGetUuidDriver | t1/v1 | t1/v1 | t1/v1 | t1/v1 | N/A | N/A | t2/v1 | t2/v2 |
| pfn_cuDeviceGetUuidRuntime | t1/v1 | t1/v1 | t1/v1 | t1/v2 | N/A | N/A | t2/v1 | t2/v2 |
| pfn_cuDeviceGetUuidDriverDriverVer | t1/v1 | t1/v2 | t1/v1 | t1/v2 | N/A | N/A | t2/v1 | t2/v2 |

```text
tX -> 编译时使用的 Typedef 版本
vX -> 运行时返回/使用的版本
```

如果应用程序是针对 CUDA 11.3 版本编译的，它将使用原始函数的 typedef；但如果针对 CUDA 11.4 版本编译，它将使用 _v2 函数的 typedef。因此，请注意存在许多 typedef 与实际返回/使用的版本不匹配的情况。

### 4.20.4.6. 允许指定 CUDA 版本的运行时 API 相关问题

除非另有说明，CUDA 运行时 API `cudaGetDriverEntryPointByVersion` 将具有与驱动程序入口点 `cuGetProcAddress` 类似的影响，因为它允许用户请求特定的 CUDA 驱动程序版本。

### 4.20.4.7. 对 API/ABI 的影响

在上述使用 `cuDeviceGetUuid` 的示例中，API 不匹配的影响很小，许多用户可能完全不会注意到，因为添加 _v2 是为了支持多实例 GPU (MIG) 模式。因此，在没有 MIG 的系统上，用户甚至可能意识不到他们获得的是不同的 API。

更成问题的是那些改变了其应用程序签名（从而改变了 ABI）的 API，例如 `cuCtxCreate`。_v2 版本在 CUDA 3.2 中引入，目前在使用 `cuda.h` 时作为默认的 `cuCtxCreate` 使用，但现在 CUDA 11.4 中引入了一个更新的版本 (`cuCtxCreate_v3`)。该 API 的签名也已修改，现在需要额外的参数。因此，在上述某些情况下，函数指针的 typedef 与返回的函数指针不匹配，可能会导致不明显的 ABI 不兼容，从而引发未定义行为。

例如，假设以下代码是针对 CUDA 11.3 工具包编译的，但安装了 CUDA 11.4 驱动程序：

```c++
PFN_cuCtxCreate cuUnknown;
CUdriverProcAddressQueryResult driverStatus;

status = cuGetProcAddress("cuCtxCreate", (void**)&cuUnknown, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && cuUnknown) {
    status = cuUnknown(&ctx, 0, dev);
}
```

当 `cudaVersion` 设置为任何 >=11040 的值（表示 CUDA 11.4）时，运行此代码可能会产生未定义行为，因为没有充分提供 `cuCtxCreate_v3` API 的 _v3 版本所需的所有参数。

## 4.20.5. 确定 cuGetProcAddress 失败原因

cuGetProcAddress 有两种类型的错误。它们是 (1) API/使用错误 和 (2) 无法找到请求的驱动程序 API。第一类错误将通过 CUresult 返回值从 API 返回错误代码。例如，将 NULL 作为 `pfn` 变量传递，或传递无效的 `flags`。
第二种错误类型编码在 `CUdriverProcAddressQueryResult *symbolStatus` 中，可用于帮助区分驱动程序无法找到请求符号的潜在问题。请看以下示例：

```c++
// cuDeviceGetExecAffinitySupport 在 CUDA 11.4 版本中引入
#include <cuda.h>
CUdriverProcAddressQueryResult driverStatus;
cudaVersion = ...;
status = cuGetProcAddress("cuDeviceGetExecAffinitySupport", &pfn, cudaVersion, 0, &driverStatus);
if (CUDA_SUCCESS == status) {
    if (CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT == driverStatus) {
        printf("当您将 cudaVersion 升级到 11.4 时，我们可以使用新功能，但 CUDA 驱动程序已准备就绪！\n");
        // 表示 cudaVersion < 11.4，但运行在 CUDA 驱动程序 >= 11.4 的环境下
    }
    else if (CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND == driverStatus) {
        printf("请将 CUDA 驱动程序和 cudaVersion 至少更新到 11.4 以使用新功能！\n");
        // 表示驱动程序 < 11.4，因为未找到字符串，与 cudaVersion 无关
    }
    else if (CU_GET_PROC_ADDRESS_SUCCESS == driverStatus && pfn) {
        printf("您正在使用 cudaVersion 和 CUDA 驱动程序 >= 11.4，正在使用新功能！\n");
        pfn();
    }
}
```

第一种情况，返回码 `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT` 表示在 CUDA 驱动程序中搜索到了 `symbol`，但该符号是在提供的 `cudaVersion` 之后添加的。在示例中，将 `cudaVersion` 指定为 11030 或更低的任何值，并在 CUDA 驱动程序 >= CUDA 11.4 的环境下运行时，会得到 `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT` 的结果。这是因为 `cuDeviceGetExecAffinitySupport` 是在 CUDA 11.4 (11040) 中添加的。

第二种情况，返回码 `CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND` 表示在 CUDA 驱动程序中未找到 `symbol`。这可能是由于几个原因造成的，例如驱动程序过旧不支持该 CUDA 函数，或者仅仅是拼写错误。对于后者，类似于上一个例子，如果用户将 `symbol` 写为 CUDeviceGetExecAffinitySupport（注意字符串以大写 CU 开头），`cuGetProcAddress` 将无法找到该 API，因为字符串不匹配。在前一种情况下，一个例子可能是用户针对支持新 API 的 CUDA 驱动程序开发应用程序，但部署在较旧的 CUDA 驱动程序上。使用上一个例子，如果开发者针对 CUDA 11.4 或更高版本进行开发，但部署在 CUDA 11.3 驱动程序上，在开发过程中他们可能成功调用了 `cuGetProcAddress`，但当应用程序运行在 CUDA 11.3 驱动程序上时，该调用将不再有效，并在 `driverStatus` 中返回 `CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`。

在本页面