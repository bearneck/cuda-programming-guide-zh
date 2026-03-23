# 5.4 C/C++ 语言扩展

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)

---

此页面是否有帮助？

# 5.4. C/C++ 语言扩展

## 5.4.1. 函数和变量注解

### 5.4.1.1. 执行空间说明符

执行空间说明符 `__host__`、`__device__` 和 `__global__` 用于指示函数是在主机上执行还是在设备上执行。

| 执行空间说明符 | 在何处执行 | 可从何处调用 |  |  |
| --- | --- | --- | --- | --- |
|  | 主机 | 设备 | 主机 | 设备 |
| __host__ , 无说明符 | ✔ | ✘ | ✔ | ✘ |
| __device__ | ✘ | ✔ | ✘ | ✔ |
| __global__ | ✘ | ✔ | ✔ | ✔ |
| __host__ __device__ | ✔ | ✔ | ✔ | ✔ |

---

`__global__` 函数的限制：

- 必须返回 `void`。
- 不能是类、结构体或联合体的成员。
- 需要如内核配置中所述的执行配置。
- 不支持递归。
- 有关其他限制，请参阅 __global__ 函数参数。

对 `__global__` 函数的调用是异步的。它们在设备完成执行之前就返回到主机线程。

---

用 `__host__ __device__` 声明的函数会同时为主机和设备编译。可以使用 `__CUDA_ARCH__`[宏](#cuda-arch-macro)来区分主机和设备代码路径：

```cuda
__host__ __device__ void func() {
#if defined(__CUDA_ARCH__)
    // 设备代码路径
#else
    // 主机代码路径
#endif
}
```

### 5.4.1.2. 内存空间说明符

内存空间说明符 `__device__`、`__managed__`、`__constant__` 和 `__shared__` 用于指示变量在设备上的存储位置。

下表总结了内存空间的属性：

| 内存空间说明符 | 位置 | 可访问者 | 生命周期 | 唯一实例 |
| --- | --- | --- | --- | --- |
| __device__ | 设备全局内存 | 设备线程（网格）/ CUDA 运行时 API | 程序/ CUDA 上下文 | 每个设备 |
| __constant__ | 设备常量内存 | 设备线程（网格）/ CUDA 运行时 API | 程序/ CUDA 上下文 | 每个设备 |
| __managed__ | 主机和设备（自动） | 主机/设备线程 | 程序 | 每个程序 |
| __shared__ | 设备（流式多处理器） | 线程块线程 | 线程块 | 线程块 |
| 无说明符 | 设备（寄存器） | 单个线程 | 单个线程 | 单个线程 |

---

- 可以使用 CUDA 运行时 API 函数 `cudaGetSymbolAddress()`、`cudaGetSymbolSize()`、`cudaMemcpyToSymbol()` 和 `cudaMemcpyFromSymbol()` 从主机访问 `__device__` 和 `__constant__` 变量。
- `__constant__` 变量在设备代码中是只读的，只能使用 CUDA 运行时 API 从主机进行修改。

以下示例说明了如何使用这些 API：

```cuda
__device__   float device_var       = 4.0f; // 设备内存中的变量
__constant__ float constant_mem_var = 4.0f; // 常量内存中的变量
                                            // 为便于阅读，以下示例主要关注设备变量。
int main() {
    float* device_ptr;
    cudaGetSymbolAddress((void**) &device_ptr, device_var);        // 获取 device_var 的地址

    size_t symbol_size;
    cudaGetSymbolSize(&symbol_size, device_var);                   // 检索符号的大小（4 字节）。

    float host_var;
    cudaMemcpyFromSymbol(&host_var, device_var, sizeof(host_var)); // 从设备复制到主机。

    host_var = 3.0f;
    cudaMemcpyToSymbol(device_var, &host_var, sizeof(host_var));   // 从主机复制到设备。
}
```
请参阅 [Compiler Explorer](https://godbolt.org/z/vYjP8GGv3) 上的示例。

#### 5.4.1.2.1. __shared__ 内存

`__shared__` 内存变量可以具有静态大小（在编译时确定）或动态大小（在内核启动时确定）。有关在运行时指定共享内存大小的详细信息，请参阅[内核配置](#execution-configuration)部分。

共享内存约束：

*   具有动态大小的变量必须声明为外部数组或指针。
*   具有静态大小的变量不能在其声明中进行初始化。

以下示例说明了如何声明和确定 `__shared__` 变量的大小：

```cuda
extern __shared__ char dynamic_smem_pointer[];
// extern __shared__ char* dynamic_smem_pointer; 替代语法

__global__ void kernel() { // 或 __device__ 函数
    __shared__ int smem_var1[4];                  // 静态大小
    auto smem_var2 = (int*) dynamic_smem_pointer; // 动态大小
}

int main() {
    size_t shared_memory_size = 16;
    kernel<<<1, 1, shared_memory_size>>>();
    cudaDeviceSynchronize();
}
```

请参阅 [Compiler Explorer](https://godbolt.org/z/nPjvd1frb) 上的示例。

#### 5.4.1.2.2. __managed__ 内存

`__managed__` 变量具有以下限制：

*   __managed__ 变量的地址不是常量表达式。
*   __managed__ 变量不得具有引用类型 T&。
*   当 CUDA 运行时可能处于无效状态时，不得使用 __managed__ 变量的地址或值，包括以下情况：在具有静态或 thread_local 存储持续时间的对象的静态/动态初始化或析构期间。在调用 exit() 后执行的代码中。例如，标记为 `__attribute__((destructor))` 的函数。在 CUDA 运行时可能未初始化的代码中。例如，标记为 `__attribute__((constructor))` 的函数。
*   __managed__ 变量不能用作 decltype() 表达式的未加括号的 id-expression 参数。
*   __managed__ 变量具有与动态分配的托管内存相同的连贯性和一致性行为。
*   另请参阅局部变量的限制。

以下是 `__managed__` 变量的合法和非法使用示例：

```cuda
#include <cassert>

__device__ __managed__ int global_var = 10; // 正确

int* ptr = &global_var;                     // 错误：在静态初始化中使用托管变量

struct MyStruct1 {
    int field;
    MyStruct1() : field(global_var) {};
};

struct MyStruct2 {
    ~MyStruct2() { global_var = 10; }
};

MyStruct1 temp1; // 错误：在动态初始化中使用托管变量

MyStruct2 temp2; // 错误：在具有静态存储持续时间的对象的析构函数中使用托管变量

__device__ __managed__ const int const_var = 10;         // 错误：const 限定类型

__device__ __managed__ int&      reference = global_var; // 错误：引用类型

template <int* Addr>
struct MyStruct3 {};

MyStruct3<&global_var> temp;     // 错误：托管变量的地址不是常量表达式

__global__ void kernel(int* ptr) {
    assert(ptr == &global_var);  // 正确
    global_var = 20;             // 正确
}

int main() {
    int* ptr = &global_var;      // 正确
    kernel<<<1, 1>>>(ptr);
    cudaDeviceSynchronize();
    global_var++;                // 正确
    decltype(global_var) var1;   // 错误：托管变量用作 decltype 的未加括号参数

    decltype((global_var)) var2; // 正确
}
```
### 5.4.1.3. 内联限定符

以下限定符可用于控制 `__host__` 和 `__device__` 函数的内联行为：

- __noinline__ : 指示 nvcc 不要内联该函数。
- __forceinline__ : 强制 nvcc 在单个翻译单元内内联该函数。
- __inline_hint__ : 在使用链接时优化时，启用跨翻译单元的激进内联。

这些限定符是互斥的。

### 5.4.1.4. __restrict__ 指针

`nvcc` 通过 `__restrict__` 关键字支持受限指针。

当两个或多个指针指向重叠的内存区域时，就会发生指针别名。这会抑制代码重排和公共子表达式消除等优化。

受限指针是程序员做出的承诺，即在指针的生命周期内，其指向的内存将仅通过该指针访问。这使得编译器能够执行更激进的优化。

- 访问设备函数的所有线程仅从中读取；或者
- 最多只有一个线程向其写入，且没有其他线程从中读取。

以下示例说明了一个别名问题，并演示了使用受限指针如何帮助编译器减少指令数量：

```cuda
__device__
void device_function(const float* a, const float* b, float* c) {
    c[0] = a[0] * b[0];
    c[1] = a[0] * b[0];
    c[2] = a[0] * b[0] * a[1];
    c[3] = a[0] * a[1];
    c[4] = a[0] * b[0];
    c[5] = b[0];
    ...
}
```

因为指针 `a`、`b` 和 `c` 可能存在别名，所以通过 `c` 的任何写入都可能修改 `a` 或 `b` 的元素。为了保证功能正确性，编译器不能将 `a[0]` 和 `b[0]` 加载到寄存器中，将它们相乘，然后将结果同时存储在 `c[0]` 和 `c[1]` 中。这是因为如果 `a[0]` 和 `c[0]` 位于同一位置，结果将与抽象执行模型不同。编译器无法利用公共子表达式。同样，编译器不能将 `c[4]` 的计算与 `c[0]` 和 `c[1]` 的计算重新排序，因为对 `c[3]` 的先前写入可能会改变 `c[4]` 计算的输入。

通过将 `a`、`b` 和 `c` 声明为受限指针，程序员告知编译器这些指针不存在别名。这意味着写入 `c` 永远不会覆盖 `a` 或 `b` 的元素。这将函数原型更改如下：

```cuda
__device__
void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
```

请注意，为了使编译器优化器生效，所有指针参数都必须是受限的。添加 `__restrict__` 关键字后，编译器可以在保持与抽象执行模型功能相同的同时，随意重新排序并执行公共子表达式消除。

```cuda
__device__
void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
    float t0 = a[0];
    float t1 = b[0];
    float t2 = t0 * t1;
    float t3 = a[1];
    c[0]     = t2;
    c[1]     = t2;
    c[4]     = t2;
    c[2]     = t2 * t3;
    c[3]     = t0 * t3;
    c[5]     = t1;
    ...
}
```
请参见 [Compiler Explorer](https://godbolt.org/z/6KeTqarnW) 上的示例。

其结果是减少了内存访问和计算量，但代价是寄存器压力因缓存加载和寄存器中的公共子表达式而增加。

由于寄存器压力是许多 CUDA 代码中的关键问题，使用限制指针可能会降低占用率，从而对性能产生负面影响。

---

对标记了 `__restrict__` 的 `__global__` 函数 `const` 指针的访问，会被编译为只读缓存加载，类似于 [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc) `ld.global.nc` 指令或 `__ldg()` [低级加载和存储函数](#low-level-load-store-functions)。

```cuda
__global__
void kernel1(const float* in, float* out) {
    *out = *in; // PTX: ld.global
}

__global__
void kernel2(const float* __restrict__ in, float* out) {
    *out = *in;  // PTX: ld.global.nc
}
```

请参见 [Compiler Explorer](https://godbolt.org/z/drsTEPa8s) 上的示例。

### 5.4.1.5. __grid_constant__ 参数

使用 `__grid_constant__` 标注 `__global__` 函数参数，可以防止编译器为该参数创建每个线程的副本。相反，网格中的所有线程都将通过单个地址访问该参数，这可以提高性能。

`__grid_constant__` 参数具有以下属性：

-   其生命周期与内核相同。
-   它对单个内核是私有的，这意味着其他网格（包括子网格）的线程无法访问该对象。
-   内核中的所有线程看到的地址相同。
-   它是只读的。修改 `__grid_constant__` 对象或其任何子对象（包括可变成员）是未定义行为。

要求：

-   使用 `__grid_constant__` 标注的内核参数必须具有 `const` 限定的非引用类型。
-   所有函数声明必须与任何 `__grid_constant__` 参数保持一致。
-   函数模板特化必须与主模板声明在 `__grid_constant__` 参数方面匹配。
-   函数模板实例化也必须与主模板声明在 `__grid_constant__` 参数方面匹配。

示例：

```cuda
struct MyStruct {
    int         x;
    mutable int y;
};

__device__ void external_function(const MyStruct&);

__global__ void kernel(const __grid_constant__ MyStruct s) {
    // s.x++; // 编译错误：试图修改只读内存
    // s.y++; // 未定义行为：试图修改只读内存

    // 编译器将不会为 "s" 创建每个线程的本地副本：
    external_function(s);
}
```

请参见 [Compiler Explorer](https://godbolt.org/z/Goq9jrEeo) 上的示例。

### 5.4.1.6. 标注摘要

下表总结了 CUDA 标注，并说明了每个标注适用于哪个执行空间以及在何处有效。

| 标注 | __host__ / __device__ / __host__ __device__ | __global__ |
| --- | --- | --- |
| __noinline__ , __forceinline__ , __inline_hint__ | 函数 | â |
| __restrict__ | 指针参数 | 指针参数 |
| __grid_constant__ | â | 参数 |
| __launch_bounds__ | â | 函数 |
| __maxnreg__ | â | 函数 |
| __cluster_dims__ | â | 函数 |

## 5.4.2. 内置类型和变量

### 5.4.2.1. 主机编译器类型扩展

只要主机编译器支持，CUDA 允许使用非标准算术类型。支持以下类型：

- 128位整数类型 __int128 。当主机编译器定义了 __SIZEOF_INT128__ 宏时，在 Linux 上支持。
- 128位浮点类型 __float128 和 _Float128 在计算能力 10.0 及更高版本的 GPU 设备上可用。__float128 类型的常量表达式可能会被编译器以较低精度的浮点表示形式处理。当主机编译器定义了 __SIZEOF_FLOAT128__ 或 __FLOAT128__ 宏时，在 Linux x86 上支持。
- _Complex 类型仅在主机代码中受支持。

### 5.4.2.2. 内置变量

用于指定和获取沿 x、y 和 z 维度的线程网格和线程块的内核配置的值为 `dim3` 类型。用于获取块索引和线程索引的变量为 `uint3` 类型。`dim3` 和 `uint3` 都是简单的结构体，由三个名为 `x`、`y` 和 `z` 的无符号值组成。在 C++11 及更高版本中，`dim3` 所有分量的默认值为 1。

仅限设备的内置变量：

- dim3 gridDim ：包含线程网格的维度，即沿 x、y 和 z 维度的线程块数量。
- dim3 blockDim ：包含线程块的维度，即沿 x、y 和 z 维度的线程数量。
- uint3 blockIdx ：包含线程网格内的块索引，沿 x、y 和 z 维度。
- uint3 threadIdx ：包含线程块内的线程索引，沿 x、y 和 z 维度。
- int warpSize ：一个运行时值，定义为线程束中的线程数，通常为 32。有关线程束的定义，另请参见 Warps and SIMT。

### 5.4.2.3. 内置类型

CUDA 提供了从基本整数和浮点类型派生的向量类型，这些类型在主机和设备上都受支持。下表显示了可用的向量类型。

| C++ 基础类型 | 向量 X1 | 向量 X2 | 向量 X3 | 向量 X4 |
| --- | --- | --- | --- | --- |
| signed char | char1 | char2 | char3 | char4 |
| unsigned char | uchar1 | uchar2 | uchar3 | uchar4 |
| signed short | short1 | short2 | short3 | short4 |
| unsigned short | ushort1 | ushort2 | ushort3 | ushort4 |
| signed int | int1 | int2 | int3 | int4 |
| unsigned | uint1 | uint2 | uint3 | uint4 |
| signed long | long1 | long2 | long3 | long4_16a/long4_32a |
| unsigned long | ulong1 | ulong2 | ulong3 | ulong4_16a/ulong4_32a |
| signed long long | longlong1 | longlong2 | longlong3 | longlong4_16a/longlong4_32a |
| unsigned long long | ulonglong1 | ulonglong2 | ulonglong3 | ulonglong4_16a/ulonglong4_32a |
| float | float1 | float2 | float3 | float4 |
| double | double1 | double2 | double3 | double4_16a/double4_32a |

请注意，`long4`、`ulong4`、`longlong4`、`ulonglong4` 和 `double4` 已在 CUDA 13 中弃用，并可能在未来的版本中移除。

---

下表详细说明了向量类型的字节大小和对齐要求：

| 类型 | 大小 | 对齐 |
| --- | --- | --- |
| char1 , uchar1 | 1 | 1 |
| char2 , uchar2 | 2 | 2 |
| char3 , uchar3 | 3 | 1 |
| char4 , uchar4 | 4 | 4 |
| short1 , ushort1 | 2 | 2 |
| short2 , ushort2 | 4 | 4 |
| short3 , ushort3 | 6 | 2 |
| short4 , ushort4 | 8 | 8 |
| int1 , uint1 | 4 | 4 |
| int2 , uint2 | 8 | 8 |
| int3 , uint3 | 12 | 4 |
| int4 , uint4 | 16 | 16 |
| long1 , ulong1 | 4/8 * | 4/8 * |
| long2 , ulong2 | 8/16 * | 8/16 * |
| long3 , ulong3 | 12/24 * | 4/8 * |
| long4 , ulong4 (已弃用) | 16/32 * | 16 * |
| long4_16a , ulong4_16a | 16/32 * | 16 |
| long4_32a , ulong4_32a | 16/32 * | 32 |
| longlong1 , ulonglong1 | 8 | 8 |
| longlong2 , ulonglong2 | 16 | 16 |
| longlong3 , ulonglong3 | 24 | 8 |
| longlong4 , ulonglong4 (已弃用) | 32 | 16 |
| longlong4_16a , ulonglong4_16a | 32 | 16 |
| longlong4_32a , ulonglong4_32a | 32 | 32 |
| float1 | 4 | 4 |
| float2 | 8 | 8 |
| float3 | 12 | 4 |
| float4 | 16 | 16 |
| double1 | 8 | 8 |
| double2 | 16 | 16 |
| double3 | 24 | 8 |
| double4 (已弃用) | 32 | 16 |
| double4_16a | 32 | 16 |
| double4_32a | 32 | 32 |

*****`long` 在 C++ LLP64 数据模型（Windows 64 位）中为 4 字节，而在 C++ LP64 数据模型（Linux 64 位）中为 8 字节。

---

向量类型是结构体。它们的第一个、第二个、第三个和第四个分量分别可以通过 `x`、`y`、`z` 和 `w` 字段访问。

```cuda
int sum(int4 value) {
    return value.x + value.y + value.z + value.w;
}
```

它们都有一个形式为 `make_<type_name>()` 的工厂函数；例如：

```cuda
int4 add_one(int x, int y, int z, int w) {
    return make_int4(x + 1, y + 1, z + 1, w + 1);
}
```

如果主机代码不是用 `nvcc` 编译的，可以通过包含 CUDA 工具包中提供的 `cuda_runtime.h` 头文件来导入向量类型和相关函数。

## 5.4.3. 内核配置

任何对 `__global__` 函数的调用都必须为该调用指定一个*执行配置*。此执行配置定义了将在设备上执行该函数时使用的线程网格和线程块的维度，以及相关的[流](../02-basics/asynchronous-execution.html#cuda-streams)。

执行配置通过在函数名和括号内的参数列表之间插入形式为 `<<<grid_dim, block_dim, dynamic_smem_bytes, stream>>>` 的表达式来指定，其中：

- grid_dim 是 dim3 类型，指定了线程网格的维度和大小，使得 grid_dim.x * grid_dim.y * grid_dim.z 等于要启动的线程块数量；
- block_dim 是 dim3 类型，指定了每个线程块的维度和大小，使得 block_dim.x * block_dim.y * block_dim.z 等于每个线程块中的线程数；
- `dynamic_smem_bytes` 是一个可选的 `size_t` 类型参数，默认值为零。它指定了在此次调用中，除了静态分配的内存外，每个线程块动态分配的共享内存字节数。此内存用于 `extern __shared__` 数组（参见 [__shared__ 内存](__shared__-memory.html)）。
- `stream` 是 `cudaStream_t`（指针）类型，用于指定关联的流。`stream` 是一个可选参数，默认为 `NULL`。

以下示例展示了一个内核函数声明及其调用：

```cuda
__global__ void kernel(float* parameter);

kernel<<<grid_dim, block_dim, dynamic_smem_bytes>>>(parameter);
```

执行配置的参数在实际函数的参数之前被求值。

如果 `grid_dim` 或 `block_dim` 超过了设备允许的最大尺寸（如 [计算能力](compute-capabilities.html#compute-capabilities) 中所规定），或者 `dynamic_smem_bytes` 在考虑静态分配的内存后大于可用的共享内存，则函数调用将失败。

### 5.4.3.1. 线程块簇

计算能力 9.0 及更高版本允许用户指定编译时线程块簇维度，以便内核可以使用 CUDA 中的 [簇层次结构](../02-basics/intro-to-cuda-cpp.html#thread-block-clusters)。编译时簇维度可以使用 `__cluster_dims__` 属性指定，语法如下：`__cluster_dims__([x, [y, [z]]])`。下面的示例展示了在 X 维度上编译时簇大小为 2，在 Y 和 Z 维度上为 1。

```cuda
__global__ void __cluster_dims__(2, 1, 1) kernel(float* parameter);
```

`__cluster_dims__()` 的默认形式指定内核将作为网格簇启动。如果未指定簇维度，用户可以在启动时指定。若在启动时也未指定维度，将导致启动时错误。

线程块簇的维度也可以在运行时指定，并且可以使用 `cudaLaunchKernelEx` API 启动带有簇的内核。此 API 接受一个 `cudaLaunchConfig_t` 类型的配置参数、一个内核函数指针以及内核参数。以下示例展示了运行时内核配置。

```cuda
__global__ void kernel(float parameter1, int parameter2) {}

int main() {
    cudaLaunchConfig_t config = {0};
    // 网格维度不受簇启动的影响，仍然使用线程块的数量来枚举。
    // 网格维度应是簇大小的倍数。
    config.gridDim          = dim3{4};  // 4 个线程块
    config.blockDim         = dim3{32}; // 每个线程块 32 个线程
    config.dynamicSmemBytes = 1024;     // 1 KB

    cudaLaunchAttribute attribute[1];
    attribute[0].id               = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2; // X 维度的簇大小
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs    = attribute;
    config.numAttrs = 1;

    float parameter1 = 3.0f;
    int   parameter2 = 4;
    cudaLaunchKernelEx(&config, kernel, parameter1, parameter2);
}
```
请查看 [Compiler Explorer](https://cuda.godbolt.org/z/M67r3a5zM) 上的示例。

### 5.4.3.2. 启动边界

正如在[内核启动与占用率](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-kernel-launch-and-occupancy)章节中所讨论的，使用更少的寄存器可以让更多的线程和线程块驻留在流式多处理器上，从而提高性能。

因此，编译器会使用启发式方法来最小化寄存器使用量，同时将[寄存器溢出](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-registers)和指令数量保持在最低水平。应用程序可以选择性地通过向编译器提供额外信息来辅助这些启发式方法，这些信息以启动边界的形式提供，在 `__global__` 函数的定义中使用 `__launch_bounds__()` 限定符指定：

```cuda
__global__ void
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
MyKernel(...) {
    ...
}
```

- `maxThreadsPerBlock` 指定应用程序将启动 `MyKernel()` 时每个线程块的最大线程数；它编译为 `.maxntid` PTX 指令。
- `minBlocksPerMultiprocessor` 是可选的，指定每个流式多处理器期望的最小驻留块数；它编译为 `.minnctapersm` PTX 指令。
- `maxBlocksPerCluster` 是可选的，指定应用程序将启动 `MyKernel()` 时每个集群期望的最大线程块数；它编译为 `.maxclusterrank` PTX 指令。

如果指定了启动边界，编译器首先会推导出内核应使用的寄存器数量的上限 `L`。这确保了 `maxThreadsPerBlock` 个线程的 `minBlocksPerMultiprocessor` 个块（如果未指定 `minBlocksPerMultiprocessor`，则为单个块）可以驻留在流式多处理器上。关于内核使用的寄存器数量与每个块分配的寄存器数量之间的关系，请参阅[占用率](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-kernel-launch-and-occupancy)章节。然后，编译器按如下方式优化寄存器使用：

- 如果初始寄存器使用量超过 `L`，编译器会减少它，直到小于或等于 `L`。这通常会导致本地内存使用量增加和/或指令数量增多。
- 如果初始寄存器使用量低于 `L`：
    - 如果指定了 `maxThreadsPerBlock` 但未指定 `minBlocksPerMultiprocessor`，编译器会使用 `maxThreadsPerBlock` 来确定在 `n` 个和 `n+1` 个驻留块之间转换时的寄存器使用量阈值。这种情况发生在减少一个寄存器使用量可以为额外的驻留块腾出空间时。然后，编译器应用与未指定启动边界时类似的启发式方法。
    - 如果同时指定了 `minBlocksPerMultiprocessor` 和 `maxThreadsPerBlock`，编译器可能会将寄存器使用量增加到 `L`，以减少指令数量并更好地隐藏单线程指令的延迟。

如果内核在以下情况下执行，启动将失败：
-   每个线程块的线程数超过其启动界限 `maxThreadsPerBlock`。
-   每个线程簇的线程块数超过其启动界限 `maxBlocksPerCluster`。

CUDA 内核所需的每个线程资源可能会以不希望的方式限制最大线程块大小。为了保持与未来硬件和工具包的向前兼容性，并确保至少有一个线程块可以在流式多处理器上运行，开发者应包含单参数 `__launch_bounds__(maxThreadsPerBlock)`，它指定了内核将启动的最大线程块大小。若不这样做，可能会导致“请求启动的资源过多”错误。在某些情况下，提供双参数版本的 `__launch_bounds__(maxThreadsPerBlock,minBlocksPerMultiprocessor)` 可以提高性能。`minBlocksPerMultiprocessor` 的最佳值应通过对每个内核的详细分析来确定。

内核的最佳启动界限通常在不同的主要架构修订版中有所不同。以下代码示例说明了如何在设备代码中使用 `__CUDA_ARCH__`[宏](#cuda-arch-macro)来管理这一点。

```cuda
#define THREADS_PER_BLOCK  256

#if __CUDA_ARCH__ >= 900
    #define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
    #define MY_KERNEL_MIN_BLOCKS   3
#else
    #define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
    #define MY_KERNEL_MIN_BLOCKS   2
#endif

__global__ void
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
MyKernel(...) {
    ...
}
```

当使用每个线程块的最大线程数（指定为 `__launch_bounds__()` 的第一个参数）调用 `MyKernel` 时，很容易想当然地在执行配置中使用 `MY_KERNEL_MAX_THREADS` 作为每个线程块的线程数：

```cuda
// Host code
MyKernel<<<blocksPerGrid, MY_KERNEL_MAX_THREADS>>>(...);
```

然而，这行不通，因为如[执行空间说明符](#execution-space-specifiers)部分所述，`__CUDA_ARCH__` 在主机代码中未定义。因此，`MyKernel` 将以每个线程块 256 个线程启动。每个线程块的线程数应通过以下方式确定：

-   要么在编译时使用不依赖于 `__CUDA_ARCH__` 的宏或常量，例如：
    ```cuda
    // Host code
    MyKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(...);
    ```
-   要么在运行时根据计算能力确定：
    ```cuda
    // Host code
    cudaGetDeviceProperties(&deviceProp, device);
    int threadsPerBlock = (deviceProp.major >= 9) ? 2 * THREADS_PER_BLOCK : THREADS_PER_BLOCK;
    MyKernel<<<blocksPerGrid, threadsPerBlock>>>(...);
    ```

`--resource-usage` 编译器选项报告寄存器使用情况。[CUDA 性能分析器](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator)报告占用率，可用于推导常驻线程块的数量。

### 5.4.3.3. 每个线程的最大寄存器数

为了支持低级性能调优，CUDA C++ 提供了 `__maxnreg__()` 函数限定符，它将性能调优信息传递给后端优化编译器。`__maxnreg__()` 限定符指定了在线程块中可分配给单个线程的最大寄存器数量。在 `__global__` 函数的定义中：

```cuda
__global__ void
__maxnreg__(maxNumberRegistersPerThread)
MyKernel(...) {
    ...
}
```

The `maxNumberRegistersPerThread` variable specifies the maximum number of registers to be allocated to a single thread in a thread block of the kernel `MyKernel()`; it compiles to the `.maxnreg` PTX directive.

The `__launch_bounds__()` and `__maxnreg__()` qualifiers cannot be applied to the same kernel together.

The `--maxrregcount <N>` compiler option can be used to control register usage for all `__global__` functions in a file. This option is ignored for kernel functions with the `__maxnreg__` qualifier.

## 5.4.4.Synchronization Primitives

### 5.4.4.1.Thread Block Synchronization Functions

```cuda
void __syncthreads();
int  __syncthreads_count(int predicate);
int  __syncthreads_and(int predicate);
int  __syncthreads_or(int predicate);
```

The intrinsics coordinate communication among threads within the same block. When threads in a block access the same addresses in shared or global memory, read-after-write, write-after-read, or write-after-write hazards can occur. These hazards can be avoided by synchronizing threads between such accesses.

The intrinsics have the following semantics:

- __syncthreads*() wait until all non-exited threads in the thread block simultaneously reach the same __syncthreads*() intrinsic call in the program or exit.
- __syncthreads*() provide memory ordering among participating threads: the call to __syncthreads*() intrinsics strongly happens before (see C++ specification [intro.races] ) any participating thread is unblocked from the wait or exits.

The following example shows how to use `__syncthreads()` to synchronize threads within a thread block and safely sum the elements of an array shared among the threads:

```cuda
// assuming blockDim.x is 128
__global__ void example_syncthreads(int* input_data, int* output_data) {
    __shared__ int shared_data[128];
    // Every thread writes to a distinct element of 'shared_data':
    shared_data[threadIdx.x] = input_data[threadIdx.x];

    // All threads synchronize, guaranteeing all writes to 'shared_data' are ordered 
    // before any thread is unblocked from '__syncthreads()':
    __syncthreads();

    // A single thread safely reads 'shared_data':
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_data[i];
        }
        output_data[blockIdx.x] = sum;
    }
}
```

The `__syncthreads*()` intrinsics are permitted in conditional code, but only if the condition evaluates uniformly across the entire thread block. Otherwise, execution may hang or produce unintended side effects.

The following example demonstrates a valid behavior:

```cuda
// assuming blockDim.x is 128
__global__ void syncthreads_valid_behavior(int* input_data, int* output_data) {
    __shared__ int shared_data[128];
    shared_data[threadIdx.x] = input_data[threadIdx.x];
    if (blockIdx.x > 0) { // CORRECT, uniform condition across all block threads
        __syncthreads();
        output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
    }
}
```
而以下示例则展示了无效行为，例如内核挂起或未定义行为：

```cuda
// 假设 blockDim.x 为 128
__global__ void syncthreads_invalid_behavior1(int* input_data, int* output_data) {
    __shared__ int shared_data[256];
    shared_data[threadIdx.x] = input_data[threadIdx.x];
    if (threadIdx.x > 0) { // 错误，非统一条件
        __syncthreads();   // 未定义行为
        output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
    }
}
```

```cuda
// 假设 blockDim.x 为 128
__global__ void syncthreads_invalid_behavior2(int* input_data, int* output_data) {
    __shared__ int shared_data[256];
    shared_data[threadIdx.x] = input_data[threadIdx.x];
    for (int i = 0; i < blockDim.x; ++i) {
        if (i == threadIdx.x) { // 错误，非统一条件
            __syncthreads();    // 未定义行为
        }
    }
    output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
}
```

---

**带谓词的 `__syncthreads()` 变体**：

```cuda
int __syncthreads_count(int predicate);
```
此函数与 `__syncthreads()` 相同，区别在于它会评估线程块中所有未退出的线程的谓词，并返回谓词评估为非零值的线程数量。

```cuda
int __syncthreads_and(int predicate);
```
此函数与 `__syncthreads()` 相同，区别在于它会评估线程块中所有未退出的线程的谓词。当且仅当所有线程的谓词评估结果均为非零值时，它才返回一个非零值。

```cuda
int __syncthreads_or(int predicate);
```
此函数与 `__syncthreads()` 相同，区别在于它会评估线程块中所有未退出的线程的谓词。当且仅当一个或多个线程的谓词评估结果为非零值时，它才返回一个非零值。

### 5.4.4.2. 线程束同步函数

```cuda
void __syncwarp(unsigned mask = 0xFFFFFFFF);
```

内置函数 `__syncwarp()` 协调同一线程束内线程之间的通信。当线程束内的某些线程访问共享内存或全局内存中的相同地址时，可能会出现读后写、写后读或写后写的数据冒险。通过在访问之间同步线程，可以避免这些数据冒险。

调用 `__syncwarp(mask)` 会在 `mask` 指定的参与线程之间提供内存排序：对 `__syncwarp(mask)` 的调用**强发生于**（参见 [C++ 规范 [intro.races]](https://eel.is/c++draft/intro.races)）`mask` 中指定的任何线程束线程从等待状态解除阻塞或退出之前。

这些函数受 [Warp __sync 内置函数约束](#warp-sync-intrinsic-constraints) 的约束。

以下示例演示了如何使用 `__syncwarp()` 同步线程束内的线程，以安全地访问共享内存数组：

```cuda
__global__ void example_syncwarp(int* input_data, int* output_data) {
    if (threadIdx.x < warpSize) {
        __shared__ int shared_data[warpSize];
        shared_data[threadIdx.x] = input_data[threadIdx.x];

        __syncwarp(); // 等价于 __syncwarp(0xFFFFFFFF)
        if (threadIdx.x == 0)
            output_data[0] = shared_data[1];
    }
}
```
### 5.4.4.3. 内存栅栏函数

CUDA 编程模型采用弱序内存模型。换句话说，一个 CUDA 线程将数据写入共享内存、全局内存、页锁定主机内存或对等设备内存的顺序，不一定是另一个 CUDA 线程或主机线程观察到这些数据被写入的顺序。在没有内存栅栏或同步的情况下，对同一内存位置进行读写会导致未定义的行为。

在以下示例中，线程 1 执行 `writeXY()`，而线程 2 执行 `readXY()`。

```cuda
__device__ int X = 1, Y = 2;

__device__ void writeXY() {
    X = 10;
    Y = 20;
}

__device__ void readXY() {
    int B = Y;
    int A = X;
}
```

这两个线程同时对相同的内存位置 `X` 和 `Y` 进行读写。任何数据竞争都会导致未定义的行为，并且没有确定的语义。因此，`A` 和 `B` 的最终值可以是任何值。

内存栅栏和同步函数强制执行内存访问的[顺序一致性](https://en.cppreference.com/w/cpp/atomic/memory_order)。这些函数在强制执行排序的[线程作用域](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes)上有所不同，但与访问的内存空间无关，包括共享内存、全局内存、页锁定主机内存和对等设备的内存。

!!! note "提示"
    出于安全性和可移植性考虑，建议尽可能使用 libcu++ 提供的 cuda::atomic_thread_fence。

**线程块级内存栅栏**

 CUDA C++

```cuda
// <cuda/atomic> 头文件
cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block);
```

确保：

- 调用线程在调用 cuda::atomic_thread_fence() 之前对所有内存进行的所有写入，对于调用线程所在线程块中的所有线程而言，都发生在调用线程在调用 cuda::atomic_thread_fence() 之后对所有内存进行的所有写入之前；
- 调用线程在调用 cuda::atomic_thread_fence() 之前对所有内存进行的所有读取，都排序在调用线程在调用 cuda::atomic_thread_fence() 之后对所有内存进行的所有读取之前。

 内部函数

```cuda
void __threadfence_block();
```

确保：

- 调用线程在调用 __threadfence_block() 之前对所有内存进行的所有写入，对于调用线程所在线程块中的所有线程而言，都发生在调用线程在调用 __threadfence_block() 之后对所有内存进行的所有写入之前；
- 调用线程在调用 __threadfence_block() 之前对所有内存进行的所有读取，都排序在调用线程在调用 __threadfence_block() 之后对所有内存进行的所有读取之前。

**设备级内存栅栏**

 CUDA C++

```cuda
cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
```

确保：

- 调用线程在调用 cuda::atomic_thread_fence() 之后对所有内存进行的任何写入，都不会被设备中的任何线程观察到发生在调用线程在调用 cuda::atomic_thread_fence() 之前对所有内存进行的任何写入之前。
## 内存栅栏函数

```cuda
void __threadfence();
```

确保：

- 调用线程在调用 `__threadfence()` 之后对所有内存的写入，不会被设备中的任何线程观察到发生在调用线程在调用 `__threadfence()` 之前对所有内存的写入之前。

**系统级内存栅栏**

CUDA C++

```cuda
cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
```

确保：

- 调用线程在调用 `cuda::atomic_thread_fence()` 之前对所有内存的写入，会被设备中的所有线程、主机线程以及对等设备中的所有线程观察到发生在调用线程在调用 `cuda::atomic_thread_fence()` 之后对所有内存的写入之前。

## 内建函数

```cuda
void __threadfence_system();
```

确保：

- 调用线程在调用 `__threadfence_system()` 之前对所有内存的写入，会被设备中的所有线程、主机线程以及对等设备中的所有线程观察到发生在调用线程在调用 `__threadfence_system()` 之后对所有内存的写入之前。

在前面的代码示例中，我们可以按如下方式在代码中插入内存栅栏：

CUDA C++

```cuda
#include <cuda/atomic>

__device__ int X = 1, Y = 2;

__device__ void writeXY() {
    X = 10;
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    Y = 20;
}

__device__ void readXY() {
    int B = Y;
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    int A = X;
}
```

## 内建函数

```cuda
__device__ int X = 1, Y = 2;

__device__ void writeXY() {
    X = 10;
    __threadfence();
    Y = 20;
}

__device__ void readXY() {
    int B = Y;
    __threadfence();
    int A = X;
}
```

对于此代码，可以观察到以下结果：

- A 等于 1 且 B 等于 2，即 `readXY()` 在 `writeXY()` 之前执行。
- A 等于 10 且 B 等于 20，即 `writeXY()` 在 `readXY()` 之前执行。
- A 等于 10 且 B 等于 2。
- A 等于 1 且 B 等于 20 的情况是不可能的，因为内存栅栏确保对 X 的写入在对 Y 的写入之前可见。

如果线程 1 和线程 2 属于同一个线程块，使用块级栅栏就足够了。如果线程 1 和线程 2 不属于同一个线程块，则当它们是来自同一设备的 CUDA 线程时，必须使用设备级栅栏；当它们是来自两个不同设备的 CUDA 线程时，必须使用系统级栅栏。

以下代码示例说明了一个常见用例，其中线程消费由其他线程产生的数据。该内核在单次调用中计算一个包含 N 个数字的数组的总和。

- 每个线程块首先对数组的一个子集求和，并将结果存储在全局内存中。
- 当所有线程块完成后，最后一个线程块从全局内存中读取每个部分和，并将它们相加以获得最终结果。
- 为了确定哪个线程块最后完成，每个线程块原子地递增一个计数器，以发出计算和存储其部分和完成的信号（有关详细信息，请参阅原子函数部分）。最后一个线程块接收到的计数器值等于 `gridDim.x - 1`。
如果在存储部分和与递增计数器之间没有内存屏障，计数器可能在部分和存储之前就递增。这可能导致计数器达到 `gridDim.x - 1`，并允许最后一个线程块在部分和于内存中更新之前就开始读取它们。

!!! note "注意"
    内存屏障仅影响内存操作的执行顺序；它并不保证这些操作对其他线程的可见性。

在下面的代码示例中，对 `result` 变量的内存操作的可见性是通过将其声明为 `volatile` 来确保的。更多细节，请参阅 `volatile`-[限定变量](cpp-language-support.html#volatile-qualifier) 章节。

```cuda
#include <cuda/atomic>

__device__ int count = 0;

__global__ void sum(const float*    array,
                    int             N,
                    volatile float* result) {
    __shared__ bool isLastBlockDone;
    // 每个线程块对输入数组的一个子集求和。
    float partialSum = calculatePartialSum(array, N);

    if (threadIdx.x == 0) {
        // 每个线程块的线程 0 将部分和存储到全局内存。
        // 由于 "result" 变量被声明为 volatile，编译器将使用绕过 L1 缓存的存储操作。
        // 这确保了最后一个线程块的线程将读取由所有其他线程块计算出的正确部分和。
        result[blockIdx.x] = partialSum;

        // 线程 0 确保只有在部分和写入全局内存后，才执行 "count" 变量的递增。
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);

        // 线程 0 发出信号，表示它已完成。
        int count_old = atomicInc(&count, gridDim.x);

        // 线程 0 判断其所在线程块是否是最后一个完成的线程块。
        isLastBlockDone = (count_old == (gridDim.x - 1));
    }
    // 同步以确保每个线程读取正确的 isLastBlockDone 值。
    __syncthreads();

    if (isLastBlockDone) {
        // 最后一个线程块对存储在 result[0 .. gridDim.x-1] 中的部分和进行求和
        float totalSum = calculateTotalSum(result);

        if (threadIdx.x == 0) {
            // 最后一个线程块的线程 0 将总和存储到全局内存，并重置 count 变量，
            // 以便下一次内核调用正常工作。
            result[0] = totalSum;
            count     = 0;
        }
    }
}
```

## 5.4.5. 原子函数

原子函数对共享数据执行读-修改-写操作，使其看起来像是单步执行的。原子性确保每个操作要么完全完成，要么根本不执行，从而为所有参与的线程提供数据的一致视图。

CUDA 以四种方式提供原子函数：

扩展的 CUDA C++ 原子函数，`cuda::atomic` 和 `cuda::atomic_ref`。

- 它们允许在主机和设备代码中使用。
- 它们遵循 C++ 标准原子操作的语义。
- 它们允许指定原子操作的线程作用域。

标准 C++ 原子函数，`cuda::std::atomic` 和 `cuda::std::atomic_ref`。

- 它们允许在主机和设备代码中使用。
- 它们遵循 C++ 标准原子操作语义。
- 它们不允许指定原子操作的线程作用域。

编译器内置原子函数，`__nv_atomic_<op>()`。

- 自 CUDA 12.8 起可用。
- 仅允许在设备代码中使用。
- 它们遵循 C++ 标准原子内存序语义。
- 它们允许指定原子操作的线程作用域。
- 它们具有与 C++ 标准原子操作相同的内存排序语义。
- 它们支持 `cuda::std::atomic` 和 `cuda::std::atomic_ref` 所允许的数据类型的子集，但不包括 128 位数据类型。

传统原子函数，`atomic<Op>()`。

- 仅允许在设备代码中使用。
- 仅支持 `memory_order_relaxed` C++ 原子内存语义。
- 它们允许将原子操作的线程作用域作为函数名的一部分来指定。
- 与内置原子函数不同，传统原子函数仅保证原子性，不引入同步点（内存栅栏）。
- 它们支持内置原子函数所允许的数据类型的子集。`atomicAdd` 操作支持额外的数据类型。

!!! note "提示"
    出于效率、安全性和可移植性考虑，建议使用 libcu++ 提供的扩展 CUDA C++ 原子函数。

### 5.4.5.1. 传统原子函数

传统原子函数对存储在全局内存或共享内存中的 32 位、64 位或 128 位字执行原子读-修改-写操作。例如，`atomicAdd()` 函数读取全局内存或共享内存中特定地址的一个字，将一个数字加到该字上，然后将结果写回同一地址。

- 原子函数只能在设备函数中使用。
- 对于向量类型，如 `__half2`、`__nv_bfloat162`、`float2` 和 `float4`，读-修改-写操作在向量的每个元素上执行。不保证对整个向量的单次访问是原子的。

本节描述的原子函数具有 `cuda::std::memory_order_relaxed` 的[内存排序](https://en.cppreference.com/w/cpp/atomic/memory_order)，并且仅在特定的[线程作用域](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes)内是原子的：

- 不带后缀的原子 API，例如 `atomicAdd`，在 `cuda::thread_scope_device` 作用域内是原子的。
- 带有 `_block` 后缀的原子 API，例如 `atomicAdd_block`，在 `cuda::thread_scope_block` 作用域内是原子的。
- 带有 `_system` 后缀的原子 API，例如 `atomicAdd_system`，如果满足特定条件，则在 `cuda::thread_scope_system` 作用域内是原子的。

以下示例展示了 CPU 和 GPU 原子地更新地址 `addr` 处的整数值：

```cuda
#include <cuda_runtime.h>

__global__ void atomicAdd_kernel(int* addr) {
    atomicAdd_system(addr, 10);
}

void test_atomicAdd(int device_id) {
    int* addr;
    cudaMallocManaged(&addr, 4);
    *addr = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    if (deviceProp.concurrentManagedAccess != 1) {
        return; // 设备无法与 CPU 并发地一致访问托管内存
    }

    atomicAdd_kernel<<<...>>>(addr);
    __sync_fetch_and_add(addr, 10);  // CPU 原子操作
}
```
---

请注意，任何原子操作都可以基于 `atomicCAS()`（比较并交换）来实现。例如，单精度浮点数的 `atomicAdd()` 可以按如下方式实现：

```cuda
#include <cuda/memory>
#include <cuda/std/bit>

__device__ float customAtomicAdd(float* d_ptr, float value) {
    volatile unsigned* d_ptr_unsigned = reinterpret_cast<unsigned*>(d_ptr);
    unsigned  old_value      = *d_ptr_unsigned;
    unsigned  assumed;
    do {
        assumed                          = old_value;
        float    assumed_float           = cuda::std::bit_cast<float>(assumed);
        float    expected_value          = assumed_float + value;
        unsigned expected_value_unsigned = cuda::std::bit_cast<unsigned>(expected_value);
        old_value                        = atomicCAS(d_ptr_unsigned, assumed, expected_value_unsigned);
    // 注意：使用整数比较以避免在 NaN 情况下挂起（因为 NaN != NaN）
    } while (assumed != old_value);
    return cuda::std::bit_cast<float>(old_value);
}
```

查看 [Compiler Explorer](https://godbolt.org/z/676e5bc7a) 上的示例。

#### 5.4.5.1.1.atomicAdd()

```cuda
T atomicAdd(T* address, T val);
```

该函数在一个原子事务中执行以下操作：

1.  读取位于全局内存或共享内存中地址 `address` 处的旧值。
2.  计算 `old + val`。
3.  将结果存储回同一地址的内存中。

该函数返回 `old` 值。

`atomicAdd()` 支持以下数据类型：

-   `int`、`unsigned`、`unsigned long long`、`float`、`double`、`__half2`、`__half`。
-   在计算能力 8.x 及更高版本的设备上支持 `__nv_bfloat16`、`__nv_bfloat162`。
-   在计算能力 9.x 及更高版本的设备上支持 `float2`、`float4`，并且仅支持全局内存地址。

应用于向量类型（例如 `__half2` 或 `float4`）的 `atomicAdd()` 的原子性，是针对每个分量分别保证的；不保证整个向量作为单次访问是原子的。

#### 5.4.5.1.2.atomicSub()

```cuda
T atomicSub(T* address, T val);
```

该函数在一个原子事务中执行以下操作：

1.  读取位于全局内存或共享内存中地址 `address` 处的旧值。
2.  计算 `old - val`。
3.  将结果存储回同一地址的内存中。

该函数返回 `old` 值。

`atomicSub()` 支持以下数据类型：

-   `int`、`unsigned`

#### 5.4.5.1.3.atomicInc()

```cuda
unsigned atomicInc(unsigned* address, unsigned val);
```

该函数在一个原子事务中执行以下操作：

1.  读取位于全局内存或共享内存中地址 `address` 处的旧值。
2.  计算 `old >= val ? 0 : (old + 1)`。
3.  将结果存储回同一地址的内存中。

该函数返回 `old` 值。

#### 5.4.5.1.4.atomicDec()

```cuda
unsigned atomicDec(unsigned* address, unsigned val);
```

该函数在一个原子事务中执行以下操作：

1.  读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 (old == 0 || old > val) ? val : (old - 1)。
3. 将结果存储回内存中的同一地址。

该函数返回 `old` 值。

#### 5.4.5.1.5.atomicAnd()

```cuda
T atomicAnd(T* address, T val);
```

该函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 old & val。
3. 将结果存储回内存中的同一地址。

该函数返回 `old` 值。

`atomicAnd()` 支持以下数据类型：

- int , unsigned , unsigned long long 。

#### 5.4.5.1.6.atomicOr()

```cuda
T atomicOr(T* address, T val);
```

该函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 old | val。
3. 将结果存储回内存中的同一地址。

该函数返回 `old` 值。

`atomicOr()` 支持以下数据类型：

- int , unsigned , unsigned long long 。

#### 5.4.5.1.7.atomicXor()

```cuda
T atomicXor(T* address, T val);
```

该函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 old ^ val。
3. 将结果存储回内存中的同一地址。

该函数返回 `old` 值。

`atomicXor()` 支持以下数据类型：

- int , unsigned , unsigned long long 。

#### 5.4.5.1.8.atomicMin()

```cuda
T atomicMin(T* address, T val);
```

该函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 old 和 val 的最小值。
3. 将结果存储回内存中的同一地址。

该函数返回 `old` 值。

`atomicMin()` 支持以下数据类型：

- int , unsigned , unsigned long long , long long 。

#### 5.4.5.1.9.atomicMax()

```cuda
T atomicMax(T* address, T val);
```

该函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 old 和 val 的最大值。
3. 将结果存储回内存中的同一地址。

该函数返回 `old` 值。

`atomicMax()` 支持以下数据类型：

- int , unsigned , unsigned long long , long long 。

#### 5.4.5.1.10.atomicExch()

```cuda
T atomicExch(T* address, T val);
```

```cuda
template<typename T>
T atomicExch(T* address, T val); // only 128-bit types, compute capability 9.x and higher
```

该函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 将 val 存储回内存中的同一地址。

该函数返回 `old` 值。

`atomicExch()` 支持以下数据类型：

- int , unsigned , unsigned long long , float 。

C++ 模板函数 `atomicExch()` 支持 128 位类型，需满足以下要求：
- 计算能力 9.x 及更高版本。
- T 必须对齐到 16 字节，即 alignof(T) >= 16。
- T 必须是可平凡复制的，即 std::is_trivially_copyable_v<T>。
- 对于 C++03 及更早版本：T 必须是可平凡构造的，即 std::is_default_constructible_v<T>。

#### 5.4.5.1.11.atomicCAS()

```cuda
T atomicCAS(T* address, T compare, T val);
```

```cuda
template<typename T>
T atomicCAS(T* address, T compare, T val);  // 仅适用于 128 位类型，计算能力 9.x 及更高版本
```

该函数在一个原子事务中执行以下操作：

1.  读取位于全局内存或共享内存中地址 `address` 处的旧值。
2.  计算 `old == compare ? val : old`。
3.  将结果存储回同一地址的内存中。

该函数返回 `old` 值。

`atomicCAS()` 支持以下数据类型：

- `int`、`unsigned`、`unsigned long long`、`unsigned short`。

C++ 模板函数 `atomicCAS()` 支持 128 位类型，需满足以下要求：

- 计算能力 9.x 及更高版本。
- T 必须对齐到 16 字节，即 alignof(T) >= 16。
- T 必须是可平凡复制的，即 std::is_trivially_copyable_v<T>。
- 对于 C++03 及更早版本：T 必须是可平凡构造的，即 std::is_default_constructible_v<T>。

### 5.4.5.2. 内置原子函数

CUDA 12.8 及更高版本支持用于原子操作的 CUDA 编译器内置函数，其遵循与 [C++ 标准原子操作](https://en.cppreference.com/w/cpp/atomic/atomic.html) 和 CUDA [线程作用域](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) 相同的内存排序语义。这些函数遵循 [GNU 的原子内置函数签名](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html)，并带有一个用于指定线程作用域的额外参数。

当支持内置原子函数时，`nvcc` 会定义宏 `__CUDACC_DEVICE_ATOMIC_BUILTINS__`。

下面列出了用作内置原子函数 `order` 和 `scope` 参数的 [内存顺序](https://en.cppreference.com/w/cpp/atomic/atomic.html) 和 [线程作用域](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) 的原始枚举值：

```cuda
// 原子内存顺序
enum {
   __NV_ATOMIC_RELAXED,
   __NV_ATOMIC_CONSUME,
   __NV_ATOMIC_ACQUIRE,
   __NV_ATOMIC_RELEASE,
   __NV_ATOMIC_ACQ_REL,
   __NV_ATOMIC_SEQ_CST
};
```

```cuda
// 线程作用域
enum {
   __NV_THREAD_SCOPE_THREAD,
   __NV_THREAD_SCOPE_BLOCK,
   __NV_THREAD_SCOPE_CLUSTER,
   __NV_THREAD_SCOPE_DEVICE,
   __NV_THREAD_SCOPE_SYSTEM
};
```

- 内存顺序对应于 C++ 标准原子操作的内存顺序。
- 线程作用域遵循 cuda::thread_scope 的定义。
- `__NV_ATOMIC_CONSUME` 内存顺序当前使用更强的 `__NV_ATOMIC_ACQUIRE` 内存顺序实现。
- `__NV_THREAD_SCOPE_THREAD` 线程作用域当前使用更宽的 `__NV_THREAD_SCOPE_BLOCK` 线程作用域实现。
示例：

```cuda
__device__ T __nv_atomic_load_n(T*  pointer,
                                int memory_order,
                                int thread_scope = __NV_THREAD_SCOPE_SYSTEM);
```

原子内置函数有以下限制：

- 它们只能在设备函数中使用。
- 它们不能操作本地内存。
- 不能获取这些函数的地址。
- 顺序（order）和范围（scope）参数必须是整数字面量；不能是变量。
- 线程范围 `__NV_THREAD_SCOPE_CLUSTER` 在计算能力 sm_90 及更高架构上受支持。

不支持情况的示例：

```cuda
 // 不允许在主机函数中使用
 __host__ void bar() {
     unsigned u1 = 1, u2 = 2;
     __nv_atomic_load(&u1, &u2, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
 }

 // 不允许应用于本地内存
__device__ void foo() {
   unsigned a = 1, b;
   __nv_atomic_load(&a, &b, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
}

 // 不允许作为模板默认参数。
 // 不能获取函数地址。
 template<void *F = __nv_atomic_load_n>
 class X {
     void *f = F; // 不能获取函数地址。
 };

 // 不允许在构造函数初始化列表中调用。
 class Y {
     int a;
 public:
     __device__ Y(int *b): a(__nv_atomic_load_n(b, __NV_ATOMIC_RELAXED)) {}
 };
```

#### 5.4.5.2.1. __nv_atomic_fetch_add(), __nv_atomic_add()

```cuda
__device__ T    __nv_atomic_fetch_add(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_add      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

这些函数在一个原子事务中执行以下操作：

1.  读取位于全局或共享内存中地址 `address` 处的旧值。
2.  计算 `old + val`。
3.  将结果存储回内存中的同一地址。

-   `__nv_atomic_fetch_add` 返回旧值。
-   `__nv_atomic_add` 没有返回值。

这些函数支持以下数据类型：

-   `int`, `unsigned`, `unsigned long long`, `float`, `double`。

#### 5.4.5.2.2. __nv_atomic_fetch_sub(), __nv_atomic_sub()

```cuda
__device__ T    __nv_atomic_fetch_sub(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_sub      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

这些函数在一个原子事务中执行以下操作：

1.  读取位于全局或共享内存中地址 `address` 处的旧值。
2.  计算 `old - val`。
3.  将结果存储回内存中的同一地址。

-   `__nv_atomic_fetch_sub` 返回旧值。
-   `__nv_atomic_sub` 没有返回值。

这些函数支持以下数据类型：

-   `int`, `unsigned`, `unsigned long long`, `float`, `double`。

#### 5.4.5.2.3. __nv_atomic_fetch_and(), __nv_atomic_and()

```cuda
__device__ T    __nv_atomic_fetch_and(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_and      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```
这些函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 `old & val`。
3. 将结果存储回同一地址的内存中。

- `__nv_atomic_fetch_and` 返回旧值。
- `__nv_atomic_and` 没有返回值。

这些函数支持以下数据类型：

- 任何大小为 4 或 8 字节的整数类型。

#### 5.4.5.2.4. `__nv_atomic_fetch_or()`, `__nv_atomic_or()`

```cuda
__device__ T    __nv_atomic_fetch_or(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_or      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 `old | val`。
3. 将结果存储回同一地址的内存中。

- `__nv_atomic_fetch_or` 返回旧值。
- `__nv_atomic_or` 没有返回值。

这些函数支持以下数据类型：

- 任何大小为 4 或 8 字节的整数类型。

#### 5.4.5.2.5. `__nv_atomic_fetch_xor()`, `__nv_atomic_xor()`

```cuda
__device__ T    __nv_atomic_fetch_xor(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_xor      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 `old ^ val`。
3. 将结果存储回同一地址的内存中。

- `__nv_atomic_fetch_xor` 返回旧值。
- `__nv_atomic_xor` 没有返回值。

这些函数支持以下数据类型：

- 任何大小为 4 或 8 字节的整数类型。

#### 5.4.5.2.6. `__nv_atomic_fetch_min()`, `__nv_atomic_min()`

```cuda
__device__ T    __nv_atomic_fetch_min(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_min      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 `old` 和 `val` 的最小值。
3. 将结果存储回同一地址的内存中。

- `__nv_atomic_fetch_min` 返回旧值。
- `__nv_atomic_min` 没有返回值。

这些函数支持以下数据类型：

- `unsigned`, `int`, `unsigned long long`, `long long`。

#### 5.4.5.2.7. `__nv_atomic_fetch_max()`, `__nv_atomic_max()`

```cuda
__device__ T    __nv_atomic_fetch_max(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_max      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局内存或共享内存中地址 `address` 处的旧值。
2. 计算 `old` 和 `val` 的最大值。
3. 将结果存储回内存中的同一地址。

- __nv_atomic_fetch_max 返回旧值。
- __nv_atomic_max 没有返回值。

这些函数支持以下数据类型：

- unsigned , int , unsigned long long , long long

#### 5.4.5.2.8.__nv_atomic_exchange(),__nv_atomic_exchange_n()

```cuda
__device__ T    __nv_atomic_exchange_n(T* address, T val,          int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_exchange  (T* address, T* val, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 `address` 处的旧值。
2. __nv_atomic_exchange_n 将 `val` 存储到 `address` 指向的位置。__nv_atomic_exchange 将 `old` 存储到 `ret` 指向的位置，并将位于地址 `val` 处的值存储到 `address` 指向的位置。

- __nv_atomic_exchange_n 返回旧值。
- __nv_atomic_exchange 没有返回值。

这些函数支持以下数据类型：

- 任何大小为 4、8 或 16 字节的数据类型。
- 16 字节数据类型在计算能力 9.x 及更高的设备上受支持。

#### 5.4.5.2.9.__nv_atomic_compare_exchange(),__nv_atomic_compare_exchange_n()

```cuda
__device__ bool __nv_atomic_compare_exchange  (T* address, T* expected, T* desired, bool weak, int success_order, int failure_order,
                                               int scope = __NV_THREAD_SCOPE_SYSTEM);

__device__ bool __nv_atomic_compare_exchange_n(T* address, T* expected, T desired, bool weak, int success_order, int failure_order,
                                               int scope = __NV_THREAD_SCOPE_SYSTEM);
```

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 `address` 处的旧值。
2. 将 `old` 与 `expected` 指向的值进行比较。
3. 如果它们相等，则返回值为 `true` 并将 `desired` 存储到 `address` 指向的位置。否则，返回 `false` 并将 `old` 存储到 `expected` 指向的位置。

参数 `weak` 被忽略，它选择 `success_order` 和 `failure_order` 之间更强的内存顺序来执行比较并交换操作。

这些函数支持以下数据类型：

- 任何大小为 2、4、8 或 16 字节的数据类型。
- 16 字节数据类型在计算能力 9.x 及更高的设备上受支持。

#### 5.4.5.2.10.__nv_atomic_load(),__nv_atomic_load_n()

```cuda
__device__ void __nv_atomic_load  (T* address, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ T    __nv_atomic_load_n(T* address,         int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

这些函数在一个原子事务中执行以下操作：

1. 读取位于全局或共享内存中地址 `address` 处的旧值。
2. __nv_atomic_load 将 `old` 存储到 `ret` 指向的位置。__nv_atomic_load_n 返回 `old`。

这些函数支持以下数据类型：
-   大小为 1、2、4、8 或 16 字节的任何数据类型。

`order` 不能是 `__NV_ATOMIC_RELEASE` 或 `__NV_ATOMIC_ACQ_REL`。

#### 5.4.5.2.11.__nv_atomic_store(),__nv_atomic_store_n()

```cuda
__device__ void __nv_atomic_store  (T* address, T* val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_store_n(T* address, T  val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

这些函数在一个原子事务中执行以下操作：

1.  读取位于全局内存或共享内存中地址 `address` 处的旧值。
2.  `__nv_atomic_store` 读取 `val` 所指向的值并存储到 `address` 所指向的位置。`__nv_atomic_store_n` 将 `val` 存储到 `address` 所指向的位置。

`order` 不能是 `__NV_ATOMIC_CONSUME`、`__NV_ATOMIC_ACQUIRE` 或 `__NV_ATOMIC_ACQ_REL`。

#### 5.4.5.2.12.__nv_atomic_thread_fence()

```cuda
__device__ void __nv_atomic_thread_fence(int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```

此原子函数根据指定的内存顺序，在本线程请求的内存访问之间建立一种排序关系。线程作用域参数指定了可能观察到该操作排序效果的一组线程。

## 5.4.6. 线程束函数

以下部分描述了允许线程束内的线程相互通信和执行计算的线程束函数。

!!! note "提示"
    出于效率、安全性和可移植性的考虑，建议尽可能使用 CUB 的 Warp-Wide "Collective" 原语来执行线程束操作。

### 5.4.6.1. 线程束活动掩码

```cuda
unsigned __activemask();
```

该函数返回一个 32 位整数掩码，表示调用线程束中所有当前活动的线程。如果调用 `__activemask()` 时线程束中的第 N 个通道处于活动状态，则第 N 位被置位。[非活动线程](../03-advanced/advanced-kernel-programming.html#simt-architecture-notes) 在返回的掩码中由 0 位表示。已退出程序的线程始终被标记为非活动状态。

!!! warning "警告"
    `__activemask()` 不能用于确定哪些线程束通道执行给定的分支。此函数旨在用于机会主义的线程束级编程，并且仅提供线程束内活动线程的瞬时快照。// 检查是否至少有一个线程的谓词计算结果为真
    if ( pred ) {
        // 无效：'at_least_one' 的值是不确定的，并且可能在多次执行之间变化。
        at_least_one = __activemask () > 0 ;
    }

请注意，在 `__activemask()` 调用处汇聚的线程，并不能保证在后续指令处保持汇聚，除非这些指令是线程束同步内部函数（`__sync`）。

例如，编译器可能会对指令重新排序，并且活动线程的集合可能无法保留：

```cuda
unsigned mask      = __activemask();              // 假设 mask == 0xFFFFFFFF（所有位已设置，所有线程活动）
int      predicate = threadIdx.x % 2 == 0;        // 偶数线程为 1，奇数线程为 0
int      result    = __any_sync(mask, predicate); // 活动线程可能无法保留
```
### 5.4.6.2. 线程束表决函数

```cuda
int      __all_sync   (unsigned mask, int predicate);
int      __any_sync   (unsigned mask, int predicate);
unsigned __ballot_sync(unsigned mask, int predicate);
```

线程束表决函数使得给定[线程束](../01-introduction/programming-model.html#programming-model-warps-simt)中的线程能够执行归约-广播操作。这些函数从线程束中每个未退出的线程接收一个整数 `predicate` 作为输入，并将这些值与零进行比较。然后，比较结果在[线程束的活动线程](../03-advanced/advanced-kernel-programming.html#simt-architecture-notes)之间通过以下方式之一进行组合（归约），并将单个返回值广播给每个参与的线程：

`__all_sync(unsigned mask, predicate)`
:

对 `mask` 中所有未退出的线程计算 `predicate`，如果所有这些线程的 `predicate` 计算结果都非零，则返回非零值。

`__any_sync(unsigned mask, predicate)`
:

对 `mask` 中所有未退出的线程计算 `predicate`，如果其中一个或多个线程的 `predicate` 计算结果非零，则返回非零值。

`__ballot_sync(unsigned mask, predicate)`
:

对 `mask` 中所有未退出的线程计算 `predicate`，并返回一个整数。如果线程束中第 N 个线程的 `predicate` 计算结果非零且该线程是活动的，则设置该整数的第 N 位。否则，第 N 位为零。

这些函数受[线程束 __sync 内部函数约束](#warp-sync-intrinsic-constraints)的约束。

!!! warning "警告"
    这些内部函数不提供任何内存排序保证。

### 5.4.6.3. 线程束匹配函数

!!! note "提示"
    建议使用 libcu++ 的 cuda::device::warp_match_all() 函数作为 __match_all_sync 函数的通用且更安全的替代方案。

```cuda
unsigned __match_any_sync(unsigned mask, T value);
unsigned __match_all_sync(unsigned mask, T value, int *pred);
```

线程束匹配函数在[线程束](../01-introduction/programming-model.html#programming-model-warps-simt)内的未退出线程之间对变量执行广播-比较操作。

`__match_any_sync`

返回 `mask` 中具有相同按位 `value` 的未退出线程的掩码。

`__match_all_sync`

如果 `mask` 中所有未退出的线程都具有相同的按位 `value`，则返回 `mask`；否则返回 0。如果 `mask` 中所有未退出的线程都具有相同的按位 `value`，则谓词 `pred` 被设置为 `true`；否则谓词被设置为 false。

`T` 可以是 `int`、`unsigned`、`long`、`unsigned long`、`long long`、`unsigned long long`、`float` 或 `double`。

这些函数受[线程束 __sync 内部函数约束](#warp-sync-intrinsic-constraints)的约束。

!!! warning "警告"
    这些内部函数不提供任何内存排序保证。

### 5.4.6.4. 线程束归约函数

!!! note "提示"
    出于效率、安全性和可移植性的考虑，建议尽可能使用 CUB 库的 Warp-Wide "Collective" 原语来执行线程束归约。
支持计算能力 8.x 或更高的设备。

```cuda
T        __reduce_add_sync(unsigned mask, T value);
T        __reduce_min_sync(unsigned mask, T value);
T        __reduce_max_sync(unsigned mask, T value);

unsigned __reduce_and_sync(unsigned mask, unsigned value);
unsigned __reduce_or_sync (unsigned mask, unsigned value);
unsigned __reduce_xor_sync(unsigned mask, unsigned value);
```

`__reduce_<op>_sync` 内部函数在同步 `mask` 指定的所有未退出的线程后，对 `value` 中提供的数据执行归约操作。

`__reduce_add_sync`
,
`__reduce_min_sync`
,
`__reduce_max_sync`

返回对 `mask` 指定的每个未退出线程在 `value` 中提供的值应用算术加法、最小值或最大值归约操作的结果。`T` 可以是 `unsigned` 或 `signed` 整数。

`__reduce_and_sync`
,
`__reduce_or_sync`
,
`__reduce_xor_sync`

返回对 `mask` 指定的每个未退出线程在 `value` 中提供的值应用按位与、或或异或归约操作的结果。

这些函数受 [Warp __sync 内部函数约束](#warp-sync-intrinsic-constraints) 的约束。

!!! warning "警告"
    这些内部函数不提供任何内存排序。

### 5.4.6.5. Warp Shuffle 函数

!!! note "提示"
    建议使用 libcu++ 的 cuda::device::warp_shuffle() 函数作为 __shfl_sync() 和 __shfl_<op>_sync() 内部函数的通用且更安全的替代方案。

```cuda
T __shfl_sync     (unsigned mask, T value, int      srcLane,  int width=warpSize);
T __shfl_up_sync  (unsigned mask, T value, unsigned delta,    int width=warpSize);
T __shfl_down_sync(unsigned mask, T value, unsigned delta,    int width=warpSize);
T __shfl_xor_sync (unsigned mask, T value, int      laneMask, int width=warpSize);
```

Warp shuffle 函数在 [warp](../01-introduction/programming-model.html#programming-model-warps-simt) 内的未退出线程之间交换值，而无需使用共享内存。

`__shfl_sync()`
:  从索引通道直接复制。

该内部函数返回由 `srcLane` 指定的线程所持有的 `value` 值。

- 如果 `width` 小于 `warpSize`，则 warp 的每个子部分表现为一个独立的实体，其起始逻辑通道 ID 为 0。
- 如果 `srcLane` 超出范围 [0, width - 1]，则结果对应于 `srcLane % width` 所持有的值，该值位于同一子部分内。

---

`__shfl_up_sync()`
: 从 ID 低于调用者的通道复制。

该内部函数通过从调用者的通道 ID 中减去 `delta` 来计算源通道 ID。返回由计算出的通道 ID 所持有的 `value` 值：实际上，`value` 在 warp 中向上移动了 `delta` 个通道。

- 如果 `width` 小于 `warpSize`，则 warp 的每个子部分表现为一个独立的实体，其起始逻辑通道 ID 为 0。
- 源通道索引不会环绕 `width` 的值，因此较低的 `delta` 个通道将保持不变。
---

`__shfl_down_sync()`
: 从 ID 高于调用者的通道复制。

该内置函数通过将 `delta` 加到调用者的通道 ID 来计算源通道 ID。返回由计算出的通道 ID 所持有的 `value` 值：这具有将 `value` 在线程束中向下移动 `delta` 个通道的效果。

- 如果宽度小于 warpSize，则线程束的每个子部分都作为一个独立的实体运行，其起始逻辑通道 ID 为 0。
- 与 __shfl_up_sync() 类似，源通道的 ID 号不会环绕宽度的值，因此上部的 delta 个通道将有效地保持不变。

---

`__shfl_xor_sync()`
: 基于自身通道 ID 的按位异或结果从某个通道复制。

该内置函数通过执行调用者通道 ID 与 `laneMask` 的按位异或来计算源通道 ID：返回由计算出的通道 ID 所持有的 `value` 值。此模式实现了一种蝶形寻址模式，常用于树形归约和广播操作。

- 如果宽度小于 warpSize，则每组连续的 width 个线程能够访问前面组中的元素。但是，如果它们试图访问后面线程组中的元素，则将返回它们自己的 value 值。

---

`T` 可以是：

- int、unsigned、long、unsigned long、long long、unsigned long long、float 或 double。
- 包含 cuda_fp16.h 头文件时的 __half 和 __half2。
- 包含 cuda_bf16.h 头文件时的 __nv_bfloat16 和 __nv_bfloat162。

线程只能从另一个正在积极参与该内置函数操作的线程读取数据。如果目标线程处于[非活动状态](../03-advanced/advanced-kernel-programming.html#simt-architecture-notes)，则获取的值是未定义的。

`width` 必须是范围 `[1, warpSize]` 内的 2 的幂，即 1、2、4、8、16 或 32。其他值将产生未定义的结果。

这些函数受[线程束 __sync 内置函数约束](#warp-sync-intrinsic-constraints)的约束。

有效的线程束洗牌用法示例：

```cuda
int laneId = threadIdx.x % warpSize;
int data   = ...

// 所有线程束线程从通道 0 获取 'data'
int result1 = __shfl_sync(0xFFFFFFFF, data, 0);

if (laneId < 4) {
    // 通道 0, 1, 2, 3 从通道 1 获取 'data'
    int result2 = __shfl_sync(0xb1111, data, 1);
}

// 通道 [0 - 15] 从通道 0 获取 'data'
// 通道 [16 - 31] 从通道 16 获取 'data'
int result3 = __shfl_sync(0xFFFFFFFF, value, warpSize / 2);

// 每个通道从其上方两个位置的通道获取 'data'
// 通道 30, 31 获取其原始值
int result4 = __shfl_down_sync(0xFFFFFFFF, data, 2);
```

无效的线程束洗牌用法示例：

```cuda
int laneId = threadIdx.x % warpSize;
int value  = ...
 // 未定义行为：通道 0 未参与调用
int result = (laneId > 0) ? __shfl_sync(0xFFFFFFFF, value, 0) : 0;

if (laneId <= 4) {
    // 未定义行为：对于通道 3, 4，目标通道 5, 6 不活动
    result = __shfl_down_sync(0b11111, value, 2);
}

// 未定义行为：宽度不是 2 的幂
__shfl_sync(0xFFFFFFFF, value, 0, /*width=*/31);
```
!!! warning "警告"
    这些内部函数不意味着内存屏障。它们不保证任何内存顺序。

示例 1：在单个线程束内广播单个值

 CUDA C++

```cuda
#include <cassert>
#include <cuda/warp>

__global__ void warp_broadcast_kernel(int input) {
    int laneId = threadIdx.x % 32;
    int value;
    if (laneId == 0) { // 除通道 0 外的所有线程中未使用的变量
        value = input;
    }
    value = cuda::device::warp_shuffle_idx(value, 0); // 同步线程束中的所有线程，并从通道 0 获取 "value"
    assert(value == input);
}

int main() {
    warp_broadcast_kernel<<<1, 32>>>(1234);
    cudaDeviceSynchronize();
    return 0;
}
```

 内部函数

```cuda
#include <assert.h>

__global__ void warp_broadcast_kernel(int input) {
    int laneId = threadIdx.x % 32;
    int value;
    if (laneId == 0) { // 除通道 0 外的所有线程中未使用的变量
        value = input;
    }
    value = __shfl_sync(0xFFFFFFFF, value, 0); // 同步线程束中的所有线程，并从通道 0 获取 "value"
    assert(value == input);
}

int main() {
    warp_broadcast_kernel<<<1, 32>>>(1234);
    cudaDeviceSynchronize();
    return 0;
}
```

在 [Compiler Explorer](https://cuda.godbolt.org/z/E3E3Y5e4e) 上查看此示例。

示例 2：在 8 个线程的子分区上进行包含式加和扫描

!!! note "提示"
    建议使用 cub::WarpScan 函数来实现高效且通用的线程束扫描功能。

 CUDA C++

```cuda
#include <cstdio>
#include <cub/cub.cuh>

__global__ void scan_sub_partition_with_8_threads_kernel() {
    using WarpScan    = cub::WarpScan<int, 8>;
    using TempStorage = typename WarpScan::TempStorage;
    __shared__ TempStorage temp_storage;

    int laneId = threadIdx.x % 32;
    int value  = 31 - laneId; // 要累加的起始值
    int partial_sum;
    WarpScan(temp_storage).InclusiveSum(value, partial_sum);
    printf("Thread %d final value = %d\n", threadIdx.x, partial_sum);
}

int main() {
    scan_sub_partition_with_8_threads_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

 内部函数

```cuda
#include <stdio.h>

__global__ void scan_sub_partition_with_8_threads_kernel() {
    int laneId = threadIdx.x % 32;
    int value  = 31 - laneId; // 要累加的起始值
    // 循环以在我的分区内累加扫描。
    // 对于 8 个线程，扫描需要 log2(8) == 3 步
    for (int delta = 1; delta <= 4; delta *= 2) {
        int tmp         = __shfl_up_sync(0xFFFFFFFF, value, delta, /*width=*/8); // 从 laneId - delta 读取
        int source_lane = laneId % 8 - delta;
        if (source_lane >= 0) // 'source_lane < 0' 的通道其值保持不变
            value += tmp;
    }
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    scan_sub_partition_with_8_threads_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

在 [Compiler Explorer](https://cuda.godbolt.org/z/Tohd38edc) 上查看此示例。

示例 3：跨线程束的归约
!!! note "提示"
    建议使用 cub::WarpReduce 函数来实现高效且通用的线程束规约功能。

CUDA C++

```cuda
#include <cstdio>
#include <cub/cub.cuh>
#include <cuda/warp>

__global__ void warp_reduce_kernel() {
    using WarpReduce  = cub::WarpReduce<int>;
    using TempStorage = typename WarpReduce::TempStorage;
    __shared__ TempStorage temp_storage;

    int laneId     = threadIdx.x % 32;
    int value      = 31 - laneId; // 要累加的起始值
    auto aggregate = WarpReduce(temp_storage).Sum(value);
    aggregate      = cuda::device::warp_shuffle_idx(aggregate, 0);
    printf("Thread %d final value = %d\n", threadIdx.x, aggregate);
}

int main() {
    warp_reduce_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

内联函数

```cuda
#include <stdio.h>

__global__ void warp_reduce_kernel() {
    int laneId = threadIdx.x % 32;
    int value  = 31 - laneId; // 要累加的起始值
    // 使用 XOR 模式执行蝶形规约
    // 全线程束规约需要 log2(32) == 5 步
    for (int i = 1; i <= 16; i *= 2)
        value += __shfl_xor_sync(0xFFFFFFFF, value, i);
    // 现在 "value" 包含所有线程的和
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    warp_reduce_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

查看 [Compiler Explorer](https://cuda.godbolt.org/z/T94nfGMzG) 上的示例。

### 5.4.6.6. 线程束 `__sync` 内联函数的约束

所有线程束 `__sync` 内联函数，例如：

- __shfl_sync , __shfl_up_sync , __shfl_down_sync , __shfl_xor_sync
- __match_any_sync , __match_all_sync
- __reduce_add_sync , __reduce_min_sync , __reduce_max_sync , __reduce_and_sync , __reduce_or_sync , __reduce_xor_sync
- __syncwarp

都使用 `mask` 参数来指示哪些线程束线程参与调用。此参数确保硬件在执行内联函数之前正确收敛。

`mask` 中的每一位对应一个线程的通道 ID (`threadIdx.x % warpSize`)。该内联函数会等待 `mask` 中指定的所有未退出的线程束线程都到达调用点。
要确保正确执行，必须满足以下约束：

- 每个调用线程必须在 mask 中设置其对应的位。
- 每个非调用线程必须在 mask 中将其对应的位设置为零。已退出的线程会被忽略。
- mask 中指定的所有未退出线程必须使用相同的 mask 值执行该内联函数。
- 线程束线程可以使用不同的 mask 值并发调用该内联函数，前提是这些 mask 是互斥的。即使在发散的控制流中，这种情况也是有效的。

如果出现以下情况，线程束 `__sync` 函数的行为将是无效的（例如内核挂起）或未定义的：

- 调用线程未在 mask 中指定。
- mask 中指定的某个未退出线程最终未能退出，或未能在同一程序点以相同的 mask 值调用该内联函数。
- 在条件代码中，mask 中指定的所有未退出线程的条件求值结果必须相同。
!!! note "注意"
    当所有线程束线程都参与调用时，即掩码设置为 0xFFFFFFFF 时，这些内联函数能达到最佳效率。

有效的线程束内联函数使用示例：

```cuda
__global__ void valid_examples() {
    if (threadIdx.x < 4) {        // 线程 0, 1, 2, 3 处于活动状态
        __all_sync(0b1111, pred); // 正确，线程 0, 1, 2, 3 参与调用
    }

    if (threadIdx.x == 0)
        return; // 退出
    // 正确，所有未退出的线程都参与调用
    __all_sync(0xFFFFFFFF, pred);
}
```

不连续 `mask` 示例：

```cuda
__global__ void example_syncwarp_with_mask(int* input_data, int* output_data) {
    if (threadIdx.x < warpSize) {
        __shared__ int shared_data[warpSize];
        shared_data[threadIdx.x] = input_data[threadIdx.x];

        unsigned mask = threadIdx.x < 16 ? 0xFFFF : 0xFFFF0000; // 正确
        __syncwarp(mask);
        if (threadIdx.x == 0 || threadIdx.x == 16)
            output_data[threadIdx.x] = shared_data[threadIdx.x + 1];
    }
}
```

```cuda
__global__ void example_syncwarp_with_mask_branches(int* input_data, int* output_data) {
    if (threadIdx.x < warpSize) {
        __shared__ int shared_data[warpSize];
        shared_data[threadIdx.x] = input_data[threadIdx.x];

        if (threadIdx.x < 16) {
            unsigned mask = 0xFFFF; // 正确
            __syncwarp(mask);
            output_data[threadIdx.x] = shared_data[15 - threadIdx.x];
        }
        else {
            unsigned mask = 0xFFFF0000; // 正确
            __syncwarp(mask);
            output_data[threadIdx.x] = shared_data[31 - threadIdx.x];
        }
    }
}
```

无效的线程束内联函数使用示例：

```cuda
if (threadIdx.x < 4) {           // 线程 0, 1, 2, 3 处于活动状态
    __all_sync(0b0000011, pred); // 错误，线程 2, 3 处于活动状态但未在掩码中设置
    __all_sync(0b1111111, pred); // 错误，线程 4, 5, 6 未处于活动状态但在掩码中设置
}

// 错误，参与线程具有不同且重叠的掩码
__all_sync(threadIdx.x == 0 ? 1 : 0xFFFFFFFF, pred);
```

## 5.4.7. CUDA 特定宏

### 5.4.7.1. __CUDA_ARCH__

宏 `__CUDA_ARCH__` 表示代码正在编译所针对的 NVIDIA GPU 的[虚拟架构](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-macros)。其值可能与设备的实际计算能力不同。此宏支持编写针对特定 GPU 架构优化的代码路径，这对于实现最佳性能或使用特定架构的功能和指令可能是必需的。该宏还可用于区分主机代码和设备代码。

`__CUDA_ARCH__` 仅在设备代码中定义，即在 `__device__`、`__host__ __device__` 和 `__global__` 函数中。该宏的值与 `nvcc` 选项 `compute_<version>` 相关联，关系为 `__CUDA_ARCH__ = <version> * 10`。

示例：

```cuda
nvcc --generate-code arch=compute_80,code=sm_90 prog.cu
```
将 `__CUDA_ARCH__` 定义为 `800`。

---

`__CUDA_ARCH__`**约束**

**1.** 以下实体的类型签名不得依赖于 `__CUDA_ARCH__` 是否被定义，也不得依赖于其值。

- __global__ 函数和函数模板。
- __device__ 和 __constant__ 变量。
- 纹理和表面。

示例：

```cuda
#if !defined(__CUDA_ARCH__)
    typedef int my_type;
#else
    typedef double my_type;
#endif

__device__ my_type my_var;           // 错误：my_var 的类型依赖于 __CUDA_ARCH__

__global__ void kernel(my_type in) { // 错误：kernel 的类型依赖于 __CUDA_ARCH__
    ...
}
```

**2.** 如果一个 `__global__` 函数模板从主机端实例化并启动，那么无论 `__CUDA_ARCH__` 是否被定义或其值如何，它都必须使用相同的模板参数进行实例化。

示例：

```cuda
__device__ int result;

template <typename T>
__global__ void kernel(T in) {
    result = in;
}

__host__ __device__ void host_device_function(void) {
#if !defined(__CUDA_ARCH__)
    kernel<<<1, 1>>>(1); // 错误：仅在 __CUDA_ARCH__ 未定义时实例化 "kernel<int>"！
#endif
}

int main(void) {
    host_device_function();
    cudaDeviceSynchronize();
    return 0;
}
```

**3.** 在单独编译模式下，具有外部链接的函数或变量定义的存在与否不得依赖于 `__CUDA_ARCH__` 的定义或其值。

示例：

```cuda
#if !defined(__CUDA_ARCH__)
    void host_function(void) {} // 错误：host_function() 的定义仅在 __CUDA_ARCH__
                                //        未定义时存在
#endif
```

**4.** 在单独编译中，预处理器宏 `__CUDA_ARCH__` 不得在头文件中使用，以防止对象具有不同的行为。或者，所有对象必须为相同的虚拟架构编译。如果一个弱函数或模板函数在头文件中定义，并且其行为依赖于 `__CUDA_ARCH__`，那么如果这些对象为不同的计算架构编译，则该函数在不同对象中的实例可能会发生冲突。

例如，如果头文件 `a.h` 包含：

```cuda
template<typename T>
__device__ T* get_ptr() {
#if __CUDA_ARCH__ == 900
    return nullptr; /* 无地址 */
#else
    __shared__ T arr[256];
    return arr;
#endif
}
```

那么，如果 `a.cu` 和 `b.cu` 都包含 `a.h` 并为同一类型实例化 `get_ptr()`，并且 `b.cu` 期望一个非 `NULL` 地址，并使用以下命令编译：

```cuda
nvcc -arch=compute_70 -dc a.cu
nvcc -arch=compute_80 -dc b.cu
nvcc -arch=sm_80 a.o b.o
```

在链接时只会使用 `get_ptr()` 函数的一个版本，因此行为取决于选择了哪个版本。为避免此问题，要么 `a.cu` 和 `b.cu` 必须为相同的计算架构编译，要么不应在共享的头文件函数中使用 `__CUDA_ARCH__`。

编译器不保证会为上述不支持的 `__CUDA_ARCH__` 用法生成诊断信息。
### 5.4.7.2. __CUDA_ARCH_SPECIFIC__ 和 __CUDA_ARCH_FAMILY_SPECIFIC__

宏 `__CUDA_ARCH_SPECIFIC__` 和 `__CUDA_ARCH_FAMILY_SPECIFIC__` 分别用于识别具有[特定架构](compute-capabilities.html#compute-capabilities-architecture-specific-features)和[特定系列](compute-capabilities.html#compute-capabilities-family-specific-features)功能的 GPU 设备。更多信息请参见[功能集编译器目标](compute-capabilities.html#compute-capabilities-feature-set-compiler-targets)部分。

与 `__CUDA_ARCH__` 类似，`__CUDA_ARCH_SPECIFIC__` 和 `__CUDA_ARCH_FAMILY_SPECIFIC__` 仅在设备代码中定义，即在 `__device__`、`__host__ __device__` 和 `__global__` 函数中。这些宏与 `nvcc` 选项 `compute_<version>a` 和 `compute_<version>f` 相关联。

```cuda
nvcc --generate-code arch=compute_100a,code=sm_100a prog.cu
```

- __CUDA_ARCH__ == 1000 .
- __CUDA_ARCH_SPECIFIC__ == 1000 .
- __CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 .

```cuda
nvcc --generate-code arch=compute_100f,code=sm_103f prog.cu
```

- __CUDA_ARCH__ == 1000 .
- __CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 .
- __CUDA_ARCH_SPECIFIC__ 未定义。

```cuda
nvcc -arch=sm_100 prog.cu
```

- __CUDA_ARCH__ == 1000 .
- __CUDA_ARCH_FAMILY_SPECIFIC__ 未定义。
- __CUDA_ARCH_SPECIFIC__ 未定义。

```cuda
nvcc -arch=sm_100a prog.cu
# 等价于：
nvcc --generate-code arch=sm_100a,compute_100,compute_100a prog.cu
```

- __CUDA_ARCH__ == 1000 .
- __CUDA_ARCH_FAMILY_SPECIFIC__ 未定义。
- 同时生成了 __CUDA_ARCH_SPECIFIC__ == 1000 和 __CUDA_ARCH_SPECIFIC__ 未定义的情况。

### 5.4.7.3. CUDA 功能测试宏

`nvcc` 提供了以下预处理器宏用于功能测试。当 CUDA 前端编译器支持特定功能时，会定义相应的宏。

- __CUDACC_DEVICE_ATOMIC_BUILTINS__ : 支持设备原子编译器内置函数。
- __NVCC_DIAG_PRAGMA_SUPPORT__ : 支持诊断控制编译指示。
- __CUDACC_EXTENDED_LAMBDA__ : 支持扩展 lambda 表达式。通过 `--expt-extended-lambda` 或 `--extended-lambda` 标志启用。
- __CUDACC_RELAXED_CONSTEXPR__ : 支持宽松的 constexpr 函数。通过 `--expt-relaxed-constexpr` 标志启用。

### 5.4.7.4. __nv_pure__ 属性

在 C/C++ 中，纯函数对其参数没有副作用，并且可以访问全局变量，但不会修改它们。

CUDA 提供了 `__nv_pure__` 属性，同时支持主机和设备函数。编译器将 `__nv_pure__` 转换为 GNU 的 `pure` 属性或 Microsoft Visual Studio 的 `noalias` 属性。

```cuda
__device__ __nv_pure__
int add(int a, int b) {
    return a + b;
}
```

## 5.4.8. CUDA 特定函数

### 5.4.8.1. 地址空间谓词函数

地址空间谓词函数用于确定指针的地址空间。

!!! note "提示"
    建议使用 libcu++ 提供的 `cuda::device::is_address_from()` 和 `cuda::device::is_object_from()` 函数作为地址空间谓词内置函数的可移植且更安全的替代方案。

```cuda
__device__ unsigned __isGlobal      (const void* ptr);
__device__ unsigned __isShared      (const void* ptr);
__device__ unsigned __isConstant    (const void* ptr);
__device__ unsigned __isGridConstant(const void* ptr);
__device__ unsigned __isLocal       (const void* ptr);
```

The functions return `1` if `ptr` contains the generic address of an object in the specified address space, `0` otherwise. Their behavior is unspecified if the argument is a `NULL` pointer.

- __isGlobal() : global memory space.
- __isShared() : shared memory space.
- __isConstant() : constant memory space.
- __isGridConstant() : kernel parameter annotated with __grid_constant__ .
- __isLocal() : local memory space.

### 5.4.8.2.Address Space Conversion Functions

CUDA pointers (`T*`) can access objects regardless of where the objects are stored. For example, an `int*` can access `int` objects whether they reside in global or shared memory.

Address space conversion functions are used to convert between generic addresses and addresses in specific address spaces. These functions are useful when the compiler cannot determine a pointerâs address space, for example, when crossing translation units or interacting with PTX instructions.

```cuda
__device__ size_t __cvta_generic_to_global  (const void* ptr); // PTX: cvta.to.global
__device__ size_t __cvta_generic_to_shared  (const void* ptr); // PTX: cvta.to.shared
__device__ size_t __cvta_generic_to_constant(const void* ptr); // PTX: cvta.to.const
__device__ size_t __cvta_generic_to_local   (const void* ptr); // PTX: cvta.to.local
```

```cuda
__device__ void* __cvta_global_to_generic  (size_t raw_ptr); // PTX: cvta.global
__device__ void* __cvta_shared_to_generic  (size_t raw_ptr); // PTX: cvta.shared
__device__ void* __cvta_constant_to_generic(size_t raw_ptr); // PTX: cvta.const
__device__ void* __cvta_local_to_generic   (size_t raw_ptr); // PTX: cvta.local
```

As an example of inter-operating with PTX instructions, the `ld.shared.s32 r0, [ptr];` PTX instruction expects `ptr` to refer to the shared memory address space. A CUDA program with an `int*` pointer to an object in `__shared__` memory needs to convert this pointer to the shared address space before passing it to the PTX instruction by calling `__cvta_generic_to_shared` as follows:

```cuda
__shared__ int smem_var;
smem_var        = 42;
size_t smem_ptr = __cvta_generic_to_shared(&smem_var);
int    output;
asm volatile("ld.shared.s32 %0, [%1];" : "=r"(output) : "l"(smem_ptr) : "memory");
assert(output == 42);
```

A common optimization that exploits these address representations is reducing data structure size by leveraging the fact that the address ranges of shared, local, and constant spaces are smaller than 32 bits, which allows storing 32-bit addresses instead of 64-bit pointers and save registers. Additionally, 32-bit arithmetic is faster than 64-bit arithmetic. To obtain the 32-bit integer representation of these addresses, truncate the 64-bit value to 32 bits by casting from an unsigned 64-bit integer to an unsigned 32-bit integer:

```cuda
__shared__ int smem_var;
uint32_t       smem_ptr_32bit = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_var));
```

To recover a generic address from such a 32-bit representation, zero-extend the address back to an unsigned 64-bit integer and then call the corresponding address space conversion function:

```cuda
size_t smem_ptr_64bit = static_cast<size_t>(smem_ptr_32bit); // zero-extend to 64 bits
void*  generic_ptr    = __cvta_shared_to_generic(smem_ptr_64bit);
assert(generic_ptr == &smem_var);
```

---

### 5.4.8.3.Low-Level Load and Store Functions

```cuda
T __ldg(const T* address);
```

The function `__ldg()` performs a read-only L1/Tex cache load. It supports all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

---

```cuda
T __ldcg(const T* address);
T __ldca(const T* address);
T __ldcs(const T* address);
T __ldlu(const T* address);
T __ldcv(const T* address);
```

The functions perform a load using the cache operator specified in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) guide. They support all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

---

```cuda
void __stwb(T* address, T value);
void __stcg(T* address, T value);
void __stcs(T* address, T value);
void __stwt(T* address, T value);
```

The functions perform a store using the cache operator specified in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) guide. They support all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

### 5.4.8.4.__trap()

!!! note "Hint"
    It is suggested to use the cuda::std::terminate() function provided by libcu++ ( C++ reference ) as a portable alternative to __trap() .

A trap operation can be initiated by calling the `__trap()` function from any device thread.

```cuda
void __trap();
```

Execution of the kernel is aborted, raising an interrupt in the host program. Calling `__trap()` results in a corrupted CUDA context, causing subsequent CUDA calls and kernel invocations to fail.

### 5.4.8.5.__nanosleep()

```cuda
__device__ void __nanosleep(unsigned nanoseconds);
```

The function `__nanosleep(ns)` suspends the thread for a sleep duration of approximately `ns` nanoseconds. The maximum sleep duration is approximately one millisecond.

Example:

The following code implements a mutex with exponential back-off.

```cuda
__device__ void mutex_lock(unsigned* mutex) {
    unsigned ns = 8;
    while (atomicCAS(mutex, 0, 1) == 1) {
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
    }
}

__device__ void mutex_unlock(unsigned *mutex) {
    atomicExch(mutex, 0);
}
```
### 5.4.8.6. 动态规划扩展 (DPX) 指令

DPX 函数集支持查找最小值和最大值，以及对最多三个 16 位或 32 位有符号或无符号整数参数进行融合加法与最小值/最大值运算。它包含一个可选的 ReLU（即钳位到零）功能。

比较函数：

- 三个参数。语义：max(a, b, c) , min(a, b, c) 。

```cuda
     int __vimax3_s32  (     int,      int,      int);
unsigned __vimax3_s16x2(unsigned, unsigned, unsigned);
unsigned __vimax3_u32  (unsigned, unsigned, unsigned);
unsigned __vimax3_u16x2(unsigned, unsigned, unsigned);

     int __vimin3_s32  (     int,      int,      int);
unsigned __vimin3_s16x2(unsigned, unsigned, unsigned);
unsigned __vimin3_u32  (unsigned, unsigned, unsigned);
unsigned __vimin3_u16x2(unsigned, unsigned, unsigned);
```

- 两个参数，带 ReLU。语义：max(a, b, 0) , max(min(a, b), 0) 。

```cuda
     int __vimax_s32_relu  (     int,      int);
unsigned __vimax_s16x2_relu(unsigned, unsigned);

     int __vimin_s32_relu  (     int,      int);
unsigned __vimin_s16x2_relu(unsigned, unsigned);
```

- 三个参数，带 ReLU。语义：max(a, b, c, 0) , max(min(a, b, c), 0) 。

```cuda
     int __vimax3_s32_relu  (     int,      int,      int);
unsigned __vimax3_s16x2_relu(unsigned, unsigned, unsigned);

     int __vimin3_s32_relu  (     int,      int,      int);
unsigned __vimin3_s16x2_relu(unsigned, unsigned, unsigned);
```

- 两个参数，同时返回哪个参数更小/更大：

```cuda
     int __vibmax_s32  (     int,      int, bool* pred);
unsigned __vibmax_u32  (unsigned, unsigned, bool* pred);
unsigned __vibmax_s16x2(unsigned, unsigned, bool* pred);
unsigned __vibmax_u16x2(unsigned, unsigned, bool* pred);

     int __vibmin_s32  (     int,      int, bool* pred);
unsigned __vibmin_u32  (unsigned, unsigned, bool* pred);
unsigned __vibmin_s16x2(unsigned, unsigned, bool* pred);
unsigned __vibmin_u16x2(unsigned, unsigned, bool* pred);
```

融合加法与最小值/最大值：

- 三个参数，比较（第一个 + 第二个）与第三个。语义：max(a + b, c) , min(a + b, c)

```cuda
     int __viaddmax_s32  (     int,     int,       int);
unsigned __viaddmax_s16x2(unsigned, unsigned, unsigned);
unsigned __viaddmax_u32  (unsigned, unsigned, unsigned);
unsigned __viaddmax_u16x2(unsigned, unsigned, unsigned);

     int __viaddmin_s32  (     int,     int,       int);
unsigned __viaddmin_s16x2(unsigned, unsigned, unsigned);
unsigned __viaddmin_u32  (unsigned, unsigned, unsigned);
unsigned __viaddmin_u16x2(unsigned, unsigned, unsigned);
```

- 三个参数，带 ReLU，比较（第一个 + 第二个）与第三个以及零。语义：max(a + b, c, 0) , max(min(a + b, c), 0)

```cuda
     int __viaddmax_s32_relu  (     int,      int,      int);
unsigned __viaddmax_s16x2_relu(unsigned, unsigned, unsigned);

     int __viaddmin_s32_relu  (     int,      int,      int);
unsigned __viaddmin_s16x2_relu(unsigned, unsigned, unsigned);
```
这些指令根据计算能力的不同，可能由硬件加速或软件模拟实现。有关计算能力要求，请参阅[算术指令](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions)章节。

完整的 API 可在 [CUDA 数学 API 文档](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html) 中找到。

---

DPX 是实现动态规划算法的极其有用的工具，例如基因组学中的 Smith-Waterman 和 Needleman-Wunsch 算法，以及路径优化中的 Floyd-Warshall 算法。

三个有符号 32 位整数的最大值，带 ReLU：

```cuda
int a           = -15;
int b           = 8;
int c           = 5;
int max_value_0 = __vimax3_s32_relu(a, b, c); // max(-15, 8, 5, 0) = 8
int d           = -2;
int e           = -4;
int max_value_1 = __vimax3_s32_relu(a, d, e); // max(-15, -2, -4, 0) = 0
```

两个 32 位有符号整数之和、另一个 32 位有符号整数与零（ReLU）的最小值：

```cuda
int a           = -5;
int b           = 6;
int c           = -2;
int max_value_0 = __viaddmax_s32_relu(a, b, c); // max(-5 + 6, -2, 0) = max(1, -2, 0) = 1
int d           = 4;
int max_value_1 = __viaddmax_s32_relu(a, d, c); // max(-5 + 4, -2, 0) = max(-1, -2, 0) = 0
```

两个无符号 32 位整数的最小值，并确定哪个值更小：

```cuda
unsigned a = 9;
unsigned b = 6;
bool     smaller_value;
unsigned min_value = __vibmin_u32(a, b, &smaller_value); // min_value is 6, smaller_value is true
```

三对无符号 16 位整数的最大值：

```cuda
unsigned a         = 0x00050002;
unsigned b         = 0x00070004;
unsigned c         = 0x00020006;
unsigned max_value = __vimax3_u16x2(a, b, c); // max(5, 7, 2) and max(2, 4, 6), so max_value is 0x00070006
```

## 5.4.9. 编译器优化提示

编译器优化提示通过附加信息修饰代码，以帮助编译器优化生成的代码。

- 内置函数在设备代码中始终可用。
- 主机代码的支持取决于主机编译器。

### 5.4.9.1. #pragma unroll

编译器默认会对已知迭代次数的小循环进行展开。但是，可以使用 `#pragma unroll` 指令来控制任何给定循环的展开。此指令必须紧接在循环之前，并且仅适用于该循环。

可以选择性地跟随一个整型常量表达式。以下是整型常量表达式的情况：

- 如果省略，当循环的迭代次数为常量时，该循环将被完全展开。
- 如果其求值结果为 0 或 1，则循环将不会被展开。
- 如果它是一个非正整数或大于 INT_MAX，则该编译指示将被忽略，并会发出警告。

示例：

```cuda
struct MyStruct {
    static constexpr int value = 4;
};

inline constexpr int Count = 4;

__device__ void foo(int* p1, int* p2) {
    // 未指定参数，循环将被完全展开
    #pragma unroll
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i] * 2;

    // 展开值 = 5
    #pragma unroll (Count + 1)
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i] * 4;

    // 展开值 = 1，禁用循环展开
    #pragma unroll 1
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i] * 8;

    // 展开值 = 4
    #pragma unroll (MyStruct::value)
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i] * 16;

    // 负值，忽略 #pragma unroll
    #pragma unroll -1
    for (int i = 0; i < 12; ++i)
        p1[i] += p2[i] * 2;
}
```
请参阅 [Compiler Explorer](https://godbolt.org/z/fPMK55PxE) 上的示例。

### 5.4.9.2.__builtin_assume_aligned()

!!! note "提示"
    建议使用 libcu++（C++ 参考）提供的 `cuda::std::assume_aligned()` 函数作为内置函数的可移植且更安全的替代方案。

```cuda
void* __builtin_assume_aligned(const void* ptr, size_t align)
void* __builtin_assume_aligned(const void* ptr, size_t align, <integral type> offset)
```

这些内置函数使编译器能够假设返回的指针至少按 `align` 字节对齐。

- 三参数版本使编译器能够假设 `(char*) ptr - offset` 至少按 `align` 字节对齐。

`align` 必须是 2 的幂且为整数字面量。

示例：

```cuda
void* res1 = __builtin_assume_aligned(ptr, 32);    // 编译器可以假设 'res1' 至少 32 字节对齐
void* res2 = __builtin_assume_aligned(ptr, 32, 8); // 编译器可以假设 'res2 = (char*) ptr - 8' 至少 32 字节对齐
```

### 5.4.9.3.__builtin_assume() 与 __assume()

```cuda
void __builtin_assume(bool predicate)
void __assume        (bool predicate) // 仅适用于 Microsoft 编译器
```

此内置函数使编译器能够假设布尔参数为真。如果该参数在运行时为假，则行为未定义。请注意，如果该参数有副作用，则行为未指定。

示例：

```cuda
__device__ bool is_greater_than_zero(int value) {
    return value > 0;
}

__device__ bool f(int value) {
    __builtin_assume(value > 0);
    return is_greater_than_zero(value); // 返回 true，无需评估条件
}
```

### 5.4.9.4.__builtin_expect()

```cuda
long __builtin_expect(long input, long expected)
```

此内置函数告诉编译器 `input` 预期等于 `expected`，并返回 `input` 的值。它通常用于向编译器提供分支预测信息。其行为类似于 C++20 的 `[[likely]]` 和 `[[unlikely]]` [属性](https://en.cppreference.com/w/cpp/language/attributes/likely)。

示例：

```cuda
// 向编译器指示很可能 "var == 0"
if (__builtin_expect(var, 0))
    doit();
```

### 5.4.9.5.__builtin_unreachable()

```cuda
void __builtin_unreachable(void)
```

此内置函数告诉编译器控制流永远不会到达调用该函数的位置。如果控制流在运行时确实到达此点，则程序行为未定义。

此函数可用于避免为不可达分支生成代码，并禁用针对不可达代码的编译器警告。

示例：

```cuda
// 向编译器指示永远不会到达 default case 标签。
switch (in) {
    case 1:  return 4;
    case 2:  return 10;
    default: __builtin_unreachable();
}
```

### 5.4.9.6.自定义 ABI 编译指示

`#pragma nv_abi` 指令使在[单独编译](../02-basics/nvcc.html#nvcc-separate-compilation)模式下编译的应用程序，能够通过保留函数使用的寄存器数量，实现与[全程序编译](../02-basics/nvcc.html#nvcc-separate-compilation)类似的性能。
使用此编译指示的语法如下，其中 `EXPR` 指任何整型常量表达式：

```cuda
#pragma nv_abi preserve_n_data(EXPR) preserve_n_control(EXPR)
```

- `#pragma nv_abi` 后面的参数是可选的，可以以任意顺序提供；但至少需要一个参数。
- `preserve_n` 参数限制函数调用期间保留的寄存器数量：`preserve_n_data(EXPR)` 限制数据寄存器的数量。`preserve_n_control(EXPR)` 限制控制寄存器的数量。

`#pragma nv_abi` 指令可以紧接在设备函数声明或定义之前放置。

```cuda
#pragma nv_abi preserve_n_data(16)
__device__ void dev_func();

#pragma nv_abi preserve_n_data(16) preserve_n_control(8)
__device__ int dev_func() {
    return 0;
}
```

或者，它可以放置在设备函数内部的 C++ 表达式语句中，紧接在间接函数调用之前。请注意，虽然支持对自由函数的间接调用，但不支持对函数引用或类成员函数的间接调用。

```cuda
__device__ int dev_func1();

struct MyStruct {
    __device__ int member_func2();
};

__device__ void test() {
    auto* dev_func_ptr = &dev_func1; // 类型: int (*)(void)
    #pragma nv_abi preserve_n_control(8)
    int v1 = dev_func_ptr();         // 正确，间接调用

    #pragma nv_abi preserve_n_control(8)
    int v2 = dev_func1();            // 错误，直接调用；编译指示无效
                                     // dev_func1 的类型: int(void)

    auto& dev_func_ref = &dev_func1; // 类型: int (&)(void)
    #pragma nv_abi preserve_n_control(8)
    int v3 = dev_func_ref();         // 错误，对引用的调用
                                     // 编译指示无效

    auto member_function_ptr = &MyStruct::member_func2; // 类型: int (MyStruct::*)(void)
    #pragma nv_abi preserve_n_control(8)
    int v4 = member_function_ptr();  // 错误，对成员函数的间接调用
                                     // 编译指示无效
}
```

当应用于设备函数的声明或定义时，该编译指示会修改对该函数任何调用的自定义 ABI 属性。当放置在间接函数调用点时，它仅影响该特定调用的 ABI 属性。请注意，该编译指示仅在放置在调用点时影响间接函数调用；它对直接函数调用没有影响。

```cuda
#pragma nv_abi preserve_n_control(8)
__device__ int dev_func3();

__device__ int dev_func4();

__device__ void test() {
    int v1 = dev_func3();            // 正确，编译指示影响直接调用

    auto* dev_func_ptr = &dev_func4; // 类型: int (*)(void)
    #pragma nv_abi preserve_n_control(8)
    int v2 = dev_func_ptr();         // 正确，编译指示影响间接调用

    int v3 = dev_func_ptr();         // 错误，编译指示无效
}
```

请注意，如果函数声明及其对应定义的编译指示参数不匹配，则程序格式错误。
## 5.4.10. 调试与诊断

### 5.4.10.1. 断言

```cuda
void assert(int expression);
```

如果 `expression` 等于零，`assert()` 宏将停止内核的执行。如果程序在调试器中运行，则会触发断点，允许使用调试器来检查设备的当前状态。否则，对于 `expression` 等于零的每个线程，在通过 `cudaDeviceSynchronize()`、`cudaStreamSynchronize()` 或 `cudaEventSynchronize()` 与主机同步后，会向 stderr 打印一条消息。该消息的格式如下：

```cuda
<文件名>:<行号>:<函数名>:
block: [blockIdx.x,blockIdx.y,blockIdx.z],
thread: [threadIdx.x,threadIdx.y,threadIdx.z]
Assertion `<表达式>` failed.
```

内核的执行被中止，并在主机程序中引发中断。`assert()` 宏会导致 CUDA 上下文损坏，使得任何后续的 CUDA 调用或内核启动都会失败并返回 `cudaErrorAssert`。

如果 `expression` 不等于零，则内核执行不受影响。

例如，源文件 `test.cu` 中的以下程序：

```cuda
#include <assert.h>

 __global__ void testAssert(void) {
     int is_one        = 1;
     int should_be_one = 0;

     // 这将不会产生任何效果
     assert(is_one);

     // 这将停止内核执行
     assert(should_be_one);
 }

 int main(void) {
     testAssert<<<1,1>>>();
     cudaDeviceSynchronize();
     return 0;
 }
```

将输出：

```cuda
test.cu:11: void testAssert(): block: [0,0,0], thread: [0,0,0] Assertion `should_be_one` failed.
```

断言旨在用于调试目的。由于它们可能影响性能，建议在生产代码中禁用断言。可以通过在包含 `assert.h` 或 `<cassert>` 之前定义 `NDEBUG` 预处理器宏，或使用编译器标志 `-DNDEBUG` 在编译时禁用断言。请注意，表达式不应有副作用；否则，禁用断言将影响代码的功能。

### 5.4.10.2. 断点函数

通过从任何设备线程调用 `__brkpt()` 函数，可以暂停内核函数的执行。

```cuda
void __brkpt();
```

### 5.4.10.3. 诊断编译指示

以下编译指示可用于管理当引发特定诊断消息时触发的错误的严重性。

```cuda
#pragma nv_diag_suppress
#pragma nv_diag_warning
#pragma nv_diag_error
#pragma nv_diag_default
#pragma nv_diag_once
```

这些编译指示的用法如下：

```cuda
#pragma nv_diag_xxx <错误编号1>, <错误编号2> ...
```

受影响的诊断通过警告消息中显示的错误编号来指定。任何诊断都可以被更改为错误，但只有警告可以在被更改为错误后，其严重性被抑制或恢复。`nv_diag_default` 编译指示将诊断的严重性恢复到发出任何其他编译指示之前生效的严重性，即由任何命令行选项修改后的消息的正常严重性。以下示例抑制了 `foo()` 的 `declared but never referenced` 警告：

```cuda
#pragma nv_diag_suppress 177 // "declared but never referenced"
void foo() {
    int i = 0;
}

#pragma nv_diag_default 177
void bar() {
    int i = 0;
}
```

The following pragmas may be used to save and restore the current diagnostic pragma state:

```cuda
#pragma nv_diagnostic push
#pragma nv_diagnostic pop
```

Examples:

```cuda
#pragma nv_diagnostic push
#pragma nv_diag_suppress 177 // "declared but never referenced"
void foo() {
    int i = 0;
}

#pragma nv_diagnostic pop
void bar() {
    int i = 0; // raise a warning
}
```

Note that these directives only affect the `nvcc` CUDA front-end compiler. They have no effect on the host compiler.

`nvcc` defines the macro `__NVCC_DIAG_PRAGMA_SUPPORT__` when diagnostic pragmas are supported.

## 5.4.11.Warp Matrix Functions

C++ warp matrix operations leverage Tensor Cores to accelerate matrix problems of the form `D=A*B+C`. These operations are supported on mixed-precision floating point data for devices of compute capability 7.0 or higher. This requires co-operation from all threads in a [warp](../01-introduction/programming-model.html#programming-model-warps-simt). In addition, these operations are allowed in conditional code only if the condition evaluates identically across the entire [warp](../01-introduction/programming-model.html#programming-model-warps-simt), otherwise the code execution is likely to hang.

### 5.4.11.1.Description

All following functions and types are defined in the namespace `nvcuda::wmma`. Sub-byte operations are considered preview, i.e. the data structures and APIs for them are subject to change and may not be compatible with future releases. This extra functionality is defined in the `nvcuda::wmma::experimental` namespace.

```cuda
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
```

`fragment`

An overloaded class containing a section of a matrix distributed across all threads in the warp. The mapping of matrix elements into `fragment` internal storage is unspecified and subject to change in future architectures.

Only certain combinations of template arguments are allowed. The first template parameter specifies how the fragment will participate in the matrix operation. Acceptable values for `Use` are:

- matrix_a when the fragment is used as the first multiplicand, A ,
- matrix_b when the fragment is used as the second multiplicand, B , or
- accumulator when the fragment is used as the source or destination accumulators ( C or D , respectively). The m , n and k sizes describe the shape of the warp-wide matrix tiles participating in the multiply-accumulate operation. The dimension of each tile depends on its role. For matrix_a the tile takes dimension m x k ; for matrix_b the dimension is k x n , and accumulator tiles are m x n . The data type, T , may be double , float , __half , __nv_bfloat16 , char , or unsigned char for multiplicands and double , float , int , or __half for accumulators. As documented in Element Types and Matrix Sizes , limited combinations of accumulator and multiplicand types are supported. The Layout parameter must be specified for matrix_a and matrix_b fragments. row_major or col_major indicate that elements within a matrix row or column are contiguous in memory, respectively. The Layout parameter for an accumulator matrix should retain the default value of void . A row or column layout is specified only when the accumulator is loaded or stored as described below.
`load_matrix_sync`

等待线程束中的所有通道都到达 `load_matrix_sync`，然后从内存加载矩阵片段 `a`。`mptr` 必须是一个 256 位对齐的指针，指向内存中矩阵的第一个元素。`ldm` 描述了连续行（对于行主序布局）或列（对于列主序布局）之间的元素跨度，对于 `__half` 元素类型必须是 8 的倍数，对于 `float` 元素类型必须是 4 的倍数（即，两种情况下都是 16 字节的倍数）。如果片段是 `accumulator`，则必须将 `layout` 参数指定为 `mem_row_major` 或 `mem_col_major`。对于 `matrix_a` 和 `matrix_b` 片段，布局是从片段的 `layout` 参数推断出来的。`mptr`、`ldm`、`layout` 以及 `a` 的所有模板参数的值，对于线程束中的所有线程必须相同。此函数必须由线程束中的所有线程调用，否则结果未定义。

`store_matrix_sync`

等待线程束中的所有通道都到达 `store_matrix_sync`，然后将矩阵片段 `a` 存储到内存。`mptr` 必须是一个 256 位对齐的指针，指向内存中矩阵的第一个元素。`ldm` 描述了连续行（对于行主序布局）或列（对于列主序布局）之间的元素跨度，对于 `__half` 元素类型必须是 8 的倍数，对于 `float` 元素类型必须是 4 的倍数（即，两种情况下都是 16 字节的倍数）。输出矩阵的布局必须指定为 `mem_row_major` 或 `mem_col_major`。`mptr`、`ldm`、`layout` 以及 `a` 的所有模板参数的值，对于线程束中的所有线程必须相同。

`fill_fragment`

用常量值 `v` 填充矩阵片段。由于矩阵元素到每个片段的映射是未指定的，此函数通常由线程束中的所有线程调用，并且 `v` 的值是公共的。

`mma_sync`

等待线程束中的所有通道都到达 `mma_sync`，然后执行线程束同步的矩阵乘加运算 `D=A*B+C`。也支持原地操作 `C=A*B+C`。`satf` 的值以及每个矩阵片段的模板参数，对于线程束中的所有线程必须相同。此外，片段 `A`、`B`、`C` 和 `D` 之间的模板参数 `m`、`n` 和 `k` 必须匹配。此函数必须由线程束中的所有线程调用，否则结果未定义。

如果 `satf`（饱和到有限值）模式为 `true`，则目标累加器适用以下附加数值属性：

- 如果元素结果为 +Infinity，则相应的累加器将包含 +MAX_NORM
- 如果元素结果为 -Infinity，则相应的累加器将包含 -MAX_NORM
- 如果元素结果为 NaN，则相应的累加器将包含 +0

由于矩阵元素到每个线程的 `fragment` 的映射是未指定的，因此在调用 `store_matrix_sync` 之后，必须从内存（共享内存或全局内存）访问各个矩阵元素。在特殊情况下，如果线程束中的所有线程将对所有片段元素统一应用逐元素操作，则可以使用以下 `fragment` 类成员实现直接元素访问。

```cuda
enum fragment<Use, m, n, k, T, Layout>::num_elements;
T fragment<Use, m, n, k, T, Layout>::x[num_elements];
```

As an example, the following code scales an `accumulator` matrix tile by half.

```cuda
wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;
float alpha = 0.5f; // Same value for all threads in warp
/*...*/
for(int t=0; t<frag.num_elements; t++)
frag.x[t] *= alpha;
```

### 5.4.11.2.Alternate Floating Point

Tensor Cores support alternate types of floating point operations on devices with compute capability 8.0 and higher.

`__nv_bfloat16`

This data format is an alternate fp16 format that has the same range as f32 but reduced precision (7 bits). You can use this data format directly with the `__nv_bfloat16` type available in `cuda_bf16.h`. Matrix fragments with `__nv_bfloat16` data types are required to be composed with accumulators of `float` type. The shapes and operations supported are the same as with `__half`.

`tf32`

This data format is a special floating-point format supported by Tensor Cores, with the same range as f32 and reduced precision (>=10 bits). The internal layout of this format is implementation-defined. To use this floating-point format with WMMA operations, the input matrices must be manually converted to tf32 precision.

To facilitate conversion, a new intrinsic `__float_to_tf32` is provided. While the input and output arguments to the intrinsic are of `float` type, the output will be `tf32` numerically. This new precision is intended to be used with Tensor Cores only, and if mixed with other `float`type operations, the precision and range of the result will be undefined.

Once an input matrix (`matrix_a` or `matrix_b`) is converted to tf32 precision, the combination of a `fragment` with `precision::tf32` precision, and a data type of `float` to `load_matrix_sync` will take advantage of this new capability. Both the accumulator fragments must have `float` data types. The only supported matrix size is 16x16x8 (m-n-k).

The elements of the fragment are represented as `float`, hence the mapping from `element_type<T>` to `storage_element_type<T>` is:

```cuda
precision::tf32 -> float
```

### 5.4.11.3.Double Precision

Tensor Cores support double-precision floating point operations on devices with compute capability 8.0 and higher. To use this new functionality, a `fragment` with the `double` type must be used. The `mma_sync` operation will be performed with the .rn (rounds to nearest even) rounding modifier.

### 5.4.11.4.Sub-byte Operations

Sub-byte WMMA operations provide a way to access the low-precision capabilities of Tensor Cores. They are considered a preview feature i.e. the data structures and APIs for them are subject to change and may not be compatible with future releases. This functionality is available via the `nvcuda::wmma::experimental` namespace:

```cuda
namespace experimental {
    namespace precision {
        struct u4; // 4-bit unsigned
        struct s4; // 4-bit signed
        struct b1; // 1-bit
   }
    enum bmmaBitOp {
        bmmaBitOpXOR = 1, // compute_75 minimum
        bmmaBitOpAND = 2  // compute_80 minimum
    };
    enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 };
}
```
对于 4 位精度，可用的 API 保持不变，但必须指定 `experimental::precision::u4` 或 `experimental::precision::s4` 作为片段数据类型。由于片段的元素被打包在一起，因此该片段的 `num_storage_elements` 将小于 `num_elements`。对于亚字节片段，`num_elements` 变量返回的是亚字节类型 `element_type<T>` 的元素数量。这对于单比特精度同样适用，在这种情况下，从 `element_type<T>` 到 `storage_element_type<T>` 的映射如下：

```cuda
experimental::precision::u4 -> unsigned (1 个存储元素中包含 8 个元素)
experimental::precision::s4 -> int (1 个存储元素中包含 8 个元素)
experimental::precision::b1 -> unsigned (1 个存储元素中包含 32 个元素)
T -> T  // 所有其他类型
```

对于亚字节片段，允许的布局始终是 `matrix_a` 为 `row_major`，`matrix_b` 为 `col_major`。

对于亚字节操作，在 `load_matrix_sync` 中，元素类型为 `experimental::precision::u4` 和 `experimental::precision::s4` 时，`ldm` 的值应为 32 的倍数；元素类型为 `experimental::precision::b1` 时，`ldm` 的值应为 128 的倍数（即，在这两种情况下都是 16 字节的倍数）。

!!! note "注意"
    以下 MMA 指令变体的支持已被弃用，并将在 sm_90 中移除：将 `bmmaBitOp` 设置为 `bmmaBitOpXOR` 的 `experimental::precision::u4`、`experimental::precision::s4`、`experimental::precision::b1`。

`bmma_sync`

等待所有线程束通道都执行了 `bmma_sync`，然后执行线程束同步的位矩阵乘加操作 `D = (A op B) + C`，其中 `op` 由逻辑操作 `bmmaBitOp` 和由 `bmmaAccumulateOp` 定义的累加操作组成。可用的操作有：

`bmmaBitOpXOR`，对 `matrix_a` 中的一行与 `matrix_b` 中的 128 位列进行 128 位异或操作。

`bmmaBitOpAND`，对 `matrix_a` 中的一行与 `matrix_b` 中的 128 位列进行 128 位与操作，适用于计算能力 8.0 及更高的设备。

累加操作始终是 `bmmaAccumulateOpPOPC`，它计算置位位的数量。

### 5.4.11.5. 限制

张量核心所需的特殊格式可能因每个主要和次要设备架构而异。由于线程仅持有整个矩阵的一个片段（不透明的、架构特定的 ABI 数据结构），并且开发者不允许对各个参数如何映射到参与矩阵乘加操作的寄存器做出假设，这使得情况更加复杂。

由于片段是架构特定的，如果函数 A 和函数 B 已为不同的链接兼容架构编译并链接到同一个设备可执行文件中，那么将片段从函数 A 传递到函数 B 是不安全的。在这种情况下，片段的大小和布局将特定于一种架构，而在另一种架构中使用 WMMA API 将导致不正确的结果或潜在的损坏。

片段布局不同的两个链接兼容架构的例子是 sm_70 和 sm_75。

```cuda
fragA.cu: void foo() { wmma::fragment<...> mat_a; bar(&mat_a); }
fragB.cu: void bar(wmma::fragment<...> *mat_a) { // operate on mat_a }
```

```cuda
// sm_70 fragment layout
$> nvcc -dc -arch=compute_70 -code=sm_70 fragA.cu -o fragA.o
// sm_75 fragment layout
$> nvcc -dc -arch=compute_75 -code=sm_75 fragB.cu -o fragB.o
// Linking the two together
$> nvcc -dlink -arch=sm_75 fragA.o fragB.o -o frag.o
```

This undefined behavior might also be undetectable at compilation time and by tools at runtime, so extra care is needed to make sure the layout of the fragments is consistent. This linking hazard is most likely to appear when linking with a legacy library that is both built for a different link-compatible architecture and expecting to be passed a WMMA fragment.

Note that in the case of weak linkages (for example, a CUDA C++ inline function), the linker may choose any available function definition which may result in implicit passes between compilation units.

To avoid these sorts of problems, the matrix should always be stored out to memory for transit through external interfaces (e.g. `wmma::store_matrix_sync(dst, â¦);`) and then it can be safely passed to `bar()` as a pointer type [e.g. `float *dst`].

Note that since sm_70 can run on sm_75, the above example sm_75 code can be changed to sm_70 and correctly work on sm_75. However, it is recommended to have sm_75 native code in your application when linking with other sm_75 separately compiled binaries.

### 5.4.11.6.Element Types and Matrix Sizes

Tensor Cores support a variety of element types and matrix sizes. The following table presents the various combinations of `matrix_a`, `matrix_b` and `accumulator` matrix supported:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| __half | __half | float | 16x16x16 |
| __half | __half | float | 32x8x16 |
| __half | __half | float | 8x32x16 |
| __half | __half | __half | 16x16x16 |
| __half | __half | __half | 32x8x16 |
| __half | __half | __half | 8x32x16 |
| unsigned char | unsigned char | int | 16x16x16 |
| unsigned char | unsigned char | int | 32x8x16 |
| unsigned char | unsigned char | int | 8x32x16 |
| signed char | signed char | int | 16x16x16 |
| signed char | signed char | int | 32x8x16 |
| signed char | signed char | int | 8x32x16 |

Alternate floating-point support:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| __nv_bfloat16 | __nv_bfloat16 | float | 16x16x16 |
| __nv_bfloat16 | __nv_bfloat16 | float | 32x8x16 |
| __nv_bfloat16 | __nv_bfloat16 | float | 8x32x16 |
| precision::tf32 | precision::tf32 | float | 16x16x8 |

Double Precision Support:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| double | double | double | 8x8x4 |

Experimental support for sub-byte operations:

| Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k) |
| --- | --- | --- | --- |
| precision::u4 | precision::u4 | int | 8x8x32 |
| precision::s4 | precision::s4 | int | 8x8x32 |
| precision::b1 | precision::b1 | int | 8x8x128 |

### 5.4.11.7. 示例

以下代码在单个线程束中实现了 16x16x16 的矩阵乘法。

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_ker(half *a, half *b, float *c) {
   // 声明片段
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // 将输出初始化为零
   wmma::fill_fragment(c_frag, 0.0f);

   // 加载输入
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // 执行矩阵乘法
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // 存储输出
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
```

 本页