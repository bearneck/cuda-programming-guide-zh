# 5.3 C++ 语言支持

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html)

---

本页面是否有帮助？

# 5.3. C++ 语言支持

`nvcc` 根据以下规范处理 CUDA 和设备代码：

- C++03 (ISO/IEC 14882:2003)，使用 `--std=c++03` 标志。
- C++11 (ISO/IEC 14882:2011)，使用 `--std=c++11` 标志。
- C++14 (ISO/IEC 14882:2014)，使用 `--std=c++14` 标志。
- C++17 (ISO/IEC 14882:2017)，使用 `--std=c++17` 标志。
- C++20 (ISO/IEC 14882:2020)，使用 `--std=c++20` 标志。

向 `nvcc` 传递 `-std=c++<version>` 标志会启用与指定版本相关的所有 C++ 功能，并使用相应的 C++ 方言选项调用主机预处理器、编译器和链接器。

编译器支持所有受支持标准中的语言功能，但需遵循后续章节中报告的限制。

## 5.3.1. C++11 语言特性

| 语言特性 | C++11 提案 | NVCC/CUDA Toolkit 7.x |
| --- | --- | --- |
| 右值引用 | N2118 | ✔ |
|    *this 的右值引用 | N2439 | ✔ |
| 通过右值初始化类对象 | N1610 | ✔ |
| 非静态数据成员初始化器 | N2756 | ✔ |
| 可变参数模板 | N2242 | ✔ |
|    扩展可变参数模板的模板参数 | N2555 | ✔ |
| 初始化列表 | N2672 | ✔ |
| 静态断言 | N1720 | ✔ |
| auto 类型变量 | N1984 | ✔ |
|    多声明符 auto | N1737 | ✔ |
|    移除 auto 作为存储类说明符 | N2546 | ✔ |
|    新的函数声明符语法 | N2541 | ✔ |
| Lambda 表达式 | N2927 | ✔ |
| 表达式的声明类型 | N2343 | ✔ |
|    不完整的返回类型 | N3276 | ✔ |
| 右尖括号 | N1757 | ✔ |
| 函数模板的默认模板参数 | DR226 | ✔ |
| 解决表达式的 SFINAE 问题 | DR339 | ✔ |
| 别名模板 | N2258 | ✔ |
| 外部模板 | N1987 | ✔ |
| 空指针常量 | N2431 | ✔ |
| 强类型枚举 | N2347 | ✔ |
| 枚举的前向声明 | N2764 DR1206 | ✔ |
| 标准化的属性语法 | N2761 | ✔ |
| 广义常量表达式 | N2235 | ✔ |
| 对齐支持 | N2341 | ✔ |
| 有条件支持的行为 | N1627 | ✔ |
| 将未定义行为更改为可诊断错误 | N1727 | ✔ |
| 委托构造函数 | N1986 | ✔ |
| 继承构造函数 | N2540 | ✔ |
| 显式转换运算符 | N2437 | ✔ |
| 新字符类型 | N2249 | ✔ |
| Unicode 字符串字面量 | N2442 | ✔ |
| 原始字符串字面量 | N2442 | ✔ |
| 字面量中的通用字符名称 | N2170 | ✔ |
| 用户定义字面量 | N2765 | ✔ |
| 标准布局类型 | N2342 | ✔ |
| 默认函数 | N2346 | ✔ |
| 删除的函数 | N2346 | ✔ |
| 扩展的友元声明 | N1791 | ✔ |
| 扩展 sizeof | N2253 DR850 | ✔ |
| 内联命名空间 | N2535 | ✔ |
| 无限制联合体 | N2544 | ✔ |
| 局部和未命名类型作为模板参数 | N2657 | ✔ |
| 基于范围的 for 循环 | N2930 | ✔ |
| 显式虚函数重写 | N2928 N3206 N3272 | ✔ |
| 对垃圾回收和基于可达性的泄漏检测的最小支持 | N2670 | ✘ |
| 允许移动构造函数抛出异常 [noexcept] | N3050 | ✔ |
| 定义移动特殊成员函数 | N3053 | ✔ |
| 并发 |  |  |
| 序列点 | N2239 | ✘ |
| 原子操作 | N2427 | ✘ |
| 强比较与交换 | N2748 | ✘ |
| 双向栅栏 | N2752 | ✘ |
| 内存模型 | N2429 | ✘ |
| 数据依赖排序：原子操作与内存模型 | N2664 | ✘ |
| 异常传播 | N2179 | ✘ |
| 允许在信号处理器中使用原子操作 | N2547 | ✘ |
| 线程局部存储 | N2659 | ✘ |
| 并发环境下的动态初始化与析构 | N2660 | ✘ |
| C++11 中的 C99 特性 |  |  |
| __func__ 预定义标识符 | N2340 | ✔ |
| C99 预处理器 | N1653 | ✔ |
| long long | N1811 | ✔ |
| 扩展整数类型 | N1988 | ✘ |

## 5.3.2. C++14 语言特性

| 语言特性 | C++14 提案 | NVCC/CUDA Toolkit 9.x |
| --- | --- | --- |
| 对特定 C++ 上下文转换的调整 | N3323 | ✔ |
| 二进制字面量 | N3472 | ✔ |
| 返回类型可推导的函数 | N3638 | ✔ |
| 广义 lambda 捕获（初始化捕获） | N3648 | ✔ |
| 泛型（多态）lambda 表达式 | N3649 | ✔ |
| 变量模板 | N3651 | ✔ |
| 放宽对 constexpr 函数的要求 | N3652 | ✔ |
| 成员初始化器与聚合类 | N3653 | ✔ |
| 明确内存分配 | N3664 | ✘ |
| 指定大小的释放 | N3778 | ✘ |
| [[deprecated]] 属性 | N3760 | ✔ |
| 单引号作为数字分隔符 | N3781 | ✔ |

## 5.3.3. C++17 语言特性

| 语言特性 | C++17 提案 | NVCC/CUDA Toolkit 11.x |
| --- | --- | --- |
| 移除三字符组 | N4086 | ✔ |
| u8 字符字面量 | N4267 | ✔ |
| 折叠表达式 | N4295 | ✔ |
| 命名空间和枚举项的属性 | N4266 | ✔ |
| 嵌套命名空间定义 | N4230 | ✔ |
| 允许对所有非类型模板参数进行常量求值 | N4268 | ✔ |
| 扩展 static_assert | N3928 | ✔ |
| 从大括号初始化列表推导 auto 的新规则 | N3922 | ✔ |
| 允许在模板模板参数中使用 typename | N4051 | ✔ |
| [[fallthrough]] 属性 | P0188R1 | ✔ |
| [[nodiscard]] 属性 | P0189R1 | ✔ |
| [[maybe_unused]] 属性 | P0212R1 | ✔ |
| 聚合初始化的扩展 | P0017R1 | ✔ |
| constexpr lambda 的措辞 | P0170R1 | ✔ |
| 一元折叠与空参数包 | P0036R0 | ✔ |
| 泛化基于范围的 for 循环 | P0184R0 | ✔ |
| 按值捕获 *this 的 Lambda | P0018R3 | ✔ |
| enum class 变量的构造规则 | P0138R2 | ✔ |
| C++ 的十六进制浮点数字面量 | P0245R1 | ✔ |
| 超对齐数据的动态内存分配 | P0035R4 | ✔ |
| 保证的复制消除 | P0135R1 | ✔ |
| 为惯用 C++ 优化表达式求值顺序 | P0145R3 | ✔ |
| constexpr if | P0292R2 | ✔ |
| 带初始化器的选择语句 | P0305R1 | ✔ |
| 类模板的模板参数推导 | P0091R3 P0512R0 | ✔ |
| 使用 auto 声明非类型模板参数 | P0127R2 | ✔ |
| 无需重复使用属性命名空间 | P0028R4 | ✔ |
| 忽略不支持的非标准属性 | P0283R2 | ✔ |
| 结构化绑定 | P0217R3 | ✔ |
| 移除 register 关键字的已弃用用法 | P0001R1 | ✔ |
| 移除已弃用的 operator++(bool) | P0002R1 | ✔ |
| 使异常规范成为类型系统的一部分 | P0012R1 | ✔ |
| C++17 的 __has_include | P0061R1 | ✔ |
| 重写继承构造函数（核心问题 1941 等） | P0136R1 | ✔ |
| 内联变量 | P0386R2 | ✔ |
| DR 150，模板模板参数的匹配 | P0522R0 | ✔ |
| 移除动态异常规范 | P0003R5 | ✔ |
| using 声明中的包展开 | P0195R2 | ✔ |
| 字节类型定义 | P0298R0 | ✔ |
| DR 727，类内显式实例化 | CWG727 | ✔ |

## 5.3.4. C++20 语言特性

GCC 版本 ≥ 10.0，Clang 版本 ≥ 10.0，Microsoft Visual Studio ≥ 2022，以及 nvc++ 版本 ≥ 20.7。

| 语言特性 | C++20 提案 | NVCC/CUDA Toolkit 12.x |
| --- | --- | --- |
| 位域的默认成员初始化器 | P0683R1 | ✔ |
| 修复指向成员的 const 限定指针 | P0704R1 | ✔ |
| 允许 lambda 捕获 [=, this] | P0409R2 | ✔ |
| 用于预处理器逗号省略的 __VA_OPT__ | P0306R4 P1042R1 | ✔ |
| 指定初始化器 | P0329R4 | ✔ |
| 泛型 lambda 的熟悉模板语法 | P0428R2 | ✔ |
| vector 的列表推导 | P0702R1 | ✔ |
| 概念 | P0734R0 P0857R0 P1084R2 P1141R2 P0848R3 P1616R1 P1452R2 P1972R0 P1980R0 P2092R0 P2103R0 P2113R0 | ✔ |
| 带初始化器的基于范围的 for 语句 | P0614R1 | ✔ |
| 简化隐式 lambda 捕获 | P0588R1 | ✔ |
| ADL 和不可见的函数模板 | P0846R0 | ✔ |
| 默认拷贝构造函数的 const 不匹配 | P0641R2 | ✔ |
| 减少 constexpr 函数的急切实例化 | P0859R0 | ✔ |
| 一致比较（operator<=>） | P0515R3 P0905R1 P1120R0 P1185R2 P1186R3 P1630R1 P1946R0 P1959R0 P2002R1 P2085R0 | ✔ |
| 特化的访问检查 | P0692R1 | ✔ |
| 可默认构造和可赋值的无状态 lambda | P0624R2 | ✔ |
| 未求值上下文中的 lambda | P0315R4 | ✔ |
| 空对象的语言支持 | P0840R2 | ✔ |
| 放宽基于范围的 for 循环定制点查找规则 | P0962R1 | ✔ |
| 允许结构化绑定访问可访问成员 | P0969R0 | ✔ |
| 放宽结构化绑定定制点查找规则 | P0961R1 | ✔ |
| 告别 typename！ | P0634R3 | ✔ |
| 允许在 lambda 初始化捕获中进行包展开 | P0780R2 P2095R0 | ✔ |
| likely 和 unlikely 属性的建议措辞 | P0479R5 | ✔ |
| 弃用通过 [=] 隐式捕获 this | P0806R2 | ✔ |
| 非类型模板参数中的类类型 | P0732R2 | ✔ |
| 非类型模板参数的不一致性 | P1907R1 | ✔ |
| 带填充位的原子比较并交换 | P0528R3 | ✔ |
| 可变大小类的高效大小删除 | P0722R3 | ✔ |
| 允许在常量表达式中进行虚函数调用 | P1064R0 | ✔ |
| 禁止具有用户声明构造函数的聚合类型 | P1008R1 | ✔ |
| explicit(bool) | P0892R2 | ✔ |
| 有符号整数采用二进制补码表示 | P1236R1 | ✔ |
| char8_t | P0482R6 | ✔ |
| 立即函数 ( consteval ) | P1073R3 P1937R2 | ✔ |
| std::is_constant_evaluated | P0595R2 | ✔ |
| 嵌套的内联命名空间 | P1094R2 | ✔ |
| constexpr 限制的放宽 | P1002R1 P1327R1 P1330R0 P1331R2 P1668R1 P0784R7 | ✔ |
| 功能测试宏 | P0941R2 | ✔ |
| 模块 | P1103R3 P1766R1 P1811R0 P1703R1 P1874R1 P1979R0 P1779R3 P1857R3 P2115R0 P1815R2 | ❌ |
| 协程 | P0912R5 | ❌ |
| 聚合类型的括号初始化 | P0960R3 P1975R0 | ✔ |
| DR: new 表达式中的数组大小推导 | P1009R2 | ✔ |
| DR: 从 T* 到 bool 的转换应被视为窄化转换 | P1957R2 | ✔ |
| 更强的 Unicode 要求 | P1041R4 P1139R2 | ✔ |
| 结构化绑定扩展 | P1091R3 P1381R1 | ✔ |
| 弃用 a[b,c] | P1161R3 | ✔ |
| 弃用 volatile 的某些用法 | P1152R4 | ✔ |
| [[nodiscard("with reason")]] | P1301R4 | ✔ |
| using enum | P1099R5 | ✔ |
| 聚合类型的类模板参数推导 | P1816R0 P2082R1 | ✔ |
| 别名模板的类模板参数推导 | P1814R0 | ✔ |
| 允许转换为未知边界的数组 | P0388R4 | ✔ |
| constinit | P1143R2 | ✔ |
| 布局兼容性和指针可互转换性特征 | P0466R5 | ✔ |
| DR: 检查抽象类类型 | P0929R2 | ✔ |
| DR: 更多的隐式移动 | P1825R0 | ✔ |
| DR: 伪析构函数结束对象生命周期 | P0593R6 | ✔ |

## 5.3.5. CUDA C++ 标准库

CUDA 提供了一个 C++ 标准库（STL）的实现，称为 [libcu++](https://nvidia.github.io/cccl/libcudacxx/standard_api.html)。该库具有以下优点：

- 功能在主机和设备上均可用。
- 与 CUDA 工具包支持的所有 Linux 和 Windows 平台兼容。
- 与最近两个主要版本的 CUDA 工具包支持的所有 GPU 架构兼容。
- 与当前和上一个主要版本的所有 CUDA 工具包兼容。
- 提供 C++20、C++23 和 C++26 等近期标准版本中可用的 C++ 标准库功能的 C++17 向后移植。
- 支持扩展数据类型，例如 128 位整数（__int128）、半精度浮点数（__half）、Bfloat16（__nv_bfloat16）和四精度浮点数（__float128）。
- 针对设备代码进行了高度优化。

此外，`libcu++` 还提供了 C++ 标准库中不可用的[扩展功能](https://nvidia.github.io/cccl/libcudacxx/extended_api.html)，以提高生产力和应用程序性能。这些功能包括数学函数、内存操作、同步原语、容器扩展、CUDA 内置函数的高级抽象、C++ PTX 包装器等。

`libcu++` 作为 [CUDA 工具包](https://developer.nvidia.com/cuda-downloads)的一部分以及开源 [CCCL](https://nvidia.github.io/cccl/) 仓库的一部分提供。
## 5.3.6. C 标准库函数

### 5.3.6.1. clock() 和 clock64()

```cuda
__host__ __device__ clock_t   clock();
__device__          long long clock64();
```

在设备代码中执行时，它返回一个每个多处理器计数器的值，该计数器在每个时钟周期递增。在内核开始和结束时采样此计数器，将两个值相减，并记录每个线程的结果，可以估算出设备执行该线程所花费的时钟周期数。然而，这个值并不代表设备实际执行线程指令所花费的时钟周期数。前者大于后者，因为线程是分时执行的。

!!! note "提示"
    相应的 CUDA C++ 函数 `cuda::std::clock()` 在 `<cuda/std/ctime>` 头文件中提供。为了类似的目的，在 `<cuda/std/chrono>` 头文件中也提供了一个可移植的 C++ `<chrono>` 实现。

### 5.3.6.2. printf()

```cuda
int printf(const char* format[, arg, ...]);
```

该函数将来自内核的格式化输出打印到主机端的输出流。

内核内的 `printf()` 函数行为类似于标准 C 库的 `printf()` 函数。用户应参考其主机系统的手册页以获取 `printf()` 行为的完整描述。本质上，作为 `format` 传入的字符串被输出到主机上的一个流。

`printf()` 命令像任何其他设备端函数一样执行：每个线程执行，并在调用线程的上下文中执行。在多线程内核中，对 `printf()` 的直接调用将由每个线程使用该线程指定的数据来执行。因此，主机流上会出现多个版本的输出字符串，每个版本对应一个遇到 `printf()` 的线程。

与返回打印字符数的 C 标准 `printf()` 不同，CUDA 的 `printf()` 返回解析的参数数量。如果格式字符串后没有参数，则返回 0。如果格式字符串为 `NULL`，则返回 `-1`。如果发生内部错误，则返回 -2。

在内部，`printf()` 使用一个共享的数据结构，因此调用 `printf()` 可能会改变线程的执行顺序。具体来说，调用 `printf()` 的线程可能比不调用 `printf()` 的线程执行路径更长，并且该路径的长度取决于 `printf()` 的参数。但是，请注意，除了在显式的 `__syncthreads()` 屏障处，CUDA 不保证线程的执行顺序。因此，无法判断执行顺序是否被 `printf()` 或硬件中的其他调度行为所修改。

---

**格式说明符**

与标准 `printf()` 一样，格式说明符的形式为：`%[flags][width][.precision][size]type`

支持以下字段。有关所有行为的完整描述，请参阅广泛可用的文档。

- 标志： # , ' ' , 0 , + , -
- 宽度： * , 0-9
- 精度：0-9
- 大小：h、l、ll
- 类型：%cdiouxXpeEfgGaAs

---

**限制**

`printf()` 输出的最终格式化发生在主机系统上。这意味着格式字符串必须能被主机系统的编译器和 C 库理解。尽管已尽最大努力确保 CUDA `printf()` 函数支持的格式说明符是大多数常见主机编译器所支持格式的通用子集，但其确切行为将取决于主机操作系统。

`printf()` 接受所有有效的标志和类型组合。这是因为它无法确定在最终输出格式化的主机系统上哪些组合有效、哪些无效。因此，如果程序发出的格式字符串包含无效组合，输出可能是未定义的。

除了格式字符串外，`printf()` 函数最多可以接受 32 个参数。任何额外的参数将被忽略，格式说明符将按原样输出。

由于 Windows 平台（32 位）和 Linux 平台（64 位）上 `long` 类型的大小不同，在 Linux 机器上编译然后在 Windows 机器上运行的内核，对于所有包含 `%ld` 的格式字符串，将产生损坏的输出。为确保安全，建议编译和执行平台保持一致。

---

**主机端缓冲区**

`printf()` 的输出缓冲区在内核启动前被设置为固定大小。该缓冲区是循环的，因此如果内核执行期间产生的输出超过缓冲区容量，较早的输出将被覆盖。只有在执行以下操作之一时，缓冲区才会被刷新：

- 通过 `<<< >>>` 或 `cuLaunchKernel()` 启动内核：在启动开始时刷新；如果 `CUDA_LAUNCH_BLOCKING` 环境变量设置为 1，则在启动结束时也会刷新。
- 通过 `cudaDeviceSynchronize()`、`cuCtxSynchronize()`、`cudaStreamSynchronize()`、`cuStreamSynchronize()`、`cudaEventSynchronize()` 或 `cuEventSynchronize()` 进行同步。
- 通过任何阻塞版本的 `cudaMemcpy*()` 或 `cuMemcpy*()` 进行内存复制。
- 通过 `cuModuleLoad()` 或 `cuModuleUnload()` 加载/卸载模块。
- 通过 `cudaDeviceReset()` 或 `cuCtxDestroy()` 销毁上下文。
- 在执行由 `cudaLaunchHostFunc()` 或 `cuLaunchHostFunc()` 添加的流回调之前。

请注意，程序退出时缓冲区不会自动刷新。

以下 API 函数设置和检索用于将 `printf()` 参数和内部元数据传输到主机的缓冲区大小。默认大小为 1 兆字节。

- `cudaDeviceGetLimit(size_t* size, cudaLimitPrintfFifoSize)`
- `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_t size)`

---

**示例**

以下代码示例：

```cuda
#include <stdio.h>

__global__ void helloCUDA(float value) {
    printf("Hello thread %d, value=%f\n", threadIdx.x, value);
}

int main() {
    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    return 0;
}
```

将输出：

```cuda
Hello thread 2, value=1.2345
Hello thread 1, value=1.2345
Hello thread 4, value=1.2345
Hello thread 0, value=1.2345
Hello thread 3, value=1.2345
```
请注意，每个线程都会遇到 `printf()` 命令。因此，输出的行数与线程网格中的线程数一样多。

查看 [Compiler Explorer](https://cuda.godbolt.org/z/d4MPj7qG8) 上的示例。

---

以下代码示例：

```cuda
#include <stdio.h>

__global__ void helloCUDA(float value) {
    if (threadIdx.x == 0)
        printf("Hello thread %d, value=%f\n", threadIdx.x, value);
}

int main() {
    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    return 0;
}
```

将输出：

```cuda
Hello thread 0, value=1.2345
```

显然，`if()` 语句限制了哪些线程调用 `printf()`，因此只看到一行输出。

查看 [Compiler Explorer](https://cuda.godbolt.org/z/YqEss81sf) 上的示例。

### 5.3.6.3. memcpy() 和 memset()

```cuda
__host__ __device__ void* memcpy(void* dest, const void* src, size_t size);
```

该函数将 `size` 字节从 `src` 指向的内存位置复制到 `dest` 指向的内存位置。

```cuda
__host__ __device__ void* memset(void* ptr, int value, size_t size);
```

该函数将 `ptr` 指向的内存块的 `size` 字节设置为 `value`，`value` 被解释为 `unsigned char`。

!!! note "提示"
    建议使用 <cuda/std/cstring> 头文件中提供的 cuda::std::memcpy() 和 cuda::std::memset() 函数，作为 memcpy 和 memset 的更安全版本。

### 5.3.6.4. malloc() 和 free()

```cuda
__host__ __device__ void* malloc(size_t size);
// 或者 <cuda/std/cstdlib> 头文件中的 cuda::std::malloc(), cuda::std::calloc()
```

函数 `malloc()`（设备端）、`cuda::std::malloc()` 和 `cuda::std::calloc()` 从设备堆中分配至少 `size` 字节，并返回一个指向已分配内存的指针。如果没有足够的内存来满足请求，则返回 `NULL`。返回的指针保证对齐到 16 字节边界。

```cuda
__device__ void* __nv_aligned_device_malloc(size_t size, size_t align);
// 或者 <cuda/std/cstdlib> 头文件中的 cuda::std::aligned_alloc()
```

函数 `__nv_aligned_device_malloc()` 和 [C++](https://en.cppreference.com/w/cpp/memory/c/aligned_alloc)`cuda::std::aligned_alloc()` 从设备堆中分配至少 `size` 字节，并返回一个指向已分配内存的指针。如果没有足够的内存来满足请求的大小或对齐要求，则返回 `NULL`。已分配内存的地址是 `align` 的倍数。`align` 必须是非零的 2 的幂。

```cuda
__host__ __device__ void free(void* ptr);
// 或者 <cuda/std/cstdlib> 头文件中的 cuda::std::free()
```

设备端函数 `free()` 和 `cuda::std::free()` 释放 `ptr` 指向的内存，该内存必须是由之前对 `malloc()`、`cuda::std::malloc()`、`cuda::std::calloc()`、`__nv_aligned_device_malloc()` 或 `cuda::std::aligned_alloc()` 的调用返回的。如果 `ptr` 是 `NULL`，则对 `free()` 或 `cuda::std::free()` 的调用将被忽略。使用相同的 `ptr` 重复调用 `free()` 或 `cuda::std::free()` 会导致未定义行为。
由给定 CUDA 线程通过 `malloc()`、`cuda::std::malloc()`、`cuda::std::calloc()`、`__nv_aligned_device_malloc()` 或 `cuda::std::aligned_alloc()` 分配的内存，将在 CUDA 上下文的整个生命周期内保持分配状态，直到通过调用 `free()` 或 `cuda::std::free()` 显式释放。此内存可被其他 CUDA 线程使用，甚至包括后续内核启动的线程。任何 CUDA 线程都可以释放由另一个线程分配的内存；但是，应注意确保同一指针不会被释放超过一次。

---

**堆内存 API**

必须在任何在设备代码中分配或释放内存的程序（包括使用 `new` 和 `delete` 关键字）运行之前，指定设备内存堆的大小。如果任何程序使用了设备内存堆而未显式指定堆大小，则会分配一个默认的 8 兆字节堆。

以下 API 函数用于获取和设置堆大小：

- cudaDeviceGetLimit(size_t* size, cudaLimitMallocHeapSize)
- cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t size)

授予的堆大小将至少为 `size` 字节。[cuCtxGetLimit()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8) 和 [cudaDeviceGetLimit()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g720e159aeb125910c22aa20fe9611ec2) 返回当前请求的堆大小。

堆的实际内存分配发生在模块加载到上下文时，无论是通过 CUDA 驱动程序 API（参见[模块](../03-advanced/driver-api.html#driver-api-module)）显式加载，还是通过 CUDA 运行时 API 隐式加载。如果内存分配失败，模块加载将产生 `CUDA_ERROR_SHARED_OBJECT_INIT_FAILED` 错误。

堆大小在模块加载后无法更改，并且不会根据需求动态调整。

为设备堆保留的内存是独立于通过主机端 CUDA API 调用（如 `cudaMalloc()`）分配的内存的。

---

**与主机内存 API 的互操作性**

通过设备端函数 `malloc()`、`cuda::std::malloc()`、`cuda::std::calloc()`、`__nv_aligned_device_malloc()`、`cuda::std::aligned_alloc()` 或 `new` 关键字分配的内存，不能通过运行时或驱动程序 API 调用（如 `cudaMalloc`、`cudaMemcpy` 或 `cudaMemset`）来使用或释放。同样，通过主机运行时 API 分配的内存也不能使用设备端函数 `free()`、`cuda::std::free()` 或 `delete` 关键字来释放。

---

每线程分配示例：

```cuda
#include <stdlib.h>
#include <stdio.h>

__global__ void single_thread_allocation_kernel() {
    size_t size = 123;
    char*  ptr  = (char*) malloc(size);
    memset(ptr, 0, size);
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    free(ptr);
}

int main() {
    // 设置堆大小为 128 兆字节。
    // 注意，这必须在任何内核启动之前完成。
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    single_thread_allocation_kernel<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
```
将输出：

```cuda
Thread 0 got pointer: 0x20d5ffe20
Thread 1 got pointer: 0x20d5ffec0
Thread 2 got pointer: 0x20d5fff60
Thread 3 got pointer: 0x20d5f97c0
Thread 4 got pointer: 0x20d5f9720
```

请注意，每个线程都遇到了 `malloc()` 和 `memset()` 命令，因此接收并初始化了自己的分配。

在 [Compiler Explorer](https://cuda.godbolt.org/z/z7K191z58) 上查看此示例。

---

**每个线程块分配**示例：

```cuda
#include <stdlib.h>

__global__ void block_level_allocation_kernel() {
    __shared__ int* data;
    // 块中的第一个线程执行分配，并通过共享内存将指针
    // 与所有其他线程共享，以便访问可以合并。
    if (threadIdx.x == 0) {
        size_t size = blockDim.x * 64; // 每个线程分配 64 字节。
        data = (int*) malloc(size);
    }
    __syncthreads();
    // 检查是否失败
    if (data == nullptr)
        return;

    // 线程索引到内存中，确保合并访问
    for (int i = 0; i < 64; ++i)
        data[i * blockDim.x + threadIdx.x] = threadIdx.x;
    // 确保所有线程在释放前完成
    __syncthreads();

    // 只能由一个线程释放内存！
    if (threadIdx.x == 0)
        free(data);
}

int main() {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
    block_level_allocation_kernel<<<10, 128>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

在 [Compiler Explorer](https://cuda.godbolt.org/z/7s8x7oonz) 上查看此示例。

---

**内核启动间持久化分配**示例：

```cuda
#include <stdlib.h>
#include <stdio.h>

const int NUM_BLOCKS = 20;

__device__ int* data_ptrs[NUM_BLOCKS]; // 每个块的指针

__global__ void allocate_memory_kernel() {
    // 只有块中的第一个线程执行分配，
    // 因为每个块只需要一次分配。
    if (threadIdx.x == 0)
        data_ptrs[blockIdx.x] = (int*) malloc(blockDim.x * 4);
    __syncthreads();
    // 检查是否失败
    if (data_ptrs[blockIdx.x] == nullptr)
        return;
    // 所有线程并行地将数据清零
    data_ptrs[blockIdx.x][threadIdx.x] = 0;
}

// 简单示例：将线程 ID 存储到每个元素中
__global__ void use_memory_kernel() {
    int* ptr = data_ptrs[blockIdx.x];
    if (ptr != nullptr)
        ptr[threadIdx.x] += threadIdx.x;
}

// 在释放缓冲区之前打印其内容
__global__ void free_memory_kernel() {
    int* ptr = data_ptrs[blockIdx.x];
    if (ptr != nullptr)
        printf("Block %d, Thread %d: final value = %d\n",
            blockIdx.x, threadIdx.x, ptr[threadIdx.x]);
    // 只能由一个线程释放！
    if (threadIdx.x == 0)
        free(ptr);
}

int main() {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    // 分配内存
    allocate_memory_kernel<<<NUM_BLOCKS, 10>>>();

    // 使用内存
    use_memory_kernel<<<NUM_BLOCKS, 10>>>();
    use_memory_kernel<<<NUM_BLOCKS, 10>>>();
    use_memory_kernel<<<NUM_BLOCKS, 10>>>();

    // 释放内存
    free_memory_kernel<<<NUM_BLOCKS, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```
请参阅 [Compiler Explorer](https://cuda.godbolt.org/z/h7r6G3dGP) 上的示例。

### 5.3.6.5.alloca()

```cuda
__host__ __device__ void* alloca(size_t size);
```

`alloca()` 函数在调用者的栈帧内分配 `size` 字节的内存。返回值是指向已分配内存的指针。当从设备代码调用该函数时，内存的起始地址是 16 字节对齐的。当调用者从 `alloca()` 返回时，内存会自动释放。

!!! note "注意"
    在 Windows 平台上，使用 alloca() 函数之前必须包含 <malloc.h> 头文件。调用 alloca() 可能导致栈溢出；用户需要相应地调整栈大小。

示例：

```cuda
__device__ void device_function(int num_items) {
    int4* ptr = (int4*) alloca(num_items * sizeof(int4));
    // 使用 ptr
    ...
}
```

## 5.3.7. Lambda 表达式

编译器通过将 lambda 表达式或闭包类型（C++11）与最内层封闭函数作用域的执行空间相关联，来确定其执行空间。如果没有封闭函数作用域，则执行空间被指定为 `__host__`。

执行空间也可以使用扩展 lambda 语法显式指定。

示例：

```cuda
auto global_lambda = [](){ return 0; }; // __host__

void host_function() {
    auto lambda1 = [](){ return 1; };   // __host__
    [](){ return 3; };                  // __host__, 闭包类型（lambda 表达式体）
}

__device__ void device_function() {
    auto lambda2 = [](){ return 2; };   // __device__
}

__global__ void kernel_function(void) {
    auto lambda3 = [](){ return 3; };   // __device__
}

__host__ __device__ void host_device_function() {
    auto lambda4 = [](){ return 4; };   // __host__ __device__
}

using function_ptr_t = int (*)();

__device__ void device_function(float          value,
                                function_ptr_t ptr = [](){ return 4; } /* __host__ */) {}
```

请参阅 [Compiler Explorer](https://godbolt.org/z/scv4vcczr) 上的示例。

### 5.3.7.1. Lambda 表达式与 `__global__` 函数参数

只有当 lambda 表达式或闭包类型的执行空间是 `__device__` 或 `__host__ __device__` 时，才能用作 `__global__` 函数的参数。全局或命名空间作用域的 lambda 表达式不能用作 `__global__` 函数的参数。

示例：

```cuda
template <typename T>
 __global__ void kernel(T input) {}

 __device__ void device_function() {
     // 设备内核调用需要单独编译（-rdc=true 标志）
     kernel<<<1, 1>>>([](){});
     kernel<<<1, 1>>>([] __device__() {});          // 扩展 lambda
     kernel<<<1, 1>>>([] __host__ __device__() {}); // 扩展 lambda
 }

 auto global_lambda = [] __host__ __device__() {};

 void host_function() {
     kernel<<<1, 1>>>([] __device__() {});          // 正确，扩展 lambda
     kernel<<<1, 1>>>([] __host__ __device__() {}); // 正确，扩展 lambda
 //  kernel<<<1, 1>>>([](){});                      // 错误，执行空间为 host 的闭包类型
 //  kernel<<<1, 1>>>(global_lambda);               // 错误，扩展 lambda，但在全局作用域
 }
```
请参阅 [Compiler Explorer](https://godbolt.org/z/ajrsn5z5Y) 上的示例。

### 5.3.7.2. 扩展 Lambda 表达式

`nvcc` 的 `--extended-lambda` 标志允许在 lambda 表达式中显式标注执行空间。这些标注应出现在 lambda 引导符之后，以及可选的 lambda 声明符之前。当指定 `--extended-lambda` 标志时，`nvcc` 会定义宏 `__CUDACC_EXTENDED_LAMBDA__`。

- 扩展 lambda 定义在 `__host__` 或 `__host__ __device__` 函数的直接或嵌套块作用域内。
- 扩展设备 lambda 是用 `__device__` 关键字标注的 lambda 表达式。
- 扩展主机-设备 lambda 是用 `__host__ __device__` 关键字标注的 lambda 表达式。

与标准 lambda 表达式不同，扩展 lambda 可以用作 `__global__` 函数中的类型参数。

示例：

```cuda
void host_function() {
    auto lambda1 = [] {};                      // 不是扩展 lambda：没有显式的执行空间标注
    auto lambda2 = [] __device__ {};           // 扩展 lambda
    auto lambda3 = [] __host__ __device__ {};  // 扩展 lambda
    auto lambda4 = [] __host__ {};             // 不是扩展 lambda
}

__host__ __device__ void host_device_function() {
    auto lambda1 = [] {};                      // 不是扩展 lambda：没有显式的执行空间标注
    auto lambda2 = [] __device__ {};           // 扩展 lambda
    auto lambda3 = [] __host__ __device__ {};  // 扩展 lambda
    auto lambda4 = [] __host__ {};             // 不是扩展 lambda
}

__device__ void device_function() {
    // 此函数内的所有 lambda 都不是扩展 lambda，
    // 因为其外层函数不是 `__host__` 或 `__host__ __device__` 函数。
    auto lambda1 = [] {};
    auto lambda2 = [] __device__ {};
    auto lambda3 = [] __host__ __device__ {};
    auto lambda4 = [] __host__ {};
}

auto global_lambda = [] __host__ __device__ { }; // 不是扩展 lambda，因为它不是定义在
                                                 // `__host__` 或 `__host__ __device__` 函数内
```

### 5.3.7.3. 扩展 Lambda 类型特征

编译器提供了类型特征，用于在编译时检测扩展 lambda 的闭包类型。

```cuda
bool __nv_is_extended_device_lambda_closure_type(type);
```

如果 `type` 是为扩展 `__device__` lambda 创建的闭包类，则此函数返回 `true`，否则返回 `false`。

```cuda
bool __nv_is_extended_device_lambda_with_preserved_return_type(type);
```

如果 `type` 是为扩展 `__device__` lambda 创建的闭包类，并且该 lambda 是使用尾随返回类型定义的，则此函数返回 `true`，否则返回 `false`。如果尾随返回类型定义引用了任何 lambda 参数名，则返回类型不被保留。

```cuda
bool __nv_is_extended_host_device_lambda_closure_type(type);
```

如果 `type` 是为扩展 `__host__ __device__` lambda 创建的闭包类，则此函数返回 `true`，否则返回 `false`。
---

无论是否启用了 lambda 或扩展 lambda，lambda 类型特征都可在所有编译模式下使用。如果扩展 lambda 模式未激活，这些特征将始终返回 `false`。

示例：

```cuda
auto lambda0 = [] __host__ __device__ { };

void host_function() {
    auto lambda1 = [] { };
    auto lambda2 = [] __device__ { };
    auto lambda3 = [] __host__ __device__ { };
    auto lambda4 = [] __device__ () -> double { return 3.14; }
    auto lambda5 = [] __device__ (int x) -> decltype(&x) { return 0; }

    using lambda0_t = decltype(lambda0);
    using lambda1_t = decltype(lambda1);
    using lambda2_t = decltype(lambda2);
    using lambda3_t = decltype(lambda3);
    using lambda4_t = decltype(lambda4);
    using lambda5_t = decltype(lambda5);

    // 'lambda0' 不是扩展 lambda，因为它定义在函数作用域之外
    static_assert(!__nv_is_extended_device_lambda_closure_type(lambda0_t));
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda0_t));
    static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda0_t));

    // 'lambda1' 不是扩展 lambda，因为它没有执行空间注解
    static_assert(!__nv_is_extended_device_lambda_closure_type(lambda1_t));
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda1_t));
    static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda1_t));

    // 'lambda2' 是一个扩展的仅设备 lambda
    static_assert(__nv_is_extended_device_lambda_closure_type(lambda2_t));
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda2_t));
    static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda2_t));

    // 'lambda3' 是一个扩展的主机-设备 lambda
    static_assert(!__nv_is_extended_device_lambda_closure_type(lambda3_t));
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda3_t));
    static_assert(__nv_is_extended_host_device_lambda_closure_type(lambda3_t));

    // 'lambda4' 是一个具有保留返回类型的扩展仅设备 lambda
    static_assert(__nv_is_extended_device_lambda_closure_type(lambda4_t));
    static_assert(__nv_is_extended_device_lambda_with_preserved_return_type(lambda4_t));
    static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda4_t));

    // 'lambda5' 不是一个具有保留返回类型的扩展仅设备 lambda，
    // 因为它在尾部返回类型中引用了 operator() 的参数类型。
    static_assert(__nv_is_extended_device_lambda_closure_type(lambda5_t));
    static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda5_t));
    static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda5_t));
}
```

### 5.3.7.4. 扩展 Lambda 限制

在调用主机编译器之前，CUDA 编译器会将扩展 lambda 表达式替换为在命名空间作用域中定义的占位符类型的实例。该占位符类型的模板参数需要获取包含原始扩展 lambda 表达式的函数的地址。这对于正确执行任何模板参数涉及扩展 lambda 闭包类型的 `__global__` 函数模板是必需的。包含函数的计算方式如下。
根据定义，扩展 lambda 表达式存在于 `__host__` 或 `__host__ __device__` 函数的直接或嵌套块作用域内。

*   如果该函数不是 lambda 表达式的 operator()，则它被视为该扩展 lambda 的**封闭函数**。
*   否则，该扩展 lambda 定义在一个或多个封闭 lambda 表达式的 operator() 的直接或嵌套块作用域内。如果最外层的 lambda 表达式定义在函数 F 的直接或嵌套块作用域内，则 F 是计算得出的封闭函数。否则，封闭函数不存在。

示例：

```cuda
void host_function() {
    auto lambda1 = [] __device__ { }; // lambda1 的封闭函数是 "host_function()"
    auto lambda2 = [] {
        auto lambda3 = [] {
            auto lambda4 = [] __host__ __device__ { }; // lambda4 的封闭函数是 "host_function"
        };
    };
}

auto global_lambda = [] {
    auto lambda5 = [] __host__ __device__ { }; // lambda5 的封闭函数不存在
};
```

---

扩展 Lambda 限制

1.  扩展 lambda 不能在另一个扩展 lambda 表达式内部定义。
    示例：
    ```cuda
    void host_function () {
        auto lambda1 = [] __host__ __device__ {
            // 错误，扩展 lambda 定义在另一个扩展 lambda 内部
            auto lambda2 = [] __host__ __device__ { };
        };
    }
    ```
2.  扩展 lambda 不能在泛型 lambda 表达式内部定义。
    示例：
    ```cuda
    void host_function () {
        auto lambda1 = [] ( auto ) {
            // 错误，扩展 lambda 定义在泛型 lambda 内部
            auto lambda2 = [] __host__ __device__ { };
        };
    }
    ```
3.  如果一个扩展 lambda 定义在一个或多个嵌套 lambda 表达式的直接或嵌套块作用域内，那么最外层的 lambda 表达式必须定义在一个函数的直接或嵌套块作用域内。
    示例：
    ```cuda
    auto lambda1 = [] {
        // 错误，外层封闭 lambda 没有定义在非 lambda-operator() 函数内部
        auto lambda2 = [] __host__ __device__ { };
    };
    ```
4.  扩展 lambda 的封闭函数必须具有名称，并且其地址必须可访问。如果封闭函数是类成员，则必须满足以下条件：
    *   所有包含该成员函数的类都必须具有名称。
    *   该成员函数在其父类中不能具有 private 或 protected 访问权限。
    *   所有封闭类在其各自的父类中不能具有 private 或 protected 访问权限。
    示例：
    ```cuda
    void host_function () {
        auto lambda1 = [] __device__ { return 0 ; }; // 正确
        {
            auto lambda2 = [] __device__ { return 0 ; }; // 正确
            auto lambda3 = [] __device__ __host__ { return 0 ; }; // 正确
        }
    }
    struct MyStruct1 {
        MyStruct1 () {
            auto lambda4 = [] __device__ { return 0 ; }; // 错误，封闭函数的地址不可访问
        }
    };
    class MyStruct2 {
        void foo () {
            auto temp1 = [] __device__ { return 10 ; }; // 错误，封闭函数在其父类中具有 private 访问权限
        }
        struct MyStruct3 {
            void foo () {
                auto temp1 = [] __device__ { return 10 ; }; // 错误，封闭类 MyStruct3 在其父类中具有 private 访问权限
            }
        };
    };
    ```
5. 在定义扩展 lambda 的位置，必须能够明确地获取其外围函数的地址。然而，这并非总是可行的，例如，当别名声明遮蔽了同名的模板类型参数时。示例：
```cpp
template < typename T > struct A {
  using Bar = void ;
  void test ();
};
template <> struct A < void > { };
template < typename Bar >
void A < Bar >:: test () {
  // 在发送给主机编译器的代码中，nvcc 将在此处注入一个地址表达式，形式如下：
  //   (void (A< Bar> ::*)(void))(&A::test))
  // 然而，类 typedef 'Bar'（指向 void）遮蔽了模板参数 'Bar'，
  // 导致 A<int>::test 中的地址表达式实际上引用的是：
  //    (void (A< void> ::*)(void))(&A::test))
  // 这未能正确获取外围函数 'A<int>::test' 的地址。
  auto lambda1 = [] __host__ __device__ { return 4 ; };
}
int main () {
  A < int > var ;
  var . test ();
}
```

6. 扩展 lambda 不能在函数内部的局部类中定义。示例：
```cpp
void host_function () {
  struct MyStruct {
    void bar () {
      // 错误，bar() 是函数内部局部类的成员
      auto lambda2 = [] __host__ __device__ { return 0 ; };
    }
  };
}
```

7. 扩展 lambda 的外围函数不能具有推导的返回类型。示例：
```cpp
auto host_function () {
  // 错误，host_function() 的返回类型是推导得出的
  auto lambda3 = [] __host__ __device__ { return 0 ; };
}
```

8. 主机-设备扩展 lambda 不能是泛型 lambda，即不能是带有 `auto` 参数类型的 lambda。示例：
```cpp
void host_function () {
  // 错误，__host__ __device__ 扩展 lambda 不能是泛型 lambda
  auto lambda1 = [] __host__ __device__ ( auto i ) { return i ; };
  // 错误，主机-设备扩展 lambda 不能是泛型 lambda
  auto lambda2 = [] __host__ __device__ ( auto ... i ) { return sizeof ...( i ); };
}
```

9. 如果外围函数是函数模板或成员模板的实例化，或者该函数是类模板的成员，则模板必须满足以下约束：
   模板最多只能有一个可变参数，并且它必须位于模板参数列表的最后。
   模板参数必须具有名称。
   模板实例化的参数类型不能涉及函数内部的局部类型（扩展 lambda 的闭包类型除外），也不能是私有或受保护的类成员。
示例 1：
```cpp
template < template < typename ... > class T , typename ... P1 , typename ... P2 >
void bar1 ( const T < P1 ... > , const T < P2 ... > ) {
  // 错误，外围函数有多个参数包
  auto lambda = [] __device__ { return 10 ; };
}
template < template < typename ... > class T , typename ... P1 , typename T2 >
void bar2 ( const T < P1 ... > , T2 ) {
  // 错误，对于外围函数，参数包未位于模板参数列表的最后
  auto lambda = [] __device__ { return 10 ; };
}
template < typename T , T >
void bar3 () {
  // 错误，对于外围函数，第二个模板参数未命名
  auto lambda = [] __device__ { return 10 ; };
}
```
示例 2：
```cpp
template < typename T >
void bar4 () {
  auto lambda1 = [] __device__ { return 10 ; };
}
class MyStruct {
  struct MyNestedStruct {};
  friend int main ();
};
int main () {
  struct MyLocalStruct {};
  // 错误，bar4() 中设备 lambda 的外围函数使用 main 函数的局部类型进行实例化
  bar4 < MyLocalStruct > ();
  // 错误，bar4 中设备 lambda 的外围函数使用一个类型进行实例化，
  //       该类型是类的私有成员
  bar4 < MyStruct :: MyNestedStruct > ();
}
```
10. 对于 Microsoft Visual Studio 主机编译器，包含函数必须具有外部链接。存在此限制是因为主机编译器不支持将非外部链接函数的地址用作模板参数。CUDA 编译器转换需要这些地址来支持扩展 lambda。

11. 对于 Microsoft Visual Studio 主机编译器，扩展 lambda 不得在 `if constexpr` 块的主体中定义。

12. 扩展 lambda 对捕获变量有以下限制：变量在被用于直接初始化表示扩展 lambda 的闭包类型的类类型的字段之前，可能会按值传递给发送到主机编译器的代码中的一系列辅助函数。然而，C++ 标准规定捕获的变量应用于直接初始化闭包类型的字段。变量只能按值捕获。如果数组维数大于 7，则无法捕获数组类型的变量。对于数组类型变量，闭包类型的数组字段首先进行默认初始化，然后在发送到主机编译器的代码中，从捕获的数组变量的相应元素复制赋值每个数组元素。因此，数组元素类型在主机代码中必须是可默认构造且可复制赋值的。作为可变参数包元素的函数参数无法被捕获。捕获的变量类型不能是函数的局部类型（扩展 lambda 闭包类型除外），也不能是私有或受保护的类成员。主机-设备扩展 lambda 不支持初始化捕获。但是，设备扩展 lambda 支持初始化捕获，除非初始化器是数组或 `std::initializer_list` 类型。扩展 lambda 的函数调用运算符不是 `constexpr`。扩展 lambda 的闭包类型不是字面类型。声明扩展 lambda 时不能使用 `constexpr` 和 `consteval` 说明符。在词法上嵌套在扩展 lambda 内的 `if-constexpr` 块内，不能隐式捕获变量，除非该变量已在 `if-constexpr` 块外被隐式捕获，或出现在扩展 lambda 的显式捕获列表中。

示例：
```cpp
void host_function () {
    // 正确：仅设备扩展 lambda 允许初始化捕获
    auto lambda1 = [ x = 1 ] __device__ () { return x ; };

    // 错误：主机-设备扩展 lambda 不允许初始化捕获
    auto lambda2 = [ x = 1 ] __host__ __device__ () { return x ; };

    int a = 1 ;
    // 错误：扩展 __device__ lambda 不能通过引用捕获变量
    auto lambda3 = [ & a ] __device__ () { return a ; };

    // 错误：仅设备扩展 lambda 不允许通过引用捕获
    auto lambda4 = [ & x = a ] __device__ () { return x ; };

    struct MyStruct {};
    MyStruct s1 ;
    // 错误：函数的局部类型不能用于捕获变量的类型
    auto lambda6 = [ s1 ] __device__ () { };

    // 错误：初始化捕获不能是 std::initializer_list 类型
    auto lambda7 = [ x = { 11 }] __device__ () { };

    std :: initializer_list < int > b = { 11 , 22 , 33 };
    // 错误：初始化捕获不能是 std::initializer_list 类型
    auto lambda8 = [ x = b ] __device__ () { };

    int var = 4 ;
    auto lambda9 = [ = ] __device__ {
        int result = 0 ;
        if constexpr ( false ) {
            // 错误：仅设备扩展 lambda 不能在 if-constexpr 上下文中首次捕获 'var'
            result += var ;
        }
        return result ;
    };

    auto lambda10 = [ var ] __device__ {
        int result = 0 ;
        if constexpr ( false ) {
            // 正确：'var' 已列在扩展 lambda 的显式捕获列表中
            result += var ;
        }
        return result ;
    };

    auto lambda11 = [ = ] __device__ {
        int result = var ;
        if constexpr ( false ) {
            // 正确：'var' 已在 'if-constexpr' 块外被隐式捕获
            result += var ;
        }
        return result ;
    };
}
```
13. 当解析函数时，CUDA 编译器会为函数中的每个扩展 lambda 分配一个计数器值。该计数器值用于传递给主机编译器的替代命名类型中。因此，函数中是否存在扩展 lambda 不应依赖于 `__CUDA_ARCH__` 的特定值，也不应依赖于 `__CUDA_ARCH__` 是否未定义。示例：
```cpp
template < typename T >
__global__ void kernel ( T in ) { in (); }

__host__ __device__ void host_device_function () {
    // 错误：扩展 lambda 的数量和相对声明顺序依赖于 __CUDA_ARCH__
#if defined(__CUDA_ARCH__)
    auto lambda1 = [] __device__ { return 0 ; };
    auto lambda2 = [] __host__ __device__ { return 10 ; };
#endif
    auto lambda3 = [] __device__ { return 4 ; };
    kernel <<< 1 , 1 >>> ( lambda3 );
}
```

14. 如上所述，CUDA 编译器将主机函数中定义的设备扩展 lambda 替换为命名空间作用域中定义的占位符类型。除非特征 `__nv_is_extended_device_lambda_with_preserved_return_type()` 对该扩展 lambda 的闭包类型返回 true，否则该占位符类型不会定义与原始 lambda 声明等效的 operator() 函数。因此，尝试确定此类 lambda 的 operator() 函数的返回类型或参数类型可能在主机代码中无法正常工作，因为主机编译器处理的代码在语义上与 CUDA 编译器处理的输入代码不同。然而，在设备代码内自省 operator() 函数的返回类型或参数类型是可以接受的。请注意，此限制不适用于特征 `__nv_is_extended_device_lambda_with_preserved_return_type()` 返回 true 的主机或设备扩展 lambda。示例：
```cpp
#include <cuda/std/type_traits>

const char & getRef ( const char * p ) { return * p ; }

void foo () {
    auto lambda1 = [] __device__ { return "10" ; };
    // 错误：尝试在主机代码中提取设备 lambda 的返回类型
    cuda :: std :: result_of < decltype ( lambda1 )() >:: type xx1 = "abc" ;

    auto lambda2 = [] __host__ __device__ { return "10" ; };
    // 正确：lambda2 表示一个主机-设备扩展 lambda
    cuda :: std :: result_of < decltype ( lambda2 )() >:: type xx2 = "abc" ;

    auto lambda3 = [] __device__ () -> const char * { return "10" ; };
    // 正确：lambda3 表示一个保留了返回类型的设备扩展 lambda
    cuda :: std :: result_of < decltype ( lambda3 )() >:: type xx2 = "abc" ;
    static_assert ( cuda :: std :: is_same_v < cuda :: std :: result_of < decltype ( lambda3 )() >:: type , const char *> );

    auto lambda4 = [] __device__ ( char x ) -> decltype ( getRef ( & x )) { return 0 ; };
    // lambda4 的返回类型未被保留，因为它在尾部返回类型中引用了 operator() 的参数类型。
    static_assert ( ! __nv_is_extended_device_lambda_with_preserved_return_type ( decltype ( lambda4 )));
}
```

15. 对于仅限设备的扩展 lambda：
    *   operator() 参数类型的自省仅在设备代码中受支持。
    *   operator() 返回类型的自省仅在设备代码中受支持，除非特征函数 `__nv_is_extended_device_lambda_with_preserved_return_type()` 返回 true。
16. 如果扩展 lambda 从主机代码传递到设备代码作为 `__global__` 函数的参数，那么 lambda 主体中捕获变量的任何表达式都必须保持不变，无论 `__CUDA_ARCH__` 宏是否定义及其值如何。此限制源于 lambda 的闭包类布局取决于编译器在处理 lambda 表达式时遇到捕获变量的顺序。如果设备编译和主机编译的闭包类布局不同，程序可能执行不正确。示例：
```cpp
__device__ int result ;
template < typename T >
__global__ void kernel ( T in ) {
  result = in ();
}
void foo ( void ) {
  int x1 = 1 ;
  // 错误，"x1" 仅在 __CUDA_ARCH__ 定义时被捕获。
  auto lambda1 = [ = ] __host__ __device__ {
#ifdef __CUDA_ARCH__
    return x1 + 1 ;
#else
    return 10 ;
#endif
  };
  kernel <<< 1 , 1 >>> ( lambda1 );
}
```

17. 如前所述，CUDA 编译器将扩展的仅设备 lambda 表达式替换为发送到主机编译器代码中的占位符类型实例。该占位符类型在主机代码中未定义指向函数的转换运算符；但是，在设备代码中提供了转换运算符。请注意，此限制不适用于主机-设备扩展 lambda。示例：
```cpp
template < typename T >
__global__ void kernel ( T in ) {
  int ( * fp )( double ) = in ;
  fp ( 0 ); // 正确，设备代码支持转换
  auto lambda1 = []( double ) { return 1 ; };
}
void foo () {
  auto lambda_device = [] __device__ ( double ) { return 1 ; };
  auto lambda_host_device = [] __host__ __device__ ( double ) { return 1 ; };
  kernel <<< 1 , 1 >>> ( lambda_device );
  kernel <<< 1 , 1 >>> ( lambda_host_device );
  // 正确，主机代码支持 __host__ __device__ lambda 的转换
  int ( * fp1 )( double ) = lambda_host_device ;
  // 错误，主机代码不支持设备 lambda 的转换
  int ( * fp2 )( double ) = lambda_device ;
}
```

18. 如前所述，CUDA 编译器将扩展的仅设备或主机-设备 lambda 表达式替换为发送到主机编译器代码中的占位符类型实例。此占位符类型可能定义 C++ 特殊成员函数，例如构造函数和析构函数。因此，对于扩展 lambda 的闭包类型，CUDA 前端编译器中的某些标准 C++ 类型特征可能与主机编译器产生不同的结果。受影响的类型特征包括：`std::is_trivially_copyable`、`std::is_trivially_constructible`、`std::is_trivially_copy_constructible`、`std::is_trivially_move_constructible`、`std::is_trivially_destructible`。必须注意确保这些特征的结果不用于 `__global__`、`__device__`、`__constant__` 或 `__managed__` 函数或变量模板的实例化中。示例：
```cpp
#include <cstdio>
#include <type_traits>
template < bool b >
void __global__ kernel () {
  printf ( "hi" );
}
template < typename T >
void kernel_launch () {
  // 错误，此内核启动可能失败，因为 CUDA 前端编译器和主机编译器
  //       对于扩展 lambda 闭包类型的 std::is_trivially_copyable_v 特征结果
  //       可能不一致
  kernel < std :: is_trivially_copyable_v < T >><<< 1 , 1 >>> ();
  cudaDeviceSynchronize ();
}
int main () {
  int x = 0 ;
  auto lambda1 = [ = ] __host__ __device__ () { return x ; };
  kernel_launch < decltype ( lambda1 ) > ();
}
```
CUDA 编译器将为 `1-12` 中描述的部分情况生成编译器诊断信息；对于 `13-17` 的情况，不会生成诊断信息，但主机编译器可能无法编译生成的代码。

### 5.3.7.5. 主机-设备 Lambda 优化说明

与仅设备 lambda 不同，主机-设备 lambda 可以从主机代码调用。如前所述，CUDA 编译器将主机代码中定义的扩展 lambda 表达式替换为命名占位符类型的实例。扩展主机-设备 lambda 的占位符类型通过间接函数调用调用原始 lambda 的 `operator()`。如果扩展 lambda 模式未激活，其特性将始终返回 false。

间接函数调用的存在可能导致主机编译器对扩展主机-设备 lambda 的优化程度低于隐式或显式仅为 `__host__` 的 lambda。在后一种情况下，主机编译器可以轻松地将 lambda 主体内联到调用上下文中。然而，当遇到扩展主机-设备 lambda 时，主机编译器可能无法轻松地将原始 lambda 主体内联。

### 5.3.7.6. 按值捕获 *this

根据 C++11/C++14 规则，当 lambda 在非 `static` 类成员函数中定义且 lambda 主体引用类成员变量时，必须按值捕获类的 `this` 指针，而不是引用的成员变量。如果 lambda 是在主机函数中定义并在 GPU 上执行的扩展仅设备或主机-设备 lambda，并且 `this` 指针指向主机内存，则在 GPU 上访问引用的成员变量将导致运行时错误。

示例：

```cuda
#include <cstdio>

template <typename T>
__global__ void foo(T in) { printf("value = %d\n", in()); }

struct MyStruct {
    int var;

    __host__ __device__ MyStruct() : var(10) {};

    void run() {
        auto lambda1 = [=] __device__ {
            // 对 "var" 的引用导致按值捕获 'this' 指针 (MyStruct*)
            return var + 1;
        };
        // 内核启动在运行时失败，因为 GPU 无法访问 'this->var'
        foo<<<1, 1>>>(lambda1);
        cudaDeviceSynchronize();
    }
};

int main() {
    MyStruct s1;
    s1.run();
}
```

C++17 通过引入新的 `*this` 捕获模式解决了这个问题。在此模式下，编译器复制由 `*this` 表示的对象，而不是按值捕获 `this` 指针。`*this` 捕获模式在 [P0018R3](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html) 中有更详细的描述。

当使用 `--extended-lambda` 标志时，CUDA 编译器支持在 `__device__` 和 `__global__` 函数内定义的 lambda 以及在主机代码中定义的扩展仅设备 lambda 使用 `*this` 捕获模式。

以下是修改为使用 `*this` 捕获模式的上述示例：

```cuda
#include <cstdio>

template <typename T>
__global__ void foo(T in) { printf("\n value = %d", in()); }

struct MyStruct {
    int var;
    __host__ __device__ MyStruct() : var(10) { };

    void run() {
        // 注意 "*this" 捕获说明符
        auto lambda1 = [=, *this] __device__ {
            // 对 "var" 的引用导致按值捕获由 '*this' 表示的对象，
            // GPU 代码将访问 'copy_of_star_this->var'
            return var + 1;
        };
        // 内核启动成功
        foo<<<1, 1>>>(lambda1);
        cudaDeviceSynchronize();
    }
};

int main() {
    MyStruct s1;
    s1.run();
}
```
`*this` 捕获模式不允许用于在主机代码中定义的非注解 lambda，也不允许用于扩展的主机-设备 lambda，除非所选语言方言启用了 `*this` 捕获。以下是支持和不支持的用法示例：

```cuda
struct MyStruct {
    int var;
    __host__ __device__ MyStruct() : var(10) { };

    void host_function() {
        // 正确，在扩展的仅设备 lambda 中使用
        auto lambda1 = [=, *this] __device__ { return var; };

        // 在扩展的主机-设备 lambda 中使用
        // 如果语言方言未启用 *this 捕获，则错误
        auto lambda2 = [=, *this] __host__ __device__ { return var; };

        // 在主机函数的非注解 lambda 中使用
        // 如果语言方言未启用 *this 捕获，则错误
        auto lambda3 = [=, *this]  { return var; };
    }

    __device__ void device_function() {
        // 正确，在仅设备函数中定义的 lambda 中使用
        auto lambda1 = [=, *this] __device__ { return var; };

        // 正确，在仅设备函数中定义的 lambda 中使用
        auto lambda2 = [=, *this] __host__ __device__ { return var; };

        // 正确，在仅设备函数中定义的 lambda 中使用
        auto lambda3 = [=, *this]  { return var; };
    }

    __host__ __device__ void host_device_function() {
        // 正确，在扩展的仅设备 lambda 中使用
        auto lambda1 = [=, *this] __device__ { return var; };

        // 在扩展的主机-设备 lambda 中使用
        // 如果语言方言未启用 *this 捕获，则错误
        auto lambda2 = [=, *this] __host__ __device__ { return var; };

        // 在主机-设备函数的未注解 lambda 中使用
        // 如果语言方言未启用 *this 捕获，则错误
        auto lambda3 = [=, *this]  { return var; };
    }
};
```

### 5.3.7.7. 参数依赖查找 (ADL)

如前所述，CUDA 编译器在调用主机编译器之前，会用占位符类型替换扩展的 lambda 表达式。该占位符类型的一个模板参数使用了包含原始 lambda 表达式的函数的地址。这可能导致额外的命名空间参与到任何参数类型涉及扩展 lambda 表达式闭包类型的主机函数调用的[参数依赖查找 (ADL)](https://en.cppreference.com/w/cpp/language/adl.html) 中。因此，主机编译器可能会选择错误的函数。

示例：

```cuda
namespace N1 {

struct MyStruct {};

template <typename T>
void my_function(T);

}; // namespace N1

namespace N2 {

template <typename T>
int my_function(T);

template <typename T>
void run(T in) { my_function(in); }

} // namespace N2

void bar(N1::MyStruct in) {
    // 对于扩展的仅设备 lambda，发送到主机编译器的代码被替换为
    // 占位符类型实例化表达式
    //    ' __nv_dl_wrapper_t< __nv_dl_tag<void (*)(N1::MyStruct in),(&bar),1> > { }'
    //
    // 因此，命名空间 'N1' 会参与到 N2::run 函数体内对 "my_function()" 调用的 ADL 查找中，导致歧义。
    auto lambda1 = [=] __device__ { };
    N2::run(lambda1);
}
```
在上述示例中，CUDA 编译器将扩展 lambda 替换为涉及 `N1` 命名空间的占位符类型。因此，`N1` 命名空间参与了 `N2::run()` 函数体内 `my_function(in)` 的 ADL 查找，导致发现了多个重载候选函数：`N1::my_function` 和 `N2::my_function`，从而造成主机端编译失败。

## 5.3.8. 多态函数包装器

`nvfunctional` 头文件提供了一个多态函数包装器类模板 `nvstd::function`。该模板的实例可以存储、复制和调用任何可调用目标，例如 lambda 表达式。`nvstd::function` 既可用于主机代码，也可用于设备代码。

示例：

```cuda
#include <nvfunctional>

__host__            int host_function()        { return 1; }
__device__          int device_function()      { return 2; }
__host__ __device__ int host_device_function() { return 3; }

__global__ void kernel(int* result) {
    nvstd::function<int()> fn1 = device_function;
    nvstd::function<int()> fn2 = host_device_function;
    nvstd::function<int()> fn3 = [](){ return 10; };
    *result                    = fn1() + fn2() + fn3();
}

__host__ __device__ void host_device_test(int* result) {
    nvstd::function<int()> fn1 = host_device_function;
    nvstd::function<int()> fn2 = [](){ return 10; };
    *result                    = fn1() + fn2();
}

__host__ void host_test(int* result) {
    nvstd::function<int()> fn1 = host_function;
    nvstd::function<int()> fn2 = host_device_function;
    nvstd::function<int()> fn3 = [](){ return 10; };
    *result                    = fn1() + fn2() + fn3();
}
```

---

无效用例：

*   主机代码中的 `nvstd::function` 实例不能用 `__device__` 函数的地址或其 `operator()` 是 `__device__` 函数的函数对象来初始化。
*   类似地，设备代码中的 `nvstd::function` 实例不能用 `__host__` 函数的地址或其 `operator()` 是 `__host__` 函数的函数对象来初始化。
*   `nvstd::function` 实例不能在运行时从主机代码传递到设备代码（反之亦然）。
*   如果 `__global__` 函数是从主机代码启动的，则 `nvstd::function` 不能用作该 `__global__` 函数的参数类型。

无效用例示例：

```cuda
#include <nvfunctional>

__device__ int device_function() { return 1; }
__host__   int host_function() { return 3; }
auto       lambda_host  = [] { return 0; };

__global__ void k() {
    nvstd::function<int()> fn1 = host_function; // 错误，使用 __host__ 函数的地址初始化
    nvstd::function<int()> fn2 = lambda_host;   // 错误，使用其 operator() 是 __host__ 函数的函数对象的地址初始化
}

__global__ void kernel(nvstd::function<int()> f1) {}

void foo(void) {
    auto lambda_device = [=] __device__ { return 1; };

    nvstd::function<int()> fn1 = device_function; // 错误，使用 __device__ 函数的地址初始化
    nvstd::function<int()> fn2 = lambda_device;   // 错误，使用其 operator() 是 __device__ 函数的函数对象的地址初始化
    kernel<<<1, 1>>>(fn2);                        // 错误，将 nvstd::function 从主机传递到设备
}
```
---

`nvstd::function` 在 `nvfunctional` 头文件中定义如下：

```cuda
namespace nvstd {

template <typename RetType, typename ...ArgTypes>
class function<RetType(ArgTypes...)> {
public:
    // 构造函数
    __device__ __host__ function() noexcept;
    __device__ __host__ function(nullptr_t) noexcept;
    __device__ __host__ function(const function&);
    __device__ __host__ function(function&&);

    template<typename F>
    __device__ __host__ function(F);

    // 析构函数
    __device__ __host__ ~function();

    // 赋值运算符
    __device__ __host__ function& operator=(const function&);
    __device__ __host__ function& operator=(function&&);
    __device__ __host__ function& operator=(nullptr_t);
    template<typename F>
    __device__ __host__ function& operator=(F&&);

    // 交换
    __device__ __host__ void swap(function&) noexcept;

    // 函数容量
    __device__ __host__ explicit operator bool() const noexcept;

    // 函数调用
    __device__ RetType operator()(ArgTypes...) const;
};

// 空指针比较
template <typename R, typename... ArgTypes>
__device__ __host__
bool operator==(const function<R(ArgTypes...)>&, nullptr_t) noexcept;

template <typename R, typename... ArgTypes>
__device__ __host__
bool operator==(nullptr_t, const function<R(ArgTypes...)>&) noexcept;

template <typename R, typename... ArgTypes>
__device__ __host__
bool operator!=(const function<R(ArgTypes...)>&, nullptr_t) noexcept;

template <typename R, typename... ArgTypes>
__device__ __host__
bool operator!=(nullptr_t, const function<R(ArgTypes...)>&) noexcept;

// 专用算法
template <typename R, typename... ArgTypes>
__device__ __host__
void swap(function<R(ArgTypes...)>&, function<R(ArgTypes...)>&);

} // namespace nvstd
```

## 5.3.9. C/C++ 语言限制

### 5.3.9.1. 不支持的特性

- 设备代码不支持运行时类型信息（RTTI）和异常：
  - `typeid` 关键字
  - `dynamic_cast` 关键字
  - `try`/`catch`/`throw` 关键字
- 设备代码不支持 `long double` 类型。
- 在任何平台上都不支持三字符组。在 Windows 上不支持双字符组。
- 用户定义的 `operator new`、`operator new[]`、`operator delete` 或 `operator delete[]` 不能用于替换编译器提供的相应内置函数，在主机和设备上这都被视为未定义行为。

### 5.3.9.2. 保留的命名空间

除非另有说明，向顶级命名空间 `cuda::`、`nv::` 或 `cooperative_groups::`，或向其中的任何嵌套命名空间添加定义是未定义行为。我们允许 `cuda::` 作为子命名空间，如下所示：

示例：

```cuda
namespace cuda {   // 对于 "nv" 和 "cooperative_groups" 命名空间同理

struct foo;        // 错误，在 "cuda" 命名空间中声明类

void bar();        // 错误，在 "cuda" 命名空间中声明函数

namespace utils {} // 错误，在 "cuda" 命名空间中声明命名空间

} // namespace cuda
```

```cuda
namespace utils {
namespace cuda {

// 正确，命名空间 "cuda" 可以嵌套在非保留的命名空间内使用
void bar();

} // namespace cuda
} // namespace utils

// 错误，相当于在全局作用域向命名空间 "cuda" 添加符号
using namespace utils;
```
### 5.3.9.3. 指针与内存地址

指针解引用（`*pointer`、`pointer->member`、`pointer[0]`）仅允许在与关联内存所在执行空间相同的执行空间中进行。以下情况会导致未定义行为，最常见的是段错误和应用程序终止。

*   在主机代码中解引用指向全局内存、共享内存或常量内存的指针。
*   在设备代码中解引用指向主机内存的指针。

以下限制适用于函数：

*   不允许在主机代码中获取 `__device__` 函数的地址。
*   在主机代码中获取的 `__global__` 函数的地址不能在设备代码中使用。同样，在设备代码中获取的 `__global__` 函数的地址不能在主机代码中使用。

通过 `cudaGetSymbolAddress()` 获取的 `__device__` 或 `__constant__` 变量的地址（如[内存空间说明符](cpp-language-extensions.html#memory-space-specifiers)章节所述）只能在主机代码中使用。

### 5.3.9.4. 变量

#### 5.3.9.4.1. 局部变量

在主机上执行的函数内部，不允许对非 `extern` 变量声明使用 `__device__`、`__shared__`、`__managed__` 和 `__constant__` 内存空间说明符。

示例：

```cuda
__host__ void host_function() {
    int x;                   // 正确，__host__ 变量
    __device__   int y;      // 错误，在主机函数内声明 __device__ 变量
    __shared__   int z;      // 错误，在主机函数内声明 __shared__ 变量
    __managed__  int w;      // 错误，在主机函数内声明 __managed__ 变量
    __constant__ int h;      // 错误，在主机函数内声明 __constant__ 变量
    extern __device__ int k; // 正确，extern __device__ 变量
}
```

在设备上执行的函数内部，对于既非 `extern` 也非 `static` 的变量声明，不允许使用 `__device__`、`__constant__` 和 `__managed__` 内存空间说明符。

```cuda
__device__ void device_function() {
    int x;                   // 正确，__device__ 变量
    __constant__      int y; // 错误，在设备函数内声明 __constant__ 变量
    __managed__       int z; // 错误，在设备函数内声明 __managed__ 变量
    extern __device__ int k; // 正确，extern __device__ 变量
}
```

另请参阅[静态变量](#static-variables)章节。

#### 5.3.9.4.2. const 限定变量

在全局、命名空间或类作用域声明的、没有内存空间注解（`__device__` 或 `__constant__`）的 `const` 限定变量被视为主机变量。设备代码不能包含对该变量的引用或获取其地址。

如果满足以下条件，该变量可以直接在设备代码中使用：

*   在使用点之前已用常量表达式初始化，
*   其类型没有 `volatile` 限定，并且
*   其类型为以下之一：内置整型或内置浮点类型，除非主机编译器是 Microsoft Visual Studio。
从 C++14 开始，建议使用 `constexpr` 或 `inline constexpr`（C++17）变量，而不是 `const` 限定的变量。`constexpr` 变量不受相同的类型限制，可以直接在设备代码中使用。

`__managed__` 变量不支持 `const` 限定的类型。

示例：

```cuda
const            int   ConstVar          = 10;
const            float ConstFloatVar     = 5.0f;
inline constexpr float ConstexprFloatVar = 5.0f; // C++17

struct MyStruct {
    static const            int   ConstVar          = 20;
//  static const             float ConstFloatVar     = 5.0f; // 错误，静态 const 变量不能是浮点类型
    static inline constexpr float ConstexprFloatVar = 5.0f; // 正确
};

extern const int ExternVar;

__device__ void foo() {
    int array1[ConstVar];                     // 正确
    int array2[MyStruct::ConstVar];           // 正确

    const     float var1 = ConstFloatVar;     // 正确，除非主机编译器是 Microsoft Visual Studio。
    constexpr float var2 = ConstexprFloatVar; // 正确
//  int             var3 = ExternVar;          // 错误，"ExternVar" 未用常量表达式初始化
//  int&            var4 = ConstVar;           // 错误，引用主机变量
//  int*            var5 = &ConstVar;          // 错误，取主机变量地址
}
```

查看 [Compiler Explorer](https://godbolt.org/z/eWG8KxK94) 上的示例。

#### 5.3.9.4.3. volatile 限定的变量

!!! note "注意"
    支持 `volatile` 关键字是为了保持与 ISO C++ 的兼容性。然而，其剩余未弃用的用途中，很少有（如果有的话）适用于 GPU。

对 `volatile` 限定对象的读写不是原子操作，并且会被编译成一个或多个 [volatile 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation)，这些指令不保证：

- 内存操作的顺序，或
- 硬件执行的内存操作数量与 PTX 指令数量匹配。

CUDA C++ `volatile` **不**适用于：

- **线程间同步**：请改用通过 `cuda::atomic_ref`、`cuda::atomic` 或原子函数进行的原子操作。原子内存操作提供线程间同步保证，并且比 volatile 操作性能更好。
然而，CUDA C++ volatile 操作不提供任何线程间同步保证，因此不适用于此目的。
以下示例展示了如何使用原子操作在两个线程之间传递消息。

    `cuda::atomic_ref`
    ```cuda
    #include <cuda/atomic>
    __global__ void kernel ( int * flag , int * data ) {
        cuda :: atomic_ref < int , cuda :: thread_scope_device > atomic_ref { * flag };
        if ( threadIdx . x == 0 ) {
            // 消费者：阻塞直到生产者设置标志，然后读取数据
            while ( atomic_ref . load ( cuda :: memory_order_acquire ) == 0 ) ;
            if ( * data != 42 ) __trap (); // 如果读取到错误数据则报错
        } else if ( threadIdx . x == 1 ) {
            // 生产者：写入数据然后设置标志
            * data = 42 ;
            atomic_ref . store ( 1 , cuda :: memory_order_release );
        }
    }
    ```

    `cuda::atomic`
    ```cuda
    #include <cuda/atomic>
    __global__ void kernel ( cuda :: atomic < int , cuda :: thread_scope_device >* flag , int * data ) {
        if ( threadIdx . x == 0 ) {
            // 消费者：阻塞直到生产者设置标志，然后读取数据
            while ( flag -> load ( cuda :: memory_order_acquire ) == 0 ) ;
            if ( * data != 42 ) __trap (); // 如果读取到错误数据则报错
        } else if ( threadIdx . x == 1 ) {
            // 生产者：写入数据然后设置标志
            * data = 42 ;
            flag -> store ( 1 , cuda :: memory_order_release );
        }
    }
    ```

    原子函数（`atomicAdd` 和 `atomicExch`）
    ```cuda
    __global__ void kernel ( int * flag , int * data ) {
        if ( threadIdx . x == 0 ) {
            // 消费者：阻塞直到生产者设置标志，然后读取数据
            while ( atomicAdd ( flag , 0 ) == 0 ) ; // 使用 Relaxed 读-修改-写进行加载
            __threadfence (); // 顺序一致性栅栏
            if ( * data != 42 ) __trap (); // 如果读取到错误数据则报错
        } else if ( threadIdx . x == 1 ) {
            // 生产者：写入数据然后设置标志
            * data = 42 ;
            __threadfence (); // 顺序一致性栅栏
            atomicExch ( flag , 1 ); // 使用 Relaxed 读-修改-写进行存储
        }
    }
    ```
- 内存映射 I/O（MMIO）：请改用通过内联 PTX 的 PTX MMIO 操作。PTX MMIO 操作会严格保持执行的内存访问次数。
然而，CUDA C++ 的 volatile 操作不会保持执行的内存访问次数，并且可能以不确定的方式执行比请求更多或更少的访问。这使得它们不适用于 MMIO。
以下示例展示了如何使用 PTX MMIO 操作读取和写入寄存器。

```cuda
__global__ void kernel ( int * mmio_reg0 , int * mmio_reg1 ) {
    // 写入 MMIO 寄存器：
    int value = 13 ;
    asm volatile ( "st.relaxed.mmio.sys.u32 [%0], %1;" : : "l" ( mmio_reg0 ), "r" ( value ) : "memory" );
    // 读取 MMIO 寄存器：
    asm volatile ( "ld.relaxed.mmio.sys.u32 %0, [%1];" : "=r" ( value ) : "l" ( mmio_reg1 ) : "memory" );
    if ( value != 42 ) __trap (); // 如果读取到错误数据则报错
}
```

#### 5.3.9.4.4. 静态变量

在以下情况下，设备代码中允许使用 `static` 变量：

- 在 `__global__` 或仅限 `__device__` 的函数内部。
- 在 `__host__ __device__` 函数内部：没有显式内存空间（自动推断）的静态变量。具有显式内存空间的静态变量，例如 `static __device__`/`__constant__`/`__shared__`/`__managed__`，仅在定义了 `__CUDA_ARCH__` 时才允许。

`__host__ __device__` 函数内的 `static` 变量根据执行空间持有不同的值。

下面展示了函数作用域 `static` 变量的合法和非法使用示例。

```cuda
struct TrivialStruct {
    int x;
};

struct NonTrivialStruct {
    __device__ NonTrivialStruct(int x) {}
};

__device__ void device_function(int x) {
    static int v1;              // 正确，隐式 __device__ 内存空间说明符
    static int v2 = 11;         // 正确，隐式 __device__ 内存空间说明符
//  static int v3 = x;           // 错误，不允许动态初始化

    static __managed__  int v4; // 正确，显式
    static __device__   int v5; // 正确，显式
    static __constant__ int v6; // 正确，显式
    static __shared__   int v7; // 正确，显式

    static TrivialStruct    s1;     // 正确，隐式 __device__ 内存空间说明符
    static TrivialStruct    s2{22}; // 正确，隐式 __device__ 内存空间说明符
//  static TrivialStruct    s3{x};   // 错误，不允许动态初始化
//  static NonTrivialStruct s4{3};   // 错误，不允许动态初始化
}
```

查看 [Compiler Explorer](https://godbolt.org/z/TdYKaTq3f) 上的示例。

---

```cuda
__host__ __device__ void host_device_function() {
    static            int v1; // 正确，隐式 __device__ 内存空间说明符
//  static __device__ int v2;  // 错误，主机-设备函数内部不能有仅限 __device__ 的变量
#ifdef __CUDA_ARCH__
    static __device__ int v3; // 正确，该声明仅在设备编译期间可见
#else
    static int v4;            // 正确，该声明仅在主机编译期间可见
#endif
}
```
请查看 [Compiler Explorer](https://godbolt.org/z/18qhjn8P1) 上的示例。

---

```cuda
#include <cassert>

__host__ __device__ int host_device_function() {
    static int v = 0;
    v++;
    return v;
}

__global__ void kernel() {
    int ret = host_device_function(); // v = 1
    assert(ret == 4);                 // FAIL
}

int main() {
    host_device_function();           // v = 1
    host_device_function();           // v = 2
    int ret = host_device_function(); // v = 3
    assert(ret == 3);                 // OK
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```

请查看 [Compiler Explorer](https://godbolt.org/z/Wqo9WjvYY) 上的示例。

#### 5.3.9.4.5. 外部变量

在[全程序编译模式](../02-basics/nvcc.html#nvcc-separate-compilation)下编译时，`__device__`、`__shared__`、`__managed__` 和 `__constant__` 变量不能使用 `extern` 关键字定义为外部链接。

唯一的例外是[共享内存的动态分配](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-dynamic-allocation-shared-memory)章节中描述的动态分配的 `__shared__` 变量。

```cuda
__device__        int x; // 正确
extern __device__ int y; // 在全程序编译模式下错误
extern __shared__ int z; // 正确
```

### 5.3.9.5. 函数

#### 5.3.9.5.1. 递归

`__global__` 函数不支持递归，而 `__device__` 和 `__host__ __device__` 函数没有此限制。

#### 5.3.9.5.2. 外部链接

具有外部链接的设备变量或函数需要在多个翻译单元之间采用[单独编译模式](../02-basics/nvcc.html#nvcc-separate-compilation)。

在单独编译模式下，如果 `__device__` 或 `__global__` 函数定义必须存在于某个特定的翻译单元中，那么该函数的参数和返回类型在该翻译单元中必须是完整的。这个概念也被称为单一定义规则使用，或 ODR-use。

示例：

```cuda
//first.cu:
struct S;                   // 前向声明
__device__ void foo(S);     // 错误，类型 'S' 是不完整类型
__device__ auto* ptr = foo; // ODR-use，获取了地址

int main() {}
```

```cuda
//second.cu:
struct S {};               // 结构体定义
__device__ void foo(S) {}  // 函数定义
```

```cuda
# 编译器调用
$ nvcc -std=c++14 -rdc=true first.cu second.cu -o prog
nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o',
                 first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
nvlink fatal   : merge_elf failed
```

#### 5.3.9.5.3. 形式参数

`__device__`、`__shared__`、`__managed__` 和 `__constant__` 内存空间说明符不允许用于形式参数。

```cuda
void device_function1(__device__ int x) { } // 错误，__device__ 参数
void device_function2(__shared__ int x) { } // 错误，__shared__ 参数
```

#### 5.3.9.5.4. __global__ 函数参数
`__global__` 函数具有以下限制：

- 不能有可变数量的参数，即 C 语言的省略号语法 `...` 和 `va_list` 类型。允许使用 C++11 可变参数模板，但需遵守 __global__ 可变参数模板 部分描述的限制。
- 函数参数通过常量内存传递给设备，其总大小限制为 32,764 字节。
- 函数参数不能通过引用传递或通过右值引用传递。
- 函数参数不能是 `std::initializer_list` 类型。
- 多态类参数（`virtual`）被视为未定义行为。
- 允许使用 Lambda 表达式和闭包类型，但需遵守 Lambda 表达式和 __global__ 函数参数 部分描述的限制。

#### 5.3.9.5.5. __global__ 函数参数传递

当从设备代码启动 `__global__` 函数时，每个参数必须是可平凡复制且可平凡析构的。

当从主机代码启动 `__global__` 函数时，每个参数类型可以是非平凡复制或非平凡析构的。然而，如下所述，对这些类型的处理并不遵循标准的 C++ 模型。用户代码必须确保此工作流程不影响程序的正确性。该工作流程在以下两个方面与标准 C++ 不同：

1.  **原始内存复制而非调用复制构造函数**
    CUDA 运行时通过复制原始内存内容（最终使用 `memcpy`）将内核参数传递给 `__global__ 函数。如果参数是非平凡复制的，并且提供了用户定义的复制构造函数，则在主机到设备的复制过程中会跳过该调用的操作和副作用。
    示例：
    ```cpp
    #include <cassert>
    struct MyStruct {
        int value = 1;
        int *ptr;
        MyStruct() = default;
        __host__ __device__ MyStruct(const MyStruct &) { ptr = &value; }
    };
    __global__ void device_function(MyStruct my_struct) {
        // 此断言失败，因为 "my_struct" 是通过复制原始内存内容获得的，并且跳过了复制构造函数。
        assert(my_struct.ptr == &my_struct.value); // 失败
    }
    void host_function(MyStruct my_struct) {
        assert(my_struct.ptr == &my_struct.value); // 正确
    }
    int main() {
        MyStruct my_struct;
        host_function(my_struct);
        device_function<<<1, 1>>>(my_struct); // 复制构造函数仅在主机端调用
        cudaDeviceSynchronize();
    }
    ```
    参见 Compiler Explorer 上的示例。

2.  **析构函数可能在 `__global__` 函数完成之前被调用**
    内核启动与主机执行是异步的。因此，如果 `__global__` 函数参数具有非平凡析构函数，则该析构函数甚至可能在 `__global__` 函数完成执行之前就在主机代码中执行。这可能会破坏析构函数有副作用的程序。
    示例：
    ```cpp
    #include <cassert>
    __managed__ int var = 0;
    struct MyStruct {
        __host__ __device__ ~MyStruct() { var = 3; }
    };
    __global__ void device_function(MyStruct my_struct) {
        assert(var == 0); // 失败，MyStruct::~MyStruct() 将值设置为 3
    }
    int main() {
        MyStruct my_struct;
        // GPU 内核执行与主机执行是异步的。
        // 因此，MyStruct::~MyStruct() 可能在内核完成执行之前就被执行。
        device_function<<<1, 1>>>(my_struct);
        cudaDeviceSynchronize();
    }
    ```
    参见 Compiler Explorer 上的示例。
### 5.3.9.6. 类

#### 5.3.9.6.1. 类类型变量

使用 `__device__`、`__constant__`、`__managed__` 或 `__shared__` 内存空间定义的变量，其类型不能是具有非空构造函数或非空析构函数的类类型。在翻译单元的某个点上，如果一个类类型的构造函数是平凡的，或者满足以下所有条件，则被视为空构造函数：

- 构造函数已被定义。
- 构造函数没有参数，具有空的初始化列表和空的复合语句函数体。
- 其类没有虚函数、虚基类或非静态数据成员初始化器。
- 其所有基类的默认构造函数可被视为空。
- 对于该类的所有类类型（或其数组）的非静态数据成员，其默认构造函数可被视为空。

在翻译单元的某个点上，如果一个类的析构函数是平凡的，或者满足以下所有条件，则被视为空析构函数：

- 析构函数已被定义。
- 析构函数体是一个空的复合语句。
- 其类没有虚函数或虚基类。
- 其所有基类的析构函数可被视为空。
- 对于该类的所有类类型（或其数组）的非静态数据成员，其析构函数可被视为空。

#### 5.3.9.6.2. 数据成员

`__device__`、`__shared__`、`__managed__` 和 `__constant__` 内存空间说明符不允许用于 `class`、`struct` 和 `union` 的数据成员。

仅支持在编译时求值的 `static` 数据成员，例如 [const 限定](#const-variables) 的变量和 `constexpr` 变量。

```cuda
struct MyStruct {
   static inline constexpr int value1 = 10; // C++17
   static constexpr        int value2 = 10; // C++11
   static const            int value3 = 10;
// static                  int value4; // ERROR
};
```

#### 5.3.9.6.3. 函数成员

`__global__` 函数不能是 `struct`、`class` 或 `union` 的成员。

`__global__` 函数允许出现在 `friend` 声明中，但不能被定义。

示例：

```cuda
struct MyStruct {
    friend __global__ void f();   // 正确，仅为友元声明

//  friend __global__ void g() {} // 错误，友元定义
};
```

参见 [Compiler Explorer](https://godbolt.org/z/rv6cP3b9j) 上的示例。

#### 5.3.9.6.4. 隐式声明和非虚显式默认函数

隐式声明的特殊成员函数是当用户未声明时编译器为类声明的函数；显式默认函数是用户声明但用 `= default` 标记的函数。隐式声明或显式默认的特殊成员函数包括：默认构造函数、复制构造函数、移动构造函数、复制赋值运算符、移动赋值运算符和析构函数。

令 `F` 表示一个非 `virtual` 函数，该函数在其首次声明时是隐式声明或显式默认的。`F` 的执行空间说明符是所有调用它的函数的执行空间说明符的并集。请注意，在此分析中，`__global__` 调用者将被视为 `__device__` 调用者。例如：

```cuda
class Base {
    int x;
public:
    __host__ __device__ Base() : x(10) {}
};

class Derived : public Base {
    int y;
};

class Other: public Base {
    int z;
};

__device__ void foo() {
    Derived D1;
    Other D2;
}

__host__ void bar() {
    Other D3;
}
```

In this case, the implicitly declared constructor function `Derived::Derived()` will be treated as a `__device__` function because it is only invoked from the `__device__` function `foo()`. The implicitly declared constructor function `Other::Other()` will be treated as a `__host__ __device__` function since it is invoked both from both a `__device__` function `foo()` and a `__host__` function `bar()`.

Additionally, if `F` is an implicitly-declared `virtual` function (for example, a `virtual` destructor), the execution spaces of each virtual function `D` that is overridden by `F` are added to the set of execution spaces for `F` if `D` not implicitly-declared.

For example:

```cuda
struct Base1 {
    virtual __host__ __device__ ~Base1() {}
};

struct Derived1 : Base1 {}; // implicitly-declared virtual destructor
                            // ~Derived1() has __host__ __device__  execution space specifiers

struct Base2 {
    virtual __device__ ~Base2() = default;
};

struct Derived2 : Base2 {}; // implicitly-declared virtual destructor
                            // ~Derived2() has __device__ execution space specifiers
```

#### 5.3.9.6.5.Polymorphic Classes

Polymorphic classes, namely those with `virtual` functions, derived from other polymorphic classes, or with polymorphic data members, are subject to the following restrictions:

- Copying polymorphic objects from device to host or from host to device, including __global__ function arguments is undefined behavior.
- The execution space of an overridden virtual function must match the execution space of the function in the base class.

Example:

```cuda
struct MyClass {
    virtual __host__ __device__ void f() {}
};

__global__ void kernel(MyClass my_class) {
    my_class.f(); // undefined behavior
}

int main() {
    MyClass my_class;
    kernel<<<1, 1>>>(my_class);
    cudaDeviceSynchronize();
}
```

See the example on [Compiler Explorer](https://godbolt.org/z/To39sGTrW).

---

```cuda
struct BaseClass {
    virtual __host__ __device__ void f() {}
};

struct DerivedClass : BaseClass {
    __device__ void f() override {} // ERROR
};
```

See the example on [Compiler Explorer](https://godbolt.org/z/xfKhEGfdG).

#### 5.3.9.6.6.Windows-Specific Class Layout

The CUDA compiler follows the IA64 ABI for class layout, while Microsoft Visual Studio does not. This prevents bitwise copy of special objects between host and device code as described below.

Let `T` denote a pointer to member type, or a class type that satisfies any of the following conditions:

- T is a polymorphic class
- T has multiple inheritance with more than one direct or indirect empty base class .
- All direct and indirect base classes B are empty and the type of the first field F of T uses B in its definition, such that B is laid out at offset 0 in the definition of F .
当使用 Microsoft Visual Studio 编译时，类型为 `T` 的类、具有类型为 `T` 的基类的类、或具有类型为 `T` 的数据成员的类，其在主机和设备上的类布局和大小可能不同。

将此类对象从设备复制到主机或从主机复制到设备（包括 `__global__` 函数参数）是未定义行为。

### 5.3.9.7. 模板

如果满足以下任一条件，则类型不能用作 `__global__` 函数或 `__device__/__constant__` 变量（C++14）的模板参数：

- 该类型在 `__host__` 或 `__host__ __device__` 函数作用域内定义。
- 该类型是未命名的，例如匿名结构体或 lambda 表达式，除非该类型是 `__device__` 或 `__global__` 函数的局部类型。
- 该类型是具有 `private` 或 `protected` 访问权限的类成员，除非该类是 `__device__` 或 `__global__` 函数的局部类。
- 该类型由上述任何类型复合而成。

示例：

```cuda
template <typename T>
__global__ void kernel() {}

template <typename T>
__device__ int device_var; // C++14

struct {
    int v;
} unnamed_struct;

void host_function() {
    struct LocalStruct {};
//  kernel<LocalStruct><<<1, 1>>>(); // 错误，LocalStruct 在主机函数内定义
    int data = 4;
//  cudaMemcpyToSymbol(device_var<LocalStruct>, &data, sizeof(data)); // 错误，同上

    auto lambda = [](){};
//  kernel<decltype(lambda)><<<1, 1>>>();         // 错误，未命名类型
//  kernel<decltype(unnamed_struct)><<<1, 1>>>(); // 错误，未命名类型
}

class MyClass {
private:
    struct PrivateStruct {};
public:
    static void launch() {
//      kernel<PrivateStruct><<<1, 1>>>(); // 错误，私有类型
    }
};
```

查看 [Compiler Explorer](https://godbolt.org/z/EhTn3GT3z) 上的示例。

## 5.3.10. C++11 限制

### 5.3.10.1. 内联命名空间

当在封闭命名空间中定义了具有相同名称和类型签名的另一个实体时，不允许在 `inline` 命名空间内定义以下任一实体：

- `__global__` 函数。
- `__device__`、`__constant__`、`__managed__`、`__shared__` 变量。
- 具有 surface 或 texture 类型的变量，例如 `cudaSurfaceObject_t` 或 `cudaTextureObject_t`。

示例：

```cuda
__device__ int my_var; // 全局作用域

inline namespace NS {

__device__ int my_var; // 命名空间作用域

} // namespace NS
```

### 5.3.10.2. 内联未命名命名空间

以下实体不能在 `inline` 未命名命名空间的命名空间作用域内声明：

- `__global__` 函数。
- `__device__`、`__constant__`、`__managed__`、`__shared__` 变量。
- 具有 surface 或 texture 类型的变量，例如 `cudaSurfaceObject_t` 或 `cudaTextureObject_t`。

### 5.3.10.3. constexpr 函数

默认情况下，`constexpr` 函数不能从具有不兼容执行空间的函数中调用，这与标准函数相同。

- 在主机代码生成阶段（即 `__CUDA_ARCH__` 宏未定义时）从主机函数调用仅限设备的 constexpr 函数。
示例：
```cuda
constexpr __device__ int device_function () { return 0 ; }
int main () { int x = device_function (); // 错误，从主机代码调用仅限设备的 constexpr 函数 }
```
- 在设备代码生成阶段（即定义 `__CUDA_ARCH__` 宏时），从 `__device__` 或 `__global__` 函数中调用仅限主机的 `constexpr` 函数。示例：
  ```cpp
  constexpr int host_function() { return 0; }
  __device__ void device_function() {
      int x = host_function(); // 错误：从设备代码调用仅限主机的 constexpr 函数
  }
  ```
  请注意，即使相应的模板函数被标记为 `constexpr` 关键字，函数模板特化也可能不是 `constexpr` 函数。

**放宽的 constexpr 函数支持**

可以使用实验性的 `nvcc` 标志 `--expt-relaxed-constexpr` 来放宽对 `__host__` 和 `__device__` 函数的此限制。但是，`__global__` 函数不能声明为 `constexpr`。`nvcc` 还将定义宏 `__CUDACC_RELAXED_CONSTEXPR__`。

指定此标志后，编译器将支持上述跨执行空间调用，具体如下：

1. 如果跨执行空间对 `constexpr` 函数的调用发生在需要常量求值的上下文中（例如 `constexpr` 变量的初始化器），则支持该调用。示例：
   ```cpp
   constexpr __host__ int host_function(int x) { return x + 1; };
   __global__ void kernel() {
       constexpr int val = host_function(1); // 正确：调用发生在需要常量求值的上下文中。
   }
   constexpr __device__ int device_function(int x) { return x + 1; }
   int main() {
       constexpr int val = device_function(1); // 正确：调用发生在需要常量求值的上下文中。
   }
   ```

2. 在设备代码生成期间，会为仅限主机的 `constexpr` 函数体生成设备代码，除非它未被使用或仅在 `constexpr` 上下文中被调用。示例：
   ```cpp
   // 注意："host_function" 在生成的设备代码中被发出，
   //       因为它在非 constexpr 上下文中从设备代码被调用
   constexpr int host_function(int x) { return x + 1; }
   __device__ int device_function(int in) {
       return host_function(in); // 正确，即使参数不是常量表达式
   }
   ```

3. 适用于设备函数的所有代码限制也适用于从设备代码调用的仅限主机的 `constexpr` 函数。但是，编译器可能不会为与编译过程相关的限制发出任何构建时诊断信息。例如，以下代码模式在主机函数体中不受支持。这与任何设备函数类似；但是，可能不会生成编译器诊断信息。
   * 对主机变量或仅限主机的非 `constexpr` 函数进行单一定义规则（ODR）使用。示例：
     ```cpp
     int host_var1, host_var2;
     constexpr int* host_function(bool b) {
         return b ? &host_var1 : &host_var2;
     };
     __device__ int device_function(bool flag) {
         return *host_function(flag); // 错误：host_function() 试图引用主机变量
                                      //       'host_var1' 和 'host_var2'。
                                      //       代码将编译，但不会正确执行。
     }
     ```
   * 使用异常 `throw`/`catch` 和运行时类型信息 `typeid`/`dynamic_cast`。示例：
     ```cpp
     struct Base {};
     struct Derived : public Base {};
     // 注意："host_function" 在生成的设备代码中被发出
     constexpr int host_function(bool b, Base* ptr) {
         if (b) {
             return 1;
         } else if (typeid(ptr) == typeid(Derived)) { // 错误：在 GPU 上执行的代码中使用 typeid
             return 2;
         } else {
             throw int{4}; // 错误：在 GPU 上执行的代码中使用 throw
         }
     }
     __device__ void device_function(bool flag) {
         Derived d;
         int val = host_function(flag, &d); // 错误：host_function() 试图使用 typeid 和 throw()，
                                            //       这在 GPU 上执行的代码中是不允许的
     }
     ```
4. 在主机代码生成期间，仅限设备使用的 constexpr 函数体会保留在发送给主机编译器的代码中。然而，如果设备函数体试图 ODR 使用一个命名空间作用域的设备变量或一个非 constexpr 的设备函数，则不支持从主机代码调用该设备函数。虽然代码可能在编译时没有诊断信息，但在运行时可能行为不正确。示例：
```cuda
__device__ int device_var1, device_var2;
constexpr __device__ int* device_function(bool b) {
    return b ? &device_var1 : &device_var2;
};
int host_function(bool flag) {
    return *device_function(flag); // 错误，device_function() 试图引用设备变量
                                   // 'device_var1' 和 'device_var2'
                                   // 代码将编译，但不会正确执行。
}
```

!!! warning "警告"
    由于上述限制以及缺乏对错误使用的编译器诊断，建议避免在设备代码中调用标准 C++ 头文件 std:: 中的函数。此类函数的实现因主机平台而异。相反，强烈建议调用 CUDA C++ 标准库 libcu++ 中 cuda::std:: 命名空间内的等效功能。

### 5.3.10.4. constexpr 变量

默认情况下，`constexpr` 变量不能在不兼容执行空间的函数中使用，这与标准变量的规则相同。

在以下情况下，`constexpr` 变量可以直接在设备代码中使用：

-   C++ 标量类型，不包括指针和指向成员的指针类型：nullptr_t、bool、整型：char、signed char、unsigned、long long 等。浮点类型：float、double。枚举器：enum 和 enum class。
-   类类型：具有 constexpr 构造函数的 class、struct 和 union。
-   上述类型的原始数组，例如 int[]，仅当它们在 constexpr __device__ 或 __host__ __device__ 函数内部使用时。

不允许使用 `constexpr __managed__` 和 `constexpr __shared__` 变量。

示例：

```cuda
constexpr int ConstexprVar = 4; // 标量类型

struct MyStruct {
    static constexpr int ConstexprVar = 100;
};

constexpr MyStruct my_struct = MyStruct{}; // 类类型

constexpr int array[] = {1, 2, 3};

__device__ constexpr int get_value(int idx) {
    return array[idx];                      // 正确
}

__device__ void foo(int idx) {
    int        v1 = ConstexprVar;           // 正确
    int        v2 = MyStruct::ConstexprVar; // 正确
//  const int &v3 = ConstexprVar1;          // 错误，引用主机 constexpr 变量
//  const int *v4 = &ConstexprVar1;         // 错误，取主机 constexpr 变量的地址
    int        v5 = get_value(2);           // 正确，'get_value(2)' 是常量表达式。
//  int        v6 = get_value(idx);         // 错误，'get_value(idx)' 不是常量表达式
//  int        v7 = array[2];               // 错误，'array' 不是标量类型。
    MyStruct   v8 = my_struct;              // 正确
}
```
请查看 [Compiler Explorer](https://godbolt.org/z/MWa1o3c9z) 上的示例。

### 5.3.10.5. `__global__` 可变参数模板

可变参数 `__global__` 函数模板有以下限制：

- 只允许一个参数包。
- 参数包必须在模板参数列表的最后列出。

示例：

```cuda
template <typename... Pack>
__global__ void kernel1(); // 正确

// template <typename... Pack, template T>
// __global__ void kernel2(); // 错误，参数包不是最后一个参数

template <typename... TArgs>
struct MyStruct {};

// template <typename... Pack1, typename... Pack2>
// __global__ void kernel3(MyStruct<Pack1...>, MyStruct<Pack2...>); // 错误，超过一个参数包
```

请查看 [Compiler Explorer](https://godbolt.org/z/x48KnPbbY) 上的示例。

### 5.3.10.6. 默认函数 =default

CUDA 编译器推断显式默认成员函数的执行空间，如[隐式声明和显式默认函数](#compiler-generated-functions)中所述。

编译器会忽略显式默认函数上的执行空间说明符，除非该函数是外联定义的或是 `virtual` 函数。

示例：

```cuda
struct MyStruct1 {
    MyStruct1() = default;
};

void host_function() {
    MyStruct1 my_struct; // __host__ __device__ 构造函数
}

__device__ void device_function() {
    MyStruct1 my_struct; // __host__ __device__ 构造函数
}

struct MyStruct2 {
    __device__ MyStruct2() = default; // 警告：__device__ 注解被忽略
};

struct MyStruct3 {
    __host__ MyStruct3();
};
MyStruct3::MyStruct3() = default; // 外联定义，不被忽略

__device__ void device_function2() {
//  MyStruct3 my_struct; // 错误，__host__ 构造函数
}

struct MyStruct4 {
    //  MyStruct4::~MyStruct4 具有主机执行空间，因为是 virtual 函数，所以不被忽略
    virtual __host__ ~MyStruct4() = default;
};

__device__ void device_function3() {
    MyStruct4 my_struct4;
    // 对 'my_struct4' 的隐式析构函数调用：
    //    错误：从 __device__ 函数 'device_function3' 调用 __host__ 函数 'MyStruct4::~MyStruct4'
}
```

请查看 [Compiler Explorer](https://godbolt.org/z/q1M4j8YYf) 上的示例。

### 5.3.10.7. [cuda::]std::initializer_list

默认情况下，CUDA 编译器隐式地认为 `[cuda::]std::initializer_list` 的成员函数具有 `__host__ __device__` 执行空间说明符，因此可以直接从设备代码中调用它们。
`nvcc` 标志 `--no-host-device-initializer-list` 会禁用此行为；`[cuda::]std::initializer_list` 的成员函数随后将被视为 `__host__` 函数，并且不能直接从设备代码调用。

`__global__` 函数的参数不能是 `[cuda::]std::initializer_list` 类型。

示例：

```cuda
#include <initializer_list>

__device__ void foo(std::initializer_list<int> in) {}

__device__ void bar() {
    foo({4,5,6}); // (a) 仅包含常量表达式的初始化列表。
    int i = 4;
    foo({i,5,6}); // (b) 至少包含一个非常量元素的初始化列表。
                  // 这种形式可能比 (a) 具有更好的性能。
}
```
请参阅 [Compiler Explorer](https://godbolt.org/z/xeah7r44T) 上的示例。

### 5.3.10.8.[cuda::]std::move, [cuda::]std::forward

默认情况下，CUDA 编译器隐式地将 `std::move` 和 `std::forward` 函数模板视为具有 `__host__ __device__` 执行空间说明符，因此可以直接从设备代码中调用它们。`nvcc` 标志 `--no-host-device-move-forward` 会禁用此行为；`std::move` 和 `std::forward` 随后将被视为 `__host__` 函数，无法直接从设备代码调用。

!!! note "提示"
    相反，cuda::std::move 和 cuda::std::forward 始终具有 __host__ __device__ 执行空间。

## 5.3.11.C++14 限制

### 5.3.11.1.具有推导返回类型的函数

`__global__` 函数不能具有推导返回类型 `auto`。

不允许在主机代码中检查具有推导返回类型的 `__device__` 函数的返回类型。

!!! note "注意"
    CUDA 前端编译器在调用主机编译器之前，会将函数声明更改为具有 void 返回类型。这可能会破坏主机代码中对 __device__ 函数推导返回类型的检查。因此，CUDA 编译器将在设备函数体外引用此类推导返回类型时发出编译时错误。

示例：

```cuda
 __device__ auto device_function(int x) { // 推导返回类型
     return x;                            // decltype(auto) 具有相同行为
 }

 __global__ void kernel() {
     int x = sizeof(device_function(2));         // 正确，设备代码作用域
 }

 // const int size = sizeof(device_function(2)); // 错误，在主机上进行返回类型推导

 void host_function() {
 //  using T = decltype(device_function(2));     // 错误，在主机上进行返回类型推导
 }

void host_fn1() {
  // 错误，在设备函数体外引用
  int (*p1)(int) = fn1;

  struct S_local_t {
    // 错误，在设备函数体外引用
    decltype(fn2(10)) m1;

    S_local_t() : m1(10) { }
  };
}

// 错误，在设备函数体外引用
template <typename T = decltype(fn2)>
void host_fn2() { }

template<typename T> struct MyStruct { };

// 错误，在设备函数体外引用
struct S1_derived_t : MyStruct<decltype(fn1)> { };
```

### 5.3.11.2.变量模板

使用 Microsoft 编译器时，`__device__` 或 `__constant__` 变量模板不能是 `const` 限定的。

示例：

```cuda
// 在 Windows 上错误（不可移植），const 限定
template <typename T>
__device__ const T var = 0;

 // 正确，ptr1 不是 const 限定的
template <typename T>
__device__ const T* ptr1 = nullptr;

// 在 Windows 上错误（不可移植），ptr2 是 const 限定的
template <typename T>
__device__ const T* const ptr2 = nullptr;
```

请参阅 [Compiler Explorer](https://godbolt.org/z/8hM5Yh7db) 上的示例。

## 5.3.12.C++17 限制

### 5.3.12.1.inline 变量

在单个翻译单元中，使用 `inline` 变量相比普通变量不提供额外功能，也不提供任何实际优势。
`nvcc` 仅允许在**单独编译**模式下或具有内部链接的变量中，使用带有 `__device__`、`__constant__` 或 `__managed__` 内存空间的 `inline` 变量。

!!! note "注意"
    当使用 gcc/g++ 主机编译器时，使用 `__managed__` 内存空间说明符声明的内联变量可能对调试器不可见。

示例：

```cuda
inline        __device__ int device_var1;  // 正确，当在单独编译模式下编译时（-rdc=true 或 -dc）
                                           // 错误，当在整体程序编译模式下编译时

static inline __device__ int device_var2;  // 正确，内部链接

namespace {

inline __device__ int device_var3;         // 正确，内部链接

inline __shared__ int shared_var;          // 正确，内部链接

static inline __device__ int device_var4;  // 正确，内部链接

inline __device__ int device_var5;         // 正确，内部链接

} // namespace
```

查看 [Compiler Explorer](https://godbolt.org/z/oraqeGTzY) 上的示例。

### 5.3.12.2. 结构化绑定

结构化绑定不能使用内存空间说明符（如 `__device__`、`__shared__`、`__constant__` 或 `__managed__`）来声明。

示例：

```cuda
struct S {
    int x, y;
};
// __device__ auto [a, b] = S{4, 5}; // 错误
```

## 5.3.13. C++20 限制

### 5.3.13.1. 三路比较运算符

三路比较运算符 (`<=>`) 在设备代码中受支持，但某些用法隐式依赖于 C++ 标准库的功能，该功能由主机实现提供。使用这些运算符可能需要指定 `--expt-relaxed-constexpr` 标志以消除警告，并且该功能要求主机实现满足设备代码的要求。

示例：

```cuda
#include <compare> // std::strong_ordering 实现

struct S {
    int x, y;

    auto operator<=>(const S&) const = default; // (a)

    __host__ __device__ bool operator<=>(int rhs) const { return false; } // (b)
};

__host__ __device__ bool host_device_function(S a, S b) {
    if (a <=> 1)  // 正确，调用用户定义的主机-设备重载 (b)
        return true;
    return a < b; // 正确，调用隐式声明的函数 (a)
                  // 注意：它需要头文件 <compare> 中提供与设备兼容的 std::strong_ordering 实现
                  //       以及 --expt-relaxed-constexpr 标志
}
```

查看 [Compiler Explorer](https://godbolt.org/z/qzs5arfx4) 上的示例。

### 5.3.13.2. consteval 函数

`consteval` 函数可以从主机和设备代码中调用，与其执行空间无关。

示例：

```cuda
consteval int host_consteval() {
    return 10;
}

__device__ consteval int device_consteval() {
    return 10;
}

__device__ int device_function() {
    return host_consteval();   // 正确，即使从设备代码调用
}

__host__ __device__ int host_device_function() {
    return device_function();  // 正确，即使从主机-设备代码调用
}
```
在本页面