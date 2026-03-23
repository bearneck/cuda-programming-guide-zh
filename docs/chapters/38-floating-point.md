# 5.5 浮点运算

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)

---

此页面是否有帮助？

# 5.5. 浮点计算

## 5.5.1. 浮点计算简介

自1985年采用[IEEE-754二进制浮点算术标准](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229)以来，几乎所有主流计算系统，包括NVIDIA的CUDA架构，都已实现了该标准。IEEE-754标准规定了浮点算术结果应如何近似。

为了获得准确的结果并以所需的精度实现最高性能，考虑浮点行为的许多方面非常重要。这在异构计算环境中尤其重要，因为操作是在不同类型的硬件上执行的。

以下部分回顾了浮点计算的基本属性，并涵盖了融合乘加（FMA）运算和点积。这些示例说明了不同的实现选择如何影响精度。

### 5.5.1.1. 浮点格式

浮点格式和功能在[IEEE-754标准](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229)中定义。

该标准规定二进制浮点数据由三个字段编码：

- **符号位**：一位，表示正数或负数。
- **指数位**：编码以数值偏置偏移的基数为2的指数。
- **有效数**（也称为尾数或小数部分）：编码数字的小数值。

![浮点编码](../images/floating-point-encoding.drawio.png)

最新的IEEE-754标准定义了以下二进制格式的编码和属性：

- **16位**，也称为半精度，对应于CUDA中的 `__half` 数据类型。
- **32位**，也称为单精度，对应于C、C++和CUDA中的 `float` 数据类型。
- **64位**，也称为双精度，对应于C、C++和CUDA中的 `double` 数据类型。
- **128位**，也称为四倍精度，对应于CUDA中的 `__float128` 或 `_Float128` 数据类型。

这些类型具有以下位长度：

![IEEE-754浮点编码](../images/floating-point-ieee.drawio.png)

与[规约数](#normal-subnormal)的浮点编码相关的数值计算如下：

\[(-1)^\mathrm{sign} \times 1.\mathrm{mantissa} \times 2^{\mathrm{exponent} - \mathrm{bias}}\]

对于[非规约数](#normal-subnormal)，公式修改为：

\[(-1)^\mathrm{sign} \times 0.\mathrm{mantissa} \times 2^{1-\mathrm{bias}}\]

指数分别以 \(127\) 和 \(1023\) 为偏置值，对应于单精度和双精度。\(1.\) 的整数部分隐含在分数中。

例如，值 \(-192 = (-1)^1 \times 2^7 \times 1.5\)，被编码为负号、指数 \(7\) 和小数部分 \(0.5\)。因此，指数 \(7\) 由位串表示，对于 `float` 其值为 `7 + 127 = 134 = 10000110`，对于 `double` 其值为 `7 + 1023 = 1030 = 10000000110`。
对于 
`double`
 类型。尾数 
`0.5 = 2^-1`
 由二进制值表示，其第一位为 
`1`
。单精度和双精度下 
\(-192\)
 的二进制编码如下图所示：

![``-192`` 的浮点数表示](../images/floating-point-192.drawio.png)

由于小数部分字段使用的位数有限，并非所有实数都能被精确表示。例如，分数 \(2 / 3\) 的数学值的二进制表示是 `0.10101010...`，在二进制小数点后有无限多个位。因此，\(2 / 3\) 在被表示为精度有限的浮点数之前必须进行舍入。舍入规则和模式在 IEEE-754 中规定。最常用的模式是*舍入到最近偶数*，简称最近舍入。

### 5.5.1.2. 规格化值与非规格化值

任何指数字段既不全为零也不全为 1 的浮点数值都称为*规格化*值。

浮点数值的一个重要方面是，最小的可表示正规格化数 `FLT_MIN` 与零之间存在巨大的间隔。这个间隔比 `FLT_MIN` 与第二小的规格化数之间的间隔要大得多。

浮点*非规格化*数，也称为*次正规数*，就是为了解决这个问题而引入的。一个非规格化浮点数值的表示是指数部分的所有位都设置为零，并且有效数字部分至少有一位设置为 1。非规格化数是 IEEE-754 浮点标准的必要组成部分。

非规格化数允许精度逐渐损失，作为突然向零舍入的替代方案。然而，非规格化数的计算成本更高。因此，不需要严格精度的应用程序可能会选择避免使用它们以提高性能。`nvcc` 编译器允许通过设置 `-ftz=true` 选项（刷新到零）来禁用非规格化数，该选项也包含在 `--use_fast_math` 中。

单精度下最小规格化值和非规格化值的编码简化示意图如下：

![最小规格化值和非规格化值表示](../images/floating-point-subnormal.drawio.png)

其中 `X` 代表 `0` 和 `1`。

### 5.5.1.3. 特殊值

IEEE-754 标准为浮点数定义了三个特殊值：

**零：**

-   数学上的零。
-   注意，浮点零有两种可能的表示：+0 和 -0。这与整数零的表示不同。
-   +0 == -0 的计算结果为 true。
-   零的编码是指数和有效数字的所有位都设置为 0。

**无穷大：**

-   浮点数遵循饱和算术规则，其中超出可表示范围的操作会产生 +Infinity 或 -Infinity。
-   无穷大的编码是指数部分的所有位设置为 1，有效数字部分的所有位设置为 0。无穷大值恰好有两种编码。
- 涉及无穷大和有限非零值的算术运算通常会产生无穷大。不确定形式如 Inf * 0.0、Inf - Inf、Inf / Inf 和 0.0 / 0.0 会产生 NaN。

**非数（NaN）：**

- NaN 是一个特殊的符号，表示未定义或无法表示的值。常见的例子有 0.0 / 0.0、sqrt(-1.0) 或 +Inf - Inf。
- NaN 的编码是指数部分的所有位都设置为 1，且有效数字部分为任意位模式，但不能全为 0。共有 \(2^{\mathrm{mantissa} + 1} - 2\) 种可能的编码。
- 任何涉及 NaN 的算术运算都会产生 NaN。
- 任何涉及 NaN 的有序比较（<、<=、>、>=、==）都会返回 false，包括 NaN == NaN（非自反性）。无序比较 NaN != NaN 返回 true。
- NaN 以两种形式提供：静默 NaN（qNaN）用于传播由无效操作或值引起的错误。无效的算术运算通常会产生静默 NaN。其编码是有效数字的最高有效位设置为 1。信号 NaN（sNaN）旨在引发无效操作异常。信号 NaN 通常是显式创建的。其编码是有效数字的最高有效位设置为 0。静默 NaN 和信号 NaN 的确切位模式由具体实现定义。CUDA 提供了 cuda::std::numeric_limits<T>::quiet_NaN 和 cuda::std::numeric_limits<T>::signaling_NaN 常量来获取它们的特殊值。

特殊值编码的简化可视化如下图所示：

![无穷大和 NaN 的浮点数表示](../images/floating-point-special-values.drawio.png)

其中 `X` 代表 `0` 和 `1`。

### 5.5.1.4. 结合律

需要注意的是，由于浮点运算精度有限，数学算术的规则和性质不能直接应用于浮点运算。下面的示例展示了单精度值 `A`、`B` 和 `C`，以及使用不同结合律计算出的它们之和的精确数学值。

\[\begin{split}\begin{aligned}
A           &= 2^{1} \times 1.00000000000000000000001 \\
B           &= 2^{0} \times 1.00000000000000000000001 \\
C           &= 2^{3} \times 1.00000000000000000000001 \\
(A + B) + C &= 2^{3} \times 1.01100000000000000000001011 \\
A + (B + C) &= 2^{3} \times 1.01100000000000000000001011
\end{aligned}\end{split}\]

从数学上讲，\((A + B) + C\) 等于 \(A + (B + C)\)。

令 \(\mathrm{rn}(x)\) 表示对 \(x\) 进行一次舍入操作。根据 IEEE-754 标准，在单精度浮点运算中采用最近舍入模式执行相同的计算，我们得到：

\[\begin{split}\begin{aligned}
A + B                                     &= 2^{1} \times 1.1000000000000000000000110000\ldots \\
\mathrm{rn}(A+B)                          &= 2^{1} \times 1.10000000000000000000010 \\
B + C                                     &= 2^{3} \times 1.0010000000000000000000100100\ldots \\
\mathrm{rn}(B+C)                          &= 2^{3} \times 1.00100000000000000000001 \\
A + B + C                                 &= 2^{3} \times 1.0110000000000000000000101100\ldots \\
\mathrm{rn}\big(\mathrm{rn}(A+B) + C\big) &= 2^{3} \times 1.01100000000000000000010 \\
\mathrm{rn}\big(A + \mathrm{rn}(B+C)\big) &= 2^{3} \times 1.01100000000000000000001
\end{aligned}\end{split}\]

作为参考，上面也计算了精确的数学结果。根据 IEEE-754 计算的结果与精确的数学结果不同。此外，对应于求和 \(\mathrm{rn}(\mathrm{rn}(A + B) + C)\) 和 \(\mathrm{rn}(A + \mathrm{rn}(B + C))\) 的结果也彼此不同。在这种情况下，\(\mathrm{rn}(A + \mathrm{rn}(B + C))\) 比 \(\mathrm{rn}(\mathrm{rn}(A + B) + C)\) 更接近正确的数学结果。

这个例子表明，即使所有基本运算都符合 IEEE-754，看似相同的计算也可能产生不同的结果。

### 5.5.1.5. 融合乘加 (FMA)

融合乘加 (FMA) 运算仅通过一次舍入步骤计算结果。如果没有 FMA，结果将需要两次舍入步骤：一次用于乘法，一次用于加法。因为 FMA 只使用一次舍入步骤，所以它能产生更准确的结果。

融合乘加运算对 NaN 传播的影响可能与两个独立运算不同。然而，FMA 的 NaN 处理并非在所有目标平台上都完全相同。对于具有多个 NaN 操作数的不同实现，可能倾向于产生静默 NaN 或传播某个操作数的有效载荷。此外，当存在多个 NaN 操作数时，IEEE-754 并未严格规定确定性的有效载荷选择顺序。NaN 也可能出现在中间计算中，例如 \(\infty \times 0 + 1\) 或 \(1 \times \infty - \infty\)，从而产生一个由实现定义的 NaN 有效载荷。

---

为了清晰起见，首先考虑一个使用十进制算术的例子来说明 FMA 运算的工作原理。我们将使用总共五位精度（小数点后四位）来计算 \(x^2 - 1\)。

- 对于 \(x = 1.0008\)，正确的数学结果是 \(x^2 - 1 = 1.60064 \times 10^{-4}\)。仅使用小数点后四位的最接近数字是 \(1.6006 \times 10^{-4}\)。
- 融合乘加运算仅通过一次舍入步骤 \(\mathrm{rn}(x \times x - 1) = 1.6006 \times 10^{-4}\) 就得到了正确结果。
- 另一种方法是分别计算乘法和加法步骤。\(x^2 = 1.00160064\) 转换为 \(\mathrm{rn}(x \times x) = 1.0016\)。最终结果是 \(\mathrm{rn}(\mathrm{rn}(x \times x) -1) = 1.6000 \times 10^{-4}\)。

分别对乘法和加法进行舍入得到的结果偏差为 \(0.00064\)。相应的 FMA 计算仅偏差 \(0.00004\)，其结果最接近正确的数学答案。结果总结如下：

\[\begin{split}\begin{aligned}
x                                           &= 1.0008 \\
x^{2}                                       &= 1.00160064 \\
x^{2} - 1                                   &= 1.60064 \times 10^{-4} && \text{真实值} \\
\mathrm{rn}\big(x^{2} - 1\big)              &= 1.6006 \times 10^{-4} && \text{融合乘加} \\
\mathrm{rn}\big(x^{2}\big)                  &= 1.0016 \\
\mathrm{rn}\big(\mathrm{rn}(x^{2}) - 1\big) &= 1.6000 \times 10^{-4} && \text{先乘后加}
\end{aligned}\end{split}\]

---

下面是另一个使用二进制单精度值的示例：

\[\begin{split}\begin{aligned}
A                                                &= 2^{0} \times 1.00000000000000000000001 \\
B                                                &= -2^{0} \times 1.00000000000000000000010 \\
\mathrm{rn}\big(A \times A + B\big)              &= 2^{-46} \times 1.00000000000000000000000 && \text{融合乘加} \\
\mathrm{rn}\big(\mathrm{rn}(A \times A) + B\big) &= 0 && \text{先乘后加}
\end{aligned}\end{split}\]

- 分别计算乘法和加法会导致所有精度位丢失，结果为 \(0\)。
- 而计算 FMA 则能提供与数学值相等的结果。

融合乘加有助于防止在相消性减法中损失精度。当符号相反、量级相近的数相加时，就会发生相消性减法。在这种情况下，许多高位比特会相互抵消，导致有意义的比特数减少。融合乘加在乘法过程中计算双倍宽度的乘积。因此，即使在加法过程中发生相消性减法，乘积中仍有足够的有效比特来产生精确的结果。

---

**CUDA 中的融合乘加支持：**

CUDA 为 `float` 和 `double` 数据类型提供了多种实现融合乘加操作的方式：

- 使用 `-fmad=true` 或 `--use_fast_math` 标志编译时的 `x * y + z` 表达式。
- C 标准库函数 `fma(x, y, z)` 和 `fmaf(x, y, z)`。
- CUDA 数学内置函数 `__fmaf_[rd, rn, ru, rz]`、`__fmaf_ieee_[rd, rn, ru, rz]` 和 `__fma_[rd, rn, ru, rz]`。
- CUDA C++ 标准库函数 `cuda::std::fma(x, y, z)` 和 `cuda::std::fmaf(x, y, z)`。

---

**主机平台上的融合乘加支持：**

是否使用融合操作取决于平台对该操作的支持情况以及代码的编译方式。在比较 CPU 和 GPU 结果时，了解主机平台对融合乘加的支持非常重要。

- 编译器标志与融合乘加硬件支持：GCC 和 Clang 的 `-mfma`、NVC++ 的 `-Mfma` 以及 Microsoft Visual Studio 的 `/fp:contract`。例如，对于具有 AVX2 ISA 的 x86 平台，使用 GCC 或 Clang 的 `-mavx2` 标志以及 Microsoft Visual Studio 的 `/arch:AVX2` 编译的代码。具有高级 SIMD (Neon) ISA 的 Arm64 (AArch64) 平台。
- C 标准库函数 `fma(x, y, z)` 和 `fmaf(x, y, z)`。
- C++ 标准库函数 `std::fma(x, y, z)` 和 `std::fmaf(x, y, z)`。
- CUDA C++ 标准库函数 `cuda::std::fma(x, y, z)` 和 `cuda::std::fmaf(x, y, z)`。
### 5.5.1.6. 点积示例

考虑求两个短向量 \(\overrightarrow{a}\) 和 \(\overrightarrow{b}\) 的点积问题，两个向量都有四个元素。

\[\begin{split}\overrightarrow{a} = \begin{bmatrix} a_{1} \\ a_{2} \\ a_{3} \\ a_{4} \end{bmatrix}
\qquad
\overrightarrow{b} = \begin{bmatrix} b_{1} \\ b_{2} \\ b_{3} \\ b_{4} \end{bmatrix}
\qquad
\overrightarrow{a} \cdot \overrightarrow{b} = a_{1}b_{1} + a_{2}b_{2} + a_{3}b_{3} + a_{4}b_{4}\end{split}\]

尽管这个操作在数学上很容易写出来，但在软件中实现它涉及几种可能产生略微不同结果的替代方案。这里介绍的所有策略都使用完全符合 IEEE-754 标准的运算。

**示例算法 1：** 计算点积最简单的方法是使用顺序的乘积和，将乘法和加法分开进行。

> 最终结果可以表示为
> \(((((a_1 \times b_1) + (a_2 \times b_2)) + (a_3 \times b_3)) + (a_4 \times b_4))\)
> 。

**示例算法 2：** 使用融合乘加顺序计算点积。

> 最终结果可以表示为
> \((a_4 \times b_4) + ((a_3 \times b_3) + ((a_2 \times b_2) + (a_1 \times b_1 + 0)))\)
> 。

**示例算法 3：** 使用分治策略计算点积。首先，我们计算向量前半部分和后半部分的点积。然后，我们使用加法合并这些结果。该算法被称为“并行算法”，因为两个子问题可以并行计算，因为它们彼此独立。然而，该算法并不要求并行实现；它可以用单个线程实现。

> 最终结果可以表示为
> \(((a_1 \times b_1) + (a_2 \times b_2)) + ((a_3 \times b_3) + (a_4 \times b_4))\)
> 。

### 5.5.1.7. 舍入

IEEE-754 标准要求支持多种运算。这些运算包括算术运算，如加法、减法、乘法、除法、平方根、融合乘加、求余数、转换、缩放、符号和比较运算。对于给定的格式和舍入模式，这些运算的结果保证在所有标准实现中都是一致的。

---

**舍入模式**

IEEE-754 标准定义了四种舍入模式：*舍入到最近值*、*向正无穷舍入*、*向负无穷舍入* 和 *向零舍入*。CUDA 支持所有四种模式。默认情况下，运算使用*舍入到最近值*。[内部数学函数](#mathematical-functions-appendix-intrinsic-functions)可用于为单个运算选择其他舍入模式。

| 舍入模式 | 解释 |
| --- | --- |
| rn | 舍入到最近值，平局时取偶数 |
| rz | 向零舍入 |
| ru | 向 \(\infty\) 舍入 |
| rd | 向 \(-\infty\) 舍入 |

### 5.5.1.8. 关于主机/设备计算精度的说明

浮点计算结果的精度受多种因素影响。本节总结了在浮点计算中获得可靠结果的重要注意事项。其中一些方面在前面的章节中已有更详细的描述。
在比较 CPU 与 GPU 的结果时，这些方面也很重要。必须仔细解释主机与设备执行之间的差异。存在差异并不一定意味着 GPU 的结果不正确或 GPU 存在问题。

**结合性**：

> 有限精度下的浮点数加法和乘法不具有结合性，因为它们通常会产生无法在目标格式中直接表示的数学值，需要进行舍入。这些运算的求值顺序会影响舍入误差的累积方式，并可能显著改变最终结果。

**融合乘加**：

> 融合乘加在单次运算中计算 \(a \times b + c\)，从而获得更高的精度和更快的执行时间。其使用会影响最终结果的精度。融合乘加依赖于硬件支持，可以通过显式调用相关函数或通过编译器优化标志隐式启用。

**精度**：

> 提高浮点精度有可能改善结果的准确性。更高的精度可以减少有效数字的损失，并能够表示更广泛的值。然而，更高精度的类型吞吐量较低，并且消耗更多的寄存器。此外，使用它们来显式存储输入和输出会增加内存使用量和数据移动量。

**编译器标志与优化**：

> 所有主流编译器都提供多种优化标志来控制浮点运算的行为。GCC（`-O3`）、Clang（`-O3`）、nvcc（`-O3`）和 Microsoft Visual Studio（`/O2`）的最高优化级别不会影响浮点语义。然而，内联、循环展开、向量化和公共子表达式消除可能会影响结果。NVC++ 编译器也需要标志 `-Kieee` 和 `-Mnofma` 来获得符合 IEEE-754 标准的语义。有关影响浮点行为的选项的详细信息，请参阅 GCC、Clang、Microsoft Visual Studio Compiler、nvc++ 和 Arm C/C++ 编译器的文档。另请参阅 nvcc 用户手册，了解专门影响 CUDA 设备代码中浮点行为的编译器标志的详细说明：`-ftz`、`-prec-div`、`-prec-sqrt`、`-fmad`、`--use_fast_math`。除了这些浮点选项外，在用户程序的上下文中验证其他编译器优化的效果也很重要。鼓励用户通过广泛的测试来验证其结果的正确性，并比较启用优化与禁用所有设备代码优化时获得的结果；另请参阅 `-G` 编译器标志。

**库实现**：

> IEEE-754 标准之外定义的函数不保证正确舍入，并且依赖于实现定义的行为。因此，结果可能在不同的平台（包括主机、设备以及不同的设备架构）之间有所不同。
**确定性结果**：

> 确定性结果指的是在相同指定条件下，每次使用相同输入运行时，都能计算出相同的逐位数值输出。此类条件包括：
> - 硬件依赖性，例如在相同的 CPU 处理器或 GPU 设备上执行。
> - 编译器方面，例如编译器的版本以及
> - 编译器标志和优化
> - 影响计算的运行时条件，例如
> - 舍入模式
> - 或环境变量。
> - 计算的输入完全相同。
> - 线程配置，包括参与计算的线程数量及其组织方式，例如线程块和线程网格的大小。
> - 算术原子操作
> - 的顺序取决于硬件调度，这在各次运行之间可能有所不同。

**利用 CUDA 库的优势**：

> CUDA 数学库
> 、
> C 标准库数学函数
> 以及
> C++ 标准库数学函数
> 旨在提升开发者在常见功能（尤其是浮点数学和数值密集型例程）上的生产力。这些功能提供了一致的高级接口，经过优化，并在各种平台和边缘情况下经过了广泛测试。鼓励用户充分利用这些库，避免繁琐的手动重新实现。

## 5.5.2. 浮点数据类型

CUDA 支持 Bfloat16、半精度、单精度、双精度和四精度浮点数据类型。下表总结了 CUDA 支持的浮点数据类型及其要求。

| 精度 / 名称 | 数据类型 | IEEE-754 | 头文件 / 内置类型 | 要求 |
| --- | --- | --- | --- | --- |
| Bfloat16 | __nv_bfloat16 | ❌ | <cuda_bf16.h> | 计算能力 8.0 或更高。 |
| 半精度 | __half | ✔ | <cuda_fp16.h> |  |
| 单精度 | float | ✔ | 内置 |  |
| 双精度 | double | ✔ | 内置 |  |
| 四精度 | __float128 / _Float128 | ✔ | 数学函数使用内置 <crt/device_fp128_functions.h> | 主机编译器支持且计算能力 10.0 或更高。C 或 C++ 的拼写（分别为 _Float128 和 __float128）也取决于主机编译器的支持。 |

CUDA 还支持 [TensorFloat-32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) (`TF32`)、[微缩放 (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 浮点类型以及其他[低精度数值格式](https://resources.nvidia.com/en-us-blackwell-architecture)，这些格式并非用于通用计算，而是用于涉及张量核心的专门用途。其中包括 4 位、6 位和 8 位浮点类型。更多详情请参阅 [CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/structs.html)。

下图展示了支持的浮点数据类型的尾数和指数大小。

![浮点类型：尾数和指数大小](../images/floating-point.drawio.png)
下表列出了所支持的浮点数据类型的取值范围。

| 精度 / 名称 | 最大值 | 最小正值 | 最小正非规格化数 | Epsilon |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Bfloat16 | \(\approx 2^{128}\) | \(\approx 3.39 \cdot 10^{38}\) | \(2^{-126}\) | \(\approx 1.18 \cdot 10^{-38}\) | \(2^{-133}\) | \(2^{-7}\) |
| Half Precision | \(\approx 2^{16}\) | \(65504\) | \(2^{-14}\) | \(\approx 6.1 \cdot 10^{-5}\) | \(2^{-24}\) | \(2^{-10}\) |
| Single Precision | \(\approx 2^{128}\) | \(\approx 3.40 \cdot 10^{38}\) | \(2^{-126}\) | \(\approx 1.18 \cdot 10^{-38}\) | \(2^{-149}\) | \(2^{-23}\) |
| Double Precision | \(\approx 2^{1024}\) | \(\approx 1.8 \cdot 10^{308}\) | \(2^{-1022}\) | \(\approx 2.22 \cdot 10^{-308}\) | \(2^{-1074}\) | \(2^{-52}\) |
| Quad Precision | \(\approx 2^{16384}\) | \(\approx 1.19 \cdot 10^{4932}\) | \(2^{-16382}\) | \(\approx 3.36 \cdot 10^{-4932}\) | \(2^{-16494}\) | \(2^{-112}\) |

!!! note "提示"
    CUDA C++ 标准库在 <cuda/std/limits> 头文件中提供了 cuda::std::numeric_limits，用于查询所支持的浮点类型（包括微缩放格式 MX）的属性和取值范围。有关可查询属性的列表，请参阅 C++ 参考。

**复数支持：**

- CUDA C++ 标准库通过 <cuda/std/complex> 头文件中的 cuda::std::complex 类型支持复数。更多详细信息，请参阅 libcu++ 文档。
- CUDA 还通过 cuComplex.h 头文件中的 cuComplex 和 cuDoubleComplex 类型为复数提供基本支持。

---

## 5.5.3. CUDA 与 IEEE-754 合规性

所有 GPU 设备均遵循 [IEEE 754-2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229) 二进制浮点运算标准，但存在以下限制：

- 没有动态可配置的舍入模式；但是，大多数运算支持多个恒定的 IEEE 舍入模式，可通过特定命名的设备内部函数进行选择。
- 没有检测浮点异常的机制，因此所有运算的行为都如同 IEEE-754 异常始终被屏蔽。如果发生异常事件，将传递 IEEE-754 定义的默认屏蔽响应。因此，虽然支持信令 NaN SNaN 编码，但它们不会发出信号，而是作为静默异常处理。
- 浮点运算可能会改变输入 NaN 有效载荷的位模式。绝对值、取反等运算也可能不符合 IEEE 754 要求，这可能导致 NaN 的符号以实现定义的方式被更新。

为了最大限度地提高结果的可移植性，建议用户使用 `nvcc` 编译器浮点选项的默认设置：`-ftz=false`、`-prec-div=true` 和 `-prec-sqrt=true`，并且不要使用 `--use_fast_math` 选项。请注意，默认情况下允许浮点表达式重新关联和收缩，类似于 `--fmad=true` 选项。有关这些编译标志的详细说明，请参阅 `nvcc` [用户手册](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math)。
IEEE-754 和 C/C++ 语言标准并未明确规定当舍入后的整数值超出目标整数格式范围时，浮点值向整数值的转换行为。GPU 设备上的钳位行为在 [PTX ISA 转换指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt) 章节中有详细说明。然而，当超出范围的转换并非直接通过 PTX 指令调用时，编译器优化可能会利用未定义行为条款，从而导致未定义行为并产生无效的 CUDA 程序。CUDA 数学文档会针对每个函数/内部函数向用户发出警告。例如，考虑 [__double2int_rz()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html#_CPPv415__double2int_rzd) 指令。这可能与主机编译器和库实现的行为方式不同。

**原子函数的非规格化数行为**：

无论编译器标志 `-ftz` 的设置如何，原子操作在浮点非规格化数方面具有以下行为：

*   对全局内存的原子单精度浮点加法始终以刷新到零模式运行，即行为等同于 PTX add.rn.ftz.f32 语义。
*   对共享内存的原子单精度浮点加法始终支持非规格化数，即行为等同于 PTX add.rn.f32 语义。

## 5.5.4. CUDA 与 C/C++ 的合规性

**浮点异常：**

与主机实现不同，设备代码支持的数学运算符和函数不会设置全局 `errno` 变量，也不会报告[浮点异常](https://en.cppreference.com/w/cpp/numeric/fenv/FE_exceptions)来指示错误。因此，如果需要错误诊断机制，用户应针对这些函数实现额外的输入和输出检查。

**浮点运算的未定义行为：**

数学运算常见的未定义行为条件包括：

*   数学运算符和函数的无效参数：使用未初始化的浮点变量。在浮点变量的生命周期之外使用它。有符号整数溢出。解引用无效指针。
*   浮点特有的未定义行为：将浮点值转换为结果无法表示的整数类型是未定义行为。这也包括 NaN 和无穷大。

用户有责任确保 CUDA 程序的有效性。无效参数可能导致未定义行为，并受编译器优化的影响。

与整数除以零不同，浮点数除以零不是未定义行为，也不受编译器优化的影响；相反，它是实现定义的行为。符合 [IEC-60559](https://en.cppreference.com/w/cpp/types/numeric_limits/is_iec559.html) (IEEE-754) 的 C++ 实现（包括 CUDA）会产生无穷大。请注意，无效的浮点运算会产生 NaN，不应被误解为未定义行为。例如零除以零和无穷大除以无穷大。
**浮点数字面量的可移植性：**

C 和 C++ 都允许使用十进制或十六进制表示法来表示浮点值。十六进制浮点数字面量在 [C99](https://en.cppreference.com/w/c/language/floating_constant.html) 和 [C++17](https://en.cppreference.com/w/cpp/language/floating_literal.html) 中受支持，它表示一个可以用二进制精确表示的科学记数法实数。然而，这并不能保证该字面量会映射到目标变量中存储的实际值（参见下一段）。相反，十进制浮点数字面量可能表示一个无法用二进制表示的数值。

根据 [C++ 标准规则](https://eel.is/c++draft/lex.fcon#3)，十六进制和十进制浮点数字面量会被舍入到最接近的可表示值（更大或更小），具体选择方式由实现定义。这种舍入行为在主机和设备之间可能不同。

```cuda
float f1 = 0.5f;    // 0.5, '0.5f' 是十进制浮点数字面量
float f2 = 0x1p-1f; // 0.5, '0x1p-1f' 是十六进制浮点数字面量
float f3 = 0.1f;
// f1, f2 表示为 0 01111110 00000000000000000000000
// f3     表示为 0 01111011 10011001100110011001101
```

同一浮点表达式的运行时和编译时求值存在以下可移植性问题：

- 浮点表达式的运行时求值可能受所选舍入模式、浮点收缩（FMA）和重关联编译器设置以及浮点异常的影响。请注意，CUDA 不支持浮点异常，且默认情况下舍入模式设置为向偶数舍入。可以使用内部函数选择其他舍入模式。
- 编译器可能对常量表达式使用更高精度的内部表示。
- 编译器可能执行优化，例如常量折叠、常量传播和公共子表达式消除，这可能导致不同的最终值或比较结果。

**C 标准数学库注意事项：**

常见数学函数的主机实现以平台特定的方式映射到 [C 标准数学库函数](https://en.cppreference.com/w/c/header/math.html)。这些函数由主机编译器提供，如果可用，则由相应的主机 `libm` 提供。

- 主机编译器不提供的函数在 crt/math_functions.h 头文件中实现。例如，erfinv() 就在那里实现。
- 不太常见的函数，如 rhypot() 和 cyl_bessel_i0()，仅在设备代码中可用。

如前所述，数学函数的主机和设备实现是独立的。有关这些函数行为的更多详细信息，请参阅主机实现的文档。

---

## 5.5.5. 浮点功能暴露

CUDA 支持的数学函数通过以下方式暴露：
[内置 C/C++ 语言算术运算符](#builtin-math-operators)：

- x + y , x - y , x * y , x / y , x++ , x-- , x += y , x -= y , x *= y , x /= y 。
- 支持单精度、双精度和四精度类型，分别是 `float`、`double` 和 `__float128`/`_Float128`。`__half` 和 `__nv_bfloat16` 类型也受支持，分别需要包含 `<cuda_fp16.h>` 和 `<cuda_bf16.h>` 头文件。`__float128`/`_Float128` 类型的支持依赖于主机编译器和设备计算能力，请参阅支持的浮点类型表。
- 它们在主机和设备代码中均可用。
- 它们的行为受 `nvcc` 优化标志影响。

[CUDA C++ 标准库数学函数](#mathematical-functions-appendix-cxx-standard-functions)：

- 通过 `<cuda/std/cmath>` 头文件和 `cuda::std::` 命名空间公开完整的 C++ `<cmath>` 头文件函数集。
- 支持 IEEE-754 标准浮点类型 `__half`、`float`、`double`、`__float128`，以及 Bfloat16 `__nv_bfloat16`。`__float128` 的支持依赖于主机编译器和设备计算能力，请参阅支持的浮点类型表。
- 它们在主机和设备代码中均可用。
- 它们通常依赖于 CUDA 数学 API 函数。因此，主机和设备代码之间可能存在不同的精度级别。
- 它们的行为受 `nvcc` 优化标志影响。
- 根据 C++23 和 C++26 标准规范，其功能子集也支持常量表达式，例如 `constexpr` 函数。

[CUDA C 标准库数学函数](#mathematical-functions-appendix-cxx-standard-functions) ([CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/index.html))：

- 公开 C `<math.h>` 头文件函数的一个子集。
- 支持单精度和双精度类型，分别是 `float` 和 `double`。它们在主机和设备代码中均可用。它们不需要额外的头文件。它们的行为受 `nvcc` 优化标志影响。
- `<math.h>` 头文件功能的一个子集也可用于 `__half`、`__nv_bfloat16` 和 `__float128`/`_Float128` 类型。这些函数的名称类似于 C 标准库中的函数。`__half` 和 `__nv_bfloat16` 类型分别需要 `<cuda_fp16.h>` 和 `<cuda_bf16.h>` 头文件。它们在主机和设备代码中的可用性是按函数定义的。`__float128`/`_Float128` 类型的支持依赖于主机编译器和设备计算能力，请参阅支持的浮点类型表。相关函数需要 `crt/device_fp128_functions.h` 头文件，并且仅在设备代码中可用。
- 它们在主机和设备代码之间可能具有不同的精度。

[非标准 CUDA 数学函数](#mathematical-functions-appendix-additional-functions) ([CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/index.html))：

- 公开不属于 C/C++ 标准库的数学功能。
- 主要支持单精度和双精度类型，分别是 `float` 和 `double`。它们在主机和设备代码中的可用性是按函数定义的。它们不需要额外的头文件。它们在主机和设备代码之间可能具有不同的精度。
-   `__nv_bfloat16`、`__half`、`__float128`/`_Float128` 仅支持有限的一组函数。`__half` 和 `__nv_bfloat16` 类型分别需要包含 `<cuda_fp16.h>` 和 `<cuda_bf16.h>` 头文件。`__float128`/`_Float128` 类型的支持依赖于主机编译器和设备计算能力，请参阅"支持的浮点类型"表格。相关函数需要包含 `crt/device_fp128_functions.h` 头文件。它们仅在设备代码中可用。
-   它们的行为受 nvcc 优化标志影响。

[内建数学函数](#mathematical-functions-appendix-intrinsic-functions) ([CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/index.html))：

-   支持单精度和双精度类型，分别是 `float` 和 `double`。
-   它们仅在设备代码中可用。
-   它们比相应的 CUDA 数学 API 函数更快，但精度较低。
-   它们的行为不受 nvcc 浮点优化标志 `-prec-div=false`、`-prec-sqrt=false` 和 `-fmad=true` 的影响。唯一的例外是 `-ftz=true`，该标志也包含在 `-use_fast_math` 中。

| 功能 | 支持的类型 | 主机 | 设备 | 受浮点优化标志影响（仅针对 `float` 和 `double`） |
| --- | --- | --- | --- | --- |
| 内建 C/C++ 语言算术运算符 | `float`, `double`, `__half`, `__nv_bfloat16`, `__float128`/`_Float128`, `cuda::std::complex` | ✔ | ✔ | ✔ |
| CUDA C++ 标准库数学函数 | `float`, `double`, `__half`, `__nv_bfloat16`, `__float128`, `cuda::std::complex` | ✔ | ✔ | ✔ |
| `__nv_fp8_e4m3`, `__nv_fp8_e5m2`, `__nv_fp8_e8m0`, `__nv_fp6_e2m3`, `__nv_fp6_e3m2`, `__nv_fp4_e2m1` * | | | | |
| CUDA C 标准库数学函数 | `float`, `double` | ✔ | ✔ | ✔ |
| `__nv_bfloat16`, `__half`（有限支持且名称相似） | 基于逐个函数 | | | |
| `__float128`/`_Float128`（有限支持且名称相似） | ✘ | ✔ | | |
| 非标准 CUDA 数学函数 | `float`, `double` | 基于逐个函数 | ✔ | |
| `__nv_bfloat16`, `__half`, `__float128`/`_Float128`（有限支持） | ✘ | ✔ | | |
| 内建函数 | `float`, `double` | ✘ | ✔ | 仅受 `-ftz=true` 影响（该标志也包含在 `-use_fast_math` 中） |

***** [CUDA C++ 标准库函数](cpp-language-support.html#cpp-standard-library) 支持对小浮点类型进行查询，例如 [numeric_limits<T>](https://en.cppreference.com/w/cpp/types/numeric_limits.html)、[fpclassify()](https://en.cppreference.com/w/cpp/numeric/math/fpclassify)、[isfinite()](https://en.cppreference.com/w/cpp/numeric/math/isfinite.html)、[isnormal()](https://en.cppreference.com/w/cpp/numeric/math/isnormal.html)、[isinf()](https://en.cppreference.com/w/cpp/numeric/math/isinf.html) 和 [isnan()](https://en.cppreference.com/w/cpp/numeric/math/isnan.html)。

以下部分在适用时提供了其中一些函数的精度信息。它使用 ULP 进行量化。有关[最后一位的单位 (ULP)](https://en.wikipedia.org/wiki/Unit_in_the_last_place) 定义的更多信息，请参阅 Jean-Michel Muller 的论文 [On the definition of ulp(x)](https://inria.hal.science/inria-00070503v1/file/RR2005-09.pdf)。
---

## 5.5.6. 内置算术运算符

对于单精度、双精度和四精度类型，内置的 C/C++ 语言运算符，例如 `x + y`、`x - y`、`x * y`、`x / y`、`x++`、`x--` 以及倒数 `1 / x`，均符合 IEEE-754 标准。在使用*向偶数舍入*的舍入模式下，它们保证最大 ULP 误差为零。这些运算符在主机和设备代码中均可用。

`nvcc` 编译标志 `-fmad=true`（也包含在 `--use_fast_math` 中）允许将浮点乘法和加法/减法收缩为浮点乘加运算，并对单精度类型 `float` 的最大 ULP 误差产生以下影响：

- x * y + z → __fmaf_rn(x, y, z) : 0 ULP

`nvcc` 编译标志 `-prec-div=false`（也包含在 `--use_fast_math` 中）对单精度类型 `float` 的除法运算符 `/` 的最大 ULP 误差产生以下影响：

- x / y → __fdividef(x, y) : 2 ULP
- 1 / x : 1 ULP

---

## 5.5.7. CUDA C++ 数学标准库函数

CUDA 通过 `cuda::std::` 命名空间为 [C++ 标准库数学函数](https://en.cppreference.com/w/cpp/header/cmath.html) 提供全面支持。这些功能是 `<cuda/std/cmath>` 头文件的一部分。它们在主机和设备代码中均可用。

以下部分指定了与 [CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 的映射关系，以及每个函数在设备上执行时的误差界限。

- 最大 ULP 误差被表述为：函数返回值与根据向偶数舍入模式获得的相应精度的正确舍入结果之间，以 ULP 为单位的差值绝对值的最大观测值。
- 误差界限来源于广泛但非穷尽的测试。因此，它们不能得到保证。

### 5.5.7.1. 基本运算

[CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 中的基本运算函数在主机和设备代码中均可用，`__float128` 除外。

以下所有函数的最大 ULP 误差均为零。

| cuda::std 函数 | 含义 | __nv_bfloat16 | __half | float | double | __float128 |
| --- | --- | --- | --- | --- | --- | --- |
| fabs(x) | \(|x|\) | __habs(x) | __habs(x) | fabsf(x) | fabs(x) | __nv_fp128_fabs(x) |
| fmod(x, y) | \(\dfrac{x}{y}\) 的余数，计算为 \(x - \mathrm{trunc}\left(\dfrac{x}{y}\right) \cdot y\) | N/A | N/A | fmodf(x, y) | fmod(x, y) | __nv_fp128_fmod(x, y) |
| remainder(x, y) | \(\dfrac{x}{y}\) 的余数，计算为 \(x - \mathrm{rint}\left(\dfrac{x}{y}\right) \cdot y\) | N/A | N/A | remainderf(x, y) | remainder(x, y) | __nv_fp128_remainder(x, y) |
| remquo(x, y, iptr) | \(\dfrac{x}{y}\) 的余数和商 | N/A | N/A | remquof(x, y, iptr) | remquo(x, y, iptr) | N/A |
| fma(x, y, z) | \(x \cdot y + z\) | __hfma(x, y, z) , 仅设备 | __hfma(x, y, z) , 仅设备 | fmaf(x, y, z) | fma(x, y, z) | __nv_fp128_fma(x, y, z) |
| cuda::std 函数 | 含义 | __nv_bfloat16 | __half | float | double | __float128 |
| --- | --- | --- | --- | --- | --- | --- |
| fmax(x, y) | \(\max(x, y)\) | __hmax(x, y) | __hmax(x, y) | fmaxf(x, y) | fmax(x, y) | __nv_fp128_fmax(x, y) |
| fmin(x, y) | \(\min(x, y)\) | __hmin(x, y) | __hmin(x, y) | fminf(x, y) | fmin(x, y) | __nv_fp128_fmin(x, y) |
| fdim(x, y) | \(\max(x-y, 0)\) | N/A | N/A | fdimf(x, y) | fdim(x, y) | __nv_fp128_fdim(x, y) |
| nan(str) | 从字符串表示生成 NaN 值 | N/A | N/A | nanf(str) | nan(str) | N/A |

*****标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 __half 和 __nv_bfloat16）并非原生可用。在这些情况下，函数通过转换为 float 类型然后转换回结果来模拟。

### 5.5.7.2. 指数函数

[CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 中的指数函数在主机和设备代码中仅适用于 `float` 和 `double` 类型。

| cuda::std 函数 | 含义 | __nv_bfloat16 | __half | float | double | __float128 |
| --- | --- | --- | --- | --- | --- | --- |
| exp(x) | \(e^x\) | hexp(x) 0 ULP | hexp(x) 0 ULP | expf(x) 2 ULP | exp(x) 1 ULP | __nv_fp128_exp(x) 1 ULP |
| exp2(x) | \(2^x\) | hexp2(x) 0 ULP | hexp2(x) 0 ULP | exp2f(x) 2 ULP | exp2(x) 1 ULP | __nv_fp128_exp2(x) 1 ULP |
| expm1(x) | \(e^x - 1\) | N/A | N/A | expm1f(x) 1 ULP | expm1(x) 1 ULP | __nv_fp128_expm1(x) 1 ULP |
| log(x) | \(\ln(x)\) | hlog(x) 0 ULP | hlog(x) 0 ULP | logf(x) 1 ULP | log(x) 1 ULP | __nv_fp128_log(x) 1 ULP |
| log10(x) | \(\log_{10}(x)\) | hlog10(x) 0 ULP | hlog10(x) 0 ULP | log10f(x) 2 ULP | log10(x) 1 ULP | __nv_fp128_log10(x) 1 ULP |
| log2(x) | \(\log_2(x)\) | hlog2(x) 0 ULP | hlog2(x) 0 ULP | log2f(x) 1 ULP | log2(x) 1 ULP | __nv_fp128_log2(x) 1 ULP |
| log1p(x) | \(\ln(1+x)\) | N/A | N/A | log1pf(x) 1 ULP | log1p(x) 1 ULP | __nv_fp128_log1p(x) 1 ULP |

*****标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 __half 和 __nv_bfloat16）并非原生可用。在这些情况下，函数通过转换为 float 类型然后转换回结果来模拟。

### 5.5.7.3. 幂函数

[CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 中的幂函数在主机和设备代码中仅适用于 `float` 和 `double` 类型。

| cuda::std 函数 | 含义 | __nv_bfloat16 | __half | float | double | __float128 |
| --- | --- | --- | --- | --- | --- | --- |
| pow(x, y) | \(x^y\) | N/A | N/A | powf(x, y) 4 ULP | pow(x, y) 2 ULP | __nv_fp128_pow(x, y) 1 ULP |
| sqrt(x) | \(\sqrt{x}\) | hsqrt(x) 0 ULP | hsqrt(x) 0 ULP | sqrtf(x) • 0 ULP • 1 ULP with --use_fast_math | sqrt(x) 0 ULP | __nv_fp128_sqrt(x) 0 ULP |
| cbrt(x) | \(\sqrt[3]{x}\) | N/A | N/A | cbrtf(x) 1 ULP | cbrt(x) 1 ULP | N/A |
| hypot(x, y) | \(\sqrt{x^2 + y^2}\) | N/A | N/A | hypotf(x, y) 3 ULP | hypot(x, y) 2 ULP | __nv_fp128_hypot(x, y) 1 ULP |

*****标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 __half 和 __nv_bfloat16）并非原生可用。在这些情况下，函数通过转换为 float 类型然后转换回结果来模拟。
### 5.5.7.4. 三角函数

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 中的三角函数在主机和设备代码中均可用，但仅适用于 `float` 和 `double` 类型。

| cuda::std 函数 | 含义 | __nv_bfloat16 | __half | float | double | __float128 |
| --- | --- | --- | --- | --- | --- | --- |
| sin(x) | \(\sin(x)\) | hsin(x) 0 ULP | hsin(x) 0 ULP | sinf(x) 2 ULP | sin(x) 2 ULP | __nv_fp128_sin(x) 1 ULP |
| cos(x) | \(\cos(x)\) | hcos(x) 0 ULP | hcos(x) 0 ULP | cosf(x) 2 ULP | cos(x) 2 ULP | __nv_fp128_cos(x) 1  ULP |
| tan(x) | \(\tan(x)\) | N/A | N/A | tanf(x) 4 ULP | tan(x) 2 ULP | __nv_fp128_tan(x) 1 ULP |
| asin(x) | \(\sin^{-1}(x)\) | N/A | N/A | asinf(x) 2 ULP | asin(x) 2 ULP | __nv_fp128_asin(x) 1 ULP |
| acos(x) | \(\cos^{-1}(x)\) | N/A | N/A | acosf(x) 2 ULP | acos(x) 2 ULP | __nv_fp128_acos(x) 1 ULP |
| atan(x) | \(\tan^{-1}(x)\) | N/A | N/A | atanf(x) 2 ULP | atan(x) 2 ULP | __nv_fp128_atan(x) 1 ULP |
| atan2(y, x) | \(\tan^{-1}\left(\dfrac{y}{x}\right)\) | N/A | N/A | atan2f(y, x) 3 ULP | atan2(y, x) 2 ULP | N/A |

*****标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 __half 和 __nv_bfloat16）并非原生可用。在这些情况下，函数通过转换为 float 类型，然后再将结果转换回来进行模拟。

### 5.5.7.5. 双曲函数

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 中的双曲函数在主机和设备代码中均可用，但仅适用于 `float` 和 `double` 类型。

| cuda::std 函数 | 含义 | __nv_bfloat16 | __half | float | double | __float128 |
| --- | --- | --- | --- | --- | --- | --- |
| sinh(x) | \(\sinh(x)\) | N/A | N/A | sinhf(x) 3 ULP | sinh(x) 2 ULP | __nv_fp128_sinh(x) 1 ULP |
| cosh(x) | \(\cosh(x)\) | N/A | N/A | coshf(x) 2 ULP | cosh(x) 1 ULP | __nv_fp128_cosh(x) 1 ULP |
| tanh(x) | \(\tanh(x)\) | htanh(x) 0 ULP | htanh(x) 0 ULP | tanhf(x) 2 ULP | tanh(x) 1 ULP | __nv_fp128_tanh(x) 1 ULP |
| asinh(x) | \(\operatorname{sinh}^{-1}(x)\) | N/A | N/A | asinhf(x) 3 ULP | asinh(x) 3 ULP | __nv_fp128_asinh(x) 1 ULP |
| acosh(x) | \(\operatorname{cosh}^{-1}(x)\) | N/A | N/A | acoshf(x) 4 ULP | acosh(x) 3 ULP | __nv_fp128_acosh(x) 1 ULP |
| atanh(x) | \(\operatorname{tanh}^{-1}(x)\) | N/A | N/A | atanhf(x) 3 ULP | atanh(x) 2 ULP | __nv_fp128_atanh(x) 1 ULP |

*****标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（例如 __half 和 __nv_bfloat16）并非原生可用。在这些情况下，函数通过转换为 float 类型，然后再将结果转换回来进行模拟。

### 5.5.7.6. 误差函数和伽玛函数

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 中的误差函数和伽玛函数在主机和设备代码中均可用，适用于 `float` 和 `double` 类型。

误差函数和伽玛函数对于 CUDA 扩展浮点类型（例如 `__half` 和 `__nv_bfloat16`）并非原生可用。在这些情况下，函数通过转换为 `float` 类型，然后再将结果转换回来进行模拟。
| cuda::std 函数 | 含义 | float | double |
| --- | --- | --- | --- |
| erf(x) | \(\dfrac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt\) | erff(x) 2 ULP | erf(x) 2 ULP |
| erfc(x) | \(1 - \mathrm{erf}(x)\) | erfcf(x) 4 ULP | erfc(x) 5 ULP |
| tgamma(x) | \(\Gamma(x)\) | tgammaf(x) 5 ULP | tgamma(x) 10 ULP |
| lgamma(x) | \(\ln |\Gamma(x)|\) | lgammaf(x) âª 当 \(x \notin [-10.001, -2.264]\) 时为 6 ULP âª 其他情况更大 | lgamma(x) âª 当 \(x \notin [-23.0001, -2.2637]\) 时为 4 ULP âª 其他情况更大 |

### 5.5.7.7. 最近整数浮点运算

[CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 中用于最近整数浮点运算的函数在主机和设备代码中均可用，但仅适用于 `float` 和 `double` 类型。

以下所有函数的最大 ULP 误差为零。

| cuda::std 函数 | 含义 | __nv_bfloat16 | __half | float | double | __float128 |
| --- | --- | --- | --- | --- | --- | --- |
| ceil(x) | \(\lceil x \rceil\) | hceil(x) | hceil(x) | ceilf(x) | ceil(x) | __nv_fp128_ceil(x) |
| floor(x) | \(\lfloor x \rfloor\) | hfloor(x) | hfloor(x) | floorf(x) | floor(x) | __nv_fp128_floor(x) |
| trunc(x) | 截断为整数 | htrunc(x) | htrunc(x) | truncf(x) | trunc(x) | __nv_fp128_trunc(x) |
| round(x) | 舍入到最近的整数，中间值远离零 | N/A | N/A | roundf(x) | round(x) | __nv_fp128_round(x) |
| nearbyint(x) | 舍入到最近的整数，中间值取偶 | N/A | N/A | nearbyintf(x) | nearbyint(x) | N/A |
| rint(x) | 舍入到最近的整数，中间值取偶 | hrint(x) | hrint(x) | rintf(x) | rint(x) | __nv_fp128_rint(x) |
| lrint(x) | 舍入到最近的整数，中间值取偶（返回 long int） | N/A | N/A | lrintf(x) | lrint(x) | N/A |
| llrint(x) | 舍入到最近的整数，中间值取偶（返回 long long int） | N/A | N/A | llrintf(x) | llrint(x) | N/A |
| lround(x) | 舍入到最近的整数，中间值远离零（返回 long int） | N/A | N/A | lroundf(x) | lround(x) | N/A |
| llround(x) | 舍入到最近的整数，中间值远离零（返回 long long int） | N/A | N/A | llroundf(x) | llround(x) | N/A |

*****标记为“N/A”的数学函数对于 CUDA 扩展浮点类型（如 __half 和 __nv_bfloat16）并非原生可用。在这些情况下，函数通过转换为 float 类型然后转换回结果来模拟。

**性能考虑**

将单精度或双精度浮点操作数舍入为整数的推荐方法是使用函数 `rintf()` 和 `rint()`，而不是 `roundf()` 和 `round()`。这是因为 `roundf()` 和 `round()` 在设备代码中映射到多条指令，而 `rintf()` 和 `rint()` 映射到单条指令。`truncf()`、`trunc()`、`ceilf()`、`ceil()`、`floorf()` 和 `floor()` 也各自映射到单条指令。

### 5.5.7.8. 浮点操作函数

[CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 中用于浮点操作的函数在主机和设备代码中均可用，但 `__float128` 除外。
浮点操作函数并非原生支持 CUDA 扩展浮点类型，例如 `__half` 和 `__nv_bfloat16`。在这些情况下，函数通过转换为 `float` 类型，然后将结果转换回原类型来模拟实现。

以下所有函数具有零的最大 ULP 误差。

| cuda::std 函数 | 含义 | float | double | __float128 |
| --- | --- | --- | --- | --- |
| frexp(x, exp) | 提取尾数和指数 | frexpf(x, exp) | frexp(x, exp) | __nv_fp128_frexp(x, nptr) |
| ldexp(x, n) | \(x \cdot 2^{\mathrm{n}}\) | ldexpf(x, n) | ldexp(x, n) | __nv_fp128_ldexp(x, n) |
| modf(x, iptr) | 提取整数和小数部分 | modff(x, iptr) | modf(x, iptr) | __nv_fp128_modf(x, iptr) |
| scalbn(x, n) | \(x \cdot 2^n\) | scalbnf(x, n) | scalbn(x, n) | N/A |
| scalbln(x, n) | \(x \cdot 2^n\) | scalblnf(x, n) | scalbln(x, n) | N/A |
| ilogb(x) | \(\lfloor \log_2(|x|) \rfloor\) | ilogbf(x) | ilogb(x) | __nv_fp128_ilogb(x) |
| logb(x) | \(\lfloor \log_2(|x|) \rfloor\) | logbf(x) | logb(x) | N/A |
| nextafter(x, y) | 朝向 \(y\) 的下一个可表示值 | nextafterf(x, y) | nextafter(x, y) | N/A |
| copysign(x, y) | 将 \(y\) 的符号复制到 \(x\) | copysignf(x, y) | copysign(x, y) | __nv_fp128_copysign(x, y) |

### 5.5.7.9. 分类与比较

[CUDA 数学 API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 中的分类和比较函数在主机和设备代码中均可用，但 `__float128` 除外。

以下所有函数具有零的最大 ULP 误差。

| cuda::std 函数 | 含义 | __nv_bfloat16 | __half | float | double | __float128 |
| --- | --- | --- | --- | --- | --- | --- |
| fpclassify(x) | 对 \(x\) 进行分类 | N/A | N/A | N/A | N/A | N/A |
| isfinite(x) | 检查 \(x\) 是否为有限值 | N/A | N/A | isfinite(x) | isfinite(x) | N/A |
| isinf(x) | 检查 \(x\) 是否为无穷大 | __hisinf(x) | __hisinf(x) | isinf(x) | isinf(x) | N/A |
| isnan(x) | 检查 \(x\) 是否为 NaN | __hisnan(x) | __hisnan(x) | isnan(x) | isnan(x) | __nv_fp128_isnan(x) |
| isnormal(x) | 检查 \(x\) 是否为规格化数 | N/A | N/A | N/A | N/A | N/A |
| signbit(x) | 检查符号位是否被设置 | N/A | N/A | signbit(x) | signbit(x) | N/A |
| isgreater(x, y) | 检查 \(x > y\) | __hgt(x, y) | __hgt(x, y) | N/A | N/A | N/A |
| isgreaterequal(x, y) | 检查 \(x \geq y\) | __hge(x, y) | __hge(x, y) | N/A | N/A | N/A |
| isless(x, y) | 检查 \(x < y\) | __hlt(x, y) | __hlt(x, y) | N/A | N/A | N/A |
| islessequal(x, y) | 检查 \(x \leq y\) | __hle(x, y) | __hle(x, y) | N/A | N/A | N/A |
| islessgreater(x, y) | 检查 \(x < y\) 或 \(x > y\) | __hne(x, y) | __hne(x, y) | N/A | N/A | N/A |
| isunordered(x, y) | 检查 \(x\)、\(y\) 或两者是否为 NaN | N/A | N/A | N/A | N/A | __nv_fp128_isunordered(x, y) |

*****标记为“N/A”的数学函数并非原生支持 CUDA 扩展浮点类型，例如 __half 和 __nv_bfloat16。

## 5.5.8. 非标准 CUDA 数学函数
CUDA 提供了一些不属于 C/C++ 标准库的数学函数，它们作为扩展提供。对于单精度和双精度函数，主机和设备代码的可用性是按每个函数单独定义的。

本节规定了每个函数在设备上执行时的误差界限。

*   最大 ULP 误差被定义为：根据“向最近偶数舍入”模式得到的相应精度的正确舍入结果，与函数返回值之间的差值（以 ULP 计）的观测绝对值的最大值。
*   误差界限来源于广泛但非穷尽的测试。因此，不能保证其绝对性。

| 含义 | float | double |
| --- | --- | --- |
| \(\dfrac{x}{y}\) | fdividef(x, y) , 仅设备 0 ULP，与 x / y 相同 | N/A |
| \(10^x\) | exp10f(x) 2 ULP | exp10(x) 1 ULP |
| \(\sqrt{x^2 + y^2 + z^2}\) | norm3df(x, y, z) , 仅设备 3 ULP | norm3d(x, y, z) , 仅设备 2 ULP |
| \(\sqrt{x^2 + y^2 + z^2 + t^2}\) | norm4df(x, y, z, t) , 仅设备 3 ULP | norm4d(x, y, z, t) , 仅设备 2 ULP |
| \(\sqrt{\sum_{i=0}^{\mathrm{dim}-1} p_i^{2}}\) | normf(dim, p) , 仅设备 由于使用了快速算法，存在因舍入导致的精度损失，无法提供误差界限 | norm(dim, p) , 仅设备 由于使用了快速算法，存在因舍入导致的精度损失，无法提供误差界限 |
| \(\dfrac{1}{\sqrt{x}}\) | rsqrtf(x) 2 ULP | rsqrt(x) 1 ULP |
| \(\dfrac{1}{\sqrt[3]{x}}\) | rcbrtf(x) 1 ULP | rcbrt(x) 1 ULP |
| \(\dfrac{1}{\sqrt{x^2 + y^2}}\) | rhypotf(x, y) , 仅设备 2 ULP | rhypot(x, y) , 仅设备 1 ULP |
| \(\dfrac{1}{\sqrt{x^2 + y^2 + z^2}}\) | rnorm3df(x, y, z) , 仅设备 2 ULP | rnorm3d(x, y, z) , 仅设备 1 ULP |
| \(\dfrac{1}{\sqrt{x^2 + y^2 + z^2 + t^2}}\) | rnorm4df(x, y, z, t) , 仅设备 2 ULP | rnorm4d(x, y, z, t) , 仅设备 1 ULP |
| \(\dfrac{1}{\sqrt{\sum_{i=0}^{\mathrm{dim}-1} p_i^{2}}}\) | rnormf(dim, p) , 仅设备 由于使用了快速算法，存在因舍入导致的精度损失，无法提供误差界限 | rnorm(dim, p) , 仅设备 由于使用了快速算法，存在因舍入导致的精度损失，无法提供误差界限 |
| \(\cos(\pi x)\) | cospif(x) 1 ULP | cospi(x) 2 ULP |
| \(\sin(\pi x)\) | sinpif(x) 1 ULP | sinpi(x) 2 ULP |
| \(\sin(\pi x), \cos(\pi x)\) | sincospif(x, sptr, cptr) 1 ULP | sincospi(x, sptr, cptr) 2 ULP |
| \(\Phi(x)\) | normcdff(x) 5 ULP | normcdf(x) 5 ULP |
| \(\Phi^{-1}(x)\) | normcdfinvf(x) 5 ULP | normcdfinv(x) 8 ULP |
| \(\mathrm{erfc}^{-1}(x)\) | erfcinvf(x) 4 ULP | erfcinv(x) 6 ULP |
| \(e^{x^2}\mathrm{erfc}(x)\) | erfcxf(x) 4 ULP | erfcx(x) 4 ULP |
| \(\mathrm{erf}^{-1}(x)\) | erfinvf(x) 2 ULP | erfinv(x) 5 ULP |
| \(I_0(x)\) | cyl_bessel_i0f(x) , 仅设备 6 ULP | cyl_bessel_i0(x) , 仅设备 6 ULP |
| \(I_1(x)\) | cyl_bessel_i1f(x) , 仅设备 6 ULP | cyl_bessel_i1(x) , 仅设备 6 ULP |
| \(J_0(x)\) | j0f(x) âª 对于 \(|x| < 8\) 为 9 ULP âª 否则最大绝对误差 \(= 2.2 \cdot 10^{-6}\) | j0(x) âª 对于 \(|x| < 8\) 为 7 ULP âª 否则最大绝对误差 \(= 5 \cdot 10^{-12}\) |
| \(J_1(x)\) | j1f(x) âª 当 \(|x| < 8\) 时，9 ULP âª 最大绝对误差 \(= 2.2 \cdot 10^{-6}\)，否则 | j1(x) âª 当 \(|x| < 8\) 时，7 ULP âª 最大绝对误差 \(= 5 \cdot 10^{-12}\)，否则 |
| \(J_n(x)\) | jnf(n, x) 对于 \(n = 128\)，最大绝对误差 \(= 2.2 \cdot 10^{-6}\) | jn(n, x) 对于 \(n = 128\)，最大绝对误差 \(= 5 \cdot 10^{-12}\) |
| \(Y_0(x)\) | y0f(x) âª 当 \(|x| < 8\) 时，9 ULP âª 最大绝对误差 \(= 2.2 \cdot 10^{-6}\)，否则 | y0(x) âª 当 \(|x| < 8\) 时，7 ULP âª 最大绝对误差 \(= 5 \cdot 10^{-12}\)，否则 |
| \(Y_1(x)\) | y1f(x) âª 当 \(|x| < 8\) 时，9 ULP âª 最大绝对误差 \(= 2.2 \cdot 10^{-6}\)，否则 | y1(x) âª 当 \(|x| < 8\) 时，7 ULP âª 最大绝对误差 \(= 5 \cdot 10^{-12}\)，否则 |
| \(Y_n(x)\) | ynf(n, x) âª 当 \(|x| < n\) 时，\(\lceil 2 + 2.5n \rceil\) âª 最大绝对误差 \(= 2.2 \cdot 10^{-6}\)，否则 | yn(n, x) 当 \(|x| > 1.5n\) 时，最大绝对误差 \(= 5 \cdot 10^{-12}\) |

适用于 `__half`、`__nv_bfloat16` 和 `__float128/_Float128` 的非标准 CUDA 数学函数仅在设备代码中可用。

| 含义 | __nv_bfloat16 | __half | __float128/_Float128 |
| --- | --- | --- | --- |
| \(\dfrac{1}{x}\) | hrcp(x) 0 ULP | hrcp(x) 0 ULP | N/A |
| \(10^x\) | hexp10(x) 0 ULP | hexp10(x) 0 ULP | __nv_fp128_exp10(x) 1 ULP |
| \(\dfrac{1}{\sqrt{x}}\) | hrsqrt(x) 0 ULP | hrsqrt(x) 0 ULP | N/A |
| \(\tanh(x)\) (近似) | htanh_approx(x) 1 ULP | htanh_approx(x) 1 ULP | N/A |

## 5.5.9. 内置函数

内置数学函数是其对应的 [CUDA C 标准库数学函数](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html) 的更快但精度较低的版本。

-   它们具有相同的名称，但前缀为 `__`，例如 `__sinf(x)`。
-   它们仅在设备代码中可用。
-   它们更快，因为它们映射到更少的原生指令。
-   标志 `--use_fast_math` 会自动将相应的 CUDA 数学 API 函数转换为内置函数。有关受影响函数的完整列表，请参阅 `–use_fast_math` 效果部分。

### 5.5.9.1. 基本内置函数

一部分数学内置函数允许指定舍入模式：

-   后缀为 `_rn` 的函数使用“舍入到最接近的偶数”舍入模式。
-   后缀为 `_rz` 的函数使用“向零舍入”舍入模式。
-   后缀为 `_ru` 的函数使用“向上舍入”（朝向正无穷大）舍入模式。
-   后缀为 `_rd` 的函数使用“向下舍入”（朝向负无穷大）舍入模式。

`__fadd_[rn,rz,ru,rd]()`、`__dadd_[rn,rz,ru,rd]()`、`__fmul_[rn,rz,ru,rd]()` 和 `__dmul_[rn,rz,ru,rd]()` 函数映射到加法和乘法运算，编译器永远不会将这些运算合并到 `FFMA` 或 `DFMA` 指令中。相比之下，由 `*` 和 `+` 运算符生成的加法和乘法运算经常被合并到 `FFMA` 或 `DFMA` 中。
下表列出了单精度和双精度浮点内置函数。所有这些函数的最大 ULP 误差均为 0，且符合 IEEE 标准。

| 含义 | float | double |
| --- | --- | --- |
| \(x + y\) | __fadd_[rn,rz,ru,rd](x, y) | __dadd_[rn,rz,ru,rd](x, y) |
| \(x - y\) | __fsub_[rn,rz,ru,rd](x, y) | __dsub_[rn,rz,ru,rd](x, y) |
| \(x \cdot y\) | __fmul_[rn,rz,ru,rd](x, y) | __dmul_[rn,rz,ru,rd](x, y) |
| \(x \cdot y + z\) | __fmaf_[rn,rz,ru,rd](x, y, z) | __fma_[rn,rz,ru,rd](x, y, z) |
| \(\dfrac{x}{y}\) | __fdiv_[rn,rz,ru,rd](x, y) | __ddiv_[rn,rz,ru,rd](x, y) |
| \(\dfrac{1}{x}\) | __frcp_[rn,rz,ru,rd](x) | __drcp_[rn,rz,ru,rd](x) |
| \(\sqrt{x}\) | __fsqrt_[rn,rz,ru,rd](x) | __dsqrt_[rn,rz,ru,rd](x) |

### 5.5.9.2. 仅限单精度的内置函数

下表列出了单精度浮点内置函数及其最大 ULP 误差。

*   最大 ULP 误差定义为：函数返回值与根据“就近舍入，偶数优先”舍入模式得到的对应精度的正确舍入结果之间，以 ULP 为单位的差值绝对值的最大观测值。
*   误差范围源自广泛但非穷尽的测试。因此，不能保证绝对准确。

| 函数 | 含义 | 最大 ULP 误差 |
| --- | --- | --- |
| __fdividef(x, y) | \(\dfrac{x}{y}\) | 对于 \(|y| \in [2^{-126}, 2^{126}]\) 为 \(2\) |
| __frsqrt_rn(x) | \(\dfrac{1}{\sqrt{x}}\) | 0 ULP |
| __expf(x) | \(e^x\) | \(2 + \lfloor |1.173 \cdot x| \rfloor\) |
| __exp10f(x) | \(10^x\) | \(2 + \lfloor |2.97 \cdot x| \rfloor\) |
| __powf(x, y) | \(x^y\) | 源自 exp2f(y * __log2f(x)) |
| __logf(x) | \(\ln(x)\) | âª 对于 \(x \in [0.5, 2]\)，绝对误差为 \(2^{-21.41}\) âª 其他情况下为 3 ULP |
| __log2f(x) | \(\log_2(x)\) | âª 对于 \(x \in [0.5, 2]\)，绝对误差为 \(2^{-22}\) âª 其他情况下为 2 ULP |
| __log10f(x) | \(\log_{10}(x)\) | âª 对于 \(x \in [0.5, 2]\)，绝对误差为 \(2^{-24}\) âª 其他情况下为 3 ULP |
| __sinf(x) | \(\sin(x)\) | âª 对于 \(x \in [-\pi, \pi]\)，绝对误差为 \(2^{-21.41}\) âª 其他情况下更大 |
| __cosf(x) | \(\cos(x)\) | âª 对于 \(x \in [-\pi, \pi]\)，绝对误差为 \(2^{-21.41}\) âª 其他情况下更大 |
| __sincosf(x, sptr, cptr) | \(\sin(x), \cos(x)\) | 分量误差与 __sinf(x) 和 __cosf(x) 相同 |
| __tanf(x) | \(\tan(x)\) | 源自 __sinf(x) * (1 / __cosf(x)) |
| __tanhf(x) | \(\tanh(x)\) | âª 最大相对误差：\(2^{-11}\) âª 即使在 `-ftz=true` 编译器标志下，次正规结果也不会被刷新为零。 |

### 5.5.9.3. --use_fast_math 标志的影响

`nvcc` 编译器标志 `--use_fast_math` 会将设备代码中调用的 [CUDA 数学 API 函数](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html) 的一个子集转换为其对应的内置函数。请注意，[CUDA C++ 标准库函数](#mathematical-functions-appendix-cxx-standard-functions) 也会受此标志影响。关于使用内置函数替代 CUDA 数学 API 函数的影响，更多细节请参见 [内置函数](#mathematical-functions-appendix-intrinsic-functions) 章节。
> 一种更稳健的方法是，仅在性能提升值得且能够接受精度降低和特殊情形处理方式改变等属性变化时，才有选择地将数学函数调用替换为内联版本。

| 设备函数 | 内联函数 |
| --- | --- |
| x/y, fdividef(x, y) | __fdividef(x, y) |
| sinf(x) | __sinf(x) |
| cosf(x) | __cosf(x) |
| tanf(x) | __tanf(x) |
| sincosf(x, sptr, cptr) | __sincosf(x, sptr, cptr) |
| logf(x) | __logf(x) |
| log2f(x) | __log2f(x) |
| log10f(x) | __log10f(x) |
| expf(x) | __expf(x) |
| exp10f(x) | __exp10f(x) |
| powf(x,y) | __powf(x,y) |
| tanhf(x) | __tanhf(x) |

## 5.5.10. 参考文献

1. IEEE 754-2019 浮点运算标准。
2. Jean-Michel Muller. On the definition of ulp(x) . INRIA/LIP 研究报告，2005。
3. Nathan Whitehead, Alex Fit-Florea. Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs . Nvidia 报告，2011。
4. David Goldberg. What every computer scientist should know about floating-point arithmetic . ACM Computing Surveys，1991年3月。
5. David Monniaux. The pitfalls of verifying floating-point computations . ACM Transactions on Programming Languages and Systems，2008年5月。
6. Peter Dinda, Conor Hetland. Do Developers Understand IEEE Floating Point? . IEEE International Parallel and Distributed Processing Symposium (IPDPS)，2018。

 本页