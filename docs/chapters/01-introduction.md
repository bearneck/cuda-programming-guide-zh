# 1.1 简介

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)

---

本页面是否有帮助？

# 1.1. 简介

## 1.1.1. 图形处理器

图形处理器（GPU）最初是作为3D图形的专用处理器诞生的，它始于用于加速实时3D渲染中并行操作的固定功能硬件。经过几代发展，GPU变得更具可编程性。到2003年，图形管线的某些阶段已完全可编程，能够为3D场景或图像的每个组成部分并行运行自定义代码。

2006年，NVIDIA推出了统一计算设备架构（CUDA），使得任何计算工作负载都能利用GPU的吞吐能力，而无需依赖图形API。

自那时起，CUDA和GPU计算已被用于加速几乎所有类型的计算工作负载，从流体动力学或能量传输等科学模拟，到数据库和分析等商业应用。此外，GPU的能力和可编程性已成为从图像分类到扩散模型或大语言模型等生成式人工智能的新算法和技术进步的基础。

## 1.1.2. 使用GPU的优势

在相似的价格和功耗范围内，GPU能提供比CPU高得多的指令吞吐量和内存带宽。许多应用程序利用这些能力，在GPU上的运行速度显著快于在CPU上（参见[GPU应用程序](https://www.nvidia.com/en-us/accelerated-applications/)）。其他计算设备，如FPGA，虽然能效也很高，但提供的编程灵活性远不如GPU。

GPU和CPU的设计目标不同。CPU旨在尽可能快地执行一系列串行操作（称为线程），并能并行执行几十个这样的线程；而GPU则擅长并行执行数千个线程，通过牺牲较低的单线程性能来实现更高的总吞吐量。

GPU专为高度并行计算而设计，将更多晶体管用于数据处理单元，而CPU则将更多晶体管用于数据缓存和流控制。[图1](#from-graphics-processing-to-general-purpose-parallel-computing-gpu-devotes-more-transistors-to-data-processing)展示了CPU与GPU芯片资源分配的示例。

![GPU将更多晶体管用于数据处理](../images/gpu-devotes-more-transistors-to-data-processing.png)

*图1 GPU将更多晶体管用于数据处理*

## 1.1.3. 快速入门

利用GPU提供的计算能力有多种方式。本指南涵盖了使用C++等高级语言为CUDA GPU平台进行编程。然而，许多应用程序无需直接编写GPU代码即可利用GPU。

通过专门的库，可以获得来自各个领域且不断增长的算法和例程集合。当某个库（尤其是NVIDIA提供的库）已经实现时，使用它通常比从头重新实现算法更具生产力和性能。像cuBLAS、cuFFT、cuDNN和CUTLASS这样的库，只是帮助开发者避免重新实现成熟算法的众多例子中的一部分。这些库还有一个额外的好处，即针对每个GPU架构进行了优化，在生产力、性能和可移植性之间提供了理想的平衡。
此外，还有一些框架（特别是用于人工智能的框架）提供了 GPU 加速的构建模块。其中许多框架通过利用上述 GPU 加速库来实现加速。

另外，特定领域语言（DSL），例如 NVIDIA 的 Warp 或 OpenAI 的 Triton，可编译为直接在 CUDA 平台上运行。这提供了一种比本指南涵盖的高级语言更高级的 GPU 编程方法。

[NVIDIA 加速计算中心](https://github.com/NVIDIA/accelerated-computing-hub) 包含教授 GPU 和 CUDA 计算的资源、示例和教程。

本页内容