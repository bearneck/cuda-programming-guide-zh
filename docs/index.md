# CUDA 编程指南（中文版）

<div class="hero" markdown>

## 🚀 NVIDIA CUDA Programming Guide

**官方文档中文翻译版** | 基于 CUDA 最新版本

[原文地址](https://docs.nvidia.com/cuda/cuda-programming-guide/){ .md-button }
[PDF 版本](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf){ .md-button .md-button--primary }

</div>

## 📖 关于本文档

本文档是 **NVIDIA CUDA Programming Guide** 的官方中文翻译版本，旨在帮助中文开发者更好地学习和使用 CUDA 并行计算框架。

- ✅ **专业翻译**：专业术语准确，语句通顺自然
- ✅ **格式保留**：完整保留原文代码块、链接、表格结构
- ✅ **持续更新**：跟进官方文档最新版本
- ✅ **完整覆盖**：覆盖全部 5 个部分、40+ 章节

---

## 📚 文档结构

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### 第一部分：CUDA 简介

CUDA 的基本概念与编程模型，了解 GPU 并行计算的核心思想。

- 1.1 [简介](chapters/01-introduction.md)
- 1.2 [编程模型](chapters/02-programming-model.md)
- 1.3 [CUDA 平台](chapters/03-cuda-platform.md)

</div>

<div class="feature-card" markdown>

### 第二部分：CUDA GPU 编程

深入学习 CUDA C++ 编程，掌握内核编写、内存管理与编译工具。

- 2.1 [CUDA C++ 入门](chapters/04-cuda-cpp-intro.md)
- 2.2 [编写 CUDA SIMT 内核](chapters/05-simt-kernels.md)
- 2.3 [异步执行](chapters/06-async-execution.md)
- 2.4 [统一内存与系统内存](chapters/07-unified-memory.md)
- 2.5 [NVCC 编译器](chapters/08-nvcc.md)

</div>

<div class="feature-card" markdown>

### 第三部分：高级 CUDA

探索高级 API、多 GPU 编程、驱动层接口等进阶主题。

- 3.1 [高级 CUDA API 与特性](chapters/09-advanced-apis.md)
- 3.2 [高级内核编程](chapters/10-advanced-kernel.md)
- 3.3 [CUDA 驱动 API](chapters/11-driver-api.md)
- 3.4 [多 GPU 系统编程](chapters/12-multi-gpu.md)
- 3.5 [CUDA 特性全览](chapters/13-feature-survey.md)

</div>

<div class="feature-card" markdown>

### 第四部分：CUDA 高级特性

CUDA Graphs、协作组、动态并行等 20 个专题深度解析。

- 4.1 [统一内存](chapters/14-unified-memory.md)
- 4.2 [CUDA Graphs](chapters/15-cuda-graphs.md)
- 4.4 [协作组](chapters/17-cooperative-groups.md)
- 4.18 [CUDA 动态并行](chapters/31-dynamic-parallelism.md)
- [查看全部 →](chapters/part4-special.md)

</div>

<div class="feature-card" markdown>

### 技术附录

计算能力规格、环境变量参考、语言扩展完整说明。

- 5.1 [计算能力](chapters/34-compute-capabilities.md)
- 5.2 [CUDA 环境变量](chapters/35-env-variables.md)
- 5.4 [C/C++ 语言扩展](chapters/37-language-extensions.md)
- 5.7 [CUDA C++ 内存模型](chapters/40-memory-model.md)
- [查看全部 →](chapters/part5-appendices.md)

</div>

</div>

---

## 🔑 核心概念速查

| 概念 | 英文 | 说明 |
|------|------|------|
| 内核 | Kernel | 在 GPU 上并行执行的函数 |
| 线程 | Thread | CUDA 并行执行的最小单元 |
| 线程块 | Block | 线程的组织单位，可协作执行 |
| 线程网格 | Grid | 线程块的集合 |
| 线程束 | Warp | 32 个线程的执行单位 |
| 共享内存 | Shared Memory | 同一线程块内线程共享的高速内存 |
| 全局内存 | Global Memory | GPU 上所有线程可访问的主内存 |
| 流 | Stream | 按顺序执行的 CUDA 操作序列 |
| 计算能力 | Compute Capability | GPU 架构版本与特性集合 |
| 占用率 | Occupancy | SM 上活跃线程束与最大数量之比 |

---

## 📝 翻译说明

本文档使用 AI 辅助翻译，并对专业术语进行了人工审校。翻译原则：

1. **专业术语优先**：遵循业界通用中文译法
2. **代码保留原文**：所有代码块保持英文原样
3. **语句通顺自然**：在准确的前提下确保中文可读性

如发现翻译问题，欢迎在 [GitHub Issues](https://github.com/bearneck/cuda-programming-guide-zh/issues) 提出。

---

*本翻译仅供学习参考，一切以 [NVIDIA 官方文档](https://docs.nvidia.com/cuda/cuda-programming-guide/) 为准。*
