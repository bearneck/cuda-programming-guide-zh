# 4.8 错误日志管理

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html)

---

此页面是否有帮助？

# 4.8. 错误日志管理

*错误日志管理*机制允许以通俗易懂的英文格式向开发者报告 CUDA API 错误，描述问题的原因。

## 4.8.1. 背景

传统上，CUDA API 调用失败的唯一指示是返回非零代码。截至 CUDA Toolkit 12.9，CUDA 运行时为错误情况定义了超过 100 种不同的返回代码，但其中许多是通用的，无法为开发者调试问题原因提供帮助。

## 4.8.2. 激活

设置 *CUDA_LOG_FILE* 环境变量。可接受的值为 *stdout*、*stderr* 或系统上用于写入文件的有效路径。即使在程序执行前未设置 *CUDA_LOG_FILE*，也可以通过 API 转储日志缓冲区。注意：无错误的执行可能不会打印任何日志。

## 4.8.3. 输出

日志以下列格式输出：

```c++
[时间][TID][来源][严重性][API 入口点] 消息
```

如果开发者尝试将错误日志管理日志转储到未分配的缓冲区，将生成以下实际错误消息：

```c++
[22:21:32.099][25642][CUDA][E][cuLogsDumpToMemory] buffer cannot be NULL
```

在此之前，开发者只能从返回代码中得到 *CUDA_ERROR_INVALID_VALUE*，如果调用 *cuGetErrorString*，可能还会得到“invalid argument”。

## 4.8.4. API 描述

CUDA 驱动程序提供了两类 API 用于与错误日志管理功能交互。

此功能允许开发者在生成错误日志时注册要使用的回调函数，回调函数签名为：

```c++
void callbackFunc(void *data, CUlogLevel logLevel, char *message, size_t length)
```

使用此 API 注册回调：

```c++
CUresult cuLogsRegisterCallback(CUlogsCallback callbackFunc, void *userData, CUlogsCallbackHandle *callback_out)
```

其中 *userData* 会原封不动地传递给回调函数。调用方应存储 *callback_out*，以便在 *cuLogsUnregisterCallback* 中使用。

```c++
CUresult cuLogsUnregisterCallback(CUlogsCallbackHandle callback)
```

另一组 API 函数用于管理日志输出。一个重要的概念是日志迭代器，它指向缓冲区的当前末尾：

```c++
CUresult cuLogsCurrent(CUlogIterator *iterator_out, unsigned int flags)
```

在不需要转储整个日志缓冲区的情况下，调用软件可以保留迭代器位置。目前，flags 参数必须为 0，其他选项保留给未来的 CUDA 版本使用。

在任何时候，都可以使用以下函数将错误日志缓冲区转储到文件或内存：

```c++
CUresult cuLogsDumpToFile(CUlogIterator *iterator, const char *pathToFile, unsigned int flags)
CUresult cuLogsDumpToMemory(CUlogIterator *iterator, char *buffer, size_t *size, unsigned int flags)
```

如果 *iterator* 为 NULL，将转储整个缓冲区，最多 100 条条目。如果 *iterator* 不为 NULL，将从该条目开始转储日志，并且 *iterator* 的值将更新为日志的当前末尾，就像调用了 *cuLogsCurrent* 一样。如果缓冲区中的日志条目超过 100 条，将在转储开始时添加一条说明。
flags 参数必须为 0，其他选项保留供未来 CUDA 版本使用。

*cuLogsDumpToMemory* 函数还有以下注意事项：

1. 缓冲区本身将以空字符结尾，但每个单独的日志条目仅由换行符（\n）分隔。
2. 缓冲区的最大大小为 25600 字节。
3. 如果提供的 size 值不足以存储所有所需的日志，将在第一条记录处添加一条说明，并且无法容纳的最旧条目将不会被转储。
4. 返回后，size 将包含实际写入所提供缓冲区的字节数。

## 4.8.5. 限制与已知问题

1. 日志缓冲区限制为 100 条条目。达到此限制后，最旧的条目将被替换，并且日志转储将包含一行说明回滚的注释。
2. 并非所有 CUDA API 都已涵盖。这是一个持续进行的项目，旨在为所有 API 提供更好的使用错误报告。
3. 错误日志管理日志位置（如果已给出）的有效性将不会进行测试，除非/直到生成日志。
4. 错误日志管理 API 目前仅通过 CUDA 驱动程序提供。等效的 API 将在未来版本中添加到 CUDA 运行时。
5. 日志消息未本地化为任何语言，所有提供的日志均为美式英语。

 本页