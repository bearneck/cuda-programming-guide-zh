# 4.1 统一内存

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版
>
> 原文地址：[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)

---

此页面是否有帮助？

# 4.1. 统一内存

本节将详细解释每种可用统一内存范式的具体行为和使用方法。[前面关于统一内存的章节](../02-basics/understanding-memory.html#memory-unified-memory)介绍了如何确定适用的统一内存范式，并对每种范式进行了简要介绍。

如前所述，统一内存编程有四种范式：

- 对显式管理的内存分配提供完全支持
- 对具有软件一致性的所有分配提供完全支持
- 对具有硬件一致性的所有分配提供完全支持
- 有限统一内存支持

前三种涉及完全统一内存支持的范式具有非常相似的行为和编程模型，将在[具有完全 CUDA 统一内存支持的设备上的统一内存](#um-pageable-systems)中介绍，并会突出显示任何差异。

最后一种范式，即统一内存支持有限的情况，将在[Windows、WSL 和 Tegra 上的统一内存](#um-legacy-devices)中详细讨论。

## 4.1.1. 具有完全 CUDA 统一内存支持的设备上的统一内存

这些系统包括硬件一致性内存系统，例如 NVIDIA Grace Hopper 和启用了异构内存管理 (HMM) 的现代 Linux 系统。HMM 是一种基于软件的内存管理系统，提供与硬件一致性内存系统相同的编程模型。

Linux HMM 需要 Linux 内核版本 6.1.24+、6.2.11+ 或 6.3+，计算能力 7.5 或更高的设备，以及安装了 [Open Kernel Modules](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#nvidia-open-gpu-kernel-modules) 的 CUDA 驱动程序版本 535+。

!!! note "注意"
    我们将 CPU 和 GPU 共享一个页表的系统称为硬件一致性系统。CPU 和 GPU 拥有独立页表的系统则称为软件一致性系统。

像 NVIDIA Grace Hopper 这样的硬件一致性系统为 CPU 和 GPU 提供了一个逻辑上统一的页表，请参阅 [CPU 和 GPU 页表：硬件一致性与软件一致性](#um-hw-coherency)。以下部分仅适用于硬件一致性系统：

> 访问计数器迁移

### 4.1.1.1. 统一内存：深入示例

具有完全 CUDA 统一内存支持的系统（参见表格[统一内存范式概述](../02-basics/understanding-memory.html#table-unified-memory-levels)）允许设备访问主机进程中与该设备交互的任何内存。

本节展示一些高级用例，使用一个内核，该内核只是将输入字符数组的前 8 个字符打印到标准输出流：

```cuda
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```

以下标签页展示了如何使用系统分配的内存来调用此内核的各种方式：
## Malloc

```cuda
void test_malloc() {
  const char test_string[] = "Hello World";
  char* heap_data = (char*)malloc(sizeof(test_string));
  strncpy(heap_data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("malloc", heap_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  free(heap_data);
}
```

## Managed

```cuda
void test_managed() {
  const char test_string[] = "Hello World";
  char* data;
  cudaMallocManaged(&data, sizeof(test_string));
  strncpy(data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("managed", data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  cudaFree(data);
}
```

## 栈变量

```cuda
void test_stack() {
  const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("stack", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

## 文件作用域静态变量

```cuda
void test_static() {
  static const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("static", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

## 全局作用域变量

```cuda
const char global_string[] = "Hello World";

void test_global() {
  kernel<<<1, 1>>>("global", global_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

## 全局作用域 extern 变量

```cuda
// 在单独的文件中声明，见下文
extern char* ext_data;

void test_extern() {
  kernel<<<1, 1>>>("extern", ext_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

```cuda
/** 这可能是一个非 CUDA 文件 */
char* ext_data;
static const char global_string[] = "Hello World";

void __attribute__ ((constructor)) setup(void) {
  ext_data = (char*)malloc(sizeof(global_string));
  strncpy(ext_data, global_string, sizeof(global_string));
}

void __attribute__ ((destructor)) tear_down(void) {
  free(ext_data);
}
```

请注意，对于 extern 变量，它可能由完全不与 CUDA 交互的第三方库声明、拥有并管理其内存。

还需注意，栈变量以及文件作用域和全局作用域的变量只能通过指针被 GPU 访问。在这个特定示例中，这很方便，因为字符数组已经声明为指针：`const char*`。然而，请考虑以下带有全局作用域整数的示例：

```cuda
// 此变量在全局作用域声明
int global_variable;

__global__ void kernel_uncompilable() {
  // 这会导致编译错误：全局（__host__）变量不得从 __device__ / __global__ 代码访问
  printf("%d\n", global_variable);
}

// 在 pageableMemoryAccess 设置为 1 的系统上，我们可以访问全局变量的地址。
// 下面的内核将该地址作为参数
__global__ void kernel(int* global_variable_addr) {
  printf("%d\n", *global_variable_addr);
}
int main() {
  kernel<<<1, 1>>>(&global_variable);
  ...
  return 0;
}
```
在上面的示例中，我们需要确保向内核传递全局变量的*指针*，而不是在内核中直接访问全局变量。这是因为没有 `__managed__` 说明符的全局变量默认声明为仅 `__host__`，因此目前大多数编译器不允许在设备代码中直接使用这些变量。

#### 4.1.1.1.1. 文件支持的统一内存

由于具有完整 CUDA 统一内存支持的系统允许设备访问主机进程拥有的任何内存，因此它们可以直接访问文件支持的内存。

这里，我们展示了上一节中初始示例的修改版本，使用文件支持的内存，以便从 GPU 打印一个字符串，该字符串直接从输入文件中读取。在以下示例中，内存由物理文件支持，但该示例同样适用于内存支持的文件。

```cuda
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```

```cuda
void test_file_backed() {
  int fd = open(INPUT_FILE_NAME, O_RDONLY);
  ASSERT(fd >= 0, "Invalid file handle");
  struct stat file_stat;
  int status = fstat(fd, &file_stat);
  ASSERT(status >= 0, "Invalid file stats");
  char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ASSERT(mapped != MAP_FAILED, "Cannot map file into memory");
  kernel<<<1, 1>>>("file-backed", mapped);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  ASSERT(munmap(mapped, file_stat.st_size) == 0, "Cannot unmap file");
  ASSERT(close(fd) == 0, "Cannot close file");
}
```

请注意，在不支持 `hostNativeAtomicSupported` 属性的系统上（参见[主机原生原子操作](#um-host-native-atomics)），包括启用了 Linux HMM 的系统，不支持对文件支持的内存进行原子访问。

#### 4.1.1.1.2. 使用统一内存的进程间通信 (IPC)

!!! note "注意"
    截至目前，将 IPC 与统一内存结合使用可能会对性能产生显著影响。

许多应用程序倾向于每个进程管理一个 GPU，但仍然需要使用统一内存，例如用于超额订阅，并从多个 GPU 访问它。

CUDA IPC（参见[进程间通信](inter-process-communication.html#interprocess-communication)）不支持托管内存：此类内存的句柄不能通过本节讨论的任何机制共享。在具有完整 CUDA 统一内存支持的系统上，系统分配的内存支持 IPC。一旦系统分配的内存访问权限与其他进程共享，就可以应用相同的编程模型，类似于[文件支持的统一内存](#um-sam-file-backed)。

有关在 Linux 下创建支持 IPC 的系统分配内存的各种方法的更多信息，请参阅以下参考资料：

- 使用 MAP_SHARED 的 mmap
- POSIX IPC API
- Linux memfd_create

请注意，使用此技术无法在不同主机及其设备之间共享内存。

### 4.1.1.2. 性能调优

为了在使用统一内存时获得良好性能，以下几点至关重要：

- 了解系统上的分页工作原理，以及如何避免不必要的页面错误
- 了解允许将数据保持在访问处理器本地的各种机制
- 考虑根据系统的内存传输粒度来调优应用程序

作为一般建议，性能提示（参见[性能提示](#um-perf-hints)）可能会提供改进的性能，但如果使用不当，与默认行为相比可能会降低性能。同时请注意，任何提示在主机端都有相关的性能开销，因此有用的提示必须至少能将性能提升到足以抵消此开销的程度。

#### 4.1.1.2.1. 内存分页与页面大小

为了更好地理解统一内存的性能影响，理解虚拟寻址、内存页面和页面大小非常重要。本小节试图定义所有必要的术语，并解释为什么分页对性能很重要。

所有当前支持统一内存的系统都使用虚拟地址空间：这意味着应用程序使用的内存地址代表一个*虚拟*位置，该位置可能被*映射*到内存实际驻留的物理位置。

所有当前支持的处理器，包括 CPU 和 GPU，都额外使用内存*分页*。由于所有系统都使用虚拟地址空间，因此存在两种类型的内存页面：

- 虚拟页面：这表示操作系统跟踪的每个进程中固定大小的连续虚拟内存块，可以映射到物理内存中。
  请注意，虚拟页面与映射相关联：例如，单个虚拟地址可能使用不同的页面大小映射到物理内存。
- 物理页面：这表示处理器主内存管理单元（MMU）支持的固定大小的连续内存块，虚拟页面可以映射到其中。

目前，所有 x86_64 CPU 默认使用 4KiB 的物理页面大小。Arm CPU 支持多种物理页面大小——4KiB、16KiB、32KiB 和 64KiB——具体取决于确切的 CPU。最后，NVIDIA GPU 支持多种物理页面大小，但更倾向于 2MiB 或更大的物理页面。请注意，这些大小可能会在未来的硬件中发生变化。

虚拟页面的默认页面大小通常与物理页面大小相对应，但只要操作系统和硬件支持，应用程序可以使用不同的页面大小。通常，支持的虚拟页面大小必须是 2 的幂，并且是物理页面大小的倍数。

跟踪虚拟页面到物理页面映射的逻辑实体将被称为*页表*，而将具有给定虚拟大小的给定虚拟页面映射到物理页面的每个映射称为*页表项（PTE）*。所有支持的处理器都为页表提供了特定的缓存，以加速虚拟地址到物理地址的转换。这些缓存被称为*转译后备缓冲器（TLB）*。
应用程序性能调优有两个重要方面：

- 虚拟页面大小的选择
- 系统提供的是CPU和GPU共用的统一页表，还是CPU和GPU各自独立的页表

##### 4.1.1.2.1.1.选择合适的页面大小

一般来说，较小的页面大小会导致较少（虚拟）内存碎片但更多TLB未命中，而较大的页面大小会导致更多内存碎片但较少TLB未命中。此外，与较小页面相比，较大页面的内存迁移通常成本更高，因为我们通常迁移完整的内存页面。这可能导致使用大页面大小的应用程序出现更大的延迟峰值。有关页面错误的更多详细信息，请参见下一节。

性能调优的一个重要方面是，与CPU相比，GPU上的TLB未命中通常代价要高得多。这意味着如果GPU线程频繁访问使用足够小页面大小映射的统一内存的随机位置，与访问使用足够大页面大小映射的统一内存相比，速度可能会显著变慢。虽然CPU线程随机访问使用小页面大小映射的大内存区域时也可能出现类似效果，但速度下降不那么明显，这意味着应用程序可能需要在速度下降和减少内存碎片之间进行权衡。

请注意，一般来说，应用程序不应根据给定处理器的物理页面大小来调整性能，因为物理页面大小可能会因硬件而异。上述建议仅适用于虚拟页面大小。

##### 4.1.1.2.1.2.CPU和GPU页表：硬件一致性 vs. 软件一致性

硬件一致性系统（如NVIDIA Grace Hopper）为CPU和GPU提供逻辑上统一的页表。这很重要，因为为了从GPU访问系统分配的内存，GPU使用CPU为请求内存创建的任何页表项。如果该页表项使用默认的CPU页面大小（4KiB或64KiB），访问大的虚拟内存区域将导致显著的TLB未命中，从而显著降低性能。

另一方面，在CPU和GPU各自拥有独立逻辑页表的软件一致性系统上，应考虑不同的性能调优方面：为了保证一致性，这些系统通常在处理器访问映射到不同处理器物理内存的内存地址时使用*页面错误*。这样的页面错误意味着：

- 需要确保当前拥有该页面的处理器（物理页面当前所在的处理器）无法再访问此页面，可以通过删除页表项或更新页表项来实现。
- 需要确保请求访问的处理器能够访问此页面，可以通过创建新的页表项或更新现有页表项来实现，使其变为有效/活动状态。
- 支持此虚拟页面的物理页面必须被移动/迁移到请求访问的处理器
请求访问权限：这可能是一项开销较大的操作，且工作量与页面大小成正比。

总体而言，在 CPU 和 GPU 线程频繁并发访问同一内存页的情况下，硬件一致性系统相比软件一致性系统能提供显著的性能优势：

- 更少的缺页异常：这些系统无需使用缺页异常来模拟一致性或迁移内存。
- 更少的争用：这些系统以缓存行粒度而非页面粒度保持一致性，也就是说，当多个处理器在同一个缓存行内发生争用时，仅交换缓存行（其大小远小于最小页面尺寸）；而当不同处理器访问页面内的不同缓存行时，则不会发生争用。

这对以下场景的性能产生影响：

- CPU 和 GPU 并发对同一地址进行原子更新
- 从 CPU 线程向 GPU 线程发送信号，或反之。

#### 4.1.1.2.2. 主机直接访问统一内存

部分设备具备硬件支持，允许主机直接对 GPU 驻留的统一内存进行一致性读取、存储和原子访问。这些设备的 `cudaDevAttrDirectManagedMemAccessFromHost` 属性被设置为 1。请注意，所有硬件一致性系统在通过 NVLink 连接的设备上均设置了此属性。在此类系统中，主机可直接访问 GPU 驻留内存，而无需触发缺页异常和数据迁移。需要注意的是，在使用 CUDA 托管内存时，必须通过 `cudaMemAdviseSetAccessedBy` 提示并指定位置类型为 `cudaMemLocationTypeHost`，才能启用这种无需缺页异常的直接访问，参见以下示例。

 系统分配器

```cuda
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_malloc() {
  int *ret = (int*)malloc(1000 * sizeof(int));
  // 对于共享页表系统，以下提示并非必需
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);

  write<<< 1, 1000 >>>(ret, 10, 100);            // 页面在 GPU 内存中填充
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU 直接访问 GPU 内存，无需迁移
                                                  // directManagedMemAccessFromHost=0: CPU 触发缺页异常并引发设备到主机的迁移
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU 访问 GPU 内存，无需迁移
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU 触发缺页异常并引发主机到设备的迁移
  free(ret);
}
```

 托管内存

```cuda
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_managed() {
  int *ret;
  cudaMallocManaged(&ret, 1000 * sizeof(int));
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);  // 设置直接访问提示

  write<<< 1, 1000 >>>(ret, 10, 100);            // 页面在 GPU 内存中填充
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU 直接访问 GPU 内存，无需迁移
                                                  // directManagedMemAccessFromHost=0: CPU 触发缺页异常并引发设备到主机的迁移
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU 访问 GPU 内存，无需迁移
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU 触发缺页异常并引发主机到设备的迁移
  cudaFree(ret);
```
在 `write` 内核完成后，`ret` 将在 GPU 内存中被创建并初始化。接下来，CPU 将访问 `ret`，随后 `append` 内核将再次使用相同的 `ret` 内存。此代码的行为将根据系统架构和硬件一致性支持的不同而有所差异：

- 在 `directManagedMemAccessFromHost=1` 的系统上：
CPU 对托管缓冲区的访问不会触发任何迁移；
数据将保留在 GPU 内存中，任何后续的 GPU 内核都可以继续直接访问它，而不会引发缺页或迁移。
- 在 `directManagedMemAccessFromHost=0` 的系统上：
CPU 对托管缓冲区的访问将引发缺页并启动数据迁移；
任何首次尝试访问相同数据的 GPU 内核都将引发缺页，并将页面迁移回 GPU 内存。

#### 4.1.1.2.3. 主机原生原子操作

某些设备，包括硬件一致性系统中通过 NVLink 连接的设备，支持对驻留在 CPU 内存的数据进行硬件加速的原子访问。这意味着对主机内存的原子访问无需通过缺页来模拟。对于这些设备，属性 `cudaDevAttrHostNativeAtomicSupported` 被设置为 1。

#### 4.1.1.2.4. 原子访问与同步原语

CUDA 统一内存支持主机和设备线程可用的所有原子操作，使所有线程能够通过并发访问相同的共享内存位置进行协作。[libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html) 库提供了许多针对主机和设备线程之间并发使用而优化的异构同步原语，包括 `cuda::atomic`、`cuda::atomic_ref`、`cuda::barrier`、`cuda::semaphore` 等。

在软件一致性系统上，不支持设备对文件支持的主机内存进行原子访问。以下示例代码在硬件一致性系统上有效，但在其他系统上表现出未定义行为：

```cuda
#include <cuda/atomic>

#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>

#define ERR(msg, ...) { fprintf(stderr, msg, ##__VA_ARGS__); return EXIT_FAILURE; }

__global__ void kernel(int* ptr) {
  cuda::atomic_ref{*ptr}.store(2);
}

int main() {
  // this will be closed/deleted by default on exit
  FILE* tmp_file = tmpfile64();
  // need to allocate space in the file, we do this with posix_fallocate here
  int status = posix_fallocate(fileno(tmp_file), 0, 4096);
  if (status != 0) ERR("Failed to allocate space in temp file\n");
  int* ptr = (int*)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(tmp_file), 0);
  if (ptr == MAP_FAILED) ERR("Failed to map temp file\n");

  // initialize the value in our file-backed memory
  *ptr = 1;
  printf("Atom value: %d\n", *ptr);

  // device and host thread access ptr concurrently, using cuda::atomic_ref
  kernel<<<1, 1>>>(ptr);
  while (cuda::atomic_ref{*ptr}.load() != 2);
  // this will always be 2
  printf("Atom value: %d\n", *ptr);

  return EXIT_SUCCESS;
}
```
在软件一致性系统中，对统一内存的原子访问可能会引发页面错误，从而导致显著的延迟。请注意，并非这些系统中所有从 GPU 到 CPU 内存的原子操作都会如此：通过 `nvidia-smi -q | grep "Atomic Caps Outbound"` 列出的操作可能避免页面错误。

在硬件一致性系统中，主机与设备之间的原子操作不需要页面错误，但仍可能因其他原因（这些原因可能导致任何内存访问出错）而引发错误。

#### 4.1.1.2.5. 统一内存的 Memcpy()/Memset() 行为

`cudaMemcpy*()` 和 `cudaMemset*()` 接受任何统一内存指针作为参数。

对于 `cudaMemcpy*()`，指定为 `cudaMemcpyKind` 的方向是一个性能提示，如果任何参数是统一内存指针，则此提示可能对性能产生更大的影响。

因此，建议遵循以下性能建议：

*   当统一内存的物理位置已知时，使用准确的 `cudaMemcpyKind` 提示。
*   与不准确的 `cudaMemcpyKind` 提示相比，优先使用 `cudaMemcpyDefault`。
*   始终使用已填充（已初始化）的缓冲区：避免使用这些 API 来初始化内存。
*   如果两个指针都指向系统分配的内存，请避免使用 `cudaMemcpy*()`：改为启动内核或使用 CPU 内存复制算法，例如 `std::memcpy`。

#### 4.1.1.2.6. 统一内存分配器概述

对于具有完整 CUDA 统一内存支持的系统，可以使用各种不同的分配器来分配统一内存。下表概述了部分分配器及其各自的功能。请注意，本节中的所有信息在未来的 CUDA 版本中可能会发生变化。

| API | 放置策略 | 可从何处访问 | 基于访问进行迁移 [ 2 ] | 页面大小 [ 4 ] [ 5 ] |
| --- | --- | --- | --- | --- |
| `malloc`, `new`, `mmap` | 首次接触/提示 [ 1 ] | CPU, GPU | 是 [ 3 ] | 系统或大页面大小 [ 6 ] |
| `cudaMallocManaged` | 首次接触/提示 | CPU, GPU | 是 | CPU 驻留：系统页面大小<br>GPU 驻留：2MB |
| `cudaMalloc` | GPU | GPU | 否 | GPU 页面大小：2MB |
| `cudaMallocHost`, `cudaHostAlloc`, `cudaHostRegister` | CPU | CPU, GPU | 否 | 由 CPU 映射：系统页面大小<br>由 GPU 映射：2MB |
| 内存池，位置类型为 host：`cuMemCreate`, `cudaMemPoolCreate` | CPU | CPU, GPU | 否 | 由 CPU 映射：系统页面大小<br>由 GPU 映射：2MB |
| 内存池，位置类型为 device：`cuMemCreate`, `cudaMemPoolCreate`, `cudaMallocAsync` | GPU | GPU | 否 | 2MB |

[ 1 ]
对于 `mmap`，文件支持的内存默认放置在 CPU 上，除非通过 `cudaMemAdviseSetPreferredLocation`（或 `mbind`，请参见下面的要点）另行指定。

[ 2 ]
此功能可以通过 `cudaMemAdvise` 覆盖。即使禁用了基于访问的迁移，如果后备内存空间已满，内存仍可能迁移。

[ 3 ]
文件支持的内存不会基于访问进行迁移。

[ 4 ]
默认系统页面大小在大多数系统上为 4KiB 或 64KiB，除非明确指定了大页面大小（例如，使用 `mmap` 的 `MAP_HUGETLB` / `MAP_HUGE_SHIFT`）。在这种情况下，支持系统上配置的任何大页面大小。
[
5
]

GPU驻留内存的页面大小可能会在未来的CUDA版本中演进。

[
6
]

目前，当内存迁移到GPU或通过首次接触（first-touch）放置到GPU时，可能无法保留大页面（huge page）大小。

表 [不同分配器的统一内存支持概述](#table-um-allocators) 展示了几个分配器在语义上的差异，这些分配器可用于分配可被多个处理器（包括主机和设备）同时访问的数据。有关 `cudaMemPoolCreate` 的更多详细信息，请参阅 [内存池](stream-ordered-memory-allocation.html#stream-ordered-memory-pools) 部分；有关 `cuMemCreate` 的更多详细信息，请参阅 [虚拟内存管理](virtual-memory-management.html#virtual-memory-management) 部分。

在设备内存作为NUMA域暴露给系统的硬件一致性系统上，可以使用特殊的分配器（如 `numa_alloc_on_node`）将内存固定到给定的NUMA节点（可以是主机或设备）。此内存可从主机和设备访问，并且不会迁移。类似地，`mbind` 可用于将内存固定到给定的NUMA节点，并可以在首次访问之前将文件支持的内存放置到给定的NUMA节点上。

以下内容适用于共享内存的分配器：

- 系统分配器（如 mmap）允许使用 MAP_SHARED 标志在进程之间共享内存。这在 CUDA 中受支持，可用于在连接到同一主机的不同设备之间共享内存。但是，目前不支持在多个主机以及多个设备之间共享内存。有关详细信息，请参阅 [使用统一内存的进程间通信（IPC）](inter-process-communication-ipc-with-unified-memory.html)。
- 对于通过网络在多个主机上访问统一内存或其他 CUDA 内存，请查阅所用通信库的文档，例如 NCCL、NVSHMEM、OpenMPI、UCX 等。

#### 4.1.1.2.7. 访问计数器迁移

在硬件一致性系统上，访问计数器功能会跟踪 GPU 对其他处理器上内存的访问频率。这是确保内存页面迁移到最频繁访问这些页面的处理器的物理内存中所必需的。它可以指导 CPU 和 GPU 之间以及对等 GPU 之间的迁移，这个过程称为访问计数器迁移。

从 CUDA 12.4 开始，访问计数器支持系统分配的内存。请注意，文件支持的内存不会基于访问进行迁移。对于系统分配的内存，可以通过使用 `cudaMemAdviseSetAccessedBy` 提示并指定相应的设备 ID 来开启访问计数器迁移。如果访问计数器已开启，可以使用 `cudaMemAdviseSetPreferredLocation` 设置为 host 来防止迁移。默认情况下，`cudaMallocManaged` 基于故障并迁移（fault-and-migrate）机制进行迁移。[[7]](#footnote-fault-and-migrate)

驱动程序也可能使用访问计数器来实现更高效的颠簸缓解或内存超额订阅场景。

[
7
当前系统允许在设置访问设备提示时，将访问计数器迁移与托管内存结合使用。这是一个实现细节，不应依赖其实现未来兼容性。

#### 4.1.1.2.8. 避免 CPU 频繁写入 GPU 驻留内存

如果主机访问统一内存，缓存未命中可能会在主机和设备之间引入比预期更多的流量。许多 CPU 架构要求所有内存操作都通过缓存层次结构，包括写入操作。如果系统内存驻留在 GPU 上，这意味着 CPU 频繁写入此内存可能导致缓存未命中，从而在将实际值写入请求的内存范围之前，先将数据从 GPU 传输到 CPU。在软件一致性系统上，这可能会引入额外的页面错误；而在硬件一致性系统上，则可能导致 CPU 操作之间的延迟更高。因此，为了与设备共享主机产生的数据，请考虑写入 CPU 驻留内存，并直接从设备读取这些值。以下代码展示了如何使用统一内存实现这一点。

 系统分配器

```cuda
  size_t data_size = sizeof(int);
  int* data = (int*)malloc(data_size);
  // ensure that data stays local to the host and avoid faults
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
  cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);

  // frequent exchanges of small data: if the CPU writes to CPU-resident memory,
  // and GPU directly accesses that data, we can avoid the CPU caches re-loading
  // data if it was evicted in between writes
  for (int i = 0; i < 10; ++i) {
    *data = 42 + i;
    kernel<<<1, 1>>>(data);
    cudaDeviceSynchronize();
    // CPU cache potentially evicted data here
  }
  free(data);
```

 托管内存

```cuda
  int* data;
  size_t data_size = sizeof(int);
  cudaMallocManaged(&data, data_size);
  // ensure that data stays local to the host and avoid faults
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
  cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);

  // frequent exchanges of small data: if the CPU writes to CPU-resident memory,
  // and GPU directly accesses that data, we can avoid the CPU caches re-loading
  // data if it was evicted in between writes
  for (int i = 0; i < 10; ++i) {
    *data = 42 + i;
    kernel<<<1, 1>>>(data);
    cudaDeviceSynchronize();
    // CPU cache potentially evicted data here
  }
  cudaFree(data);
```

#### 4.1.1.2.9. 利用对系统内存的异步访问

如果应用程序需要与主机共享设备上的工作结果，有几种可能的选项：

1.  设备将其结果写入 GPU 驻留内存，然后使用 cudaMemcpy* 传输结果，主机读取传输的数据。
2.  设备直接将其结果写入 CPU 驻留内存，主机读取该数据。
3. 设备写入 GPU 驻留内存，主机直接访问该数据。

如果可以在设备上调度独立工作的同时由主机传输/访问结果，则首选选项 1 或 3。如果设备在主机访问结果之前一直处于空闲状态，则可能首选选项 2。这是因为除非使用多个主机线程来读取数据，否则设备的写入带宽通常高于主机的读取带宽。

 1. 显式复制

```cuda
void exchange_explicit_copy(cudaStream_t stream) {
  int* data, *host_data;
  size_t n_bytes = sizeof(int) * 16;
  // allocate receiving buffer
  host_data = (int*)malloc(n_bytes);
  // allocate, since we touch on the device first, will be GPU-resident
  cudaMallocManaged(&data, n_bytes);
  kernel<<<1, 16, 0, stream>>>(data);
  // launch independent work on the device
  // other_kernel<<<1024, 256, 0, stream>>>(other_data, ...);
  // transfer to host
  cudaMemcpyAsync(host_data, data, n_bytes, cudaMemcpyDeviceToHost, stream);
  // sync stream to ensure data has been transferred
  cudaStreamSynchronize(stream);
  // read transferred data
  printf("Got values %d - %d from GPU\n", host_data[0], host_data[15]);
  cudaFree(data);
  free(host_data);
}
```

 2. 设备直接写入

```cuda
void exchange_device_direct_write(cudaStream_t stream) {
  int* data;
  size_t n_bytes = sizeof(int) * 16;
  // allocate receiving buffer
  cudaMallocManaged(&data, n_bytes);
  // ensure that data is mapped and resident on the host
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, location);
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, location);
  kernel<<<1, 16, 0, stream>>>(data);
  // sync stream to ensure data has been transferred
  cudaStreamSynchronize(stream);
  // read transferred data
  printf("Got values %d - %d from GPU\n", data[0], data[15]);
  cudaFree(data);
}
```

 3. 主机直接读取

```cuda
void exchange_host_direct_read(cudaStream_t stream) {
  int* data;
  size_t n_bytes = sizeof(int) * 16;
  // allocate receiving buffer
  cudaMallocManaged(&data, n_bytes);
  // ensure that data is mapped and resident on the device
  cudaMemLocation device_loc = {};
  cudaGetDevice(&device_loc.id);
  device_loc.type = cudaMemLocationTypeDevice;
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, device_loc);
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, device_loc);
  kernel<<<1, 16, 0, stream>>>(data);
  // launch independent work on the GPU
  // other_kernel<<<1024, 256, 0, stream>>>(other_data, ...);
  // sync stream to ensure data may be accessed (has been written by device)
  cudaStreamSynchronize(stream);
  // read data directly from host
  printf("Got values %d - %d from GPU\n", data[0], data[15]);
  cudaFree(data);
```

最后，在上述显式复制示例中，除了使用 `cudaMemcpy*` 传输数据外，还可以使用主机或设备内核来显式执行此传输。对于连续数据，首选使用 CUDA 复制引擎，因为复制引擎执行的操作可以与主机和设备上的工作重叠。复制引擎可能在 `cudaMemcpy*` 和 `cudaMemPrefetchAsync` API 中使用，但不能保证 `cudaMemcpy*` API 调用一定使用复制引擎。出于同样的原因，对于足够大的数据，显式复制优于主机直接读取：如果主机和设备执行的工作均未使其各自的内存系统饱和，则传输可以由复制引擎与主机和设备执行的工作并发进行。
复制引擎通常用于主机与设备之间以及 NVLink 连接系统内对等设备之间的传输。由于复制引擎的总数有限，某些系统在使用 `cudaMemcpy*` 时的带宽可能低于使用设备显式执行传输的情况。在这种情况下，如果传输位于应用程序的关键路径上，则可能更倾向于使用基于设备的显式传输。

## 4.1.2. 仅支持 CUDA 托管内存的设备上的统一内存

对于计算能力为 6.x 或更高但不支持可分页内存访问的设备（参见表格[统一内存范式概述](../02-basics/understanding-memory.html#table-unified-memory-levels)），CUDA 托管内存得到完全支持且具有一致性，但 GPU 无法访问系统分配的内存。统一内存的编程模型和性能调优在很大程度上与[具有完整 CUDA 统一内存支持的设备上的统一内存](#um-pageable-systems)一节中描述的模型相似，但有一个显著例外：不能使用系统分配器来分配内存。因此，以下子章节不适用：

- 统一内存：深入示例
- CPU 和 GPU 页表：硬件一致性与软件一致性
- 原子访问和同步原语
- 访问计数器迁移
- 避免 CPU 频繁写入 GPU 驻留内存
- 利用对系统内存的异步访问

## 4.1.3. Windows、WSL 和 Tegra 上的统一内存

!!! note "注意"
    本节仅关注计算能力低于 6.0 的设备或 Windows 平台，以及 `concurrentManagedAccess` 属性设置为 0 的设备。

计算能力低于 6.0 的设备或 Windows 平台，以及 `concurrentManagedAccess` 属性设置为 0 的设备（参见[统一内存范式概述](../02-basics/understanding-memory.html#table-unified-memory-levels)），支持 CUDA 托管内存，但存在以下限制：

- 数据迁移和一致性：不支持按需将托管数据细粒度地迁移到 GPU。每当启动 GPU 内核时，通常必须将所有托管内存传输到 GPU 内存，以避免内存访问出错。仅支持从 CPU 端进行页面错误处理。
- GPU 内存超额订阅：它们无法分配超过 GPU 内存物理大小的托管内存。
- 一致性和并发性：无法同时访问托管内存，因为如果 GPU 内核处于活动状态时 CPU 访问统一内存分配，由于缺少 GPU 页面错误处理机制，无法保证一致性。

### 4.1.3.1. 多 GPU

在计算能力低于 6.0 的设备或 Windows 平台上，托管分配通过 GPU 的对等能力自动对系统中的所有 GPU 可见。托管内存分配的行为类似于使用 `cudaMalloc()` 分配的非托管内存：当前活动设备是物理分配的主设备，但系统中的其他 GPU 将通过 PCIe 总线以降低的带宽访问该内存。
在 Linux 系统上，只要程序当前使用的所有 GPU 都支持点对点（peer-to-peer）访问，托管内存就会分配在 GPU 内存中。如果应用程序在任何时候开始使用一个 GPU，而该 GPU 与任何其他已分配托管内存的 GPU 之间不支持点对点访问，那么驱动程序会将所有托管内存分配迁移到系统内存。在这种情况下，所有 GPU 都会受到 PCIe 带宽的限制。

在 Windows 系统上，如果点对点映射不可用（例如，在不同架构的 GPU 之间），那么系统将自动回退到使用映射内存，无论程序是否实际使用这两个 GPU。如果实际上只使用一个 GPU，则需要在启动程序之前设置 `CUDA_VISIBLE_DEVICES` 环境变量。这会限制可见的 GPU，并允许托管内存分配在 GPU 内存中。

或者，在 Windows 上，用户也可以将 `CUDA_MANAGED_FORCE_DEVICE_ALLOC` 设置为非零值，以强制驱动程序始终使用设备内存作为物理存储。当此环境变量设置为非零值时，该进程中使用的所有支持托管内存的设备必须彼此点对点兼容。如果使用了支持托管内存的设备，并且该设备与该进程中先前使用的任何其他支持托管内存的设备不点对点兼容，即使已对这些设备调用 `::cudaDeviceReset`，也会返回 `::cudaErrorInvalidDevice` 错误。这些环境变量在 [CUDA 环境变量](../05-appendices/environment-variables.html#cuda-environment-variables) 中有详细描述。

### 4.1.3.2. 一致性与并发性

为确保一致性，统一内存编程模型在 CPU 和 GPU 并发执行时对数据访问施加了约束。实际上，在执行任何内核操作期间，无论特定内核是否正在主动使用数据，GPU 都拥有对所有托管数据的独占访问权，而 CPU 则不允许访问。并发的 CPU/GPU 访问，即使是对不同的托管内存分配，也会导致段错误，因为该页面被视为 CPU 无法访问。

例如，以下代码在计算能力 6.x 的设备上运行成功，因为 GPU 页面错误处理能力取消了对同时访问的所有限制；但在 6.x 之前的架构和 Windows 平台上会失败，因为当 CPU 访问 `y` 时，GPU 程序内核仍在运行：

```cuda
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    y = 20;            // 在不支持并发访问的 GPU 上会出错

    cudaDeviceSynchronize();
    return  0;
}
```

程序在访问 `y` 之前，必须与 GPU 显式同步（无论 GPU 内核是否实际触及 `y` 或任何托管数据）：

```cuda
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    cudaDeviceSynchronize();
    y = 20;            //  Success on GPUs not supporting concurrent access
    return  0;
}
```

Note that any function call that logically guarantees the GPU completes its work is valid to ensure logically that the GPU work is completed, see [Explicit Synchronization](../03-advanced/advanced-host-programming.html#advanced-host-explicit-synchronization).

Note that if memory is dynamically allocated with `cudaMallocManaged()` or `cuMemAllocManaged()` while the GPU is active, the behavior of the memory is unspecified until additional work is launched or the GPU is synchronized. Attempting to access the memory on the CPU during this time may or may not cause a segmentation fault. This does not apply to memory allocated using the flag `cudaMemAttachHost` or `CU_MEM_ATTACH_HOST`.

### 4.1.3.3.Stream Associated Unified Memory

The CUDA programming model provides streams as a mechanism for programs to indicate dependence and independence among kernel launches. Kernels launched into the same stream are guaranteed to execute consecutively, while kernels launched into different streams are permitted to execute concurrently.  See section [CUDA Streams](../02-basics/asynchronous-execution.html#cuda-streams).

#### 4.1.3.3.1.Stream Callbacks

It is legal for the CPU to access managed data from within a stream callback, provided no other stream that could potentially be accessing managed data is active on the GPU. In addition, a callback that is not followed by any device work can be used for synchronization: for example, by signaling a condition variable from inside the callback; otherwise, CPU access is valid only for the duration of the callback(s). There are several important points of note:

1. It is always permitted for the CPU to access non-managed mapped memory data while the GPU is active.
2. The GPU is considered active when it is running any kernel, even if that kernel does not make use of managed data. If a kernel might use data, then access is forbidden
3. There are no constraints on concurrent inter-GPU access of managed memory, other than those that apply to multi-GPU access of non-managed memory.
4. There are no constraints on concurrent GPU kernels accessing managed data.

Note how the last point allows for races between GPU kernels, as is currently the case for non-managed GPU memory. In the perspective of the GPU, managed memory functions are identical to non-managed memory. The following code example illustrates these points:

```cuda
int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    int *non_managed, *managed, *also_managed;
    cudaMallocHost(&non_managed, 4);    // Non-managed, CPU-accessible memory
    cudaMallocManaged(&managed, 4);
    cudaMallocManaged(&also_managed, 4);
    // Point 1: CPU can access non-managed data.
    kernel<<< 1, 1, 0, stream1 >>>(managed);
    *non_managed = 1;
    // Point 2: CPU cannot access any managed data while GPU is busy,
    //          unless concurrentManagedAccess = 1
    // Note we have not yet synchronized, so "kernel" is still active.
    *also_managed = 2;      // Will issue segmentation fault
    // Point 3: Concurrent GPU kernels can access the same data.
    kernel<<< 1, 1, 0, stream2 >>>(managed);
    // Point 4: Multi-GPU concurrent access is also permitted.
    cudaSetDevice(1);
    kernel<<< 1, 1 >>>(managed);
    return  0;
}
```
#### 4.1.3.3.2. 与流关联的托管内存允许更精细的控制

统一内存建立在流无关模型之上，允许 CUDA 程序显式地将托管分配与 CUDA 流关联起来。通过这种方式，程序员可以根据内核是否在指定流中启动，来指示其对数据的使用。这使得基于程序特定数据访问模式的并发成为可能。控制此行为的函数是：

```cuda
cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream,
                                     void *ptr,
                                     size_t length=0,
                                     unsigned int flags=0);
```

`cudaStreamAttachMemAsync()` 函数将从 `ptr` 开始的 `length` 字节内存与指定的流关联。只要流中的所有操作都已完成，无论其他流是否处于活动状态，CPU 都可以访问该内存区域。实际上，这将活动 GPU 对托管内存区域的独占所有权限制为每个流的活动，而不是整个 GPU 的活动。最重要的是，如果一个分配没有与特定流关联，那么它对所有正在运行的内核都是可见的，无论它们属于哪个流。这是 `cudaMallocManaged()` 分配或 `__managed__` 变量的默认可见性；因此，简单的规则是，当任何内核正在运行时，CPU 可能无法触碰数据。

!!! note "注意"
    通过将分配与特定流关联，程序保证只有在该流中启动的内核才会触碰该数据。统一内存系统不执行错误检查。

!!! note "注意"
    除了允许更大的并发性之外，使用 `cudaStreamAttachMemAsync()` 还可以在统一内存系统内实现数据传输优化，这可能会影响延迟和其他开销。

以下示例展示了如何显式地将 `y` 与主机可访问性关联，从而允许 CPU 随时访问。（注意内核调用后没有 `cudaDeviceSynchronize()`。）现在，运行内核的 GPU 对 `y` 的访问将产生未定义的结果。

```cuda
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &y, 0, cudaMemAttachHost);
    cudaDeviceSynchronize();          // 等待主机关联完成。
    kernel<<< 1, 1, 0, stream1 >>>(); // 注意：在 stream1 中启动。
    y = 20;                           // 成功 —— 内核正在运行，但 "y"
                                      // 未与任何流关联。
    return  0;
}
```

#### 4.1.3.3.3. 关于多线程主机程序的更详细示例

`cudaStreamAttachMemAsync()` 的主要用途是使用 CPU 线程实现独立的任务并行性。通常，在此类程序中，CPU 线程会为其生成的所有工作创建自己的流，因为使用 CUDA 的 NULL 流会导致线程之间的依赖关系。托管数据对任何 GPU 流的默认全局可见性，使得在多线程程序中很难避免 CPU 线程之间的交互。因此，使用 `cudaStreamAttachMemAsync()` 函数将线程的托管分配与该线程自己的流关联起来，并且这种关联通常在线程的生命周期内不会改变。这样的程序只需添加一个对 `cudaStreamAttachMemAsync()` 的调用，即可为其数据访问使用统一内存：

```cuda
// This function performs some task, in its own , in its own private stream and can be run in parallel
void run_task(int *in, int *out, int length) {
    // Create a stream for us to use.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Allocate some managed data and associate with our stream.
    // Note the use of the host-attach flag to cudaMallocManaged();
    // we then associate the allocation with our stream so that
    // our GPU kernel launches can access it.
    int *data;
    cudaMallocManaged((void **)&data, length, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, data);
    cudaStreamSynchronize(stream);
    // Iterate on the data in some way, using both Host & Device.
    for(int i=0; i<N; i++) {
        transform<<< 100, 256, 0, stream >>>(in, data, length);
        cudaStreamSynchronize(stream);
        host_process(data, length);    // CPU uses managed data.
        convert<<< 100, 256, 0, stream >>>(out, data, length);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(data);
}
```

In this example, the allocation-stream association is established just once, and then data is used repeatedly by both the host and device. The result is much simpler code than occurs with explicitly copying data between host and device, although the result is the same.

The function `cudaMallocManaged()` specifies the cudaMemAttachHost flag, which creates an allocation that is initially invisible to device-side execution. (The default allocation would be visible to all GPU kernels on all streams.) This ensures that there is no accidental interaction with another threadâs execution in the interval between the data allocation and when the data is acquired for a specific stream.

Without this flag, a new allocation would be considered in-use on the GPU if a kernel launched by another thread happens to be running. This might impact the threadâs ability to access the newly allocated data from the CPU before it is able to explicitly attach it to a private stream. To enable safe independence between threads, therefore, allocations should be made specifying this flag.

An alternative would be to place a process-wide barrier across all threads after the allocation has been attached to the stream. This would ensure that all threads complete their data/stream associations before any kernels are launched, avoiding the hazard. A second barrier would be needed before the stream is destroyed because stream destruction causes allocations to revert to their default visibility. The `cudaMemAttachHost` flag exists both to simplify this process, and because it is not always possible to insert global barriers where required.

#### 4.1.3.3.4.Data Movement of Stream Associated Unified Memory

Memcpy()/Memset() with stream associated unified memory behaves different on devices where `concurrentManagedAccess` is not set, the following rules apply:

If `cudaMemcpyHostTo*` is specified and the source data is unified memory, then it will be accessed from the host if it is coherently accessible from the host in the copy stream [(1)](#um-legacy-memcpy-cit1); otherwise it will be accessed from the device. Similar rules apply to the destination when `cudaMemcpy*ToHost` is specified and the destination is unified memory.
如果指定了 `cudaMemcpyDeviceTo*` 并且源数据是统一内存，那么将从设备访问该数据。源数据必须在复制流中可从设备一致访问 [(2)](#um-legacy-memcpy-cit2)；否则，将返回错误。当指定了 `cudaMemcpy*ToDevice` 且目标是统一内存时，类似规则适用于目标。

如果指定了 `cudaMemcpyDefault`，那么统一内存将在以下情况下从主机访问：要么它无法在复制流中从设备一致访问 [(2)](#um-legacy-memcpy-cit2)，要么数据的首选位置是 `cudaCpuDeviceId` 并且它可以在复制流中从主机一致访问 [(1)](#um-legacy-memcpy-cit1)；否则，它将从设备访问。

当对统一内存使用 `cudaMemset*()` 时，数据必须在用于 `cudaMemset()` 操作的流中可从设备一致访问 [(2)](#um-legacy-memcpy-cit2)；否则，将返回错误。

当数据通过 `cudaMemcpy*` 或 `cudaMemset*` 从设备访问时，操作流被视为在 GPU 上处于活动状态。在此期间，如果 GPU 的设备属性 `concurrentManagedAccess` 值为零，则 CPU 访问与该流关联的数据或具有全局可见性的数据将导致段错误。程序必须适当地同步，以确保在从 CPU 访问任何关联数据之前操作已完成。

> 在给定流中可从主机一致访问意味着该内存既不具有全局可见性，也不与给定流关联。

> 在给定流中可从设备一致访问意味着该内存要么具有全局可见性，要么与给定流关联。

## 4.1.4. 性能提示

性能提示允许程序员向 CUDA 提供更多关于统一内存使用的信息。CUDA 使用性能提示来更有效地管理内存并提高应用程序性能。性能提示绝不会影响应用程序的正确性。性能提示只影响性能。

!!! note "注意"
    应用程序应仅在性能提示能提高性能时使用它们。

性能提示可用于任何统一内存分配，包括 CUDA 托管内存。在具有完整 CUDA 统一内存支持的系统上，性能提示可应用于所有系统分配的内存。

### 4.1.4.1. 数据预取

`cudaMemPrefetchAsync` API 是一个异步的流序 API，可以将数据迁移到更靠近指定处理器的位置。数据在预取期间仍可被访问。迁移操作直到流中所有先前的操作完成后才开始，并在流中任何后续操作开始前完成。

```cuda
cudaError_t cudaMemPrefetchAsync(const void *devPtr,
                                 size_t count,
                                 struct cudaMemLocation location,
                                 unsigned int flags,
                                 cudaStream_t stream=0);
```
当预取任务在给定的 `stream` 中执行时，包含 `[devPtr, devPtr + count)` 的内存区域可能会被迁移到目标设备 `location.id`（如果 `location.type` 是 `cudaMemLocationTypeDevice`），或者迁移到 CPU（如果 `location.type` 是 `cudaMemLocationTypeHost`）。有关 `flags` 的详细信息，请参阅当前的 [CUDA 运行时 API 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)。

考虑以下简单的代码示例：

系统分配器

```cuda
void test_prefetch_sam(const cudaStream_t& s) {
  // 在 CPU 上初始化数据
  char *data = (char*)malloc(dataSizeBytes);
  init_data(data, dataSizeBytes);                                     
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

  // 鼓励数据在使用前移动到 GPU
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);      

  // 在 GPU 上使用数据
  const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);  

  // 鼓励数据移回 CPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);      
  
  cudaStreamSynchronize(s);

  // 在 CPU 上使用数据
  use_data(data, dataSizeBytes);                                      
  free(data);
}
```

托管内存

```cuda
void test_prefetch_managed(const cudaStream_t& s) {
  // 在 CPU 上初始化数据
  char *data;
  cudaMallocManaged(&data, dataSizeBytes);
  init_data(data, dataSizeBytes);                                     
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

  // 鼓励数据在使用前移动到 GPU
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  // 在 GPU 上使用数据
  const uinsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes); 

  // 鼓励数据移回 CPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s); 

  cudaStreamSynchronize(s);

  // 在 CPU 上使用数据
  use_data(data, dataSizeBytes);
  cudaFree(data);
}
```

### 4.1.4.2. 数据使用提示

当多个处理器同时访问相同数据时，可以使用 `cudaMemAdvise` 来提示 `[devPtr, devPtr + count)` 处的数据将如何被访问：

```cuda
cudaError_t cudaMemAdvise(const void *devPtr,
                          size_t count,
                          enum cudaMemoryAdvise advice,
                          struct cudaMemLocation location);
```

以下示例展示了如何使用 `cudaMemAdvise`：

```cuda
  init_data(data, dataSizeBytes);                                     
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

  // 鼓励数据在使用前移动到 GPU
  const unsigned int flags = 0;
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

  // 在 GPU 上使用数据
  const uinsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
  mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes); 

  // 鼓励数据移回 CPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s); 

  cudaStreamSynchronize(s);

  // 在 CPU 上使用数据
  use_data(data, dataSizeBytes);
  cudaFree(data);
}
// test-prefetch-managed-end

static const int maxDevices = 1;
static const int maxOuterLoopIter = 3;
static const int maxInnerLoopIter = 4;

// test-advise-managed-begin
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB
```
其中 `advice` 可接受以下值：

- cudaMemAdviseSetReadMostly：表示该数据区域主要进行读取操作，仅偶尔写入。通常，这允许在该区域以写入带宽换取读取带宽。

- cudaMemAdviseSetPreferredLocation：此提示将数据的首选位置设置为指定设备的物理内存。该提示鼓励系统将数据保留在首选位置，但不保证一定如此。当 `location.type` 传入 `cudaMemLocationTypeHost` 时，会将首选位置设置为 CPU 内存。其他提示（如 `cudaMemPrefetchAsync`）可能会覆盖此提示，并允许内存从其首选位置迁移。

- cudaMemAdviseSetAccessedBy：在某些系统中，在从给定处理器访问数据之前建立内存映射可能对性能有益。此提示告诉系统，当 `location.type` 为 `cudaMemLocationTypeDevice` 时，`location.id` 指定的设备将频繁访问该数据，从而使系统能够判断建立这些映射是值得的。此提示并不暗示数据应驻留在何处，但可以与 `cudaMemAdviseSetPreferredLocation` 结合使用来指定位置。在硬件一致性系统上，此提示会启用访问计数器迁移，详见访问计数器迁移。

每个建议也可以通过使用以下值之一来取消设置：`cudaMemAdviseUnsetReadMostly`、`cudaMemAdviseUnsetPreferredLocation` 和 `cudaMemAdviseUnsetAccessedBy`。

以下示例展示了如何使用 `cudaMemAdvise`：

系统分配器

```cuda
void test_advise_sam(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB
  
  // 使用 malloc 或 cudaMallocManaged 分配内存
  dataPtr = (char*)malloc(dataSize);

  // 在内存区域上设置建议
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // 每次外层循环迭代由 CPU 写入数据
    init_data(dataPtr, dataSize);

    // 通过预取使数据对所有 GPU 可用。
    // 此处的预取会导致数据的读取复制，而不是数据迁移
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      const unsigned int flags = 0;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
    }

    // 内核仅在内层循环中读取此数据
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }

  free(dataPtr);
}
```

托管内存

```cuda
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB

  // 使用 cudaMallocManaged 分配内存
  //（在完全支持 CUDA 统一内存的系统上，也可以使用 malloc）
  cudaMallocManaged(&dataPtr, dataSize);

  // 在内存区域上设置建议
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // 每次外层循环迭代由 CPU 写入数据
    init_data(dataPtr, dataSize);

    // 通过预取使数据对所有 GPU 可用。
    // 此处的预取会导致数据的读取复制，而不是数据迁移
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      const unsigned int flags = 0;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
    }

    // 内核仅在内层循环中读取此数据
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }
  
  cudaFree(dataPtr);
}
```
### 4.1.4.3. 内存丢弃

`cudaMemDiscardBatchAsync` API 允许应用程序通知 CUDA 运行时，指定内存范围的内容不再有用。统一内存驱动程序会执行自动内存传输，这是由于基于缺页的迁移或内存逐出，以支持设备内存超额订阅。这些自动内存传输有时可能是冗余的，这会严重降低性能。将一个地址范围标记为“丢弃”将通知统一内存驱动程序，应用程序已使用完该范围内的内容，无需在预取或页面逐出时迁移此数据以为其他分配腾出空间。在没有后续写入访问或预取的情况下读取已丢弃的页面将产生不确定的值。而丢弃操作之后的任何新写入都保证能被后续的读取访问看到。对正在被丢弃的地址范围进行并发访问或预取将导致未定义行为。

```cuda
cudaError_t cudaMemDiscardBatchAsync(void **dptrs,
                                    size_t *sizes,
                                    size_t count,
                                    unsigned long long flags,
                                    cudaStream_t stream);
```

该函数对 `dptrs` 和 `sizes` 数组中指定的地址范围执行批量内存丢弃操作。两个数组的长度必须与 `count` 指定的相同。每个内存范围必须引用通过 `cudaMallocManaged` 分配或通过 `__managed__` 变量声明的托管内存。

`cudaMemDiscardAndPrefetchBatchAsync` API 结合了丢弃和预取操作。调用 `cudaMemDiscardAndPrefetchBatchAsync` 在语义上等同于先调用 `cudaMemDiscardBatchAsync` 再调用 `cudaMemPrefetchBatchAsync`，但更优化。这在应用程序需要内存位于目标位置但不需要内存内容时非常有用。

```cuda
cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void **dptrs,
                                               size_t *sizes,
                                               size_t count,
                                               struct cudaMemLocation *prefetchLocs,
                                               size_t *prefetchLocIdxs,
                                               size_t numPrefetchLocs,
                                               unsigned long long flags,
                                               cudaStream_t stream);
```

`prefetchLocs` 数组指定了预取的目标位置，而 `prefetchLocIdxs` 指示每个预取位置应用于哪些操作。例如，如果一个批次有 10 个操作，其中前 6 个应预取到一个位置，而剩余 4 个预取到另一个位置，那么 `numPrefetchLocs` 应为 2，`prefetchLocIdxs` 应为 {0, 6}，并且 `prefetchLocs` 应包含两个目标位置。

**重要注意事项：**
- 在未进行后续写入或预取的情况下，从已丢弃的范围读取将返回不确定的值
- 通过对该范围进行写入或通过 `cudaMemPrefetchAsync` 进行预取，可以撤销丢弃操作
- 任何与丢弃操作同时发生的读取、写入或预取操作将导致未定义行为
- 所有设备必须具有非零的 `cudaDevAttrConcurrentManagedAccess` 属性值

### 4.1.4.4. 查询托管内存的数据使用属性

程序可以通过以下 API 查询在 CUDA 托管内存上通过 `cudaMemAdvise` 或 `cudaMemPrefetchAsync` 分配的内存范围属性：

```cuda
cudaMemRangeGetAttribute(void *data,
                         size_t dataSize,
                         enum cudaMemRangeAttribute attribute,
                         const void *devPtr,
                         size_t count);
```

此函数查询起始于 `devPtr`、大小为 `count` 字节的内存范围的属性。内存范围必须引用通过 `cudaMallocManaged` 分配或通过 `__managed__` 变量声明的托管内存。可以查询以下属性：

- `cudaMemRangeAttributeReadMostly`：如果整个内存范围设置了 `cudaMemAdviseSetReadMostly` 属性，则返回 1，否则返回 0。
- `cudaMemRangeAttributePreferredLocation`：如果整个内存范围将相应处理器设为首选位置，则返回 GPU 设备 ID 或 `cudaCpuDeviceId`，否则返回 `cudaInvalidDeviceId`。
  应用程序可以使用此查询 API，根据托管指针的首选位置属性，决定通过 CPU 还是 GPU 暂存数据。
  请注意，查询时内存范围的实际位置可能与首选位置不同。
- `cudaMemRangeAttributeAccessedBy`：将返回已为该内存范围设置该建议的设备列表。
- `cudaMemRangeAttributeLastPrefetchLocation`：将返回使用 `cudaMemPrefetchAsync` 显式预取内存范围的最后一个位置。请注意，这仅返回应用程序请求预取内存范围的最后一个位置。它不指示到该位置的预取操作是否已完成甚至已开始。
- `cudaMemRangeAttributePreferredLocationType`：返回首选位置的位置类型，其值如下：
  - `cudaMemLocationTypeDevice`：如果内存范围中的所有页具有相同的 GPU 作为其首选位置，
  - `cudaMemLocationTypeHost`：如果内存范围中的所有页具有 CPU 作为其首选位置，
  - `cudaMemLocationTypeHostNuma`：如果内存范围中的所有页具有相同的主机 NUMA 节点 ID 作为其首选位置，
  - `cudaMemLocationTypeInvalid`：如果所有页不具有相同的首选位置，或者某些页根本没有首选位置。
- `cudaMemRangeAttributePreferredLocationId`：如果对相同地址范围的 `cudaMemRangeAttributePreferredLocationType` 查询返回 `cudaMemLocationTypeDevice`，则返回设备序号。如果首选位置类型是主机 NUMA 节点，则返回主机 NUMA 节点 ID。否则，应忽略该 ID。
-   **cudaMemRangeAttributeLastPrefetchLocationType**：返回内存范围中所有页面通过 `cudaMemPrefetchAsync` 显式预取到的最后一个位置类型。返回以下值：
    -   `cudaMemLocationTypeDevice`：如果内存范围中的所有页面都预取到了同一个 GPU。
    -   `cudaMemLocationTypeHost`：如果内存范围中的所有页面都预取到了 CPU。
    -   `cudaMemLocationTypeHostNuma`：如果内存范围中的所有页面都预取到了同一个主机 NUMA 节点 ID。
    -   `cudaMemLocationTypeInvalid`：如果并非所有页面都预取到了同一位置，或者某些页面从未被预取过。
-   **cudaMemRangeAttributeLastPrefetchLocationId**：如果对相同地址范围的 `cudaMemRangeAttributeLastPrefetchLocationType` 查询返回 `cudaMemLocationTypeDevice`，则此 ID 将是一个有效的设备序号；如果返回 `cudaMemLocationTypeHostNuma`，则此 ID 将是一个有效的主机 NUMA 节点 ID。否则，应忽略此 ID。

此外，可以通过使用相应的 `cudaMemRangeGetAttributes` 函数来查询多个属性。

### 4.1.4.5. GPU 内存超额订阅

统一内存使应用程序能够*超额订阅*任何单个处理器的内存：换句话说，它们可以分配和共享大于系统中任何单个处理器内存容量的数组，从而能够对单个 GPU 无法容纳的数据集进行核外处理，而无需显著增加编程模型的复杂性。

此外，可以通过使用相应的 `cudaMemRangeGetAttributes` 函数来查询多个属性。

 本页