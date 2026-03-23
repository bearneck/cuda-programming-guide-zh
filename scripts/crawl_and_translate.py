#!/usr/bin/env python3
"""
CUDA Programming Guide 抓取 + DeepSeek 翻译脚本
支持多线程并行翻译，保留原始格式/代码块/链接结构
"""

import os
import re
import time
import json
import requests
import concurrent.futures
from bs4 import BeautifulSoup
from pathlib import Path

# ==================== 配置 ====================
DEEPSEEK_API_KEY = "sk-2b575d01b6fe4db593a0ed83b06b7c4c"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
BASE_URL = "https://docs.nvidia.com/cuda/cuda-programming-guide/"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "chapters"
MAX_WORKERS = 4
CHUNK_SIZE = 3000

# ==================== 章节定义（基于真实 URL 结构）====================
CHAPTERS = [
    # Part 1: Introduction
    {"id": "part1",       "url": BASE_URL + "part1.html",                                          "filename": "part1-intro.md",              "title": "第一部分：CUDA 简介"},
    {"id": "intro",       "url": BASE_URL + "01-introduction/introduction.html",                   "filename": "01-introduction.md",          "title": "1.1 简介"},
    {"id": "prog-model",  "url": BASE_URL + "01-introduction/programming-model.html",              "filename": "02-programming-model.md",     "title": "1.2 编程模型"},
    {"id": "cuda-plat",   "url": BASE_URL + "01-introduction/cuda-platform.html",                  "filename": "03-cuda-platform.md",         "title": "1.3 CUDA 平台"},
    # Part 2: Programming GPUs
    {"id": "part2",       "url": BASE_URL + "part2.html",                                          "filename": "part2-basics.md",             "title": "第二部分：CUDA GPU 编程"},
    {"id": "cuda-cpp",    "url": BASE_URL + "02-basics/intro-to-cuda-cpp.html",                    "filename": "04-cuda-cpp-intro.md",        "title": "2.1 CUDA C++ 入门"},
    {"id": "simt",        "url": BASE_URL + "02-basics/writing-cuda-kernels.html",                 "filename": "05-simt-kernels.md",          "title": "2.2 编写 CUDA SIMT 内核"},
    {"id": "async-exec",  "url": BASE_URL + "02-basics/asynchronous-execution.html",               "filename": "06-async-execution.md",       "title": "2.3 异步执行"},
    {"id": "memory",      "url": BASE_URL + "02-basics/understanding-memory.html",                 "filename": "07-unified-memory.md",        "title": "2.4 统一内存与系统内存"},
    {"id": "nvcc",        "url": BASE_URL + "02-basics/nvcc.html",                                 "filename": "08-nvcc.md",                  "title": "2.5 NVCC 编译器"},
    # Part 3: Advanced CUDA
    {"id": "part3",       "url": BASE_URL + "part3.html",                                          "filename": "part3-advanced.md",           "title": "第三部分：高级 CUDA"},
    {"id": "adv-host",    "url": BASE_URL + "03-advanced/advanced-host-programming.html",          "filename": "09-advanced-apis.md",         "title": "3.1 高级 CUDA API 与特性"},
    {"id": "adv-kernel",  "url": BASE_URL + "03-advanced/advanced-kernel-programming.html",        "filename": "10-advanced-kernel.md",       "title": "3.2 高级内核编程"},
    {"id": "driver-api",  "url": BASE_URL + "03-advanced/driver-api.html",                         "filename": "11-driver-api.md",            "title": "3.3 CUDA 驱动 API"},
    {"id": "multi-gpu",   "url": BASE_URL + "03-advanced/multi-gpu-systems.html",                  "filename": "12-multi-gpu.md",             "title": "3.4 多 GPU 系统编程"},
    {"id": "features",    "url": BASE_URL + "03-advanced/feature-survey.html",                     "filename": "13-feature-survey.md",        "title": "3.5 CUDA 特性全览"},
    # Part 4: Special Topics
    {"id": "part4",       "url": BASE_URL + "part4.html",                                          "filename": "part4-special.md",            "title": "第四部分：CUDA 高级特性"},
    {"id": "um",          "url": BASE_URL + "04-special-topics/unified-memory.html",               "filename": "14-unified-memory.md",        "title": "4.1 统一内存"},
    {"id": "graphs",      "url": BASE_URL + "04-special-topics/cuda-graphs.html",                  "filename": "15-cuda-graphs.md",           "title": "4.2 CUDA Graphs"},
    {"id": "stream-mem",  "url": BASE_URL + "04-special-topics/stream-ordered-memory-allocation.html", "filename": "16-stream-memory.md",   "title": "4.3 流有序内存分配"},
    {"id": "coop-groups", "url": BASE_URL + "04-special-topics/cooperative-groups.html",           "filename": "17-cooperative-groups.md",    "title": "4.4 协作组"},
    {"id": "dep-launch",  "url": BASE_URL + "04-special-topics/programmatic-dependent-launch.html","filename": "18-dependent-launch.md",      "title": "4.5 可编程依赖启动与同步"},
    {"id": "green-ctx",   "url": BASE_URL + "04-special-topics/green-contexts.html",               "filename": "19-green-contexts.md",        "title": "4.6 绿色上下文"},
    {"id": "lazy-load",   "url": BASE_URL + "04-special-topics/lazy-loading.html",                 "filename": "20-lazy-loading.md",          "title": "4.7 延迟加载"},
    {"id": "error-log",   "url": BASE_URL + "04-special-topics/error-log-management.html",         "filename": "21-error-log.md",             "title": "4.8 错误日志管理"},
    {"id": "async-bar",   "url": BASE_URL + "04-special-topics/async-barriers.html",               "filename": "22-async-barriers.md",        "title": "4.9 异步屏障"},
    {"id": "pipelines",   "url": BASE_URL + "04-special-topics/pipelines.html",                    "filename": "23-pipelines.md",             "title": "4.10 流水线"},
    {"id": "async-copy",  "url": BASE_URL + "04-special-topics/async-copies.html",                 "filename": "24-async-copies.md",          "title": "4.11 异步数据拷贝"},
    {"id": "cluster",     "url": BASE_URL + "04-special-topics/cluster-launch-control.html",       "filename": "25-cluster-launch.md",        "title": "4.12 集群启动控制"},
    {"id": "l2-cache",    "url": BASE_URL + "04-special-topics/l2-cache-control.html",             "filename": "26-l2-cache.md",              "title": "4.13 L2 缓存控制"},
    {"id": "mem-sync",    "url": BASE_URL + "04-special-topics/memory-sync-domains.html",          "filename": "27-memory-sync.md",           "title": "4.14 内存同步域"},
    {"id": "ipc",         "url": BASE_URL + "04-special-topics/inter-process-communication.html",  "filename": "28-ipc.md",                   "title": "4.15 进程间通信"},
    {"id": "virt-mem",    "url": BASE_URL + "04-special-topics/virtual-memory-management.html",    "filename": "29-virtual-memory.md",        "title": "4.16 虚拟内存管理"},
    {"id": "ext-gpu-mem", "url": BASE_URL + "04-special-topics/extended-gpu-memory.html",          "filename": "30-extended-gpu-memory.md",   "title": "4.17 扩展 GPU 内存"},
    {"id": "dyn-para",    "url": BASE_URL + "04-special-topics/dynamic-parallelism.html",          "filename": "31-dynamic-parallelism.md",   "title": "4.18 CUDA 动态并行"},
    {"id": "interop",     "url": BASE_URL + "04-special-topics/graphics-interop.html",             "filename": "32-interop.md",               "title": "4.19 CUDA 与图形 API 互操作"},
    {"id": "drv-entry",   "url": BASE_URL + "04-special-topics/driver-entry-point-access.html",    "filename": "33-driver-entry.md",          "title": "4.20 驱动入口点访问"},
    # Part 5: Appendices
    {"id": "part5",       "url": BASE_URL + "part5.html",                                          "filename": "part5-appendices.md",         "title": "第五部分：技术附录"},
    {"id": "cc",          "url": BASE_URL + "05-appendices/compute-capabilities.html",             "filename": "34-compute-capabilities.md",  "title": "5.1 计算能力"},
    {"id": "env-vars",    "url": BASE_URL + "05-appendices/environment-variables.html",            "filename": "35-env-variables.md",         "title": "5.2 CUDA 环境变量"},
    {"id": "cpp-support", "url": BASE_URL + "05-appendices/cpp-language-support.html",             "filename": "36-cpp-support.md",           "title": "5.3 C++ 语言支持"},
    {"id": "cpp-ext",     "url": BASE_URL + "05-appendices/cpp-language-extensions.html",          "filename": "37-language-extensions.md",   "title": "5.4 C/C++ 语言扩展"},
    {"id": "fp",          "url": BASE_URL + "05-appendices/mathematical-functions.html",           "filename": "38-floating-point.md",        "title": "5.5 浮点运算"},
    {"id": "dev-api",     "url": BASE_URL + "05-appendices/device-callable-apis.html",             "filename": "39-device-apis.md",           "title": "5.6 设备端可调用 API"},
    {"id": "mem-model",   "url": BASE_URL + "05-appendices/cuda-cpp-memory-model.html",            "filename": "40-memory-model.md",          "title": "5.7 CUDA C++ 内存模型"},
    {"id": "exec-model",  "url": BASE_URL + "05-appendices/cuda-cpp-execution-model.html",         "filename": "41-execution-model.md",       "title": "5.8 CUDA C++ 执行模型"},
]


# ==================== 爬虫函数 ====================
def fetch_page(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[ERROR] 抓取 {url} 失败: {e}")
        return ""


def html_to_markdown(element) -> str:
    """将 BS4 元素转换为 Markdown，保护代码块"""
    result = []

    def process(el, list_depth=0, list_type=None, list_idx=None):
        tag = getattr(el, 'name', None)
        if tag is None:
            text = str(el)
            if text.strip():
                result.append(text.replace('\n', ' '))
            return

        if tag in ['script', 'style', 'nav', 'footer', 'head']:
            return

        # 标题
        if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(tag[1])
            text = el.get_text(strip=True)
            # 移除末尾的 # 锚点符号
            text = text.rstrip('#').strip()
            if text:
                result.append(f"\n{'#' * level} {text}\n")
            return

        # 代码块 — 不翻译
        if tag == 'pre':
            code_el = el.find('code')
            code = (code_el or el).get_text()
            lang = "cuda"
            if code_el:
                for cls in (code_el.get('class') or []):
                    if cls.startswith('language-'):
                        lang = cls[9:]
                        break
            result.append(f"\n```{lang}\n{code.rstrip()}\n```\n")
            return

        # 行内代码
        if tag == 'code':
            if el.parent and el.parent.name == 'pre':
                return
            result.append(f"`{el.get_text()}`")
            return

        # 段落
        if tag == 'p':
            parts = []
            for child in el.children:
                ctag = getattr(child, 'name', None)
                if ctag == 'code':
                    parts.append(f"`{child.get_text()}`")
                elif ctag == 'a':
                    href = child.get('href', '#')
                    txt = child.get_text(strip=True)
                    parts.append(f"[{txt}]({href})")
                elif ctag in ['strong', 'b']:
                    parts.append(f"**{child.get_text(strip=True)}**")
                elif ctag in ['em', 'i']:
                    parts.append(f"*{child.get_text(strip=True)}*")
                elif ctag in ['sub', 'sup']:
                    parts.append(child.get_text(strip=True))
                elif ctag is None:
                    t = str(child).replace('\n', ' ')
                    if t.strip():
                        parts.append(t)
                else:
                    t = child.get_text(separator=' ', strip=True)
                    if t:
                        parts.append(t)
            text = ''.join(parts).strip()
            if text:
                result.append(f"\n{text}\n")
            return

        # 无序列表
        if tag == 'ul':
            result.append("")
            for li in el.find_all('li', recursive=False):
                indent = "  " * list_depth
                li_text = li.get_text(separator=' ', strip=True)
                result.append(f"{indent}- {li_text}")
            result.append("")
            return

        # 有序列表
        if tag == 'ol':
            result.append("")
            for i, li in enumerate(el.find_all('li', recursive=False), 1):
                indent = "  " * list_depth
                li_text = li.get_text(separator=' ', strip=True)
                result.append(f"{indent}{i}. {li_text}")
            result.append("")
            return

        # 表格
        if tag == 'table':
            rows = el.find_all('tr')
            if not rows:
                return
            table_lines = []
            # 提取所有行
            all_rows = []
            for row in rows:
                cells = [c.get_text(separator=' ', strip=True) for c in row.find_all(['th', 'td'])]
                if cells:
                    all_rows.append(cells)
            if not all_rows:
                return
            # 统一列数
            max_cols = max(len(r) for r in all_rows)
            for i, row in enumerate(all_rows):
                while len(row) < max_cols:
                    row.append("")
                row_str = "| " + " | ".join(row) + " |"
                table_lines.append(row_str)
                if i == 0:
                    table_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
            result.append("\n" + "\n".join(table_lines) + "\n")
            return

        # 分隔线
        if tag == 'hr':
            result.append("\n---\n")
            return

        # 块引用
        if tag == 'blockquote':
            for line in el.get_text(separator='\n', strip=True).split('\n'):
                result.append(f"> {line}")
            result.append("")
            return

        # 强调
        if tag in ['strong', 'b']:
            text = el.get_text(strip=True)
            if text:
                result.append(f"**{text}**")
            return

        if tag in ['em', 'i']:
            text = el.get_text(strip=True)
            if text:
                result.append(f"*{text}*")
            return

        # div/section/article — 递归处理子元素
        for child in el.children:
            process(child, list_depth)

    process(element)
    # 合并、清理多余空行
    text = '\n'.join(result)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_main_content(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # 移除干扰元素
    for sel in ["nav", "header", "footer", ".breadcrumb", ".feedback",
                ".sidebar", ".toc", "[role='navigation']", ".prevnext",
                "script", "style", ".admonition-title"]:
        for el in soup.select(sel):
            el.decompose()

    # 找主内容区
    main = (
        soup.find("main")
        or soup.find("div", class_=re.compile(r"\bcontent\b|\bmain\b", re.I))
        or soup.find("article")
        or soup.find("body")
    )
    return html_to_markdown(main) if main else ""


# ==================== DeepSeek 翻译 ====================
TRANSLATE_SYSTEM = """你是专业的技术文档翻译专家，专精于 NVIDIA CUDA 并行计算领域。

翻译规则（严格遵守）：
1. 将英文内容翻译为流畅、准确的简体中文
2. 严格保留所有代码块（```...```）内的内容，完全不翻译
3. 严格保留所有行内代码（`...`）内的内容，完全不翻译
4. 严格保留所有 Markdown 格式符号：# ## ### **粗体** *斜体* - 列表 1. 有序 | 表格 | [链接](url)
5. 专业术语翻译规范：
   kernel→内核, thread→线程, block→线程块, grid→线程网格,
   warp→线程束（warp）, shared memory→共享内存, global memory→全局内存,
   constant memory→常量内存, texture memory→纹理内存,
   device→设备, host→主机, stream→流, event→事件,
   compute capability→计算能力, occupancy→占用率,
   register→寄存器, latency→延迟, throughput→吞吐量,
   SM (Streaming Multiprocessor)→流式多处理器（SM）,
   PTX/SASS/NVCC/CUDA/GPU/CPU→保留原文不译
6. 只输出翻译结果，不添加任何说明文字
7. 保持原文段落结构和换行"""


def translate_with_deepseek(text: str, retries: int = 3) -> str:
    if not text.strip():
        return text
    for attempt in range(retries):
        try:
            resp = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": TRANSLATE_SYSTEM},
                        {"role": "user", "content": f"请翻译以下 CUDA 技术文档：\n\n{text}"}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096
                },
                timeout=90
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [WARN] 翻译失败 attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return text  # 返回原文


def split_chunks(text: str, size: int = CHUNK_SIZE) -> list:
    """按段落分块，不打断代码块"""
    chunks, cur, cur_len, in_code = [], [], 0, False
    for line in text.split('\n'):
        if line.strip().startswith('```'):
            in_code = not in_code
        cur.append(line)
        cur_len += len(line) + 1
        if cur_len >= size and not in_code:
            chunks.append('\n'.join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append('\n'.join(cur))
    return chunks


def translate_document(content: str) -> str:
    if not content.strip():
        return content
    chunks = split_chunks(content, CHUNK_SIZE)
    translated = []
    print(f"  → {len(chunks)} 个翻译块")
    for i, chunk in enumerate(chunks, 1):
        print(f"  翻译 {i}/{len(chunks)}...", end=' ', flush=True)
        stripped = chunk.strip()
        # 纯代码块不翻译
        if stripped.startswith('```') and stripped.count('```') >= 2:
            translated.append(chunk)
            print("(代码块，跳过)")
        else:
            t = translate_with_deepseek(chunk)
            translated.append(t)
            print("✓")
        time.sleep(0.3)
    return '\n'.join(translated)


# ==================== 主处理流程 ====================
def process_chapter(chapter: dict) -> dict:
    cid = chapter["id"]
    url = chapter["url"]
    filename = chapter["filename"]
    title = chapter["title"]
    output_path = OUTPUT_DIR / filename

    if output_path.exists() and output_path.stat().st_size > 200:
        print(f"[SKIP] {title}")
        return {"id": cid, "status": "skipped", "title": title}

    print(f"\n{'='*50}")
    print(f"[处理] {title}")
    print(f"  URL: {url}")

    html = fetch_page(url)
    if not html:
        return {"id": cid, "status": "error", "title": title}

    content = extract_main_content(html)
    if not content.strip():
        print(f"  [WARN] 内容为空")
        content = BeautifulSoup(html, "lxml").get_text(separator='\n', strip=True)[:6000]

    print(f"  内容长度: {len(content)} 字符")
    translated = translate_document(content)

    header = f"""# {title}

> 本文档为 [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 官方文档中文翻译版，基于最新版本翻译。
>
> 原文地址：[{url}]({url})

---

"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + translated, encoding="utf-8")
    print(f"  [✓] 已保存: {output_path.name}")
    return {"id": cid, "status": "success", "title": title}


def main():
    print("=" * 60)
    print("  CUDA Programming Guide 汉化工程  ")
    print("  共", len(CHAPTERS), "个章节  ")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    # 使用线程池并行处理，但 API 调用有速率限制，用 4 线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chapter, ch): ch for ch in CHAPTERS}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                ch = futures[future]
                print(f"[ERROR] {ch['title']}: {e}")
                results.append({"id": ch["id"], "status": "error", "title": ch["title"]})

    ok = sum(1 for r in results if r["status"] == "success")
    skip = sum(1 for r in results if r["status"] == "skipped")
    err = sum(1 for r in results if r["status"] == "error")
    print(f"\n{'='*60}")
    print(f"完成！✓成功:{ok}  ⏭跳过:{skip}  ✗失败:{err}")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    main()
