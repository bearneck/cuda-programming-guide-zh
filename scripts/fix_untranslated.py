#!/usr/bin/env python3
"""
补翻脚本 v2：检测并修复 Markdown 文件中的英文残留段落
逻辑：按空行分段，逐段判断是否为英文，若是则翻译
"""

import re
import sys
import time
import requests
from pathlib import Path

DEEPSEEK_API_KEY = "sk-2b575d01b6fe4db593a0ed83b06b7c4c"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
CHAPTERS_DIR = Path(__file__).parent.parent / "docs" / "chapters"

TRANSLATE_SYSTEM = """你是专业的技术文档翻译专家，专精于 NVIDIA CUDA 并行计算领域。

翻译规则（严格遵守）：
1. 将英文内容翻译为流畅、准确的简体中文
2. 严格保留所有代码块（```...```）内的内容，完全不翻译
3. 严格保留所有行内代码（`...`）内的内容，完全不翻译
4. 严格保留 LaTeX 数学公式 \\(...\\) 完全不修改
5. 严格保留图片引用 ![alt](path) 格式不变
6. 严格保留所有 Markdown 格式符号：# ## ### **粗体** *斜体* - 列表 | 表格 | [链接](url)
7. 专业术语：kernel→内核, thread→线程, block→线程块, grid→线程网格,
   warp→线程束（warp）, shared memory→共享内存, global memory→全局内存,
   device→设备, host→主机, stream→流, compute capability→计算能力,
   occupancy→占用率, SM→流式多处理器（SM）, PTX/SASS/NVCC/CUDA/GPU→保留不译
8. 只输出翻译结果，不添加任何说明文字"""


def is_english_text(text: str) -> bool:
    """判断文本是否主要是英文"""
    clean = re.sub(r'`[^`\n]+`', '', text)
    clean = re.sub(r'https?://\S+', '', clean)
    clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)
    clean = re.sub(r'[^a-zA-Z\u4e00-\u9fff]', ' ', clean)
    clean = clean.strip()
    if not clean:
        return False
    ascii_alpha = sum(1 for c in clean if c.isalpha() and ord(c) < 128)
    cjk = sum(1 for c in clean if '\u4e00' <= c <= '\u9fff')
    total = ascii_alpha + cjk
    if total < 20:
        return False
    return ascii_alpha / total > 0.75 and ascii_alpha > 40


def translate_text(text: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = requests.post(
                DEEPSEEK_API_URL,
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": TRANSLATE_SYSTEM},
                        {"role": "user", "content": f"翻译以下 CUDA 文档段落：\n\n{text}"}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048
                },
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    [WARN] {attempt+1}/{retries}: {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return text


def split_into_blocks(content: str) -> list:
    """
    将 Markdown 内容按语义块分割。
    每个块是连续的非空行组成的段落/代码块/标题等。
    返回 list of (block_text, is_code_block)
    """
    blocks = []
    lines = content.split('\n')
    current = []
    in_code = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('```'):
            if not in_code:
                # 开始代码块：先保存当前段落
                if current:
                    blocks.append(('\n'.join(current), False))
                    current = []
                in_code = True
                current = [line]
            else:
                # 结束代码块
                current.append(line)
                blocks.append(('\n'.join(current), True))
                current = []
                in_code = False
        elif in_code:
            current.append(line)
        elif not stripped:
            # 空行 = 段落分隔
            if current:
                blocks.append(('\n'.join(current), False))
                current = []
            blocks.append(('', False))  # 保留空行
        else:
            current.append(line)

    if current:
        blocks.append(('\n'.join(current), in_code))

    return blocks


def fix_file(filepath: Path) -> int:
    """修复文件中的英文残留，返回修复数量"""
    content = filepath.read_text(encoding='utf-8')
    blocks = split_into_blocks(content)

    fixed_blocks = []
    fix_count = 0

    for block_text, is_code in blocks:
        if is_code or not block_text.strip():
            fixed_blocks.append(block_text)
            continue

        stripped = block_text.strip()

        # 跳过标题、引用、图片链接、表格行
        first_line = stripped.split('\n')[0]
        if (first_line.startswith('#') or first_line.startswith('>') or
                first_line.startswith('!') or first_line.startswith('|') or
                first_line.startswith('---')):
            fixed_blocks.append(block_text)
            continue

        if is_english_text(stripped):
            preview = stripped[:60].replace('\n', ' ')
            print(f"  → 翻译 ({len(stripped)}字符): {preview}...", flush=True)
            translated = translate_text(stripped)
            fixed_blocks.append(translated)
            fix_count += 1
            time.sleep(0.2)
        else:
            fixed_blocks.append(block_text)

    if fix_count > 0:
        new_content = '\n'.join(fixed_blocks)
        filepath.write_text(new_content, encoding='utf-8')
        print(f"  [✓] {filepath.name}: 修复 {fix_count} 处", flush=True)
    else:
        print(f"  [OK] {filepath.name}: 无需修复", flush=True)

    return fix_count


PRIORITY_FILES = [
    "37-language-extensions.md",
    "39-device-apis.md",
    "16-stream-memory.md",
    "06-async-execution.md",
    "15-cuda-graphs.md",
    "14-unified-memory.md",
    "19-green-contexts.md",
    "29-virtual-memory.md",
    "04-cuda-cpp-intro.md",
    "41-execution-model.md",
    "24-async-copies.md",
    "33-driver-entry.md",
    "36-cpp-support.md",
    "09-advanced-apis.md",
    "11-driver-api.md",
    "30-extended-gpu-memory.md",
    "12-multi-gpu.md",
    "32-interop.md",
    "35-env-variables.md",
]


def main():
    # 支持命令行指定单个文件
    target = sys.argv[1] if len(sys.argv) > 1 else None
    files = [target] if target else PRIORITY_FILES

    print("=" * 60, flush=True)
    print(f"  英文残留补翻脚本 v2 - 共 {len(files)} 个文件", flush=True)
    print("=" * 60, flush=True)

    total_fixes = 0
    for fname in files:
        fpath = CHAPTERS_DIR / fname
        if not fpath.exists():
            print(f"[SKIP] {fname} 不存在", flush=True)
            continue
        print(f"\n{'='*50}", flush=True)
        print(f"[处理] {fname}", flush=True)
        try:
            n = fix_file(fpath)
            total_fixes += n
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}", flush=True)
    print(f"完成！共修复 {total_fixes} 处英文残留", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
