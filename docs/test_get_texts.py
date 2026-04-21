"""
用最小脚本演示 trainer/train_tokenizer.py 里的 get_texts(data_path) 是如何读取 jsonl 的。

默认行为：
1. 先构造一份临时 demo jsonl，便于观察哪些行会被 yield，哪些行会被跳过。
2. 再读取仓库里的真实数据文件，查看前几条输出。

这个脚本只依赖 Python 标准库，方便教学演示。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFT_PATH = REPO_ROOT / "dataset" / "sft_t2t_mini.jsonl"
DEFAULT_PRETRAIN_PATH = REPO_ROOT / "dataset" / "pretrain_t2t_mini.jsonl"
DEFAULT_DEMO_PATH = REPO_ROOT / "tmp" / "get_texts_demo.jsonl"


def get_texts_like_train_tokenizer(
    data_path: Path, max_lines: int = 10000
) -> Iterable[str]:
    """
    复刻 trainer/train_tokenizer.py 中 get_texts() 的核心逻辑：
    - 逐行读取 jsonl
    - 每行 json.loads
    - 只提取 conversations[*].content
    - 用换行拼接后 yield
    """
    with data_path.open("r", encoding="utf-8", errors="ignore") as file:
        for index, line in enumerate(file):
            if index >= max_lines:
                break
            try:
                data = json.loads(line)
                contents = [
                    item.get("content")
                    for item in data.get("conversations", [])
                    if item.get("content")
                ]
                if contents:
                    yield "\n".join(contents)
            except json.JSONDecodeError:
                continue


def write_demo_jsonl(output_path: Path) -> None:
    """
    写一份最小样例，覆盖 4 种情况：
    - 标准 conversations 数据
    - conversations 中有空 content / 缺字段
    - pretrain 风格 text 字段
    - 非法 JSON
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            "conversations": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，我是 MiniMind。"},
                {"role": "user", "content": "请解释一下 get_texts。"},
            ]
        },
        {
            "conversations": [
                {"role": "system", "content": ""},
                {"role": "user"},
                {"role": "assistant", "content": "这一轮只有我会被保留。"},
            ]
        },
        {"text": "这是一条 pretrain 风格数据，get_texts() 不会读取它。"},
        '{"conversations": [invalid json]}',
    ]

    with output_path.open("w", encoding="utf-8") as file:
        for item in lines[:-1]:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
        file.write(lines[-1] + "\n")


def inspect_raw_lines(data_path: Path, raw_lines: int) -> None:
    print(f"\n=== 原始文件预览: {data_path} ===")
    with data_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line_no, raw_line in enumerate(file, start=1):
            if line_no > raw_lines:
                break
            print(f"[raw line {line_no}] {raw_line.rstrip()[:220]}")
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(f"  -> JSON 解析失败: {exc}")
                continue

            print(f"  -> 顶层字段: {list(data.keys())}")
            conversations = data.get("conversations", [])
            if conversations:
                extracted = [
                    item.get("content") for item in conversations if item.get("content")
                ]
                print(f"  -> conversations 条数: {len(conversations)}")
                print(f"  -> 提取出的 content 条数: {len(extracted)}")
                for index, text in enumerate(extracted, start=1):
                    print(f"     content[{index}]: {text[:80]}")
            else:
                print("  -> 没有 conversations 字段，get_texts() 会直接跳过这一行。")


def preview_generator_output(data_path: Path, preview_count: int) -> None:
    print(f"\n=== get_texts() 输出预览: {data_path} ===")
    texts = get_texts_like_train_tokenizer(data_path)
    found = False
    for index, text in enumerate(texts, start=1):
        found = True
        print(f"[yield {index}]")
        print(text[:300])
        print("-" * 80)
        if index >= preview_count:
            break

    if not found:
        print("没有读到任何文本。通常说明这份数据不是 conversations 格式。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="测试 MiniMind train_tokenizer.py 中 get_texts() 的读取逻辑"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_SFT_PATH,
        help="要测试的 jsonl 文件路径，默认使用 dataset/sft_t2t_mini.jsonl",
    )
    parser.add_argument(
        "--raw-lines",
        type=int,
        default=2,
        help="预览前几条原始 jsonl 行",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=2,
        help="预览前几条 get_texts() 的输出",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="跳过内置 demo 数据，只测试 --data-path 指定的文件",
    )
    args = parser.parse_args()

    if not args.skip_demo:
        write_demo_jsonl(DEFAULT_DEMO_PATH)
        inspect_raw_lines(DEFAULT_DEMO_PATH, raw_lines=4)
        preview_generator_output(DEFAULT_DEMO_PATH, preview_count=4)

    inspect_raw_lines(args.data_path, raw_lines=args.raw_lines)
    preview_generator_output(args.data_path, preview_count=args.preview_count)

    if args.data_path.resolve() == DEFAULT_SFT_PATH.resolve():
        print("\n=== 对比提示 ===")
        print(
            "当前 train_tokenizer.py 里的 get_texts() 只读取 conversations[*].content。"
        )
        print(
            f"如果你改测 {DEFAULT_PRETRAIN_PATH}，由于它是 text 字段，通常不会有任何 yield。"
        )


if __name__ == "__main__":
    main()
