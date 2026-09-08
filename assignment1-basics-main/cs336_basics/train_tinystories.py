"""训练 BPE 的实验驱动脚本:计时、校验长度、保存 vocab/merges。

用法:
    uv run python cs336_basics/train_tinystories.py <输入文件> <vocab_size>

只负责调用 train_bpe_version2.run_train_bpe_ 并做计时/校验/持久化,不含任何 BPE 逻辑。
"""

import os
import pickle
import sys
import time

from cs336_basics.train_bpe_version4 import run_train_bpe_


SPECIAL_TOKENS = ["<|endoftext|>"]


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "用法: uv run python cs336_basics/train_tinystories.py <输入文件> <vocab_size>",
            file=sys.stderr,
        )
        return 2

    input_path = sys.argv[1]
    vocab_size = int(sys.argv[2])
    expected_merges = vocab_size - 256 - len(SPECIAL_TOKENS)

    t0 = time.perf_counter()
    vocab, merges = run_train_bpe_(input_path, vocab_size, SPECIAL_TOKENS)
    elapsed = time.perf_counter() - t0

    print(f"输入: {input_path}")
    print(f"vocab_size: {vocab_size}")
    print(f"耗时: {elapsed:.2f} 秒")
    print(f"len(vocab) = {len(vocab)}(期望 {vocab_size})")
    print(f"len(merges) = {len(merges)}(期望 {expected_merges})")

    os.makedirs("experiments", exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_path = f"experiments/{stem}_vocab{vocab_size}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump((vocab, merges), f)
    print(f"已保存: {out_path}")

    if len(vocab) != vocab_size or len(merges) != expected_merges:
        print(
            "警告: 长度不符,检查 pair 计数是否中途变负被 break 截断。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
