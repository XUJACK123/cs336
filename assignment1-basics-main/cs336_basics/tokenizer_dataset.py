import os
import pickle
import sys
import time
from array import array
import numpy as np
from cs336_basics.tokenizer import Tokenizer
SPECIAL_TOKEN = ["<|endoftext|>"]
def tokenizer_file(input_path: str, tokenizer: Tokenizer) -> np.ndarray:
    tokens = array("H")
    t0 = time.perf_counter()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            for token_id in tokenizer.encode_iterable([line]):
                tokens.append(token_id)
    arr = np.frombuffer(tokens, dtype = np.uint16)
    print(f"{input_path}: {len(arr)} token, {time.perf_counter() - t0:.1f}s, "f"{os.path.getsize(input_path) / len(arr):.2f} bytes/token")
    return arr

def main()->int:
    pkl_path, input_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(pkl_path, "rb") as f:
        vocab, merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, special_tokens=SPECIAL_TOKEN)
    arr = tokenizer_file(input_path, tokenizer)
    assert arr.max() < len(vocab), "token id 超出词表"
    np.save(output_path, arr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())