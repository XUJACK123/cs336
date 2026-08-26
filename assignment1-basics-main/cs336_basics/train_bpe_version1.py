import regex as re
from collections import Counter

GPT2_SPLIT_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def run_train_bpe_(fp: str, num_words: int, special_tokens: list[str] = None, **kwargs)->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 创立词表
    vocab_list: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for idx, st in enumerate(special_tokens):
        vocab_list[256 + idx] = st.encode("utf-8") if isinstance(st, str) else st
    # 开文件
    with open(fp, 'r', encoding='utf-8') as f:
        text = f.read()
    # 遇到special_roken断开
    if special_tokens:
        special_pattern = "|".join(re.escape(st) for st in special_tokens)
        chunks = re.split(f"({special_pattern})", text)
    else:
        chunks = [text]
    # 分开所有的字节
    word_counts = Counter()
    compile_pat = re.compile(GPT2_SPLIT_REGEX)
    for chunk in chunks:
        if not chunk or (chunk in special_tokens):
            continue
        for match in compile_pat.finditer(chunk):
            token_bytes = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
            word_counts[token_bytes] += 1
    merges: list[tuple[bytes, bytes]] = []
    merge_count = num_words - len(special_tokens) - 256
    """
    merges:就是新增的组合在一起的token
    merge_count:要重复多少次
    pair_freqs:每一字节的重复次数
    most_pair:重复最多的字节
    new_word_counts:新序列中的字节统计
    new_tokens:新的词表
    new_token:新的词
    """
    # 查看每一次的重复
    for _ in range(merge_count):
        pair_freqs = Counter()
        # 查看所有词的重复
        for word, count in word_counts.items():
            # 查看每一个词的字节的重复
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += count
        if not pair_freqs:
            break
        # 找到最多的
        most_pair = max(pair_freqs.keys(), key=lambda p: (pair_freqs[p], p))
        merges.append(most_pair)
        new_token = most_pair[0]+most_pair[1]
        vocab_list[len(vocab_list)] = new_token
        # 更新词频表中的序列结构
        new_word_counts = Counter()
        for word, count in word_counts.items():
            new_tokens = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and word[i] == most_pair[0] and word[i+1] == most_pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(word[i])
                    i += 1
            new_word_counts[tuple(new_tokens)] += count
        word_counts = new_word_counts
        
    return vocab_list, merges