import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import regex as re
from collections import Counter, defaultdict

GPT2_SPLIT_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Node:
    __slots__ = ('val','prev','next','word_id','pos')
    def __init__(self, val:bytes, word_id: int, pos: int):
        self.val: bytes | None = val
        self.prev: Node | None = None
        self.next: Node | None = None
        self.word_id: int = word_id
        self.pos: int = pos

def _worker_process_chunk(fp: str, start: int, end: int, special_tokens: list[str] = None) -> Counter:
    # 分开所有的字节,同时统计有多少个初始的词
    word_counts = Counter()
    compile_pat = re.compile(GPT2_SPLIT_REGEX)
    special_pat = None
    special_set = set(special_tokens) if special_tokens else set()
    if special_tokens:
        special_pattern = "|".join(re.escape(st) for st in special_tokens)
        special_pat = re.compile(f"({special_pattern})")

    # 对一段文本（多行拼成的完整 chunk）进行预分词并统计词频
    def process_chunk(text:str):
        if not text:
            return
        for match in compile_pat.finditer(text):
            token_bytes = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
            word_counts[token_bytes] += 1
    buffer = []

    # 开文件，逐行读取，去除掉special tokens
    with open(fp, 'rb') as f:
        f.seek(start)
        while True:
            curr_pos = f.tell()
            if curr_pos >= end:
                break
            line_bytes = f.readline()
            if not line_bytes:
                break
            if curr_pos + len(line_bytes) > end:
                line_bytes = line_bytes[: end - curr_pos]
            line = line_bytes.decode('utf-8', errors='replace')
            if special_pat and special_pat.search(line):
                chunks = special_pat.split(line)
                for chunk in chunks:
                    if chunk in special_set:
                        # 遇到特殊 token：拿出暂存箱里的整段文字进行处理，随后清空暂存箱
                        process_chunk("".join(buffer))
                        buffer = []
                    else:
                        buffer.append(chunk)
            else:
                buffer.append(line)
    if buffer:
        process_chunk("".join(buffer))
    return word_counts


def find_chunk_boundaries(fp: str, num_chunks: int, special_tokens: list[str] = None) -> list[tuple[int, int]]:
    file_size = os.path.getsize(fp)
    if file_size == 0 or num_chunks <= 1:
        return [(0, file_size)]
    target_bytes = b"<|endoftext|>"
    if special_tokens and len(special_tokens) > 0:
        st = special_tokens[0]
        target_bytes = st.encode("utf-8") if isinstance(st, str) else st
    step = file_size // num_chunks
    raw_boundaries = [0]
    with open(fp, 'rb') as f:
        for i in range(1, num_chunks):
            target_pos = i * step
            f.seek(target_pos)
            chunk_size = 8192
            pos = target_pos
            found_boundary = None
            while pos < file_size:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                idx = chunk.find(target_bytes)
                if idx != -1:
                    found_boundary = pos + idx + len(target_bytes)
                    break
                pos += len(chunk)
            if found_boundary and found_boundary < file_size:
                raw_boundaries.append(found_boundary)
    raw_boundaries.append(file_size)
    sorted_bounds = sorted(list(set(raw_boundaries)))
    ranges = [
        (sorted_bounds[i], sorted_bounds[i + 1]) 
        for i in range(len(sorted_bounds) - 1) 
        if sorted_bounds[i] < sorted_bounds[i + 1]
    ]
    return ranges

    
def run_train_bpe_(fp: str, num_words: int, special_tokens: list[str] = None, **kwargs) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []
    vocab_list: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for idx, st in enumerate(special_tokens):
        vocab_list[256 + idx] = st.encode("utf-8") if isinstance(st, str) else st

    # Num workers configuration
    num_workers = kwargs.get("num_workers", 12)
    chunk_ranges = find_chunk_boundaries(fp, num_workers, special_tokens)
    # 多进程并行预分词 (多进程仅传递 start, end 坐标)
    word_counts = Counter()
    with ProcessPoolExecutor(max_workers=len(chunk_ranges)) as executor:
        futures = [
            executor.submit(_worker_process_chunk, fp, start, end, special_tokens)
            for start, end in chunk_ranges
        ]
        for future in futures:
            word_counts.update(future.result())
    # 建立双向链表和pair的频率，pair_freq就是pair->总频次
    # 有多少个词就有多少个链表，而pair_positions就是Pair->set of (left_node, word_count)，把链表中每个组合和组合的次数都记录下来，方便更改的时候快速找到
    pair_freqs = Counter()
    pair_positions = defaultdict(set)
    for word_id, (word_bytes, count) in enumerate(word_counts.items()):
        if len(word_bytes) < 2:
            continue
        nodes = [Node(b, word_id=word_id, pos=i) for i, b in enumerate(word_bytes)]
        for i in range(len(nodes)-1):
            nodes[i].next = nodes[i+1]
            nodes[i+1].prev = nodes[i]
            # 统计所有的pair算出多少新的token
            pair = (nodes[i].val, nodes[i+1].val)
            pair_freqs[pair] += count
            pair_positions[pair].add((nodes[i],count))
            
    merges: list[tuple[bytes, bytes]] = []
    merge_count = num_words - len(special_tokens) - 256
    """
    merges:就是新增的组合在一起的token
    merge_count:要重复多少次
    pair_freqs:每一字节的重复次数
    pair_positions:每一个字节在链表中的位置
    most_pair:重复最多的字节
    new_word_counts:新序列中的字节统计
    new_tokens:新的词表
    new_token:新的词

    分为两个表，一个是每个词自己有一个链表，一个是查看pair_freq的表
    """
    # 查看每一次的重复
    for _ in range(merge_count):
        # 找到最多的
        if not pair_freqs:
            break
        most_pair = max(pair_freqs.keys(), key=lambda p: (pair_freqs[p], p))
        if pair_freqs[most_pair] <= 0:
            break
        merges.append(most_pair)
        new_token = most_pair[0]+most_pair[1]
        vocab_list[len(vocab_list)] = new_token
        nodes_to_merge = sorted(pair_positions[most_pair], key=lambda item: (item[0].word_id, item[0].pos))
        del pair_positions[most_pair]
        del pair_freqs[most_pair]
        # 针对每一个词依次修改
        for left_node, count in nodes_to_merge:
            if (left_node.val is None or left_node.next is None or left_node.val != most_pair[0] or left_node.next.val != most_pair[1]):
                continue
            right_node = left_node.next
            prev_node = left_node.prev
            next_node = right_node.next
            # 处理left和prev的关系，删除原本的left和prev这个pair
            if prev_node and prev_node.val is not None:
                old_prev_pair = (prev_node.val, left_node.val)
                pair_freqs[old_prev_pair] -= count
                if pair_freqs[old_prev_pair] <= 0:
                    del pair_freqs[old_prev_pair]
                pair_positions[old_prev_pair].discard((prev_node, count))
            # 处理right和next
            if next_node and next_node.val is not None:
                old_next_pair = (right_node.val, next_node.val)
                # 减去原本的贡献
                pair_freqs[old_next_pair] -= count
                if pair_freqs[old_next_pair] <= 0:
                    del pair_freqs[old_next_pair]
                pair_positions[old_next_pair].discard((right_node, count))
            left_node.val = new_token
            left_node.next = next_node
            if next_node:
                next_node.prev = left_node
            right_node.val = None
            right_node.prev = None
            right_node.next = None

            if prev_node and prev_node.val is not None:
                new_prev_pair = (prev_node.val, left_node.val)
                pair_freqs[new_prev_pair] += count
                pair_positions[new_prev_pair].add((prev_node, count))
            if next_node and next_node.val is not None:
                new_next_pair = (left_node.val, next_node.val)
                pair_freqs[new_next_pair] += count
                pair_positions[new_next_pair].add((left_node, count))
        
    return vocab_list, merges