"""Byte-level BPE tokenizer matching tiktoken's ``gpt2`` encoding.

The tokenizer is constructed from an already-trained vocabulary and merge
list, mirroring the interface expected by ``tests/adapters.get_tokenizer``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.train_bpe_version4 import GPT2_SPLIT_REGEX


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self._id_to_bytes = dict(vocab)
        self._bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self._merge_rank = {pair: rank for rank, pair in enumerate(merges)}
        self._special_tokens = list(special_tokens) if special_tokens else []

        self._pretokenizer = re.compile(GPT2_SPLIT_REGEX)
        if self._special_tokens:
            # Longest first so overlapping special tokens (e.g. "<|endoftext|>"
            # vs "<|endoftext|><|endoftext|>") match as one atomic token.
            ordered = sorted(self._special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(token) for token in ordered)
            self._special_regex = re.compile(pattern)
            self._special_ids = {
                token: self._bytes_to_id[token.encode("utf-8")] for token in self._special_tokens
            }
            self._max_special_len = max(len(token) for token in self._special_tokens)
        else:
            self._special_regex = None
            self._special_ids = {}
            self._max_special_len = 0

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids."""
        ids: list[int] = []
        for segment, special in self._split_text(text):
            if special is not None:
                ids.append(self._special_ids[special])
            else:
                ids.extend(self._encode_regular(segment))
        return ids

    def decode(self, ids: list[int] | Iterable[int]) -> str:
        """Decode a sequence of token ids back into a string."""
        raw = b"".join(self._id_to_bytes[token_id] for token_id in ids)
        return raw.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str | bytes]) -> Iterator[int]:
        """Streaming encode: yields token ids without holding the whole input in memory.

        Text is drained in two stages:
        1. Split off complete special-token occurrences (they are atomic).
        2. Tokenize the remaining regular text, always holding back the last
           regex match and a tail of ``max_special_len - 1`` characters so that
           a word or special token cut by a chunk boundary is never emitted
           prematurely.
        """
        buf = ""
        for chunk in iterable:
            if chunk is None:
                continue
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="replace")
            if not chunk:
                continue
            buf += chunk

            while True:
                # Stage 1: any complete special token can be emitted right away,
                # together with the regular text before it.
                if self._special_regex is not None:
                    pos = 0
                    last_end = -1
                    for match in self._special_regex.finditer(buf):
                        yield from self._encode_regular(buf[pos : match.start()])
                        yield self._special_ids[match.group()]
                        pos = match.end()
                        last_end = match.end()
                    if last_end >= 0:
                        buf = buf[last_end:]
                        continue

                # Stage 2: drain regular text, deferring the trailing partial
                # token / possible special-token prefix.
                cut = self._drain_cut(buf)
                if cut <= 0:
                    break
                yield from self._encode_regular(buf[:cut])
                buf = buf[cut:]

        # End of input: everything left is final, no need to defer.
        yield from self._encode_regular(buf)

    def _split_text(self, text: str) -> Iterator[tuple[str, str | None]]:
        """Yield ``(segment, special_token_or_None)`` pairs covering ``text``."""
        if self._special_regex is None or not text:
            if text:
                yield (text, None)
            return
        pos = 0
        for match in self._special_regex.finditer(text):
            if match.start() > pos:
                yield (text[pos : match.start()], None)
            yield (match.group(), match.group())
            pos = match.end()
        if pos < len(text):
            yield (text[pos:], None)

    def _encode_regular(self, text: str) -> list[int]:
        """Pretokenize ``text`` with the GPT-2 regex and BPE-encode each piece."""
        ids: list[int] = []
        for match in self._pretokenizer.finditer(text):
            piece = match.group().encode("utf-8")
            ids.extend(self._encode_piece(piece))
        return ids

    def _encode_piece(self, piece: bytes) -> list[int]:
        """Byte-level BPE on a single pretokenized piece.

        Matches tiktoken's ``_byte_pair_encode``: repeatedly find the
        lowest-rank mergeable adjacent pair and merge its leftmost occurrence.
        """
        parts = [bytes([b]) for b in piece]
        while len(parts) >= 2:
            min_idx = -1
            min_rank = -1
            for i in range(len(parts) - 1):
                rank = self._merge_rank.get((parts[i], parts[i + 1]))
                if rank is not None and (min_rank < 0 or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_idx < 0:
                break
            parts = (
                parts[:min_idx]
                + [parts[min_idx] + parts[min_idx + 1]]
                + parts[min_idx + 2 :]
            )
        return [self._bytes_to_id[part] for part in parts]

    def _drain_cut(self, buf: str) -> int:
        """Return the safe prefix length of ``buf`` that can be emitted now.

        The regex always tiles its input, so the final match of the safe region
        is the only one that could be a truncated token; it is deferred together
        with a small tail that may contain a partial special token.
        """
        if not buf:
            return 0
        holdback = max(0, self._max_special_len - 1)
        region_end = len(buf) - holdback
        if region_end <= 0:
            return 0
        last_start = 0
        for match in self._pretokenizer.finditer(buf, 0, region_end):
            last_start = match.start()
        return last_start
