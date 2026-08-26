from cs336_basics.train_bpe_version1 import run_train_bpe_

vocab, merges = run_train_bpe_('tests/fixtures/tinystories_sample.txt', 300, ['<|endoftext|>'])

print('len(vocab) =', len(vocab))
print('check1 词表满额:', len(vocab) == 300)
print('check2 merge 数目:', len(merges) == 300 - 256 - 1)
print('check3 值都是 bytes:', all(isinstance(v, bytes) for v in vocab.values()))
print('特殊 token 在词表:', b'<|endoftext|>' in vocab.values())
print('前 5 个 merge:', merges[:5])