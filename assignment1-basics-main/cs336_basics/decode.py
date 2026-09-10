import argparse
import pickle

import torch
import torch.nn.functional as F

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained LM")
    parser.add_argument("--vocab", type=str,
                        default="experiments/TinyStoriesV2-GPT4-train_vocab10000.pkl")
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/checkpoints/checkpoint-0020000.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def decode(model, tokenizer, prompt, max_tokens, temperature, top_p, device, seed):
    torch.manual_seed(seed)
    endoftext_id = tokenizer.encode("<|endoftext|>")[0]   # 256,循环外算一次
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for i in range(max_tokens):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x)[0,-1]
            probs = F.softmax(logits/temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = (cumsum <= top_p)
            mask[0] = True
            kept = sorted_probs*mask
            kept /= kept.sum()
            nxt = int(sorted_idx[torch.multinomial(kept, 1)].item())
            if nxt == tokenizer.encode("<|endoftext|>")[0]:
                break
            ids.append(nxt)
    return tokenizer.decode(ids)

def main():
    args = parse_args()
    with open(args.vocab, "rb") as f:
        vocab, merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    model = TransformerLM(10000, 1024, 512, 4, 16, 1344, 10000.0)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)
    text = decode(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_p, args.device, args.seed)
    print(text)


if __name__ == "__main__":
    main()