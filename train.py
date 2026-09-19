"""CPU-only LoRA fine-tuning of TinyLlama-1.1B (Sec. 3, Appendix A).

Manual PyTorch loop on purpose: on CPU-only hosts the HF Trainer probes CUDA
devices at init and crashes (Sec. 3.4 / Appendix B).
"""
import argparse, json, os, random, time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)


class BugDataset(Dataset):
    """Prompt+target in one sequence; right-pad with </s>; mask ONLY pad positions
    with -100 so loss accumulates over real code+response tokens (Sec. 3.2)."""

    def __init__(self, path, tokenizer, max_len=128):
        with open(path, encoding="utf-8") as fh:
            self.rows = [json.loads(l) for l in fh]
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text = self.rows[i]["prompt"] + " " + self.rows[i]["target"]
        ids = self.tok(text, truncation=True, max_length=self.max_len)["input_ids"]
        n = len(ids)
        ids += [self.tok.eos_token_id] * (self.max_len - n)
        mask = [1] * n + [0] * (self.max_len - n)
        labels = [t if m else -100 for t, m in zip(ids, mask)]
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(mask),
                "labels": torch.tensor(labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.float32)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32)

    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    loader = DataLoader(BugDataset(args.data, tok, args.max_len),
                        batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    history, t0 = [], time.time()
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        bar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in bar:
            out = model(**batch)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            total += out.loss.item()
            bar.set_postfix(loss=out.loss.item())
        avg = total / len(loader)
        history.append(avg)
        print(f"Epoch {epoch+1} average loss: {avg:.4f}")

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6.5, 4))
    plt.plot(range(1, args.epochs + 1), history, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss per Epoch: LoRA-adapted TinyLlama-1.1B (CPU)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("figures/loss_curve.png", dpi=300)
    json.dump(history, open("figures/losses.json", "w"))

    model.save_pretrained(args.out)      # ~4 MB adapter (Sec. 9)
    tok.save_pretrained(args.out)
    print(f"Done in {(time.time()-t0)/60:.1f} min. Adapter saved to {args.out}/")


if __name__ == "__main__":
    main()