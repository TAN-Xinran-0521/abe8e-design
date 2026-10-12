"""
dnabert_extract.py
用 DNABERT-2 对 train/val/test 的 grna(20nt) 批量提 768维 mean-pooling embedding。
绕过 AutoModel 注册冲突(直接用 remote BertModel)；用 last_hidden_state(非随机pooler)。
输出 npz 到 /mnt/d/TZS/data/features_dnabert/，供下游 XGBoost 对标 B'。
"""
import os, time
os.environ.setdefault("HF_HOME", "/mnt/d/TZS/hf_cache")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL = "zhihan1996/DNABERT-2-117M"
TARGET = "efficiency_full_grna_reported"
DATA = "/mnt/d/TZS/data/processed"
OUT = "/mnt/d/TZS/data/features_dnabert"


def load_dnabert():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    BertModel = get_class_from_dynamic_module("bert_layers.BertModel", MODEL)
    model = BertModel.from_pretrained(MODEL, config=config, trust_remote_code=True)
    return tok, model.eval().to("cpu")


@torch.no_grad()
def embed_batch(tok, model, seqs):
    """对一批序列提 mean-pooling embedding，用 attention_mask 做加权池化。"""
    enc = tok(seqs, return_tensors="pt", padding=True)
    out = model(**enc)
    hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state  # (B,T,768)
    mask = enc["attention_mask"].unsqueeze(-1).float()                            # (B,T,1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).cpu().numpy()                                        # (B,768)


def extract_split(tok, model, split, batch_size=64):
    df = pd.read_csv(f"{DATA}/{split}.csv")[["grna", TARGET]].dropna()
    df = df[df["grna"].str.len() == 20].reset_index(drop=True)
    seqs = df["grna"].tolist()
    y = df[TARGET].values.astype(np.float32)

    embs = []
    t0 = time.time()
    for i in range(0, len(seqs), batch_size):
        embs.append(embed_batch(tok, model, seqs[i:i + batch_size]))
        done = min(i + batch_size, len(seqs))
        if done % (batch_size * 10) == 0 or done == len(seqs):
            el = time.time() - t0
            print(f"  {split}: {done}/{len(seqs)}  ({el:.0f}s, "
                  f"{done/max(el,1e-9):.1f} seq/s)")
    X = np.vstack(embs).astype(np.float32)
    return X, y


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading DNABERT-2...")
    tok, model = load_dnabert()
    print("Loaded.\n")

    for split in ["val", "test", "train"]:   # 先小后大，早发现问题
        print(f"Extracting {split} ...")
        X, y = extract_split(tok, model, split)
        np.savez(f"{OUT}/{split}.npz", X=X, y=y)
        print(f"  saved {split}: X={X.shape} y={y.shape} -> {OUT}/{split}.npz\n")

    print("Done. embeddings -> /mnt/d/TZS/data/features_dnabert/")


if __name__ == "__main__":
    main()