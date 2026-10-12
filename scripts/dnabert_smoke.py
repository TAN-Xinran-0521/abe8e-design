"""
dnabert_smoke.py
DNABERT-2 冒烟测试：CPU 加载模型 + 对少量 grna 提 embedding。
绕过 transformers 4.38 的 AutoModel 注册冲突：直接用 remote code 的 BertModel 类。
所有缓存/输出走 D 盘 (/mnt/d/TZS)。
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/d/TZS/hf_cache")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL = "zhihan1996/DNABERT-2-117M"


def load_dnabert():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    BertModel = get_class_from_dynamic_module("bert_layers.BertModel", MODEL)
    model = BertModel.from_pretrained(MODEL, config=config, trust_remote_code=True)
    model = model.eval().to("cpu")
    return tok, model


def main():
    print("Loading DNABERT-2 (CPU)...")
    tok, model = load_dnabert()
    print("Model loaded OK.")

    seqs = [
        "TACGTACCAGGTGGAGCACC",
        "GTCGACGCTGTAAACAATCG",
        "AAAAAAAAAAAAAAAAAAAA",
    ]
    for s in seqs:
        enc = tok(s, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        emb = hidden.mean(dim=1).squeeze().numpy()
        print(f"seq={s[:12]}... tokens={enc['input_ids'].shape[1]} "
              f"emb_dim={emb.shape[0]} emb[:5]={np.round(emb[:5],3)}")

    print("\nSmoke test passed. embedding dim above = feature width for batch extraction.")


if __name__ == "__main__":
    main()