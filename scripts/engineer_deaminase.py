# D:\TZS\scripts\engineer_deaminase.py
"""
脚本功能: 用 ESM 蛋白语言模型对 TadA-8e 提名效应模块改造突变
          (无需结构, 无需 torch_geometric; 用进化序列信息做突变效应打分)

方法学:
  用 masked-marginal 打分:
    score(mut) = log P(mut | 上下文) - log P(wt | 上下文)
  score>0 表示模型认为该突变比野生型更可能, 即"进化上更合理".
  默认用轻量 ESM2-35M (~150MB), 可用 --model 换更大模型.

流程:
  1. 载入 TadA-8e 序列 (166 aa, 已从 Addgene 138489 确认)
  2. 逐位点做 masked-marginal 打分, 得到 (L, 20) 矩阵
  3. 正对照: 检验模型能否给 8e 已知 8 个演化突变打正分
  4. 提名候选: 功能区内模型显著偏好的非野生型替换
  5. 生成变体卡片

依赖: fair-esm, torch, biopython, numpy, pandas
用法:
  python engineer_deaminase.py --fasta D:/TZS/data/structures/tada8e_sequence.fasta
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import esm
from Bio import SeqIO


AA_LIST = "ACDEFGHIKLMNPQRSTVWY"

# ---------------- 领域知识配置 ---------------- #
FUNCTIONAL_REGIONS = {
    "catalytic_core":    (50, 90),
    "substrate_channel": (95, 130),
    "specificity_loop":  (140, 166),
}


# 正对照: TadA-8e 相对 ABE7.10 脱氨酶的 8 个演化突变
# 编号 = tada8e_sequence.fasta 的位置 (无起始 Met, 位置1=S), 与序列文件严格一致.
# 文献习惯用含 Met 编号 (= 此处编号 + 1), 例如此处 A108S = 文献 A109S.
# 来源: 从 Addgene 138489 质粒序列比对得出 (extract_tada8e_domain.py).
KNOWN_MUTATIONS = {
    "A108S": "7.10->8e evolution mutation (lit. A109S)",
    "T110R": "7.10->8e, enhances DNA binding (lit. T111R)",
    "D118N": "7.10->8e evolution mutation (lit. D119N)",
    "H121N": "7.10->8e evolution mutation (lit. H122N)",
    "Y146D": "7.10->8e evolution mutation (lit. Y147D)",
    "F148Y": "7.10->8e evolution mutation (lit. F149Y)",
    "T165I": "7.10->8e evolution mutation (lit. T166I)",
    "D166N": "7.10->8e evolution mutation (lit. D167N)",
}


class DeaminaseEngineerLM:
    def __init__(self, fasta_file, device="cpu", pos_shift=0,
                 model_name="esm2_t12_35M_UR50D"):
        self.fasta_file = Path(fasta_file)
        self.device = device
        self.pos_shift = pos_shift
        self.model_name = model_name

        if not self.fasta_file.exists():
            raise FileNotFoundError(f"FASTA not found: {self.fasta_file}")

        record = SeqIO.read(self.fasta_file, "fasta")
        self.seq = str(record.seq)
        self.seq_len = len(self.seq)
        print(f"TadA-8e sequence: {self.seq_len} aa")
        print(f"Sequence: {self.seq}")

        print(f"\nLoading ESM model ({model_name})...")
        if model_name not in esm.pretrained.__dict__:
            raise ValueError(
                f"未知模型: {model_name}\n"
                f"常用选项: esm2_t12_35M_UR50D (~150MB, 快), "
                f"esm2_t30_150M_UR50D (~600MB), "
                f"esm2_t33_650M_UR50D (~2.5GB), "
                f"esm1v_t33_650M_UR90S_1 (~2.6GB)"
            )
        self.model, self.alphabet = esm.pretrained.__dict__[model_name]()
        self.model = self.model.eval().to(device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @torch.no_grad()
    def compute_masked_marginal(self):
        data = [("tada8e", self.seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        mask_idx = self.alphabet.mask_idx
        aa_token_idx = [self.alphabet.get_idx(a) for a in AA_LIST]

        mat = np.zeros((self.seq_len, len(AA_LIST)), dtype=np.float32)

        print("Masked-marginal scoring (per-position)...")
        for i in range(self.seq_len):
            masked = tokens.clone()
            masked[0, i + 1] = mask_idx
            logits = self.model(masked)["logits"]  # (1, L+2, vocab)
            log_probs = torch.log_softmax(logits[0, i + 1], dim=-1)
            mat[i] = log_probs[aa_token_idx].cpu().numpy()
            if (i + 1) % 30 == 0:
                print(f"  scored {i+1}/{self.seq_len}")

        return mat

    def positive_control(self, logprob_mat):
        aa2col = {a: i for i, a in enumerate(AA_LIST)}
        rows = []
        for mut_str, note in KNOWN_MUTATIONS.items():
            wt, pos1, mut = mut_str[0], int(mut_str[1:-1]), mut_str[-1]
            idx = pos1 - 1 + self.pos_shift
            if idx < 0 or idx >= self.seq_len:
                rows.append({"mutation": mut_str, "status": "OUT_OF_RANGE",
                             "delta_logp": np.nan, "rank_in_20aa": np.nan,
                             "note": note})
                continue
            seq_res = self.seq[idx]
            if seq_res != mut:
                rows.append({
                    "mutation": mut_str,
                    "status": f"SEQ_MISMATCH(seq={seq_res},expect_8e={mut})",
                    "delta_logp": np.nan, "rank_in_20aa": np.nan, "note": note})
                continue
            wt_lp = logprob_mat[idx, aa2col[wt]]
            mut_lp = logprob_mat[idx, aa2col[mut]]
            delta = mut_lp - wt_lp
            rank = int(np.sum(logprob_mat[idx] > mut_lp)) + 1
            rows.append({
                "mutation": mut_str,
                "status": "RECALLED" if delta > 0 else "not_favored",
                "top5_recall": rank <= 5,          # 更宽松合理的召回判据
                "delta_logp": round(float(delta), 3),
                "rank_in_20aa": rank,
                "note": note})
        return pd.DataFrame(rows)

    def nominate_mutations(self, logprob_mat, min_delta=1.0, top_per_region=5):
        aa2col = {a: i for i, a in enumerate(AA_LIST)}
        known_positions = {int(m[1:-1]) for m in KNOWN_MUTATIONS}
        rows = []

        for region, (start, end) in FUNCTIONAL_REGIONS.items():
            for pos1 in range(start, end + 1):
                idx = pos1 - 1 + self.pos_shift
                if idx < 0 or idx >= self.seq_len:
                    continue
                wt = self.seq[idx]
                if wt not in aa2col:
                    continue
                wt_lp = logprob_mat[idx, aa2col[wt]]

                order = np.argsort(-logprob_mat[idx])
                for col in order:
                    mut = AA_LIST[col]
                    if mut == wt:
                        continue
                    mut_lp = logprob_mat[idx, col]
                    delta = mut_lp - wt_lp
                    if delta >= min_delta:
                        rows.append({
                            "region": region,
                            "position": pos1,
                            "wt_aa": wt,
                            "mut_aa": mut,
                            "mutation": f"{wt}{pos1}{mut}",
                            "logp_wt": round(float(wt_lp), 3),
                            "logp_mut": round(float(mut_lp), 3),
                            "delta_logp": round(float(delta), 3),
                            "is_known_8e_site": pos1 in known_positions,
                        })
                    break

        if not rows:
            return pd.DataFrame(columns=[
                "region", "position", "wt_aa", "mut_aa", "mutation",
                "logp_wt", "logp_mut", "delta_logp", "is_known_8e_site"])

        df = pd.DataFrame(rows).sort_values("delta_logp", ascending=False)
        df = df.groupby("region", group_keys=False).head(top_per_region)
        return df.reset_index(drop=True)

    @staticmethod
    def make_variant_cards(cand_df):
        region_mech = {
            "catalytic_core":
                "位于催化核心, 改变可能调节脱氨催化速率与整体活性",
            "substrate_channel":
                "位于 ssDNA 底物通道, 改变可能收窄编辑窗口、降低 bystander 编辑",
            "specificity_loop":
                "位于 C 端特异性环区, 改变可能调节序列偏好与靶向特异性",
        }
        cards = []
        for _, r in cand_df.iterrows():
            cards.append({
                "variant_name": f"TadA-8e-{r['mutation']}",
                "mutation": r["mutation"],
                "region": r["region"],
                "delta_logp": r["delta_logp"],
                "expected_effect": region_mech.get(r["region"], ""),
                "rationale":
                    f"ESM 语言模型在该位点更偏好 {r['mut_aa']} 而非当前 "
                    f"{r['wt_aa']} (delta logP={r['delta_logp']}), 提示 "
                    f"{r['mut_aa']} 在进化序列约束下更合理, 可能提升稳定性"
                    f"或改变底物相互作用。",
            })
        return pd.DataFrame(cards)


def main():
    ap = argparse.ArgumentParser(description="TadA-8e 脱氨酶效应模块改造 (ESM 语言模型)")
    ap.add_argument("--fasta", default=__import__('repo_paths').repo_path('data/structures/tada8e_sequence.fasta'))
    ap.add_argument("--model", default="esm2_t12_35M_UR50D",
                    help="ESM 模型名. 轻量默认 esm2_t12_35M_UR50D (~150MB). "
                         "更高精度可用 esm2_t33_650M_UR50D 或 esm1v_t33_650M_UR90S_1")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--shift", type=int, default=0,
                    help="序列编号偏移校正 (正对照 SEQ_MISMATCH 时调整)")
    ap.add_argument("--min_delta", type=float, default=1.0)
    ap.add_argument("--top_per_region", type=int, default=5)
    ap.add_argument("--outdir", default=__import__('repo_paths').repo_path('results/deaminase'))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eng = DeaminaseEngineerLM(args.fasta, args.device, args.shift, args.model)

    print("\n=== Step 1: masked-marginal 突变打分 ===")
    logprob_mat = eng.compute_masked_marginal()
    np.save(outdir / "logprob_matrix.npy", logprob_mat)

    print("\n=== Step 2: 正对照 — 模型能否召回 8e 已知演化突变 ===")
    pc = eng.positive_control(logprob_mat)
    print(pc.to_string(index=False))
    pc.to_csv(outdir / "positive_control.csv", index=False)

    valid = pc[~pc["status"].str.contains("MISMATCH|OUT_OF_RANGE")]
    n_valid = len(valid)
    n_recalled = int((valid["status"] == "RECALLED").sum())
    if n_valid > 0:
        print(f"\n召回率 (delta_logp>0): {n_recalled}/{n_valid} = {n_recalled/n_valid:.0%}")
    else:
        print("\n[警告] 全部 SEQ_MISMATCH/越界 -> 序列与编号不匹配, 试 --shift 微调。")
        print(f"       序列: {eng.seq}")

    print("\n=== Step 3: 提名候选改造突变 ===")
    cand = eng.nominate_mutations(logprob_mat, args.min_delta, args.top_per_region)
    if len(cand) == 0:
        print("未提名到候选 (min_delta 太高?)。试降低 --min_delta, 例如 0.5")
    else:
        print(cand.to_string(index=False))
    cand.to_csv(outdir / "candidate_mutations.csv", index=False)

    print("\n=== Step 4: 生成变体卡片 ===")
    cards = eng.make_variant_cards(cand)
    cards.to_csv(outdir / "variant_cards.csv", index=False)
    print(f"提名候选数: {len(cand)}  (>=3 满足赛题效应模块改造要求)")

    print(f"\n=== 输出 ({outdir}) ===")
    for f in ["logprob_matrix.npy", "positive_control.csv",
              "candidate_mutations.csv", "variant_cards.csv"]:
        print(f"  {f}")
    
    n_top5 = int(valid["top5_recall"].sum())
    print(f"Top-5 召回率: {n_top5}/{n_valid} = {n_top5/n_valid:.0%}")


if __name__ == "__main__":
    main()