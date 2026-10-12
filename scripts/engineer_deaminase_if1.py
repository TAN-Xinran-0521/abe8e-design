#!/usr/bin/env python3
"""ESM-IF1 (inverse folding) mutation scoring for TadA-8e.
Structure-conditioned counterpart to engineer_deaminase.py (ESM2/ESM1v).
Outputs are format-aligned with Day3 so the three models are directly comparable.
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
import esm
import esm.inverse_folding
from esm.inverse_folding.util import CoordBatchConverter

AA20 = list("ACDEFGHIKLMNPQRSTVWY")

# 8 known TadA-8e evolved mutations vs ABE7.10 (sequence-native 1-based numbering)
KNOWN = ["A108S", "T110R", "D118N", "H121N", "Y146D", "F148Y", "T165I", "D166N"]
KNOWN_SITES = {int(m[1:-1]) for m in KNOWN}

# functional regions (sequence-native 1-based), aligned with Day3 assignments
def region_of(pos):  # pos is 1-based
    if 50 <= pos <= 105:  return "catalytic_core"
    if 106 <= pos <= 140: return "substrate_channel"
    if 141 <= pos <= 166: return "specificity_loop"
    return "n_terminal"


def get_logprob_matrix(pdb, chain, device):
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)
    structure = esm.inverse_folding.util.load_structure(pdb, chain)
    coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

    bc = CoordBatchConverter(alphabet)
    coords_t, confidence, _, tokens, padding_mask = bc([(coords, None, seq)], device=device)
    prev = tokens[:, :-1]
    with torch.no_grad():
        logits, _ = model.forward(coords_t, padding_mask, confidence, prev)  # (B, vocab, L)
    logprobs = torch.log_softmax(logits, dim=1)[0]  # (vocab, L)

    L = len(seq)
    idx = {aa: alphabet.get_idx(aa) for aa in AA20}
    mat = np.zeros((L, 20), dtype=np.float32)
    for j, aa in enumerate(AA20):
        mat[:, j] = logprobs[idx[aa], :L].cpu().numpy()
    return seq, mat


def lp(mat, pos1, aa):  # pos1 is 1-based
    return float(mat[pos1 - 1, AA20.index(aa)])


def rank_in_20(mat, pos1, aa):
    row = mat[pos1 - 1]
    order = np.argsort(-row)  # descending
    target = AA20.index(aa)
    return int(np.where(order == target)[0][0]) + 1


def positive_control(seq, mat):
    rows = []
    for m in KNOWN:
        wt7, pos, res8e = m[0], int(m[1:-1]), m[-1]
        seq_res = seq[pos - 1]
        if seq_res != res8e:
            rows.append({"mutation": m, "status": f"SEQ_MISMATCH(seq={seq_res})",
                         "top5_recall": False, "delta_logp": np.nan, "rank_in_20aa": -1})
            continue
        delta = lp(mat, pos, res8e) - lp(mat, pos, wt7)
        rank = rank_in_20(mat, pos, res8e)
        rows.append({"mutation": m,
                     "status": "RECALLED" if delta > 0 else "not_favored",
                     "top5_recall": rank <= 5,
                     "delta_logp": round(delta, 3),
                     "rank_in_20aa": rank})
    return pd.DataFrame(rows)


def nominate(seq, mat, min_delta, top_per_region):
    cands = []
    for i in range(len(seq)):
        pos1 = i + 1
        cur = seq[i]
        if cur not in AA20:
            continue
        cur_lp = lp(mat, pos1, cur)
        for aa in AA20:
            if aa == cur:
                continue
            delta = lp(mat, pos1, aa) - cur_lp
            if delta >= min_delta:
                cands.append({"mutation": f"{cur}{pos1}{aa}",
                              "region": region_of(pos1),
                              "delta_logp": round(delta, 3),
                              "is_known_8e_site": pos1 in KNOWN_SITES})
    df = pd.DataFrame(cands)
    if df.empty:
        return df
    df = (df.sort_values("delta_logp", ascending=False)
            .groupby("region", group_keys=False)
            .head(top_per_region)
            .sort_values(["region", "delta_logp"], ascending=[True, False])
            .reset_index(drop=True))
    return df


def variant_cards(df):
    eff = {"catalytic_core": "stability / fold-consensus (activity residue caution)",
           "substrate_channel": "substrate-channel remodeling (window/specificity)",
           "specificity_loop": "specificity-loop tuning (window narrowing)",
           "n_terminal": "N-terminal, effect uncertain"}
    rows = []
    for _, r in df.iterrows():
        rows.append({"variant_name": f"TadA-8e_{r['mutation']}",
                     "mutation": r["mutation"],
                     "region": r["region"],
                     "delta_logp": r["delta_logp"],
                     "expected_effect": eff[r["region"]],
                     "rationale": ("structure-conditioned IF1 favors this substitution; "
                                   + ("overlaps known 8e site (method sanity)" if r["is_known_8e_site"]
                                      else "novel candidate, verify with structure+wetlab"))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min_delta", type=float, default=1.0)
    ap.add_argument("--top_per_region", type=int, default=5)
    ap.add_argument("--outdir", default=__import__('repo_paths').repo_path('results/deaminase_if1'))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seq, mat = get_logprob_matrix(args.pdb, args.chain, args.device)
    print(f"Loaded structure: chain {args.chain}, {len(seq)} residues")

    np.save(os.path.join(args.outdir, "logprob_matrix.npy"), mat)

    pc = positive_control(seq, mat)
    pc.to_csv(os.path.join(args.outdir, "positive_control.csv"), index=False)
    valid = pc[pc["delta_logp"].notna()]
    n_delta = int((valid["delta_logp"] > 0).sum())
    n_top5 = int(valid["top5_recall"].sum())
    print("\n=== Positive control (ESM-IF1) ===")
    print(pc.to_string(index=False))
    print(f"\nRecall (delta>0): {n_delta}/8   Top-5 recall: {n_top5}/8")

    cand = nominate(seq, mat, args.min_delta, args.top_per_region)
    cand.to_csv(os.path.join(args.outdir, "candidate_mutations.csv"), index=False)
    print(f"\n=== Candidates (min_delta={args.min_delta}, top_per_region={args.top_per_region}) ===")
    print(cand.to_string(index=False) if not cand.empty else "(none)")

    if not cand.empty:
        variant_cards(cand).to_csv(os.path.join(args.outdir, "variant_cards.csv"), index=False)

    print(f"\nOutputs written to {args.outdir}")


if __name__ == "__main__":
    main()
