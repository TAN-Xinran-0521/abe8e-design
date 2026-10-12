# D:\TZS\scripts\extract_tada8e_from_gbk.py
"""
脚本功能：解析 Addgene GenBank 文件，定位并提取 TadA-8e 脱氨酶结构域序列
用法：
    python extract_tada8e_from_gbk.py --gbk <path> --out <fasta_path>
"""
import argparse
from pathlib import Path
from Bio import SeqIO

LINKER_PATTERNS = ["GGGGS", "SGGS", "GGSSGG", "GGSGG", "GSGGS"]

def has_linker_signature(seq: str) -> bool:
    return any(p in seq for p in LINKER_PATTERNS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbk", required=True)
    ap.add_argument("--out", default=__import__('repo_paths').repo_path('data/structures/tada8e_sequence.fasta'))
    args = ap.parse_args()

    gbk_path = Path(args.gbk)
    if not gbk_path.exists():
        raise FileNotFoundError(f"GenBank file not found: {gbk_path}")

    record = SeqIO.read(gbk_path, "genbank")
    print(f"=== Plasmid: {record.name} | {record.description} ===")
    print(f"Total length: {len(record.seq)} bp")
    print(f"Feature count: {len(record.features)}\n")

    print("=== All CDS features (overview) ===")
    cds_list = []
    for i, feat in enumerate(record.features):
        if feat.type != "CDS":
            continue
        quals = feat.qualifiers
        label = quals.get("label", quals.get("gene", quals.get("product", ["?"])))[0]
        translation = quals.get("translation", [None])[0]
        length_aa = len(translation) if translation else None
        cds_list.append((i, label, length_aa, feat))
        print(f"  [{i}] label='{label}'  aa_len={length_aa}")

    print("\n=== Searching for TadA-related CDS ===")
    candidates = []
    for i, label, length_aa, feat in cds_list:
        if "tada" in label.lower():
            candidates.append((i, label, length_aa, feat))
            print(f"  MATCH -> [{i}] '{label}' ({length_aa} aa)")

    if not candidates:
        print("\n[警告] 没有找到包含 'TadA' 关键字的 CDS label。")
        print("请检查上面 '所有 CDS 概览'，手动确认哪个 label 对应脱氨酶结构域")
        print("（常见命名：ecTadA(8e), TadA-8e, TadA7.10, deaminase 等）。")
        return

    print("\n=== Extracted candidate sequences ===")
    with open(args.out, "w") as f:
        for i, label, length_aa, feat in candidates:
            translation = feat.qualifiers["translation"][0]
            has_linker = has_linker_signature(translation)
            print(f"\n--- Feature [{i}] label='{label}' ---")
            print(f"Length: {length_aa} aa")
            print(f"Contains linker signature: {has_linker}")
            print(f"Sequence: {translation}")

            status = "OK" if (150 <= length_aa <= 175 and not has_linker) else "CHECK_MANUALLY"
            print(f"Sanity check: {status}")

            f.write(f">{label}_from_{record.name}_len{length_aa}_{status}\n")
            f.write(translation + "\n")

    print(f"\n已写入: {args.out}")
    print("请检查上面每条候选的 Sanity check：")
    print("  - OK：长度在 150-175 aa 且不含常见 linker 特征，可直接使用")
    print("  - CHECK_MANUALLY：需要你人工确认是否截取范围有误")

if __name__ == "__main__":
    main()