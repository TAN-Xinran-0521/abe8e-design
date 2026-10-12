# D:\TZS\scripts\list_all_features.py
"""
脚本功能：打印 GenBank 文件的全部 feature（不限类型），按起始位置排序，
用于定位未被单独标注为 CDS 的脱氨酶结构域所在区间（"缺口区域"）。

思路：
  1. 列出所有 feature（gene/misc_feature/CDS/primer_bind/promoter 等），
     附带起止坐标、长度、strand。
  2. 按起始坐标排序，打印相邻 feature 之间的"间隙"（gap），
     TadA-8e 很可能就位于 NLS 与 Cas9(D10A) 之间的某个未加框区间，
     或者被标注为 misc_feature / gene 而非 CDS。
  3. 对每个间隙，直接从 record.seq 切出对应核苷酸并翻译成氨基酸，
     供人工核对是否为 TadA-8e（~166-167 aa，起始密码子 M，
     不含长串 GGGGS 型 linker）。
"""
import argparse
from pathlib import Path
from Bio import SeqIO

LINKER_PATTERNS = ["GGGGS", "SGGS", "GGSSGG", "GGSGG", "GSGGS"]


def has_linker_signature(seq: str) -> bool:
    return any(p in seq for p in LINKER_PATTERNS)


def translate_region(record, start, end, strand):
    sub = record.seq[start:end]
    if strand == -1:
        sub = sub.reverse_complement()
    # 去掉可能的非3倍数余数，避免翻译报错
    trim = len(sub) - (len(sub) % 3)
    prot = sub[:trim].translate(to_stop=False)
    return str(prot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbk", required=True)
    ap.add_argument("--min_gap", type=int, default=100,
                     help="只显示核苷酸长度 >= min_gap 的间隙 (默认100bp)")
    args = ap.parse_args()

    gbk_path = Path(args.gbk)
    if not gbk_path.exists():
        raise FileNotFoundError(f"GenBank file not found: {gbk_path}")

    record = SeqIO.read(gbk_path, "genbank")
    print(f"=== Plasmid: {record.name} | {record.description} ===")
    print(f"Total length: {len(record.seq)} bp")
    print(f"Feature count: {len(record.features)}\n")

    # ---------- 1. 列出全部 feature，按起始坐标排序 ----------
    rows = []
    for i, feat in enumerate(record.features):
        start = int(feat.location.start)
        end = int(feat.location.end)
        strand = feat.location.strand
        quals = feat.qualifiers
        label = quals.get("label", quals.get("gene", quals.get("note", ["?"])))[0]
        aa_len = None
        if feat.type == "CDS" and "translation" in quals:
            aa_len = len(quals["translation"][0])
        rows.append({
            "idx": i, "type": feat.type, "label": label,
            "start": start, "end": end, "strand": strand,
            "length_nt": end - start, "aa_len": aa_len,
        })

    rows_sorted = sorted(rows, key=lambda r: r["start"])

    print("=== All features sorted by start position ===")
    for r in rows_sorted:
        strand_sym = "+" if r["strand"] == 1 else ("-" if r["strand"] == -1 else "?")
        aa_info = f", aa_len={r['aa_len']}" if r["aa_len"] else ""
        print(f"  [{r['idx']:2d}] {r['type']:15s} '{r['label']}'"
              f"  {r['start']}-{r['end']} ({strand_sym}, {r['length_nt']}bp{aa_info})")

    # ---------- 2. 找相邻 feature 之间的空隙 ----------
    print(f"\n=== Gaps between consecutive features (>= {args.min_gap} bp) ===")
    print("(这些空隙可能包含未单独标注的 TadA-8e CDS)\n")

    gap_found = False
    for j in range(len(rows_sorted) - 1):
        cur_end = rows_sorted[j]["end"]
        nxt_start = rows_sorted[j + 1]["start"]
        gap_len = nxt_start - cur_end
        if gap_len >= args.min_gap:
            gap_found = True
            prev_label = rows_sorted[j]["label"]
            next_label = rows_sorted[j + 1]["label"]
            print(f"--- Gap: {cur_end}-{nxt_start} ({gap_len} bp) "
                  f"between '{prev_label}' and '{next_label}' ---")

            # 分别按 +/- 链翻译，供人工比对
            for strand, tag in [(1, "+strand"), (-1, "-strand")]:
                prot = translate_region(record, cur_end, nxt_start, strand)
                aa_len = len(prot.rstrip("*"))
                has_linker = has_linker_signature(prot)
                print(f"  [{tag}] aa_len(approx)={aa_len}  "
                      f"contains_linker={has_linker}")
                print(f"  [{tag}] translation: {prot}\n")

    if not gap_found:
        print(f"没有找到长度 >= {args.min_gap} bp 的空隙。")
        print("尝试调低 --min_gap 参数重新运行，例如 --min_gap 50")

    # ---------- 3. 特别检查 Cas9(D10A) 之前的区域 ----------
    print("\n=== Special check: region immediately upstream of Cas9(D10A) ===")
    cas9_feats = [r for r in rows_sorted if "Cas9" in r["label"] and r["aa_len"] and r["aa_len"] > 1000]
    if cas9_feats:
        cas9 = cas9_feats[0]
        print(f"Cas9(D10A) location: {cas9['start']}-{cas9['end']} (strand {cas9['strand']})")
        # 找紧邻在 Cas9 之前的 NLS
        upstream_candidates = [r for r in rows_sorted if r["end"] <= cas9["start"]]
        if upstream_candidates:
            nearest = max(upstream_candidates, key=lambda r: r["end"])
            print(f"Nearest upstream feature: '{nearest['label']}' "
                  f"ends at {nearest['end']}")
            gap = cas9["start"] - nearest["end"]
            print(f"Gap between it and Cas9(D10A) start: {gap} bp")
            if gap > 0:
                for strand, tag in [(1, "+strand"), (-1, "-strand")]:
                    prot = translate_region(record, nearest["end"], cas9["start"], strand)
                    print(f"  [{tag}] translation: {prot}")
    else:
        print("未找到 aa_len > 1000 的 Cas9 feature，跳过此项检查。")


if __name__ == "__main__":
    main()