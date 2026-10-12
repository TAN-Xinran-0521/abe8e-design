# D:\TZS\scripts\extract_tada8e_domain.py
"""
脚本功能：
  用已确认坐标 (501-999, +strand) 从 GenBank 精确提取 TadA-8e 脱氨酶结构域，
  与 linker (999-1095) 分离保存，并做健全性检查后写入 fasta。

坐标依据（已通过 list_all_features.py 交叉验证）：
  SV40 NLS      : 480-501
  TadA-8e domain: 501-999   (166 aa, 无起始 Met, 直接续接 NLS 读码框)
  Linker        : 999-1095  (32 aa, SGGSSGGSSGSETPGTSESATPESSGGSSGGS)
  Cas9(D10A)    : 1095-5196 (1367 aa)
"""
import argparse
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq

# 已知野生型 E. coli TadA 参考序列 (UniProt, 167 aa, 含起始 Met)
# 用于计算 TadA-8e 相对野生型的突变位点
ECTADA_WT = (
    "MSEVEFSHEYWMRHALTLAKRARDEREVPVGAVLVLNNRVIGEGWNRAIGLHDPTAHAE"
    "IMALRQGGLVMQNYRLIDATLYVTFEPCVMCAGAMIHSRIGRVVFGVRNAKTGAAGSLM"
    "DVLHYPGMNHRVEITEGILADECAALLCYFFRMPRQVFNAQKKAQSSTD"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbk", required=True)
    ap.add_argument("--tada_start", type=int, default=501)
    ap.add_argument("--tada_end", type=int, default=999)
    ap.add_argument("--linker_end", type=int, default=1095)
    ap.add_argument("--out_domain",
                     default=__import__('repo_paths').repo_path('data/structures/tada8e_sequence.fasta'))
    ap.add_argument("--out_linker",
                     default=__import__('repo_paths').repo_path('data/structures/linker_sequence.fasta'))
    args = ap.parse_args()

    gbk_path = Path(args.gbk)
    if not gbk_path.exists():
        raise FileNotFoundError(f"GenBank file not found: {gbk_path}")

    record = SeqIO.read(gbk_path, "genbank")
    print(f"=== Plasmid: {record.name} | {record.description} ===")
    print(f"Total length: {len(record.seq)} bp\n")

    # ---------- 1. 切出 TadA-8e 结构域 ----------
    tada_nt = record.seq[args.tada_start:args.tada_end]
    tada_aa = str(tada_nt.translate(to_stop=False))

    print("=== TadA-8e domain ===")
    print(f"Coordinates: {args.tada_start}-{args.tada_end} "
          f"({args.tada_end - args.tada_start} bp)")
    print(f"Amino acid length: {len(tada_aa)} aa")
    print(f"Sequence: {tada_aa}")

    # 健全性检查
    has_stop = "*" in tada_aa
    linker_hits = [p for p in ["GGGGS", "SGGS", "GGSSGG"] if p in tada_aa]
    print(f"Contains premature stop codon: {has_stop}")
    print(f"Contains linker signature: {linker_hits if linker_hits else 'None'}")

    # ---------- 2. 切出 linker ----------
    linker_nt = record.seq[args.tada_end:args.linker_end]
    linker_aa = str(linker_nt.translate(to_stop=False))

    print("\n=== Linker (should NOT be used for structure modeling) ===")
    print(f"Coordinates: {args.tada_end}-{args.linker_end} "
          f"({args.linker_end - args.tada_end} bp)")
    print(f"Amino acid length: {len(linker_aa)} aa")
    print(f"Sequence: {linker_aa}")

    # ---------- 3. 与野生型 TadA 对齐比较（找出 TadA-8e 特有突变）----------
    print("\n=== Alignment vs wild-type E. coli TadA ===")
    # TadA-8e 没有起始 Met，WT 参考序列去掉首位 M 后逐位比对
    wt_aligned = ECTADA_WT[1:]  # 去掉起始 M，166 aa
    print(f"WT (no start-Met) length: {len(wt_aligned)} aa")
    print(f"TadA-8e length          : {len(tada_aa)} aa")

    if len(wt_aligned) == len(tada_aa):
        diffs = []
        for i, (wt_res, mut_res) in enumerate(zip(wt_aligned, tada_aa), start=1):
            if wt_res != mut_res:
                diffs.append(f"{wt_res}{i}{mut_res}")
        print(f"Mutations relative to WT ({len(diffs)} total): {diffs}")
    else:
        print("[警告] 长度不一致，无法逐位对齐，请人工核对野生型参考序列版本。")

    # ---------- 4. 写文件 ----------
    Path(args.out_domain).parent.mkdir(parents=True, exist_ok=True)

    status = "OK" if (not has_stop and not linker_hits and len(tada_aa) in (166, 167)) \
        else "CHECK_MANUALLY"

    with open(args.out_domain, "w") as f:
        f.write(f">TadA-8e_domain_from_Addgene138489_coord{args.tada_start}-{args.tada_end}_{status}\n")
        f.write(tada_aa + "\n")

    with open(args.out_linker, "w") as f:
        f.write(f">Linker_from_Addgene138489_coord{args.tada_end}-{args.linker_end}\n")
        f.write(linker_aa + "\n")

    print(f"\n已写入: {args.out_domain}  (status={status})")
    print(f"已写入: {args.out_linker}  (仅供记录，不用于建模)")


if __name__ == "__main__":
    main()