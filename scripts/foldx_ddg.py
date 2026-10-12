"""
foldx_ddg.py
用 FoldX 5.1 对 TadA-8e 脱氨酶候选突变算 ΔΔG 稳定性。

流程（FoldX 标准三段）：
  1. RepairPDB : 能量最小化修复 ESMFold 预测结构（必做，否则 ΔΔG 是噪声）
  2. 生成 individual_list.txt : 从 candidate_consensus_exact.csv 转 FoldX 突变格式
       FoldX 格式: <WT><chain><resnum><MUT>;  例如 F83L -> FA83L;
  3. BuildModel : 逐突变算 ΔΔG（numberOfRuns 次取均值），解析 Average_*.fxout

编号：PDB 残基 1..166 = 序列原生编号，与候选表 position 逐一对齐（零偏移，已核实）。
符号约定（FoldX）：ΔΔG = G(mutant) - G(wildtype)
  ΔΔG < 0 -> 稳定化（stabilizing）
  ΔΔG > 0 -> 去稳定化（destabilizing）
口径边界：结构为 ESMFold 预测（非实验），ΔΔG 为力场估计，取定性排序 +
  与语言模型 delta 的方向一致性对照，不声称精确 kcal/mol。
"""
import argparse, os, sys, subprocess, shutil
import pandas as pd

# WT 序列（166 aa，与 PDB 残基 1..166 对齐），用于闸门校验突变位点 WT 氨基酸
TADA8E_SEQ = ("SEVEFSHEYWMRHALTLAKRARDEREVPVGAVLVLNNRVIGEGWNRAIGLHDPTAHAEIMALR"
              "QGGLVMQNYRLIDATLYVTFEPCVMCAGAMIHSRIGRVVFGVRNSKRGAAGSLMNVLNYPGMN"
              "HRVEITEGILADECAALLCDFYRMPRQVFNAQKKAQSSIN")

AA3 = set("ACDEFGHIKLMNPQRSTVWY")


def run_foldx(exe, args_list, cwd):
    """调 FoldX exe，实时打印，返回 returncode。"""
    cmd = [exe] + args_list
    print(f"[FoldX] {' '.join(cmd)}\n(cwd={cwd})")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout[-2000:])   # 只打尾部，避免刷屏
    if proc.returncode != 0:
        print("[FoldX STDERR]", proc.stderr[-2000:])
    return proc.returncode


def parse_mutation(mut):
    """'F83L' -> (wt='F', pos=83, mutaa='L')。"""
    wt = mut[0]
    mutaa = mut[-1]
    pos = int(mut[1:-1])
    return wt, pos, mutaa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foldx", required=True)
    ap.add_argument("--pdb", default=__import__('repo_paths').repo_path('data/structures/tada8e_structure.pdb'))
    ap.add_argument("--candidates",
                    default=__import__('repo_paths').repo_path('results/deaminase_compare/candidate_consensus_exact.csv'))
    ap.add_argument("--chain", default="A")
    ap.add_argument("--runs", type=int, default=5, help="FoldX numberOfRuns (BuildModel)")
    ap.add_argument("--workdir", default=__import__('repo_paths').repo_path('results/foldx'))
    ap.add_argument("--skip_repair", action="store_true",
                    help="若已 Repair 过，跳过 RepairPDB 直接算")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    exe = os.path.abspath(args.foldx)
    pdb_name = os.path.basename(args.pdb)

    # 把原始 pdb 拷进 workdir（FoldX 在 cwd 里读写）
    src_pdb = os.path.join(args.workdir, pdb_name)
    shutil.copy(args.pdb, src_pdb)

    # -------- 1. RepairPDB --------
    repaired_name = pdb_name.replace(".pdb", "_Repair.pdb")
    repaired_path = os.path.join(args.workdir, repaired_name)
    if args.skip_repair and os.path.exists(repaired_path):
        print(f"[skip] 已存在 {repaired_name}，跳过 RepairPDB")
    else:
        rc = run_foldx(exe, ["--command=RepairPDB", f"--pdb={pdb_name}",
                             f"--pdb-dir={args.workdir}",
                             f"--output-dir={args.workdir}"], cwd=args.workdir)
        if rc != 0 or not os.path.exists(repaired_path):
            print("[ABORT] RepairPDB 失败或未生成 _Repair.pdb")
            sys.exit(1)
    print(f"[ok] repaired structure -> {repaired_name}\n")

    # -------- 2. 读候选 + 闸门校验 WT 氨基酸 + 生成 individual_list --------
    cand = pd.read_csv(args.candidates)
    muts = []
    bad = []
    for m in cand["mutation"]:
        wt, pos, mutaa = parse_mutation(m)
        if mutaa not in AA3 or wt not in AA3:
            bad.append((m, "非标准氨基酸")); continue
        if pos < 1 or pos > len(TADA8E_SEQ):
            bad.append((m, "位点越界")); continue
        # 闸门：候选表 WT 必须与序列一致（防编号错位）
        if TADA8E_SEQ[pos - 1] != wt:
            bad.append((m, f"WT不符: 序列[{pos}]={TADA8E_SEQ[pos-1]} vs 候选{wt}"))
            continue
        muts.append(m)

    if bad:
        print("[WARN] 以下候选被跳过：")
        for m, why in bad:
            print(f"    {m}: {why}")
    if not muts:
        print("[ABORT] 无有效候选（编号可能全错位）"); sys.exit(1)
    print(f"[ok] {len(muts)} 个有效候选通过 WT 闸门校验\n")

    # FoldX individual_list: 每行 <WT><chain><pos><MUT>;
    ind_list = os.path.join(args.workdir, "individual_list.txt")
    with open(ind_list, "w") as f:
        for m in muts:
            wt, pos, mutaa = parse_mutation(m)
            f.write(f"{wt}{args.chain}{pos}{mutaa};\n")
    print(f"[ok] individual_list.txt 写入 {len(muts)} 条突变\n")

    # -------- 3. BuildModel 逐突变算 ΔΔG --------
    rc = run_foldx(exe, ["--command=BuildModel",
                         f"--pdb={repaired_name}",
                         f"--pdb-dir={args.workdir}",
                         "--mutant-file=individual_list.txt",
                         f"--numberOfRuns={args.runs}",
                         f"--output-dir={args.workdir}"], cwd=args.workdir)
    if rc != 0:
        print("[ABORT] BuildModel 失败"); sys.exit(1)

    # 解析 Average_*.fxout（每行对应 individual_list 的一条突变，顺序一致）
    avg_files = [f for f in os.listdir(args.workdir)
                 if f.startswith("Average_") and f.endswith(".fxout")]
    if not avg_files:
        print("[ABORT] 未找到 Average_*.fxout"); sys.exit(1)
    avg_path = os.path.join(args.workdir, sorted(avg_files)[0])

    # fxout 格式：前若干行是表头/注释，数据行含 Pdb, total energy(ΔΔG), SD...
    rows = []
    with open(avg_path) as f:
        lines = f.readlines()
    # 找数据起始（含 tab 的行、且第 2 列可转 float）
    data_lines = []
    for ln in lines:
        parts = ln.rstrip("\n").split("\t")
        if len(parts) >= 3:
            try:
                float(parts[1]); float(parts[2])
                data_lines.append(parts)
            except ValueError:
                continue
    if len(data_lines) != len(muts):
        print(f"[WARN] Average 行数 {len(data_lines)} != 突变数 {len(muts)}，"
              f"按顺序对齐可能错位，请核对 {avg_path}")

    cand_map = {m: (parse_mutation(m), cand.set_index("mutation").loc[m])
                for m in muts}
    for m, parts in zip(muts, data_lines):
        ddg = float(parts[1])       # total energy = ΔΔG
        sd = float(parts[2])
        (wt, pos, mutaa), crow = cand_map[m]
        rows.append({
            "mutation": m,
            "mutation_lit": f"{wt}{pos+1}{mutaa}",   # 文献编号(+1)
            "region": crow["region"],
            "lm_mean_delta": crow["mean_delta"],
            "n_models": crow["n_models"],
            "ddg_foldx": round(ddg, 3),
            "ddg_sd": round(sd, 3),
            "stability": "stabilizing" if ddg < 0 else "destabilizing",
        })

    out = pd.DataFrame(rows).sort_values("ddg_foldx").reset_index(drop=True)
    out_csv = os.path.join(args.workdir, "ddg_results.csv")
    out.to_csv(out_csv, index=False)
    print("\n=== FoldX ΔΔG 结果（按 ΔΔG 升序，越负越稳定）===")
    print(out.to_string(index=False))
    print(f"\nSaved -> {out_csv}")

    # 方向一致性快报：语言模型高 delta 是否倾向稳定化
    stab = out[out["ddg_foldx"] < 0]
    print(f"\n稳定化候选(ΔΔG<0): {len(stab)}/{len(out)}")
    print("语言模型 delta 与 FoldX ΔΔG 的方向对照见 ddg_results.csv，"
          "报告用 Spearman/散点做一致性分析（下一步出图）。")


if __name__ == "__main__":
    raise SystemExit(
        "This historical FoldX entry point is temporarily disabled pending "
        "migration to the validated shared parser. It may overwrite research "
        "outputs or use obsolete parsing logic. "
        "For read-only saved-result verification, use "
        "scripts/validate_protein_saved_results.py with a new --outdir."
    )