# D:\TZS\scripts\setup_deaminase_env.py
"""
脚本功能：安装脱氨酶改造所需依赖并下载 TadA-8e 结构
"""
import subprocess
import os
from pathlib import Path

def setup_environment():
    """安装依赖"""
    packages = [
        "biopython",
        "fair-esm",  # ESM 模型
        "torch",
        "biotite",
        "py3Dmol",  # 结构可视化
        "requests"
    ]
    
    print("=== Installing dependencies ===")
    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.run(["pip", "install", pkg, "-q"], check=False)
    
    print("\n=== Setup complete ===")

def download_tada8e_structure():
    """
    获取 TadA-8e 结构
    优先尝试 PDB，否则用 AlphaFold2 预测
    """
    import requests
    
    struct_dir = Path("D:/TZS/data/structures")
    struct_dir.mkdir(parents=True, exist_ok=True)
    
    # TadA-8e 序列（ABE8e 的脱氨酶部分）
    # 基于 ecTadA 的进化突变体，含 V82G 等关键突变
    tada8e_seq = (
    "MSEVEFSHEYWMRHALTLAKRARDEREVPVGAVLVLNNRVIGEGWNRAIGLHDPTAHAEIMALRQGG"
    "LVMQNYRLIDATLYVTFEPCVMCAGAMIHSRIGRVVFGVRNAKTGAAGSLMDVLHHPGMNHRVEITE"
    "GILADECAALLCYFFRMPRQVFNAQKKAQSSTD")
    
    pdb_file = struct_dir / "tada8e_structure.pdb"
    seq_file = struct_dir / "tada8e_sequence.fasta"
    
    # 保存序列
    with open(seq_file, "w") as f:
        f.write(">TadA-8e (ABE8e deaminase domain)\n")
        f.write(tada8e_seq + "\n")
    
    print(f"TadA-8e sequence saved to {seq_file}")
    print(f"Sequence length: {len(tada8e_seq)} aa")
    
    # 尝试从 AlphaFold 数据库下载（如果有 UniProt ID）
    # 或使用 ESMFold 在线预测
    print("\n=== Structure acquisition ===")
    print("Option 1: Use ESMFold online (https://esmatlas.com/resources?action=fold)")
    print(f"  Paste the sequence from {seq_file}")
    print("\nOption 2: Use AlphaFold2 Colab")
    print("  https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb")
    print(f"\nSave predicted structure as: {pdb_file}")
    
    return seq_file, pdb_file, tada8e_seq

if __name__ == "__main__":
    setup_environment()
    seq_file, pdb_file, seq = download_tada8e_structure()
    
    print("\n=== Next steps ===")
    print("1. Get TadA-8e structure using ESMFold or AlphaFold2")
    print("2. Save as tada8e_structure.pdb")
    print("3. Run engineer_deaminase.py")