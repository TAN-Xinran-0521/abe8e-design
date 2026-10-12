"""
bystander_prior.py
ABE8e position-weighted bystander risk.
把"窗口内 A 计数"升级为"按位点活性加权"：核心位(5-6)的 bystander A 危害
远大于窗口边缘。默认用文献口径 ABE8e 编辑窗曲线(核心4-8/峰值5-7,峰归一=1.0),
与本项目 SHAP 位点重要性(5>6>7>4>8)内部印证；也可用 calibrate_bystander_prior.py
产出的经验曲线替换,做到完全数据自洽。
"""
from __future__ import annotations
import json
import math

# 1-indexed protospacer 位置(1 = PAM-distal 5' 端, 20 = PAM-proximal)
# 相对 A->G 活性, 峰归一到 1.0。近似值, 可调。
ABE8E_POSITION_PRIOR = {
    1: 0.02, 2: 0.05, 3: 0.20, 4: 0.60, 5: 0.95, 6: 1.00, 7: 0.90,
    8: 0.65, 9: 0.30, 10: 0.12, 11: 0.05, 12: 0.03, 13: 0.02, 14: 0.02,
    15: 0.02, 16: 0.01, 17: 0.01, 18: 0.01, 19: 0.01, 20: 0.01,
}


def load_prior(path=None):
    """path=None 用手工近似先验；否则加载 calibrate 产出的 empirical_prior.json"""
    if path is None:
        return dict(ABE8E_POSITION_PRIOR)
    with open(path) as f:
        d = json.load(f)
    prior = {int(k): float(v) for k, v in d.items()}
    if set(prior) != set(range(1, 21)):
        raise ValueError("Prior must contain positions 1 through 20.")
    if any(not math.isfinite(v) or v < 0 for v in prior.values()):
        raise ValueError("Prior weights must be finite and nonnegative.")
    return prior

def bystander_risk(protospacer, target_pos, prior=None, eps=1e-9):
    """
    位置加权 bystander 风险。
    protospacer : 20nt guide (5'->3', position 1 = PAM-distal)
    target_pos  : 目标 A 的 1-indexed 位置
    prior       : {position: relative_activity}, 默认 ABE8E_POSITION_PRIOR

    返回 dict:
      weighted_target     : 目标 A 处的先验活性
      weighted_bystanders : 其余所有 A 的先验活性之和
      bystander_ratio     : w_bys / (w_target + w_bys) ∈ [0,1)
                            = 期望落在非目标 A 上的编辑事件占比(可直接进 composite)
      n_bystander_A       : 原始计数(向后兼容/报告用)
      bystander_positions : [(pos, weight), ...] 便于可视化/解释
    """
    if prior is None:
        prior = ABE8E_POSITION_PRIOR
    protospacer = protospacer.upper().strip()
    if len(protospacer) != 20 or set(protospacer) - set("ACGT"):
        raise ValueError("Expected a 20nt A/C/G/T protospacer.")
    if not isinstance(target_pos, int) or not 1 <= target_pos <= 20:
        raise ValueError("target_pos must be an integer from 1 to 20.")
    if set(prior) != set(range(1, 21)):
        raise ValueError("Prior must cover positions 1 through 20.")
    if any(not math.isfinite(v) or v < 0 for v in prior.values()):
        raise ValueError("Prior weights must be finite and nonnegative.")
    if protospacer[target_pos - 1] != 'A':
        raise ValueError(f"target_pos {target_pos} 不是 protospacer 中的 'A'")

    w_target = prior.get(target_pos, 0.0)
    bys = [(i, prior.get(i, 0.0)) for i in range(1, len(protospacer) + 1)
           if protospacer[i - 1] == 'A' and i != target_pos]
    w_bys = sum(w for _, w in bys)
    ratio = w_bys / (w_target + w_bys + eps)
    return {
        "weighted_target": float(w_target),
        "weighted_bystanders": float(w_bys),
        "bystander_ratio": float(ratio),
        "n_bystander_A": len(bys),
        "bystander_positions": [(p, float(w)) for p, w in bys],
    }