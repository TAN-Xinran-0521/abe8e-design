# 项目研究与结果复现

## AI驱动的ABE8e碱基编辑候选设计与精准A→G矫正研究框架

本文档侧重：
- 项目研究问题；
- 已完成工作；
- 核心结论；
- 结果来源；
- 复现命令；
- 已知限制。

如果希望为自己的位点生成候选，请先阅读[README.md](README.md)。

---

## 1. 研究背景与目标

ABE能够实现A·T→G·C转换，为部分致病点突变的矫正提供工具。

但候选选择不能只依赖最高预测效率，还需要考虑：
- 目标位置和PAM可及性；
- 旁观者编辑；
- 潜在基因组匹配；
- 预测不确定性；
- 编辑器本体的候选改造；
- 后续功能与递送验证。

本项目构建两个相关的计算研究模块：

### gRNA层

数据处理与效率预测
→ 位置和序列规律解释
→ 旁观者风险估计
→ 潜在匹配评价
→ 候选排序。

### 蛋白层

蛋白模型提名
→ 结构条件评分
→ FoldX稳定性计算
→ 共识与分歧分析
→ 待实验验证候选。

两模块共同提供计算设计依据。
不能将“某位点Cas相关匹配负担较高”直接解释为
“改造TadA稳定性就能解决该问题”。

当前结果以公开数据和计算分析为主，未完成新候选湿实验验证。

---

## 2. 当前复现范围

已经验证：

- 核心B′推理；
- 新建Conda环境中的HFE示例一致性；
- 22项自动测试；
- 固定数据上的B′原始／截断指标；
- 保存的HFE／LMNA搜索结果精确目标排除；
- HFE三个候选的全局评分；
- 保存FoldX Raw、Dif、Average文件交叉解析；
- 蛋白模型评分矩阵与候选表一致性；
- 完整精度蛋白相关性。

未宣称完成：

- 全部模型从头重训；
- 全基因组搜索在新环境重新执行；
- 新FoldX计算；
- ESM模型重新推理；
- 独立外部用户复现；
- 跨平台支持；
- 实验功能和安全性验证。

---

## 3. 数据与模型体系

### 3.1 效率数据

数据来源：BE-dataHIVE。

主要条件：
- ABE8e；
- HEK293T；
- study_id=3；
- 标签：`efficiency_full_grna_reported`。

质控后核心数据：22,187条。

| 集合 | 样本量 |
|---|---:|
| Train | 15,530 |
| Validation | 3,328 |
| Test | 3,329 |

原划分按完全相同的`full_context_sequence`分组。

原划分依赖数据快照和输入顺序，复现历史指标应使用固定文件，不能重新下载后直接随机划分。

### 3.2 模型对照

历史同源测试结果：

| 模型 | 特征 | Spearman | MAE |
|---|---|---:|---:|
| XGBoost全特征基准 | 107列 | 0.751 | 0.100 |
| Transformer全特征 | 107列 | 0.735 | 0.103 |
| Transformer纯序列 | 80列 | 0.668 | 0.109 |
| B′ | 83列 | 0.670 | 0.109 |

全特征模型依赖外部energy信息，不能直接用其0.751代表实际
新候选设计工具的表现。

### 3.3 B与B′的区别

- 中间方案B：去除energy，但保留数据库target-Tm口径，历史Spearman约0.685。
- 最终B′：所有物理特征由输入20 nt序列计算，历史Spearman约0.670。

B′特征：
- 80维位置one-hot；
- MFE；
- Tm；
- 相同Tm的重复列。


---

## 4. B′指标与后处理审计

原研究训练脚本评价未截断输出；
设计和区间工具将输出截断到[0,1]。

修正版分别报告：

| Test政策 | Spearman | MAE |
|---|---:|---:|
| Raw | 0.670049192932404 | 0.10876274617229263 |
| Clip [0,1] | 约0.6700606317 | 约0.1085982630 |

验证集90条、测试集108条原始预测小于0。
两组均无大于1的预测。

共享B′特征模块重新计算后，验证集与测试集原始四项指标与历史JSON一致。

截断带来的微小误差变化属于输出范围约束，不是重新训练收益。

### 复现

在仓库根目录的Anaconda Prompt中：

```bat
python scripts\validate_bprime_prediction_policy.py --outdir results\user_runs\bprime_policy
```

所需文件：

```text
models/xgboost_design_v2.pkl
data/processed/val.csv
data/processed/test.csv
results/xgboost_design_v2_results.json
```

输出：

```text
metrics_by_policy.csv
historical_comparison.csv
val_predictions.csv
test_predictions.csv
run_manifest.json
```

正式核验结果：

```text
results/release_validation/bprime_prediction_policy/
```

---

## 5. 解释分析与旁观者先验

### 5.1 位置重要性

B′的SHAP位置贡献在第4—8位较高，第5—6位附近为峰值。

该结果描述模型预测依赖哪些位置。

历史结果：

```text
results/shap/
```

### 5.2 经验位置先验

从训练集中选择：
> 第1—12位归因区间内恰好含一个A的样本。

样本数：2,672。

这不是要求整条20 nt序列只有一个A。

经验均值归一化后：
- 第5—6位附近为峰值；
- 第9—10位仍有非零reported效率；
- 某些位置无直接估计，使用插值、边界补齐和0.01下限。

第13—20位的0.01不能解释为这些位置的实测编辑活性。

相关文件：

```text
results/bystander/position_activity.csv
results/bystander/empirical_prior.json
scripts/calibrate_bystander_prior.py
scripts/bystander_prior.py
```

### 5.3 邻碱基偏好

当前单A分析中，位置归一化结果支持：

```text
TA > CA > GA
```

AA不在该单A归因分析范围内。

位置、motif和SHAP属于同源数据上的多方法交叉支持，
不是三组完全独立实验。

### 5.4 风险计算

旁观者风险：

```text
非目标A的位置权重之和
÷
目标A与非目标A的位置权重之和
```

该分数依赖位置先验和归因假设，不是直接监督学习得到的
逐位点编辑概率。

历史HFE示例中，一条窗口内额外A计数为0的候选，
考虑第9位A后得到约0.380的加权风险。

---

## 6. HFE／LMNA潜在匹配结果

### 6.1 搜索条件

- 所用hg38参考文件；
- 对应PAM规则；
- 最多4处错配；
- 保存的Cas-OFFinder输出；
- 排除预期目标，而不是删除全部零错配匹配。

### 6.2 精确坐标核验

HFE：
- 正链protospacer：chr6:26092909—26092928；
- Cas-OFFinder完整匹配0-based起点：26092908；
- 目标A在protospacer第5位；
- 目标坐标：26092913。

LMNA：
- 负链protospacer：chr1:156138599—156138618；
- 完整匹配0-based起点：156138595；
- 需加3 nt PAM偏移后转换；
- 目标A在protospacer第6位；
- 对应正链目标坐标：156138613。

预期靶点存在1处参考／变异差异，因此不能要求on-target必须零错配。

### 6.3 结果

| 项目 | HFE／SpCas9 | LMNA／VRQR |
|---|---:|---:|
| PAM | NGG | NGA |
| 原始解析记录 | 80 | 2,075 |
| 排除预期靶点 | 1 | 1 |
| 剩余匹配 | 79 | 2,074 |
| 2错配 | 0 | 8 |
| 3错配 | 5 | 89 |
| 4错配 | 74 | 1,977 |

这是保存输出的重新解析，不是新一轮基因组搜索。

### 复现HFE

`--genome`填写原搜索参考目录；保存输出模式不会重新搜索该目录。

```bat
python scripts\run_true_offtarget.py --candidates results\hfe_offtarget_real\run_SpCas9\candidates_normalized_input.csv --genome YOUR_HG38_DIRECTORY --pam NGG --max_mismatches 4 --cas_output results\hfe_offtarget_real\run_SpCas9\casoffinder_output.txt --outdir results\user_runs\hfe_exact --tolerance 0 --summary_only
```

### 复现LMNA

```bat
python scripts\run_true_offtarget.py --candidates results\lmna_offtarget_real\run_VRQR\candidates_normalized_input.csv --genome YOUR_HG38_DIRECTORY --pam NGA --max_mismatches 4 --cas_output results\lmna_offtarget_real\run_VRQR\casoffinder_output.txt --outdir results\user_runs\lmna_exact --tolerance 0 --summary_only
```

不能将计数比值解释成实测安全性倍数差异。

---

## 7. HFE统一全局排序

### 7.1 为什么需要修正历史流程？

历史实现存在：
- 设计综合分使用加权旁观者指标，但Pareto仍用旧计数；
- 全基因组重评分重新使用旧旁观者计数；
- 按变体单独归一化后无法直接跨组比较。

修正版统一：
- B′完整精度输出；
- 位置加权旁观者风险；
- 对应PAM原始匹配结果；
- 合并后一次归一化；
- 一致的评分与Pareto定义。

### 7.2 修正版结果

| 排名 | 标签 | 效率 | 旁观者风险 | 匹配数 | 综合分 |
|---|---|---:|---:|---:|---:|
| 1 | SpCas9 | 0.334684 | 0.469109 | 79 | -0.154089 |
| 2 | SpRY | 0.245277 | 0.379537 | 360 | -0.306582 |
| 3 | SpCas9-NG | 0.288351 | 0.539391 | 336 | -0.346477 |

SpCas9与SpRY均位于Pareto前沿。

基因组评分还考虑错配位置，因此匹配数较少不必然意味着加权分更低。

### 复现

先确保已生成包含加权旁观者指标的HFE候选表，
或使用已发布的核验输入：

```bat
python scripts\validate_hfe_global_ranking.py --candidates results\release_validation\hfe_candidates_patched.csv --outdir results\user_runs\hfe_global
```

正式核验结果：

```text
results/release_validation/hfe_global_exact/
```

该脚本检查：
- 原始输入与输出对应；
- 解析行数；
- PAM匹配；
- 错配数；
- 重复基因组记录；
- 精确目标排除；
- 全局归一化；
- 综合分计算。

不验证原基因组搜索是否覆盖全部理论目标。

---

## 8. TadA-8e候选与FoldX结果

### 8.1 候选来源

三种模型：
- ESM2-35M；
- ESM1v-650M；
- ESM-IF1。

候选表条目数：
- ESM2：15；
- ESM1v：15；
- ESM-IF1：20。

合并去重后：44个替换。

`candidate_consensus_exact.csv`是历史文件名。
该表包含单模型候选，不是44个严格多模型共识。

### 8.2 稳定性结果

每个替换5次计算重复。

通过核验：
- Raw突变体−匹配WT≈Dif；
- Dif重复均值≈Average；
- Average SD≈总体标准差，ddof=0；
- 23／44个平均ΔΔG小于0。

部分结果：

| 候选 | 平均ΔΔG |
|---|---:|
| C145E | -1.064260 |
| C145R | -1.032530 |
| C145K | -0.952558 |
| C145N | -0.943586 |
| C145D | -0.836135 |
| C145Q | -0.802883 |
| S108P | -0.479730 |
| F83L | +0.017003 |
| P151R | +1.451300 |

负值表示当前计算的稳定化倾向，不等于编辑功能改善。

F83L接近零，不应继续列为已获得稳定化支持的代表。
P151R显示序列模型提名与FoldX稳定性预测之间的分歧。

### 8.3 正式解析复现

```bat
python scripts\parse_foldx_ddg.py --outdir results\user_runs\foldx_reparse
```

所需输入：

```text
results/foldx/individual_list.txt
results/foldx/Average_tada8e_structure_Repair.fxout
results/foldx/Dif_tada8e_structure_Repair.fxout
results/foldx/Raw_tada8e_structure_Repair.fxout
results/deaminase_compare/candidate_consensus_exact.csv
data/structures/tada8e_sequence.fasta
```

输出：
- `ddg_results.csv`
- `repeat_energy_checks.csv`
- `run_manifest.json`

解析器：
- 按表头名称读取；
- 按模型编号匹配突变；
- 检查重复和缺失；
- 保留完整精度；
- 不自动生成文献编号＋1。

正式核验目录：

```text
results/release_validation/foldx_parser_release/
```

---

## 9. 蛋白评分与ΔΔG相关性

### 9.1 完整精度结果

| 模型分数 | n | Spearman | p值 |
|---|---:|---:|---:|
| ESM2 | 44 | -0.0384778013 | 0.8041522826 |
| ESM1v | 44 | 0.2545454545 | 0.0954253896 |
| 序列模型均分 | 44 | 0.2191684285 | 0.1528916115 |
| ESM-IF1 | 44 | -0.4786469345 | 0.0010139509 |


### 9.2 与历史数字的区别

历史脚本先将分数舍入到3位小数，再计算相关性：

```text
ρ = -0.4784523771
p = 0.0010194590
```

修正版保留完整精度。

结论方向不变，但正式新版应使用完整精度结果，历史结果保留用于追溯。

### 9.3 复现命令

先完成第8节的FoldX解析，再运行：

```bat
python scripts\ddg_lm_consistency.py --foldx results\user_runs\foldx_reparse\ddg_results.csv --outdir results\user_runs\protein_consistency
```

所需评分文件：

```text
results/deaminase/logprob_matrix.npy
results/deaminase/candidate_mutations.csv

results/deaminase_650m/logprob_matrix.npy
results/deaminase_650m/candidate_mutations.csv

results/deaminase_if1/logprob_matrix.npy
results/deaminase_if1/candidate_mutations.csv
```

输出：
- `lm_ddg_merged.csv`
- `correlations.csv`
- `matrix_validation.csv`
- `run_manifest.json`

正式核验目录：

```text
results/release_validation/protein_consistency_release/
```

### 9.4 解释边界

- 三种矩阵均为166×20；
- 与候选CSV的评分在给定容差内一致；
- 这种核验是内部一致性检查，不是实验验证；
- 两序列模型均分是未经效应校准的探索汇总；
- 未检出显著相关不等于证明独立或正交；
- ESM-IF1与FoldX共享预测结构输入；
- 当前相关性受候选选择影响；
- 稳定性与编辑功能是不同评价目标。

新版不再生成旧的中位数四象限“共识”或混合量纲优先分。

---

## 10. 预测区间与统计限制

历史校准结果：

| 名义水平 | 测试集经验覆盖率 |
|---|---:|
| 80% | 0.8005407029 |
| 90% | 0.8954641033 |

文件：

```text
results/conformal/conformal_qhat.json
```

校准数据与早停验证数据复用，因此不能直接主张标准split-conformal独立校准条件下的有限样本覆盖保证。

这些结果可用于描述当前数据上的经验区间表现。

---

## 11. 递送与应用定位

历史组件估算中，当前构建假设下的表达盒约5,949 bp，
高于约4,700 bp的典型单AAV载荷范围。

这表示需要进一步评估：
- 分体递送；
- 紧凑底盘；
- 实际表达组件与功能适配。

近期应用定位：
> 为实验团队提供研究性候选、风险信息与计算依据，
> 通过后续实验验证筛选范围和模型假设。

---

## 12. 环境与测试复现

安装：

```bat
conda env create -f environments\environment-core.yml
conda activate abe8e-core-check
python -m pip check
```

测试：

```bat
python -m unittest discover -s tests -p "test_*.py" -v
```

当前：22项通过。

最新日志：

```text
results/release_validation/release_core/tests_22.txt
```

新环境记录：

```text
results/release_validation/clean_environment/
```

这些记录来自作者同一台Windows电脑的新环境，
不是独立外部用户复现。

---

## 13. 历史结果与修正版管理

必须保留而不静默覆盖：
- 原始模型指标；
- 固定数据划分；
- 原始搜索输出；
- FoldX原始文件；
- 历史三位小数蛋白结果；
- 早期测试日志。

修正版结果集中于：

```text
results/release_validation/
```

历史研究脚本不全部属于推荐用户入口。

尤其：
- `foldx_ddg.py`新计算入口仍禁用；
- 旧`priority_quadrant.csv`仅作探索历史；
- 旧启发式HFE排序不能当作全基因组最终排序；
- 所有绘图必须明确使用历史还是修正版数据。

---

## 14. 复现层级说明

| 层级 | 本项目示例 | 当前状态 |
|---|---|---|
| 使用训练模型推理 | HFE候选设计 | 已验证 |
| 固定测试集评价 | B′原始／截断指标 | 已验证 |
| 保存搜索结果重分析 | 精确坐标排除与HFE全局排序 | 已验证 |
| 保存FoldX结果重分析 | 44候选均值与SD | 已验证 |
| 保存矩阵统计分析 | 四组相关性 | 已验证 |
| 原始数据重新下载并重建全部训练流程 | 完整训练复现 | 已验证 |
| 重新执行全基因组搜索 | 新Cas-OFFinder运行 | 已验证 |
| 重新运行FoldX和ESM | 新蛋白计算 | 已发布验证 |
| 湿实验验证 | 编辑功能、安全性、递送 | 尚未完成 |


---

## 15. 发布与引用

数据、模型及第三方资源的公开条件见：

- [docs/data_sources.md](docs/data_sources.md)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)