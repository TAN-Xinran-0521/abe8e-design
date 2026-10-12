# ABE8e候选设计与编辑方案评价工具

**AI-driven ABE8e Candidate Design and Evaluation**

本仓库提供面向ABE8e腺嘌呤碱基编辑的研究性计算工具，帮助使用者：

- 从包含目标位点的DNA序列中生成候选sgRNA；
- 使用已训练的B′模型估计编辑效率；
- 评价位置加权的旁观者编辑风险；
- 根据效率、旁观者风险和序列启发式评分筛选候选；
- 输出供后续基因组搜索和实验验证使用的候选表；
- 重分析已保存的基因组搜索及蛋白稳定性计算结果。

> **研究用途声明**
>
> 本工具输出的是候选优先级和计算评价，不是已验证的编辑方案。
> 它不能替代实验室对靶序列、编辑结果、脱靶风险、递送和生物安全性的独立验证，不应用于直接作出临床决策。
>
> 模型主要基于ABE8e／HEK293T同源数据训练。输入一个新的位点能够得到计算结果，

项目研究结论及其复现方法见：[PROJECT.md](PROJECT.md)。

---

## 1. 这个工具适合什么任务？

- 对一个已知目标A进行研究性候选sgRNA生成；
- 对参考正链上表现为T的目标，自动转到互补链寻找目标A；
- 比较当前实现的PAM规则下有哪些可生成候选；
- 对候选同时查看效率估计、旁观者风险和序列启发式评分；
- 为后续实验筛选输出序列、方向和可选的基因组坐标；
- 在固定数据和保存的外部工具输出上复现本项目分析。

---

## 2. 当前可运行的主要入口

| 入口 | 用途 | 当前状态 |
|---|---|---|
| `scripts/design_sgrna.py` | 自定义context的候选设计与启发式排序 | 核心入口；HFE示例已验证 |
| `scripts/run_true_offtarget.py` | 调用或解析Cas-OFFinder结果 | 已验证部分保存输出解析；通用新搜索仍需自行验证 |
| `scripts/run_real_pam_offtarget.py` | 按变体拆分搜索并合并评分 | 提供实现；不属于完整通用验证保证 |
| `scripts/validate_hfe_global_ranking.py` | 复现项目HFE全局排序 | 固定案例复现入口，不是任意位点接口 |
| `scripts/parse_foldx_ddg.py` | 严格解析保存的FoldX结果 | 已验证 |
| `scripts/ddg_lm_consistency.py` | 比较蛋白模型分数与FoldX结果 | 保存结果分析已验证 |

完整范围说明见：[docs/release_scope.md](docs/release_scope.md)。

---

## 3. 安装与启动

### 3.1 已测试环境

- Windows
- Python 3.10.20
- 新建Conda环境
- 22项自动测试通过


### 3.2 安装核心环境

下载或克隆仓库后，在 **Anaconda Prompt** 中进入仓库根目录：

```bat
cd /d "你的仓库目录"
```

创建并激活环境：

```bat
conda env create -f environments\environment-core.yml
conda activate abe8e-core-check
python -m pip check
```

预期：

```text
No broken requirements found.
```

查看设计入口：

```bat
python scripts\design_sgrna.py --help
```

核心环境用于模型推理、测试和部分保存结果分析。
它不包含FoldX、Cas-OFFinder、ESM推理或全部研究绘图依赖。

详细说明见：[docs/installation_core.md](docs/installation_core.md)。

### 3.3 检查所需模型与先验

默认候选设计需要：

```text
models/xgboost_design_v2.pkl
results/bystander/empirical_prior.json
```

可选预测区间需要：

```text
results/conformal/conformal_qhat.json
```

如果模型或数据以独立资产发布，应先按对应发布说明下载到上述路径。
缺少这些文件时，代码仓库本身不能完成相应推理。

只加载可信来源的pickle模型文件。

### 3.4 运行测试

```bat
python -m unittest discover -s tests -p "test_*.py" -v
```

当前预期：

```text
Ran 22 tests
OK
```

测试通过证明已覆盖功能在当前环境正常，不代表所有生物学假设和任意输入均已验证。

---

## 4. 使用自己的位点前，准备哪些信息？

核心入口需要：

1. **目标已经处于待编辑状态的DNA context序列**；
2. **目标碱基在context中的0-based位置**；
3. 选择PAM规则或Cas标签；
4. 如需后续坐标感知搜索，提供context的基因组位置。

### 4.1 context应是什么？

输入为参考正链方向的DNA序列字符串：

- 当前版本仅接受A、C、G、T，不接受N或其他模糊碱基；
- context需要包含目标以及足够的两侧序列，以覆盖20 nt protospacer与PAM；
- 如果context太短，可能漏掉实际存在的候选；
- 应由使用者核查目标附近是否存在其他变异。

对于变异矫正场景：

> 输入应包含需要矫正的变异碱基，而不是未经修改的参考序列。

工具不会根据疾病名称或变异编号自动构建该输入。

### 4.2 目标碱基要求

| context中的目标碱基 | 工具行为 |
|---|---|
| A | 在正链设计，目标是A→G |
| T | 在反向互补链设计目标A，对应输入正链的T→C |
| C或G | 当前入口拒绝；不属于该输入方向下的ABE目标 |

例如：

- 正链G>A变异：待编辑context中的目标应为A。
- 正链C>T变异：待编辑context中的目标应为T，工具在互补链上设计A→G。

必须自行核对变异方向及预期矫正结果。

### 4.3 坐标约定

| 参数／字段 | 坐标约定 |
|---|---|
| `--target_pos` | 目标在context中的位置，0-based |
| 输出`target_pos` | 目标A在protospacer中的位置，1-based |
| `--context_start` | context第一个碱基的基因组坐标，1-based |
| `ot_start`、`ot_end` | protospacer在参考正链上的区间，1-based inclusive |

输入参数和输出字段都叫`target_pos`，但含义不同，使用时特别注意。

若`context_start = S`，目标在context中的索引为`i`，则目标基因组坐标为：

```text
S + i
```

当前候选生成要求目标A位于protospacer第4—8位。
旁观者风险评价可以考虑更宽的位置范围；两者不是同一个窗口定义。

---

## 5. 使用自己的context运行设计

以下命令均在仓库根目录的 **Anaconda Prompt** 中执行。

### 5.1 直接输入序列

将下面的`YOUR_CONTEXT_SEQUENCE`和`TARGET_INDEX_0BASED`替换为自己的真实输入：

```bat
python scripts\design_sgrna.py --context YOUR_CONTEXT_SEQUENCE --target_pos TARGET_INDEX_0BASED --cas_variant SpCas9 --out_csv results\user_runs\my_target.csv
```
输出目录会自动创建。

### 5.2 从单条FASTA读取context

当前设计入口没有直接提供`--context_fasta`参数，
可以使用以下Python包装命令读取单条FASTA。

将：
- `my_context.fa`改为自己的文件；
- `target_index=40`改为自己的0-based目标位置。

```bat
python -c "from Bio import SeqIO; import subprocess,sys; record=SeqIO.read('my_context.fa','fasta'); target_index=40; subprocess.run([sys.executable,'scripts/design_sgrna.py','--context',str(record.seq),'--target_pos',str(target_index),'--cas_variant','SpCas9','--out_csv','results/user_runs/my_target.csv'],check=True)"
```

`SeqIO.read`要求文件只有一条序列，避免把多条FASTA误拼接成一个context。

### 5.3 提供基因组坐标

```bat
python scripts\design_sgrna.py --context YOUR_CONTEXT_SEQUENCE --target_pos TARGET_INDEX_0BASED --cas_variant SpCas9 --chrom YOUR_CHROMOSOME --context_start CONTEXT_START_1BASED --out_csv results\user_runs\my_target_with_coordinates.csv
```

`--chrom`和`--context_start`必须同时提供。

染色体名称应与后续参考基因组一致，例如`chr6`与`6`不是同一个字符串。

### 5.4 比较PAM可及性

```bat
python scripts\design_sgrna.py --context YOUR_CONTEXT_SEQUENCE --target_pos TARGET_INDEX_0BASED --compare_variants --out_csv results\user_runs\pam_accessibility.csv
```

输出各规则下的候选数量和模型预测摘要。

这不是不同Cas底盘效率已经经过校准的比较实验。

### 5.5 合并当前支持的变体候选

```bat
python scripts\design_sgrna.py --context YOUR_CONTEXT_SEQUENCE --target_pos TARGET_INDEX_0BASED --all_variants --out_csv results\user_runs\all_variant_candidates.csv
```

当前工具按预设顺序选择代表性Cas标签，不保留全部编辑器—guide组合。
帮助文本中的历史“most PAM-restrictive”措辞应理解为实现中的代表标签规则，
而不是所有PAM之间都具有严格包含关系。

### 5.6 当前实现的PAM规则

| Cas标签 | 搜索规则 |
|---|---|
| SpCas9 | NGG |
| SpCas9-NG | NGN |
| SpG | NGN |
| VRQR | NGA |
| SpRY | NRN |

这些是工具当前实现的规则范围，不等于各蛋白全部生物学PAM识别能力。

自定义规则可以通过`--pam`传入：

```bat
python scripts\design_sgrna.py --context YOUR_CONTEXT_SEQUENCE --target_pos TARGET_INDEX_0BASED --pam NGG --out_csv results\user_runs\custom_pam_candidates.csv
```

自定义PAM只改变候选生成规则，不会自动建立对应底盘的效率模型。

---

## 6. 输出文件包含什么？

普通设计输出是一张候选CSV。

| 字段 | 含义 |
|---|---|
| `rank` | 当前综合分排序 |
| `grna` | 20 nt候选protospacer，5′→3′ |
| `pam` | context中匹配到的PAM序列 |
| `cas_variant` | 当前代表性Cas标签 |
| `strand` | 相对输入正链的guide方向 |
| `target_pos` | 目标A在protospacer中的1-based位置 |
| `pred_efficiency` | B′预测，工具截断到[0,1] |
| `pred_eff_low/high` | 可选经验校准区间 |
| `bystander_A` | 第4—8位窗口内除目标外的A计数 |
| `bystander_weighted` | 位置先验加权的旁观者风险分 |
| `offtarget_risk` | 历史列名；实际为序列启发式评分 |
| `ot_seed/gc/homopolymer/complexity` | 启发式评分分项 |
| `composite_score` | 当前权重下的综合分 |
| `ot_chrom/start/end` | 可选的protospacer基因组位置 |
| `pareto_optimal` | 当前目标定义下是否非支配 |

### 重要解释

- `grna`是protospacer，不是完整表达构建。
- 输出没有自动加入启动子需要的额外碱基、sgRNA scaffold或克隆接头。
- 不应直接把该表当作已经验证的实验构建设计。
- `pred_efficiency`对应模型学习的reported标签，不保证等于新实验中目标A的精确编辑率。
- `bystander_weighted=0.38`不等于实测38%的旁观者编辑。
- `offtarget_risk`不是实际脱靶概率。
- 负的`composite_score`正常，它不是概率，也没有通用的合格阈值。
- `pareto_optimal=True`不等于已证明安全或有效。

---

## 7. 如何理解排序？

设计工具采用：

```text
综合分 = w1 × 预测效率
       − w2 × 加权旁观者风险
       − w3 × 序列启发式评分
```

当前历史权重预设：

| 名称 | 效率 | 旁观者 | 序列启发式 |
|---|---:|---:|---:|
| research | 0.8 | 0.1 | 0.1 |
| therapy | 0.4 | 0.3 | 0.3 |
| clinical | 0.3 | 0.4 | 0.3 |

默认是`therapy`。

这些名称只表示人为设定的偏好，**不是经治疗或临床效用验证的模式**。

例如：

```bat
python scripts\design_sgrna.py --context YOUR_CONTEXT_SEQUENCE --target_pos TARGET_INDEX_0BASED --scenario research --out_csv results\user_runs\efficiency_priority.csv
```

应同时检查各分量和Pareto前沿，而不是只取第一行就认定最佳。

---

## 8. 可选预测区间

```bat
python scripts\design_sgrna.py --context YOUR_CONTEXT_SEQUENCE --target_pos TARGET_INDEX_0BASED --conformal --conformal_level 90 --out_csv results\user_runs\candidates_with_intervals.csv
```

需要：

```text
results/conformal/conformal_qhat.json
```

当前区间来自历史验证集残差校准，而该验证集也用于模型早停。

因此：

- 可以展示经验不确定性；
- 不主张已经满足标准独立校准条件下的覆盖保证；
- 不保证任意新细胞、新Cas底盘或疾病位点的90%覆盖；
- 区间重叠时，不应宣称候选效率已被确定地区分。

---

## 9. 从预筛走向自己的编辑方案评价

建议按以下顺序使用结果：

1. 独立核对输入变异、链方向和参考基因组坐标。
2. 生成候选，检查是否存在PAM和窗口限制。
3. 同时查看效率、加权旁观者分和序列特征。
4. 检查候选窗口内可能产生的其他碱基改变及其功能后果。
5. 使用适合目标基因组和编辑器的外部方法进行潜在脱靶评价。
6. 结合项目场景开展实验验证。
7. 根据实测结果校准或更新候选选择。

本工具完成前3步后用户可根据结果进行调整并继续4-7步。

### 无候选时

可能原因：

- 当前PAM规则不满足；
- 目标A不能落在第4—8位；
- context不足以覆盖候选；
- 输入目标位置或变异方向错误。

无候选只代表当前输入和规则下没有找到，不代表该位点在所有编辑系统中不可编辑。

---

## 10. 关于全基因组潜在匹配分析

核心设计命令不自动运行全基因组搜索。

仓库提供Cas-OFFinder相关脚本，但当前充分验证的是：
- 部分保存结果的解析；
- 精确目标排除；
- 项目HFE候选集合的全局合并评分。

面向自己的新搜索，还需自行确认：

- 参考基因组文件、版本和染色体命名；
- 候选序列与基因组坐标；
- 工具安装、设备及输出格式；
- PAM和错配设置；
- 同一guide是否涉及多个候选实例；
- 个体变异和参考序列差异；
- 搜索是否包含预期范围之外的重复或替代序列。

### 重分析自己的单PAM保存输出

以下是接口示意，需替换路径：

```bat
python scripts\run_true_offtarget.py --candidates YOUR_CANDIDATES.csv --genome YOUR_GENOME_DIRECTORY --pam NGG --max_mismatches 4 --cas_output YOUR_CASOFFINDER_OUTPUT.txt --outdir results\user_runs\my_saved_search --tolerance 0 --summary_only
```

这个模式不重新运行Cas-OFFinder。

候选必须具备准确的：
- `grna`
- `strand`
- `ot_chrom`
- `ot_start`
- `ot_end`

精确坐标转换针对当前测试过的3′ PAM、无bulge输出约定。其他格式不能未经验证直接套用。


### 不要跨分组比较归一化分数

先按不同PAM搜索，再将原始匹配结果合并，
最后在共同尺度下计算排序。

单候选组的归一化分数可能恒为1，
不能据此判断不同变体风险相同。

项目HFE的完整保存输出复现见[PROJECT.md](PROJECT.md)。

---

## 11. 蛋白候选分析的用途

本仓库包含TadA-8e候选替换的历史计算结果，以及经过验证的保存结果解析工具。

它可以用于：
- 检查突变输入与结果对应关系；
- 读取FoldX计算重复；
- 比较已有蛋白模型分数与平均ΔΔG；
- 识别不同计算方法的支持与分歧。

它不是面向任意蛋白的通用自动优化服务。

当前正式流程仍围绕已提供的166残基参考及相应模型矩阵。
更换蛋白时，需要重新验证序列、结构、编号、模型输入和评分口径。

---

## 12. 可复现性与项目结果

项目结论、数据划分、主要数字及逐步复现命令见：

- [PROJECT.md](PROJECT.md)
- [docs/results_index.md](docs/results_index.md)
- [docs/validation_status.md](docs/validation_status.md)
- [docs/model_card.md](docs/model_card.md)

当前验证包括：
- 新建Conda环境安装；
- 22项自动测试；
- HFE新旧环境结果一致；
- B′指标政策审计；
- 保存基因组搜索的坐标复核；
- FoldX原始输出交叉检查；
- 完整精度蛋白相关性复现。

---

## 13. 常见问题

### 为什么换PAM后效率预测没有成为“该Cas专属预测”？

当前B′输入不包含Cas变体特征。
PAM决定候选能否生成，不代表模型已学习各Cas底盘的专属效率。

### 可以输入其他物种的序列吗？

程序可以处理符合格式的序列，但模型适用性、染色体坐标和
潜在脱靶参考必须重新评估。没有跨物种效果保证。

### 为什么SpRY在预筛中排第一，而项目最终HFE排序是SpCas9？

两者使用的脱靶评价不同：
- 预筛使用序列启发式；
- 项目全局重分析使用对应PAM的基因组匹配结果。

不能把两个阶段的分数混为同一指标。

### 为什么模型有两个相同Tm特征？

为保持现有83列已训练模型兼容。
直接删除一列会改变输入结构，需要重新训练和验证。

### 为什么一些结果目录已经存在，程序拒绝运行？

保存结果复核脚本会避免覆盖已有结果。
请换一个新的`--outdir`。

### 模型文件缺失怎么办？

先获取该发布版本对应的模型资产。
不要随意下载其他同名pickle替换。

---

## 14. 数据、许可证与贡献

原创代码计划采用MIT，以根目录实际发布的`LICENSE`为准。

第三方数据、预训练权重、工具和资源保留各自许可，
不自动适用项目代码许可证。

详见：

- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- [docs/data_sources.md](docs/data_sources.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

公开前仍需确认数据和模型资产的再分发条件。
缺少资产下载方式时，不应宣称用户可以完整复现相关流程。

反馈问题时请提供：
- 版本或commit；
- 操作系统及Python环境；
- 命令；
- 最小非敏感输入；
- 完整错误；
- 预期与实际结果。

不要上传患者个人信息、访问令牌或未授权数据。