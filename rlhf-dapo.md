## DAPO 论文详解
论文《DAPO: An Open-Source LLM Reinforcement Learning System at Scale》

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）是字节跳动Seed团队在2025年提出的**大规模LLM强化学习系统**，基于Qwen2.5-32B基座模型在AIME 2024上达到了**50分**，超越了DeepSeek-R1-Zero-Qwen-32B的47分，且仅用了50%的训练步数。它是GRPO在**长思维链（Long-CoT）**场景下的重要升级版。

---

### 1. GRPO的失败与DAPO的诞生：为什么GRPO在长CoT场景中失效？

论文明确指出，使用**Naive GRPO**在Qwen2.5-32B上进行RL训练，AIME分数仅为**30分**，远低于DeepSeek R1报告中的47分。经过深入分析，发现了以下几个关键问题：

- **熵崩溃（Entropy Collapse）**：策略的生成概率分布迅速变得“尖锐”，模型几乎只输出极少数固定模式，探索能力丧失。
- **梯度消失（Gradient Decreasing）**：当某些问题的所有采样输出全对或全错时，组内奖励的均值和标准差导致优势为0，这些样本对梯度贡献为0，批次中有效样本越来越少。
- **长序列处理不当**：GRPO采用**样本级（Sample-Level）**损失聚合——先对每个输出内部的Token损失求平均，再跨样本平均。长输出中的每个Token贡献被“稀释”，导致模型无法有效学习高质量长推理中的有用模式，也无法有效惩罚低质量长输出中的重复/乱码。
- **截断奖励噪声（Overlong Reward Noise）**：超长样本被截断后统一给予负奖励，但某些正确推理只是因为过长而被惩罚，这种噪声严重干扰训练。

---

### 2. DAPO的四项核心技术

#### 2.1 Clip-Higher：解耦上下裁剪边界，对抗熵崩溃

**问题**：GRPO使用对称的裁剪范围\([1-\epsilon, 1+\epsilon]\)（通常\(\epsilon=0.2\)）。对于低概率Token（概率约0.01），其上限仅为0.012，几乎不可能被显著提升——这严重抑制了“探索性”Token的生成，导致熵迅速崩塌。

**DAPO的解法**：将裁剪范围解耦为**独立的上下界**：
\[
\epsilon_{\text{low}} = 0.2,\quad \epsilon_{\text{high}} = 0.28
\]

目标函数变为：
\[
\mathcal{I}_{\text{DAPO}}(\theta) =
\mathbb{E}\left[
\frac{1}{\sum |o_i|}\sum_i\sum_t
\min\left(
r_{i,t}(\theta)\hat{A}_{i,t},
\text{clip}\left(r_{i,t}(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\right)\hat{A}_{i,t}
\right)
\right]
\]

**原理**：提高\(\epsilon_{\text{high}}\)为低概率“探索性”Token留下更多概率提升空间（上限从1.2提升到1.28），有效对抗熵崩溃，促进生成多样性。

---

#### 2.2 Dynamic Sampling：动态采样，消除零梯度样本

**问题**：在GRPO中，若某问题的所有采样输出都正确（准确率=1），组内奖励全部相同（均为+1），归一化后优势全为0。随着训练推进，这类问题越来越多，有效样本数持续减少，梯度信号被“稀释”，训练效率急剧下降。

**DAPO的解法**：增加约束条件：
\[
0 < \left|\{o_i \mid \text{is\_equivalent}(a, o_i)\}\right| < G
\]

即**过滤掉全部正确和全部错误的样本**，只保留组内既有正确也有错误的那些问题。采样过程中持续采样直到Batch被填满。虽然采样成本增加，但生成时间通常由长尾样本主导，且实验证明该策略反而加快了收敛（图6）。

---

#### 2.3 Token-Level Policy Gradient Loss：Token级损失聚合，解决长序列问题

**GRPO的问题（Sample-Level）**：
\[
\text{Loss} = \frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \text{loss}_{i,t}
\]
每个样本先对Token求平均再跨样本平均——所有样本**等权重**。长输出的每个Token贡献被稀释，导致模型无法学习高质量长推理中的有效模式，也无法有效惩罚低质量长输出中的重复/乱码。

**DAPO的解法（Token-Level）**：
\[
\mathcal{I}_{\text{DAPO}}(\theta) =
\mathbb{E}\left[
\frac{1}{\sum_i |o_i|}\sum_i\sum_t
\min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(\cdots)\hat{A}_{i,t}\right)
\right]
\]

所有Token**展平后统一求平均**，每个Token等权重。长序列中的Token无论出现在短输出还是长输出中，其贡献完全一致。这使得高质量长推理模式能被有效学习，低质量乱码也能被充分惩罚。

- **GRPO（样本级）**：先对每个样本内部的 Token 损失求平均，再跨样本平均。每个样本（无论长短）在最终损失中权重相同。
  - **问题（论文 3.3 节）**：
    - 长样本中的每个 Token 贡献被稀释，高质量长推理模式难以被学习。
    - 低质量长样本（重复/乱码）无法被充分惩罚，导致长度和熵不健康增长（见图 4）。

- **DAPO（Token 级）**：将所有样本的所有 Token **“压平”成一整个集合**，统一除以总 Token 数。每个 Token（无论来自短样本还是长样本）在损失中完全等权。
  - **优势**：若某种模式（如自我反思）能带来奖励提升，无论它出现在长输出还是短输出中，都会被同等程度地强化或抑制。

---

#### 2.4 Overlong Reward Shaping：超长奖励塑形，消除奖励噪声

**问题**：在RL训练中通常设置最大生成长度，超长样本被截断并给予统一的负奖励。这引入了严重噪声——一个推理过程完全正确但只是稍长，却被错误地惩罚。

**DAPO的两个策略**：

1. **Overlong Filtering**：直接将截断样本的Loss掩码掉（不参与梯度更新）。这大幅稳定了训练，如图5所示。

2. **Soft Overlong Punishment（最终采用）**：定义惩罚区间：
\[
R_{\text{length}}(y)=
\begin{cases}
0, & |y| \leq L_{\max} - L_{\text{cache}}\\
\frac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < |y| \leq L_{\max}\\
-1, & L_{\max} < |y|
\end{cases}
\]

**含义**：
- 未超长：无惩罚
- 接近上限（\(L_{\max}-L_{\text{cache}}\)到\(L_{\max}\)之间）：**软惩罚**，越长惩罚越大（从0线性降至-1）
- 超过\(L_{\max}\)：截断，惩罚固定为-1

**核心参数**：论文设置\(L_{\max}=20480\)，\(L_{\text{cache}}=4096\)。

**备注**：GRPO 设计目标是通用的数学推理，平均推理长度较短（论文设定最大长度仅 1024 tokens）。长 CoT 场景并非其核心考量。因此，GRPO不需要这项技术。

---

### 3. DAPO 分步总结

| 步骤 | 对应环节 | **依赖模型** | **计算公式** | **总结（与GRPO的核心差异）** |
| :--- | :--- | :--- | :--- | :--- |
| **①** | **组采样 + 奖励生成** | **旧策略（采样G个输出）+ Reward模型（规则打分）** | **无KL惩罚**（已移除）<br>\(R_i = \pm 1\)（正确/错误） | **移除KL惩罚**：长CoT场景中模型分布已大幅偏离初始模型，不再适用KL约束。<br>**移除Ref模型**：无需参考模型，省显存。 |
| **②** | **动态采样过滤** | **旧策略 + Reward模型** | 约束条件：<br>\(0 < \text{正确数} < G\) | **Dynamic Sampling**：只保留组内**既有正确也有错误**的样本，消除零梯度样本，维持稳定梯度信号。 |
| **③** | **组内相对优势估计** | **旧策略 + Reward模型（组内分数）** | **结果监督（OS）**：<br>\( \hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})} \) | **同GRPO**：组内归一化替代GAE，无需Value网络。 |
| **④** | **Actor反向传播**<br>（内含Clip-Higher + Token-Level Loss） | **新策略（更新主体）+ 旧策略（参考比率）** | **目标函数（公式8）**：<br>\(\min(r_{i,t}\hat{A}_{i,t},\ \text{clip}(r_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\hat{A}_{i,t})\)<br>**Token级聚合** | **两个关键差异**：<br>① **Clip-Higher**：\(\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28\)，上下界解耦，提升低概率Token上限，对抗熵崩溃。<br>② **Token-Level Loss**：所有Token等权重聚合，解决长序列Token贡献被稀释问题，避免模型长输出退化。 |
| **⑤** | **Overlong Reward Shaping** | **规则函数** | **软惩罚（公式13）**：<br>\(0 \to\) 线性衰减至 \(-1 \to -1\) | **消除奖励噪声**：对超长样本不简单给予统一惩罚，而是在接近上限时引入线性软惩罚，使模型能够区分“正确但稍长”和“异常超长”。 |
| **⑥** | ~~Critic反向传播~~ | ❌ **无Critic模型** | ❌ 无 | **彻底移除Critic**：继承GRPO，无需Value网络。 |
| **⑦** | ~~KL惩罚~~ | ❌ **无Ref模型** | ❌ 无 | **彻底移除KL惩罚**：长CoT训练中模型分布已大幅偏离初始模型，该约束不再必要。 |

---

### 4. DAPO vs GRPO 对比速查

| 特性 | **GRPO (DeepSeekMath)** | **DAPO (ByteDance Seed)** |
| :--- | :--- | :--- |
| **价值网络 (Critic)** | **舍弃** | **舍弃** |
| **参考模型 (Ref)** | **需要**（KL锚点） | **移除**（KL惩罚已移除） |
| **KL散度** | 作为正则项加入Loss（公式3） | **完全移除**（论文2.3节） |
| **裁剪范围** | **对称**：\([1-\epsilon, 1+\epsilon]\)（\(\epsilon=0.2\)） | **解耦**：\(\epsilon_{\text{low}}=0.2\)，\(\epsilon_{\text{high}}=0.28\)（Clip-Higher） |
| **损失聚合方式** | **样本级**（Sample-Level） | **Token级**（Token-Level） |
| **采样过滤** | 无 | **Dynamic Sampling**（过滤全对/全错样本） |
| **超长奖励** | 统一负奖励 | **Soft Overlong Punishment**（线性软惩罚） |
| **AIME 2024分数** | ~30分（Naive GRPO） | **50分** |
| **训练步数** | 基准 | **DeepSeek R1的50%** |

---

### 5. DAPO的核心理念总结

DAPO不是一个完全推翻GRPO的新算法，而是**针对长CoT场景中对GRPO的4个关键bug fix**：

1. **Clip-Higher**：提高低概率Token的提升上限 → 对抗熵崩溃
2. **Dynamic Sampling**：只保留“部分对部分错”的样本 → 维持稳定梯度
3. **Token-Level Loss**：让每个Token等权重 → 解决长序列Token稀释问题
4. **Soft Overlong Punishment**：用线性软惩罚替代统一负奖励 → 消除奖励噪声

再加上：
- **完全移除KL惩罚和Ref模型**：长CoT中模型分布已大幅偏离初始模型，该约束已无必要
- **继承GRPO的优势**：无Critic，无GAE，组内归一化优势

最终，这些改进使DAPO在Qwen2.5-32B上达到了**AIME 2024 50分**的SOTA结果，超越了DeepSeek R1 Zero，且仅用50%训练步数。