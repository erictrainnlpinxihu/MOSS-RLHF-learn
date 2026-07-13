## GSPO（Group Sequence Policy Optimization）详解
基于论文《Group Sequence Policy Optimization》

GSPO 是阿里巴巴 Qwen 团队在 2025 年提出的**序列级强化学习算法**，针对 GRPO 在长序列和多轮 RL 中暴露的**根本性缺陷**进行了重新设计。其最核心的改进是：**将重要性采样比率从 Token 级提升至 Sequence 级，从根本上消除了 GRPO 中累积梯度噪声导致模型崩溃的隐患**，并极大简化了 MoE 模型的 RL 训练流程。

---

### 1. 为什么 GRPO 会在长序列和 MoE 训练中崩溃？（GSPO 的动机与洞察）

在标准 PPO 和 GRPO 中，虽然摒弃了价值网络（Critic），但它们仍保留了**逐 Token 的重要性采样比率** \( w_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t})}{\pi_{\theta_{old}}(y_{i,t})} \)。GSPO 论文通过严谨的理论分析（**论文第 3 节**），指出这一定义存在**致命的理论缺陷**：

- **重要性采样的基本前提（方程 4）**：重要性采样（IS）必须基于**多个样本的平均**，才能有效修正分布偏差。即 \( \mathbb{E}_{z \sim \pi_{tar}}[f(z)] \approx \frac{1}{N}\sum \frac{\pi_{tar}}{\pi_{beh}} f(z) \)。
- **GRPO 的误用**：GRPO 对**单个 Token**（单样本）施加重要性权重。这意味着它不是在“修正分布”，而是在引入**高方差噪声**。
- **级联放大效应**：当序列变长时，这种逐 Token 的噪声不断累积，导致梯度估计极度不稳定。在 MoE（混合专家）模型中，每次更新后专家激活路径发生剧烈变化（约 10% 的专家改变），进一步加剧了该问题，导致**不可逆的模型崩溃（Collapse）**。一旦崩溃，即使回退检查点或精细调参也无法恢复。

**GSPO 的核心洞察**：既然奖励是针对**整个序列**给出的，那么**“优化的基本单位”和“奖励的基本单位”必须一致**。因此，必须放弃 Token 级操作，转向**序列级（Sequence-Level）** 操作。

---

### 2. 核心机制：序列级重要性比率与序列级裁剪（Sequence-Level Importance Ratio & Clipping）

GSPO 定义了基于**序列似然**（Sequence Likelihood）的重要性比率 \( s_i(\theta) \)：

\[
s_i(\theta) = \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \right)^{\frac{1}{|y_i|}}
\]

**解析**：
1.  **\( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \)**：整个回复序列的新旧策略概率之比。
2.  **\( ()^{\frac{1}{|y_i|}} \)**：**长度归一化（Length Normalization）**。这防止了长序列因为连乘导致的数值极端膨胀或塌缩，确保不同长度的回复具有可比的数值范围（实践中 GSPO 的 clip 范围约为 \( 10^{-4} \) 级，而 GRPO 约为 \( 0.2 \) 级）。

**序列级裁剪**：GSPO 直接对 **“整个回复”** 应用裁剪，而非对单个 Token：
\[
\mathcal{J}_{\mathrm{GSPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^{G} \min \left( s_i(\theta)\hat{A}_i, \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i \right) \right]
\]
这意味着：如果某个回复的序列概率比偏离旧策略太远，**整个回复的所有 Token 都会一起被“安全隔离”**，不参与梯度更新。这完美匹配了“奖励是给整个序列”的特性，避免了 Token 级裁剪带来的内部不一致性。

---

### 3. 梯度分析（GSPO vs GRPO）：为什么 GSPO 更稳定？

通过对比两者的梯度公式，可以直观看到稳定性的根本来源。

#### GRPO 的梯度（方程 11）：
\[
\nabla_{\theta}\mathcal{J}_{\mathrm{GRPO}} \propto \sum_{i} \hat{A}_i \cdot \frac{1}{|y_i|}\sum_{t} \underbrace{\frac{\pi_{\theta}(y_{i,t})}{\pi_{\theta_{old}}(y_{i,t})}}_{\text{不等权重 } w_{i,t}} \nabla_{\theta}\log \pi_{\theta}(y_{i,t})
\]
**问题**：每个 Token 被赋予**不均匀的权重** \( w_{i,t} \)。这些权重本身就是高噪声的，且相互独立，导致梯度方向极不稳定。

#### GSPO 的梯度（方程 10）：
\[
\nabla_{\theta}\mathcal{J}_{\mathrm{GSPO}} \propto \sum_{i} \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \right)^{\frac{1}{|y_i|}} \hat{A}_i \cdot \frac{1}{|y_i|}\sum_{t} \nabla_{\theta}\log \pi_{\theta}(y_{i,t})
\]
**优势**：
- **等权重**：GSPO 对同一回复内的**所有 Token 施加完全相等的梯度系数**。
- **单一声量**：整个回复只用一个信号（序列比率）驱动梯度，极大地降低了梯度估计的方差，消除了 GRPO 累积噪声导致崩溃的根源。

---

### 4. 在 MoE 训练与基础设施上的巨大收益

#### MoE 训练稳定性的革命
- **GRPO 的困境**：MoE 模型每次前向都会动态激活不同的专家。由于 GRPO 的 Token 级权重极其敏感，专家激活的变化导致 Token 概率剧烈波动，GRPO 必须依赖复杂的 **Routing Replay（路由重放）** 策略（强制新旧策略使用相同专家）才能勉强收敛，但这增加了额外开销且限制了模型表达能力。
- **GSPO 的优势**：GSPO 只关注**整个序列的似然**。即使个别 Token 的专家路径发生变化，只要模型整体语言建模能力不变，序列似然就不会剧烈波动。因此，**GSPO 彻底摆脱了对 Routing Replay 的依赖**，让 MoE 模型在 RL 训练中既能稳定收敛，又能发挥完整的专家容量。

#### 对 RL 基础设施的简化
由于 GSPO 使用序列级似然而非 Token 级概率，它对训练引擎与推理引擎之间的**浮点数精度差异（Precision Discrepancy）** 具有极高的容忍度。因此，GSPO **可以直接使用推理引擎返回的似然值进行优化，无需像 GRPO 那样强制在训练引擎中重算（Recompute）**，极大地简化了“训推分离”框架下的系统设计。

---

### 5. GSPO 分步总结

| 步骤 | 对应环节 | **依赖模型** | **计算公式** | **总结（与 GRPO 的核心差异）** |
| :--- | :--- | :--- | :--- | :--- |
| **①** | **组采样 + 奖励生成** | **旧策略（为每个问题采样 G 个输出）+ Reward模型（打分）** | 结果监督下，生成  \( G \) 个输出，奖励为 \( r_i \) | **继承 GRPO**：无 Critic，无 GAE。 |
| **②** | **组内相对优势估计** | **旧策略 + Reward模型（组内分数）** | **结果监督（OS）**：<br>\( \hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})} \) | **同 GRPO**：组内归一化作为优势。 |
| **③** | **序列级重要性比率计算** | **新策略 + 旧策略** | **序列似然归一化**：<br>\( s_i(\theta) = \left( \frac{\pi_\theta(y_i\|x)}{\pi_{\theta_{old}}(y_i\|x)} \right)^{1/\|y_i\|} \) | **GSPO 独创**：<br>① **粒度提升**：从 Token 级变为 **整个序列级**。<br>② **长度归一化**：消除了长序列的概率极端值。 |
| **④** | **Actor 反向传播**<br>（序列级裁剪） | **新策略（更新主体） + 旧策略（参考比率）** | **目标函数（方程 5）**：<br>\( \min \left( s_i(\theta)\hat{A}_i,\ \text{clip}(s_i, 1\pm\epsilon)\hat{A}_i \right) \) | **裁剪对象变更**：<br>直接将 **整个回复** 的比率进行裁剪，将过偏的“离群样本”整条剔除，而不是裁剪单个 Token。 |
| **⑤** | ~~Critic 反向传播~~ | ❌ **无 Critic 模型** | ❌ 无 | **继承 GRPO**。 |
| **⑥** | ~~KL 惩罚与 Ref 模型~~ | ❌ **无 KL/Ref 模型** | ❌ 无 | **继承 GRPO/高级变体**：论文为强调核心差异省略了 KL，实际可根据需要加入。 |

---
基于 GSPO 论文《Group Sequence Policy Optimization》第 4.1 节，GSPO 的优势（Advantage）计算过程可以拆解为 **4 个明确的数学步骤**。它与 GRPO 共享“组内归一化”的公式结构，但在“应用粒度”上完全不同。

---

### 📐 GSPO 优势计算的 4 个步骤

#### 第 1 步：采样组（Group Sampling）
对于每一个查询（Query）\( x \)，从**旧策略** \( \pi_{\theta_{\text{old}}} \) 中采样一组（Group）完整的回复序列：
\[
\{y_1, y_2, \dots, y_G\} \sim \pi_{\theta_{\text{old}}}(\cdot|x)
\]
其中 \( G \) 是组大小（Group Size）。

#### 第 2 步：获取序列级奖励（Sequence-Level Rewards）
使用验证器（Verifier）或奖励模型（Reward Model）对**每一个完整的回复序列** \( y_i \) 进行打分，得到对应的奖励集合：
\[
\mathbf{r} = \{r_1, r_2, \dots, r_G\}
\]
- **关键点**：每个奖励 \( r_i \) 是针对**整个回复** \( y_i \) 的，而不是针对单个 Token 的（例如数学题中答对得 1 分，答错得 0 分，或 RM 给出的总分）。

#### 第 3 步：组内相对归一化（Intra-Group Relative Normalization）
计算该组奖励的均值 \( \mu \) 和标准差 \( \sigma \)：
\[
\mu = \text{mean}(\mathbf{r}) = \frac{1}{G}\sum_{i=1}^G r_i
\]
\[
\sigma = \text{std}(\mathbf{r}) = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \mu)^2}
\]
然后，对第 \( i \) 个回复的奖励进行标准化（减去均值除以标准差），得到该回复的**组内相对优势**：
\[
\hat{A}_i = \frac{r_i - \mu}{\sigma} \tag{6}
\]

#### 第 4 步：向序列内所有 Token 广播（Broadcast）
将计算出的标量优势 \( \hat{A}_i \) **直接赋给该回复序列中的所有 Token**（位置 \( t=1 \) 到 \( |y_i| \)）：
\[
\hat{A}_{i,t} = \hat{A}_i \quad (\text{对于 } t = 1, 2, \dots, |y_i|)
\]

---

#### ⭐️ GSPO 与 GRPO 的优势计算过程对比

**视觉上**，GSPO 的优势计算步骤（1-4）与 GRPO **看起来完全一样**（GRPO 论文中也是将组内归一化奖励赋给所有 Token）。**然而，GSPO 的创新并非在于“如何计算优势”，而是在于“优势计算出来后，如何被使用（应用粒度）”。**

| 阶段 | **GRPO（组相对策略优化）** | **GSPO（组序列策略优化）** |
| :--- | :--- | :--- |
| **① 优势计算过程** | 组内归一化奖励：\( \hat{A}_{i,t} = \frac{r_i - \mu}{\sigma} \) | **完全相同**：\( \hat{A}_i = \frac{r_i - \mu}{\sigma} \) |
| **② 优势赋给谁？** | 赋给该回复的**所有 Token**（数值相同） | **完全相同**：赋给该回复的所有 Token |
| **③ 优势（在损失中）与谁相乘？** | 乘以 **Token 级重要性比率** \( w_{i,t} \) <br>（每个 Token 有自己的 \( w_{i,t} \)） | 乘以 **序列级重要性比率** \( s_i \) <br>（整个回复只有一个 \( s_i \)） |
| **④ 结果** | 即使优势相同，但 Token 级的 \( w \) 权重不均，导致**同一个回复内部的 Token 梯度强度天差地别**（内部噪声大） | 整个回复的 Token **共享完全相同的梯度权重** \( s_i \)，实现**优化粒度的完美自洽** |

#### 🏁 总结

GSPO 的优势计算过程**在数学上**与 GRPO 完全一致，都是“组内归一化奖励”的简易机制。**GSPO 真正的技术突破发生在“损失函数应用层”**——它确保这个优势 \( \hat{A}_i \) 只与 **“整个回复的单一比率”** \( s_i \) 互动，而不是像 GRPO 那样与 **“每个 Token 的各异比率”** \( w_{i,t} \) 互动。

---

### 6. 总结：GRPO vs GSPO 对比速查

| 特性 | **GRPO (DeepSeekMath)** | **GSPO (Qwen Team)** |
| :--- | :--- | :--- |
| **优势计算** | 组内归一化奖励（无 Critic） | **同 GRPO**（组内归一化奖励） |
| **重要性采样比率** | **Token 级**（每个 Token 独立计算） | **序列级**（整个回复算一个比率，经长度归一化） |
| **优化基本单位** | Token | **整个序列（Sequence）** |
| **裁剪（Clip）对象** | 超出范围的**单个 Token** | 超出范围的**整个回复** |
| **梯度噪声/方差** | **高**（Token 级权重不均，长序列累积放大） | **低**（序列级单一声量，等权重驱动梯度） |
| **MoE 训练稳定性** | 极差（需依赖复杂的 **Routing Replay** 保底） | **极优**（直接训练，无特殊策略，专家容量完全释放） |
| **硬件/推理兼容性** | 需重算概率（对精度敏感） | **高容忍度**（可用推理引擎概率直接优化） |
| **核心缺陷/优势** | 存在导致模型崩溃的理论隐患（论文第 3 节） | **根除了 Token 级噪声，稳如磐石，效率与性能双提升** |

---

