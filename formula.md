# MOSS-RLHF PPO 论文公式与代码实现对照

> 基于论文《Secrets of RLHF in Large Language Models Part I: PPO》公式体系，阐述代码实现。
---

## 一、3.1 奖励建模（Reward Modeling）

### 公式 (1)：基础 Bradley-Terry 排序损失

\[
\mathcal{L}(\psi) = -\log \sigma \big( r_\psi(x, y_w) - r_\psi(x, y_l) \big)
\]

**代码位置**：`rm/reward_trainer.py:138-142`

```python
preferred_rewards = logits[:bs]
rejected_rewards = logits[bs:]
probs = torch.sigmoid(preferred_rewards - rejected_rewards)
loss = (-torch.log(probs + 1e-5)).mean()
```

**实现原理**：
- 将 batch 按前半/后半拆分为 chosen 和 rejected 两组
- 计算得分差值 `preferred_rewards - rejected_rewards`，过 sigmoid 得到 chosen 优于 rejected 的概率
- 负对数似然作为损失：当差值越大（模型判定越正确），损失越小；反之差值趋向负无穷则损失急剧增大
- 本质是一个二分类排序损失，让 RM 学会区分同一 prompt 下两个回复的相对好坏

---

### 公式 (2)：引入模仿学习的混合损失

\[
\mathcal{L}(\psi) = -\lambda \mathbb{E}_{(x,y_w,y_l)} \big[ \log \sigma (r(x,y_w) - r(x,y_l)) \big] + \beta_{\mathrm{rm}} \mathbb{E}_{(x,y_w)} \big[ \log (r'(x,y_w)) \big]
\]

**代码位置**：`rm/reward_trainer.py:142-156`

```python
if self.calculate_lm_loss:
    lm_logits, *_ = outputs
    scores = lm_logits[:bs, :-1, :]  # lm loss for chosen only
    preds = scores.argmax(dim=-1)
    # ... compute cross-entropy loss on chosen response
    lm_loss = self._lm_loss(scores, preds, label_vec, training)
    loss = loss + self.lm_loss_factor * lm_loss
```

**实现原理**：
- 模型保留一个共享底层 Transformer 的「兄弟头」`r'`，输出维度对应词表大小
- 第一项：排序损失（公式 1），系数由 `λ`（代码中隐式为 1）控制
- 第二项：LM 损失（自回归交叉熵），仅对 chosen（偏好回复）计算，系数 `β_rm` 对应 `--reward_lm_loss_factor`
- 第二项只在 `calculate_lm_loss=True`（即 `reward_lm_loss_factor > 0`）时启用
- 辅助 LM 损失有效防止 RM 在训练中丢失文本生成能力（表征坍缩）

---

### 公式 (3)：总奖励公式（KL 约束）

\[
r_{\mathrm{total}}(x, y) = r(x, y) - \eta \cdot \mathrm{KL}\big( \pi_{\phi}^{\mathrm{RL}}(y|x) \;\|\; \pi^{\mathrm{SFT}}(y|x) \big)
\]

**代码位置**：`ppo/ppo_trainer.py:330-334, 362-363`

```python
# 计算 token 级 KL 散度近似
ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], sampled_vec[:, 1:])
logprobs = logprobs_from_logits(logits[:, :-1, :], sampled_vec[:, 1:])
kl_penalty = (-self.kl_penalty_weight * (logprobs - ref_logprobs)).cpu()

# 组装 token 级奖励
penalized_rewards = kl_penalty[i].clone()
penalized_rewards[-1] += rewards[i]  # RM 总分只加在最后 token
```

**实现原理**：
- 公式 (3) 在论文中是**概念性公式**，说明 RL 中的总奖励 = RM 打分 − KL 惩罚
- 代码在**逐 token 级别**实现了这个公式（即公式 19）：
  - 每个中间 token：`r_t = -η · (log π_θ - log π_SFT)`
  - 最后一个 token：`r_T = r(x,y) - η · (log π_θ - log π_SFT)`
- KL 散度用 `logprobs - ref_logprobs` 近似，即 \(\log\frac{\pi_\theta}{\pi_{\text{ref}}}\)
- η 对应 `--kl_penalty_weight`（英文 0.01，中文 0.1）

---

## 二、3.2.1 策略梯度方法（Policy Gradient Methods）

### 公式 (4)：策略梯度更新规则

\[
\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)
\]

**代码位置**：`ppo/ppo_trainer.py:527-548`

```python
def train_step(self, batch, **kwargs):
    self.optimizer.zero_grad()
    model_output = self.RLHF_model_forward(batch)
    loss = self.criterion(model_output, batch)
    self.accelerator.backward(loss)
    self.optimizer.step()
    self.scheduler.step()
```

**实现原理**：
- 公式 (4) 是策略梯度的抽象数学表达，代码中由 PyTorch 优化器执行
- Policy 学习率 `5e-7`，Critic 学习率 `1.5e-6`（3 倍，因为 Critic 需要更快适应变化的价值分布）
- `self.optimizer = AdamW` 或 `DeepSpeedCPUAdam`（ZeRO-2 offload 时）
- 学习率调度：warmup + inverse sqrt 衰减
- 全局梯度裁剪由 DeepSpeed 配置 `gradient_clipping: 1.0` 实现

---

### 公式 (5)：策略梯度的一般形式

\[
\nabla_{\theta}J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t = 0}^{T}\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)\Phi_t\right]
\]

**代码位置**：`ppo/ppo_trainer.py:412, 436-446`

```python
logprobs = logprobs_from_logits(policy_logits[:, :-1, :], batch['text_vec'][:, 1:]) * loss_mask
log_ratio = (logprobs - old_logprobs) * loss_mask
ratio = torch.exp(log_ratio)  # r_t(θ) = π_new / π_old

pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - pg_clip, 1.0 + pg_clip)
pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * loss_mask) / n
```

**实现原理**：
- 公式 (5) 是策略梯度的理论基础，Φ_t 是累积奖励估计量
- 代码中 Φ_t 用的是 GAE 计算的优势函数 Â_t（对应公式 12 的完整形式）
- `logprobs_from_logits()` 对 logits 做 log_softmax，然后 gather 出实际生成 token 的对数概率
- 不直接计算 Σ ∇_θ log π，而是通过 PPO-clip 损失函数隐式实现策略梯度

---

### 公式 (6)：优势函数作为 Φ_t

\[
\Phi_t = A(s_t, a_t)
\]

**代码位置**：`ppo/ppo_datahelper.py:192-208`

计算 GAE 优势后，在 `ppo_trainer.py:401-408` 中用作 PPO 损失的权重：

```python
advantages = batch['advantages']
if self.use_advantage_norm:
    advantages = whiten(advantages, loss_mask, accelerator=self.accelerator)
if self.use_advantage_clip:
    advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
```

**实现原理**：
- 公式 (6) 从理论上证明了用优势函数 A 替代原始回报 R 可以降低策略梯度方差
- Q 无法精确计算，所以用 GAE（公式 9）估计 Â_t 作为 A 的近似
- 代码中优势值还经过两个后处理步骤：
  - **优势归一化**（`whiten()`）：减均值除以标准差，跨所有 GPU 聚合统计量
  - **优势裁剪**（`--advantage_clip=0.5`）：限制在 [-0.5, 0.5]，防止大优势主导梯度

---

## 三、3.2.2 广义优势估计（GAE）

### 公式 (7)：k-步回报

\[
\hat{R}_t^k = r_t + \gamma r_{t+1} + \dots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k})
\]

**代码位置**：这是概念公式，代码中不直接计算 k-步回报，而是直接用 GAE（公式 9）计算优势

---

### 公式 (8)：k-步优势

\[
\hat{A}_t^k = \hat{R}_t^k - V(s_t) = -V(s_t) + r_t + \gamma r_{t+1} + \dots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k})
\]

**代码位置**：概念公式，不直接实现，由 GAE 统一处理

---

### 公式 (9)：GAE 定义

\[
\hat{A}_t^{\mathrm{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
\]

其中 \[δ_t = r_t + γV(s_{t+1}) - V(s_t)\]

**代码位置**：`ppo/ppo_datahelper.py:192-208`

```python
def get_advantages_and_returns(self, rewards, values):
    response_length = len(values)
    advantages_reversed = []
    lastgaelam = 0
    for t in reversed(range(response_length)):
        nextvalues = values[t + 1] if t < response_length - 1 else 0.0
        delta = rewards[t] + self.gamma * nextvalues - values[t]   # TD error
        lastgaelam = delta + self.gamma * self.lam * lastgaelam    # GAE 递推
        advantages_reversed.append(lastgaelam)
    advantages = advantages_reversed[::-1]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

**实现原理（从后往前遍历）**：
1. **TD Error 计算**（第 201 行）：`δ_t = r_t + γ·V(s_{t+1}) - V(s_t)`
   - 最后位置 `V(s_{T+1}) = 0`，所以 `δ_T = r_T - V(s_T)`
2. **GAE 递推**（第 202 行）：`A_t = δ_t + γλ·A_{t+1}`
   - 初始 `lastgaelam = 0`（即 `A_{T+1} = 0`）
   - 这等价于公式 (9) 的指数加权和：`A_t = Σ (γλ)^l δ_{t+l}`
3. **Return 计算**（第 206 行）：`R̂_t = A_t + V(s_t)`
   - 用作 Critic 的回归目标
4. **关键参数**：`γ = 1.0`（不折扣，因对话有固定长度上限），`λ = 0.95`（论文推荐，平衡偏差-方差）

**偏差-方差权衡**：
- `λ = 0`（公式 10）：GAE 退化为单步 TD，`Â_t = δ_t`，方差最小但偏差最大
- `λ = 1`（公式 11）：GAE 退化为蒙特卡洛，`Â_t = Σ γ^l r_{t+l} - V(s_t)`，偏差最小但方差最大
- `λ = 0.95`：在两者之间取得平衡

---

### 公式 (12)：带 GAE 的策略梯度

\[
\nabla_{\theta}\hat{J} (\theta) = \frac{1}{|\mathcal{D}|}\sum_{\tau \in \mathcal{D}}\sum_{t = 1}^{T}\nabla_{\theta}\log \pi_{\theta}(a_{t}|s_{t})\hat{A}_{t}
\]

**代码位置**：`ppo/ppo_trainer.py:436-446`（通过 PPO-clip loss 隐式实现）

**实现原理**：
- 公式 (12) 是将 GAE 优势 Â_t 代入策略梯度公式 (5) 的结果
- 代码不直接计算这个梯度，而是通过 `pg_loss = -advantages * ratio` 让 autograd 自动求导
- 负号是因为 loss 是最小化，而公式 (12) 是梯度上升（最大化期望回报）
- `ratio = exp(log π_new - log π_old)` 是重要性采样比率，用于 off-policy 修正

---

## 四、3.2.3 近端策略优化（PPO）

### 公式 (13)：TRPO 的 KL 硬约束

\[
\begin{array}{rl}
\mathrm{maximize}_{\theta} & \hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\mathrm{old}}}(a_{t}|s_{t})}\hat{A}_{t}\right], \\
\mathrm{subject~to} & \hat{\mathbb{E}}_{t}[\mathrm{KL}(\pi_{\theta_{\mathrm{old}}}(\cdot |s_{t}),\pi_{\theta}(\cdot |s_{t}))]\leq \delta
\end{array}
\]

**代码位置**：不直接实现。PPO-max 选择 PPO-Clip 替代 TRPO 的硬约束

---

### 公式 (14)：PPO-惩罚（PPO-Penalty）

\[
\mathcal{L}_{\mathrm{ppo - penalty}}(\theta) = \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}\hat{A}_t\right] - \beta \cdot \mathrm{KL}(\pi_{\theta_{\mathrm{old}}}(\cdot |s_t),\pi_\theta(\cdot |s_t))
\]

**代码位置**：不直接实现。代码选用 PPO-Clip 作为默认策略优化方法

---

### 公式 (15)：PPO-裁剪（PPO-Clip）⭐

\[
\mathcal{L}_{\mathrm{ppo - clip}}(\theta) = \hat{\mathbb{E}}_t\left[\min \left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}\hat{A}_t,\;\mathrm{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)},1 - \epsilon,1 + \epsilon\right)\hat{A}_t\right)\right]
\]

**代码位置**：`ppo/ppo_trainer.py:431-447`

```python
log_ratio = (logprobs - old_logprobs) * loss_mask
ratio = torch.exp(log_ratio)                                    # r_t(θ)

pg_loss1 = -advantages * ratio                                  # 未裁剪项
pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - pg_clip, 1.0 + pg_clip)  # 裁剪项

if self.use_policy_loss_clip:
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * loss_mask) / n
else:
    pg_loss = torch.sum(pg_loss1 * loss_mask) / n
```

**实现原理**：
- `ratio` 即重要性采样比率 r_t(θ) = π_new / π_old
- `pg_clip = 0.2`（即 ε = 0.2），ratio 被限制在 [0.8, 1.2]
- **`torch.max(pg_loss1, pg_loss2)` 等价于论文中的 `min`**：
  - 原始：`min(r·A, clip(r)·A)` 取保守估计
  - 代码（带负号）：`max(-r·A, -clip(r)·A)` = `-min(r·A, clip(r)·A)`
- **Clip 机制**：
  - 当 A > 0（好动作）：ratio 被限制不超过 1+ε，防止过度增大概率
  - 当 A < 0（坏动作）：ratio 被限制不低于 1-ε，防止过度减小概率
- `pg_clipfrac` 监控有多少 token 的 ratio 被裁剪了

---

### 公式 (16)：价值函数损失（Critic Loss）

\[
\mathcal{L}_{\mathrm{critic}}(\phi) = \hat{\mathbb{E}}_t\left[\| V_{\phi}(s_t) - \hat{R}_t\| ^2\right]
\]

**代码位置**：`ppo/ppo_trainer.py:414-429`

```python
values_clipped = torch.clamp(values, old_values - value_clip, old_values + value_clip)
vf_loss1 = (values - returns) ** 2                    # 未裁剪的 MSE
vf_loss2 = (values_clipped - returns) ** 2            # 裁剪后的 MSE

if self.use_critic_loss_clip:
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * loss_mask) / n
else:
    vf_loss = 0.5 * torch.sum(vf_loss1 * loss_mask) / n
```

**实现原理**：
- 基础形式是 MSE：`(V(s_t) - R̂_t)²`
- 代码额外实现了 **Value Clipping**（PPO 原始论文的技巧）：
  - `values_clipped = clamp(V_new, V_old - 0.2, V_old + 0.2)`
  - 当新旧估值差超过 0.2 时，裁剪后的损失被使用
  - `torch.max(vf_loss1, vf_loss2)` 取更保守的估计
- 0.5 因子是标准 MSE 梯度惯例
- 仅在 response 部分（`loss_mask=1`）计算损失

---

### 公式 (17)：预训练损失混合（PPO-ptx）

\[
\mathcal{L}_{\mathrm{ppo - ptx}}(\theta) = \mathcal{L}_{\mathrm{ppo - clip}}(\theta) + \lambda_{\mathrm{ptx}}\mathbb{E}_{x\sim \mathcal{D}_{\mathrm{pretrain}}}\left[\log (\pi_{\theta}^{\mathrm{RL}}(x))\right]
\]

**代码位置**：`ppo/ppo_trainer.py:455-489`

```python
if self.use_ppo_pretrain_loss:
    pretrain_sampled_vec = batch['ppo_context_vec']
    scores, *_ = self.policy_model_forward(pretrain_sampled_vec)
    scores = scores[:, :-1, :]
    # ... compute cross-entropy loss on pretrain data
    pretrain_loss = self.loss_fn(score_view, labels.reshape(-1)).sum()
    pretrain_loss = pretrain_loss / target_tokens

    loss1 = pg_loss + vf_loss_weight * vf_loss    # PPO loss
    loss2 = ppo_pretrain_loss_weight * pretrain_loss  # PTX loss
    loss = loss1 + loss2
```

**实现原理**：
- 第一项：PPO-clip 损失（公式 15）
- 第二项：标准自回归语言模型损失（next-token prediction），数据来自 `PPOSFTDataset`
- `λ_ptx` 对应 `--ppo_pretrain_loss_weight`
- 两项分别调用 `self.accelerator.backward()` 进行反向传播
- 作用：对抗「对齐税」——防止模型为了迎合 RM 而牺牲通用语言能力

---

## 五、PPO-max 改进（论文 5.4 节）

### 公式 (18)：奖励归一化与裁剪

\[
\tilde{r}(x, y) = \text{clip}\left( \frac{r_n(x, y) - \overline{r(x, y)}}{\sigma(r(x, y))}, -\delta, \delta \right)
\]

**代码位置**：`ppo/ppo_trainer.py:308-321` + `utils.py:88-125`

```python
# 步骤 1：更新滑动统计量
if self.use_reward_scaling:
    rewards_mean, rewards_std = self.running.update(rewards)
    if self.use_reward_norm:
        rewards = (rewards - self.running.mean) / self.running.std  # 标准化
    else:
        rewards /= self.running.std  # 仅除以标准差

# 步骤 2：裁剪
if self.use_reward_clip:
    rewards = torch.clip(rewards, -self.reward_clip, self.reward_clip)
```

**实现原理**：
- `RunningMoments` 类（`utils.py:88-125`）维护全局奖励的滑动均值/标准差
  - 使用 Welford 在线算法更新，跨所有 GPU 聚合统计量
  - 公式：`new_mean = old_mean + δ * xs_count / tot_count`
- 标准化后奖励被裁剪到 `[-reward_clip, reward_clip]`（默认 10.0）
- 论文推荐 δ=0.3，代码默认 10.0 更宽松，配合其他约束使用
- 解决了 RM 打分分布随训练漂移的问题

---

### 公式 (19)：Token 级 KL 惩罚

\[
r_{\mathrm{total}}(x, y_i) = r(x, y_i) - \eta \cdot \text{KL}\left( \pi_\theta^{\mathrm{RL}}(y_i|x) \;\|\; \pi^{\mathrm{SFT}}(y_i|x) \right)
\]

**代码位置**：`ppo/ppo_trainer.py:330-334, 360-363`

```python
# 计算逐 token KL 近似
ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], sampled_vec[:, 1:])
logprobs = logprobs_from_logits(logits[:, :-1, :], sampled_vec[:, 1:])
kl_penalty = (-self.kl_penalty_weight * (logprobs - ref_logprobs)).cpu()

# 组装 token 级奖励
for i in range(bsz):
    penalized_rewards = kl_penalty[i].clone()
    penalized_rewards[-1] += rewards[i]  # RM 总分只加在最后 token
```

**实现原理**：
- KL 散度用 `logprobs - ref_logprobs` 近似（即 \(\log\frac{\pi_\theta}{\pi_{\text{SFT}}}\)）
- 每个 token 都减去 KL 惩罚：`r_t = -η · KL_t`
- RM 总分只在最后一个 token（EOS 位置）加上：`r_T = r(x,y) - η · KL_T`
- 这是论文识别出的**最关键改进**：防止策略模型生成 OOD 文本（模式坍塌）
- η 参数：英文 0.01，中文 0.1

---

### 公式 (20)：PPO-max 完整目标

\[
\mathcal{L}_{\text{ppo-max}}(\theta) = \mathcal{L}_{\text{ppo-clip}}(\theta) + \lambda_{\text{ptx}} \cdot \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log \pi_\theta^{\text{RL}}(x) \right]
\]

**代码位置**：`ppo/ppo_trainer.py:479-489`

```python
if self.use_entropy_loss:
    loss1 = pg_loss + vf_loss_weight * vf_loss + entropy_loss_weight * entro_loss
else:
    loss1 = pg_loss + vf_loss_weight * vf_loss
loss2 = ppo_pretrain_loss_weight * pretrain_loss
loss = loss1 + loss2
```

**实现原理**：
- 与公式 (17) 结构相同，PPO-max 继承了 PPO-ptx 的预训练损失混合
- 代码中还支持可选的 entropy bonus（`--use_entropy_loss`），但论文推荐 KL 惩罚优于熵正则

---