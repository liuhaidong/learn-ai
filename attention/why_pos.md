# 📄 Transformer 中注意力计算与位置编码的数学原理结构化整理

---

## ☑️ 简要

本文围绕 Transformer 模型中的注意力计算机制，系统性地解释以下关键主题：

1. Token 向量点积的数学意义
2. 位置编码的设计原则
3. RoPE 与正余弦位置编码的对比
4. 位置编码的“平移等价性”

---

## 1️⃣ Token 向量点积的数学意义

### ▶️ 基本公式

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

* Q: 查询向量 (Query)
* K: 键向量 (Key)
* V: 值向量 (Value)

### ▶️ 点积代表“轨向相似性”

$$
Q \cdot K = \|Q\| \cdot \|K\| \cdot \cos(\theta)
$$

* $\theta$: Q 和 K 之间的旋转角
* $\cos(\theta)$ 越大，表示方向越相似，点积越大，被认为相关度高

### ▶️ 和正交间距离的对比

| 特性         | 点积 | 正交间距离 |
| ---------- | -- | ----- |
| 表达方向相似性    | ☑️ | ❌     |
| 对形状敏感      | 较低 | ☑️高   |
| 适合 softmax | ☑️ | ❌     |

---

## 2️⃣ 位置编码设计原则

### ▶️ Transformer 需要位置编码

* Attention 本质是 permutation-invariant，即无顺序效应
* 需要将位置信息以编码形式引入

### ▶️ 正余弦位置编码 PE(pos, i)

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

* 可出现周期性和多频率编码
* 输入保持在 \[-1,1]，易于 softmax 处理

### ▶️ “平移等价性”

利用三角函数公式，可以得到：

$$
PE(pos + \Delta) - PE(pos) \approx \text{只与 } \Delta \text{ 有关，与 pos 无关}
$$

即相对位置在向量空间中的代表是常规的，有利于统一形式表达轨径结构

---

## 3️⃣ RoPE (Rotary Position Embedding)

### ▶️ 基本思想

* RoPE 将位置信息以 **处理角度的方式合并到向量** 中
* 对每个向量分组做处理，对对儿向量做处理：

$$
z = x^1 + i x^2,\quad \text{RoPE}(z, \theta_p) = z \cdot e^{i\theta_p}
$$

* $\theta_p = \frac{1}{10000^{2i/d}} \cdot p$

### ▶️ 在 Attention 中表示相对位置

$$
Q_p \cdot K_q^* = \|x\|^2 e^{i(\theta_p - \theta_q)}
$$

* 内积的角度只与 $\theta_p - \theta_q$ 有关
* 即内积分数是相对位置的函数，内置地表达相对位置

### ▶️ RoPE 优势

| 特性      | RoPE   | 正余弦 PE |
| ------- | ------ | ------ |
| 相对位置    | ☑️明确表示 | ❌隐含表示  |
| 多频率     | ☑️     | ☑️     |
| 应用于 Q/K | ☑️     | ❌      |
| 解释力     | 高，旋转函数 | 低，动态差值 |

---

## 4️⃣ 数学结论

### ▶️ Attention 点积本质

* 轨向相似性 = 词之间语义关联
* 利用点积 = 解析 token 关联性

### ▶️ 位置编码策略

| 策略           | 类型 | 特性                    |
| ------------ | -- | --------------------- |
| 正余弦 PE       | 静态 | 多频率 + 平移等价            |
| Learnable PE | 可训 | 容易过拟，不可续              |
| RoPE         | 静态 | 相对位置运算，应用应对 Attention |

---

## 5️⃣ 总结一句话

> Transformer 通过 token 向量点积控制 attention 分布，位置编码贡献于倾向优先级，RoPE 通过旋转形式明确地表达相对位置信息，是现代大型 LLM 最接地气的策略之一。
