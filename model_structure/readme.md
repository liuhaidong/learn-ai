# LLM 模型架构迭代

---

## **1. LLaMA（Meta）**
### **模型架构**
- **LLaMA-1**（2023）：
  - 基于Transformer架构，采用**RMSNorm**（Root Mean Square Layer Normalization）替代LayerNorm。
  - 使用**SwiGLU**（Swish-Gated Linear Unit）激活函数替代ReLU。
  - **RoPE**（Rotary Positional Embedding）位置编码，取代绝对位置编码。
- **LLaMA-2**（2023）：
  - 架构与LLaMA-1类似，但优化了训练数据、上下文长度（扩展到4k tokens）和微调策略。
  - 增加了**Grouped Query Attention (GQA)**，减少推理时的显存占用。

### **激活函数**
- **SwiGLU**（LLaMA-1/2）：
  - 相比ReLU，能提供更平滑的梯度，提高模型表达能力。

### **损失函数**
- 标准的**交叉熵损失（Cross-Entropy Loss）**，用于自回归语言建模。

---

## **2. GPT系列（OpenAI）**
### **模型架构**
- **GPT-1**（2018）：
  - 12层Transformer，使用**ReLU**激活函数。
  - 绝对位置编码。
- **GPT-2**（2019）：
  - 扩展到48层，仍使用ReLU，但数据量和参数大幅增加。
- **GPT-3**（2020）：
  - 1750亿参数，改用**GeLU**（Gaussian Error Linear Unit）激活函数。
  - 仍使用绝对位置编码，但引入稀疏注意力（Sparse Transformer）。
- **GPT-4**（2023）：
  - 具体架构未公开，但推测使用**SwiGLU**或类似改进的激活函数。
  - 可能采用**混合专家（MoE）**架构，提高计算效率。

### **激活函数**
- **ReLU → GeLU → SwiGLU（推测）**  
  - GeLU比ReLU更平滑，适合深层网络。

### **损失函数**
- 始终使用**交叉熵损失**，但训练数据规模和方法（RLHF）不断优化。

---

## **3. Qwen（阿里云）**
### **模型架构**
- **Qwen-7B/14B**（2023）：
  - 类似LLaMA架构，采用**RoPE**位置编码。
  - 使用**SwiGLU**激活函数。
- **Qwen-72B**（2023）：
  - 更大参数规模，优化训练策略，支持更长上下文（32k tokens）。
  - 可能引入**FlashAttention**加速计算。

### **激活函数**
- **SwiGLU**（与LLaMA类似）。

### **损失函数**
- 标准**交叉熵损失**，但可能结合RLHF进行微调。

---

## **4. DeepSeek（深度求索）**
### **模型架构**
- **DeepSeek-MoE**（2024）：
  - 采用**混合专家（MoE）**架构，部分参数动态激活，提高计算效率。
  - 仍基于Transformer，但引入**专家选择路由（Router）**机制。
  - 可能使用**RoPE**或改进的位置编码。
- **DeepSeek-V2/V3**：
  - 优化训练策略，支持更长上下文（128k+ tokens）。
  - 可能采用**GQA**或**FlashAttention-2**加速推理。

### **激活函数**
- 推测使用**SwiGLU**或类似变体。

### **损失函数**
- 标准交叉熵损失 + **MoE辅助损失**（专家负载均衡）。

---

## **总结对比**
| 模型系列 | 架构变化 | 激活函数 | 损失函数 |
|----------|----------|----------|----------|
| **LLaMA** | RMSNorm + RoPE + GQA | SwiGLU | 交叉熵 |
| **GPT** | ReLU → GeLU → (SwiGLU?) | GeLU/SwiGLU | 交叉熵 + RLHF |
| **Qwen** | RoPE + SwiGLU | SwiGLU | 交叉熵 |
| **DeepSeek** | MoE + RoPE + GQA | SwiGLU? | 交叉熵 + MoE负载均衡 |

### **主要趋势**
1. **激活函数**：从ReLU → GeLU → SwiGLU，更平滑、表达能力更强。
2. **位置编码**：绝对位置 → RoPE（旋转位置编码），提升长文本建模能力。
3. **注意力优化**：GQA、FlashAttention等加速计算。
4. **架构扩展**：MoE架构（GPT-4、DeepSeek）提高计算效率。

不同版本的改进主要集中在**训练效率、长上下文支持、推理优化**等方面，而核心的Transformer架构仍然保持稳定。