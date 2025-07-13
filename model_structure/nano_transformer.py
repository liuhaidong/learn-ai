import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional

# ======================= 1. 基础组件 =======================

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 线性变换并重塑为多头形式
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重新组合多头结果
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attention_output)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 多头注意力 + 残差连接
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# ======================= 2. 简单Tokenizer =======================

class SimpleTokenizer:
    """简单的tokenizer"""
    
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        
        # 特殊token
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"  # 开始token
        self.eos_token = "<EOS>"  # 结束token
        
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """构建词汇表"""
        # 统计词频
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        
        # 构建词汇表
        vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        for word, freq in word_counts.most_common():
            if freq >= min_freq:
                vocab.append(word)
        
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"词汇表大小: {self.vocab_size}")
        
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词"""
        # 转换为小写并按空格和标点分割
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # 替换标点为空格
        return text.split()
    
    def encode(self, text: str) -> List[int]:
        """编码文本为token ids"""
        words = self._tokenize(text)
        return [self.vocab.get(word, self.vocab[self.unk_token]) for word in words]
    
    def decode(self, token_ids: List[int]) -> str:
        """解码token ids为文本"""
        words = [self.inverse_vocab.get(idx, self.unk_token) for idx in token_ids]
        return ' '.join(words)

# ======================= 3. 数据集 =======================

class QADataset(Dataset):
    """问答数据集"""
    
    def __init__(self, qa_pairs: List[Tuple[str, str]], tokenizer: SimpleTokenizer, 
                 max_len: int = 128):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        
        # 编码问题和答案
        question_tokens = self.tokenizer.encode(question)
        answer_tokens = self.tokenizer.encode(answer)
        
        # 构造输入序列: <BOS> + question + answer + <EOS>
        input_sequence = ([self.tokenizer.vocab[self.tokenizer.bos_token]] + 
                         question_tokens + answer_tokens + 
                         [self.tokenizer.vocab[self.tokenizer.eos_token]])
        
        # 截断或填充
        if len(input_sequence) > self.max_len:
            input_sequence = input_sequence[:self.max_len]
        else:
            input_sequence.extend([self.tokenizer.vocab[self.tokenizer.pad_token]] * 
                                (self.max_len - len(input_sequence)))
        
        # 构造标签（用于语言模型训练）
        labels = input_sequence[1:] + [self.tokenizer.vocab[self.tokenizer.pad_token]]
        
        return {
            'input_ids': torch.tensor(input_sequence, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor([1 if x != self.tokenizer.vocab[self.tokenizer.pad_token] 
                                          else 0 for x in input_sequence], dtype=torch.long)
        }

# ======================= 4. 主要模型 =======================

class SimpleTransformer(nn.Module):
    """简单的Transformer语言模型"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 1024, max_len: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入和位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self.init_weights()
    
    def init_weights(self):
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.02)
    
    def create_causal_mask(self, seq_len: int):
        """创建因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        
        # 词嵌入
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 创建因果掩码
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # 如果有attention mask，结合使用
        if attention_mask is not None:
            # 扩展attention_mask维度以匹配causal_mask
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.expand(-1, -1, seq_len, -1)
            causal_mask = causal_mask * extended_attention_mask
        
        # 通过Transformer块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # 层归一化和输出投影
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        
        return logits

# ======================= 5. 训练器 =======================

class TransformerTrainer:
    """Transformer训练器"""
    
    def __init__(self, model: SimpleTransformer, tokenizer: SimpleTokenizer, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab[tokenizer.pad_token])
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 前向传播
            logits = self.model(input_ids, attention_mask)
            
            # 计算损失
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, dataloader: DataLoader, num_epochs: int = 10):
        """训练模型"""
        print("开始训练...")
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")
        print("训练完成!")

# ======================= 6. 推理生成器 =======================

class TextGenerator:
    """文本生成器"""
    
    def __init__(self, model: SimpleTransformer, tokenizer: SimpleTokenizer, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.95) -> str:
        """生成文本回答"""
        self.model.eval()
        
        # 根据词汇表大小调整top_k
        actual_top_k = min(top_k, self.tokenizer.vocab_size)
        
        # 编码prompt
        input_ids = [self.tokenizer.vocab[self.tokenizer.bos_token]] + self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 获取预测
                outputs = self.model(generated_ids)
                logits = outputs[:, -1, :] / temperature
                
                # Top-k和top-p采样
                if actual_top_k > 0:
                    logits = self.top_k_filter(logits, actual_top_k)
                if top_p > 0:
                    logits = self.top_p_filter(logits, top_p)
                
                # 采样下一个token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # 如果生成了EOS token，停止生成
                if next_token.item() == self.tokenizer.vocab[self.tokenizer.eos_token]:
                    break
                
                # 添加到生成序列
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        
        # 移除特殊token并返回prompt后的部分
        generated_text = generated_text.replace(self.tokenizer.bos_token, "")
        generated_text = generated_text.replace(self.tokenizer.eos_token, "")
        generated_text = generated_text.replace(self.tokenizer.pad_token, "")
        
        # 提取答案部分（去掉prompt）
        prompt_text = prompt.lower()
        if prompt_text in generated_text:
            answer_start = generated_text.find(prompt_text) + len(prompt_text)
            answer = generated_text[answer_start:].strip()
        else:
            answer = generated_text.strip()
        
        return answer if answer else "我不确定如何回答"
    
    def top_k_filter(self, logits, k):
        """Top-k过滤 - 修复版本"""
        if k <= 0:
            return logits
        
        # 确保k不超过词汇表大小
        vocab_size = logits.size(-1)
        k = min(k, vocab_size)
        
        values, indices = torch.topk(logits, k)
        min_values = values[:, -1, None]
        return torch.where(logits < min_values, torch.ones_like(logits) * -float('inf'), logits)
    
    def top_p_filter(self, logits, p):
        """Top-p (nucleus) 过滤"""
        if p <= 0 or p >= 1:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 找到累积概率超过p的位置
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('inf'))
        
        return logits

# ======================= 7. 主函数和示例 =======================

def create_sample_data():
    """创建示例训练数据"""
    qa_pairs = [
        ("你好", "你好！很高兴见到你"),
        ("今天天气怎么样", "今天天气很好，阳光明媚"),
        ("你是谁", "我是一个AI助手，可以回答你的问题"),
        ("什么是人工智能", "人工智能是计算机科学的一个分支"),
        ("如何学习编程", "学习编程需要多练习多实践"),
        ("Python是什么", "Python是一种编程语言"),
        ("机器学习有什么用", "机器学习可以用于数据分析和预测"),
        ("深度学习是什么", "深度学习是机器学习的一个子领域"),
        ("谢谢", "不客气！很高兴能帮助你"),
        ("再见", "再见！期待下次交流"),
        ("你能做什么", "我可以回答问题和进行对话"),
        ("计算机是什么", "计算机是用于处理数据的电子设备"),
        ("互联网的作用", "互联网连接了全世界的信息"),
        ("编程语言有哪些", "常见的编程语言有Python Java C++等"),
        ("什么是算法", "算法是解决问题的步骤和方法"),
        ("数据结构重要吗", "数据结构是编程的基础非常重要"),
        ("如何提高编程能力", "多写代码多看优秀的代码"),
        ("什么是开源", "开源是指源代码公开可以自由使用"),
        ("云计算是什么", "云计算是通过网络提供计算资源"),
        ("区块链的应用", "区块链可以用于数字货币和数据安全"),
    ]
    
    # 扩展数据集
    extended_pairs = []
    for q, a in qa_pairs:
        extended_pairs.append((q, a))
        # 添加一些变体
        extended_pairs.append((q + "？", a))
        extended_pairs.append(("请问" + q, a))
    
    return extended_pairs

def main():
    """主函数"""
    print("=== 初始化简单Transformer模型 ===")
    
    # 创建训练数据
    qa_pairs = create_sample_data()
    print(f"训练数据量: {len(qa_pairs)}")
    
    # 构建tokenizer
    tokenizer = SimpleTokenizer()
    all_texts = []
    for q, a in qa_pairs:
        all_texts.extend([q, a])
    
    tokenizer.build_vocab(all_texts, min_freq=1)
    
    # 创建数据集和dataloader
    dataset = QADataset(qa_pairs, tokenizer, max_len=64)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 创建模型
    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,  # 较小的模型维度
        num_heads=4,
        num_layers=3,
        d_ff=256,
        max_len=64,
        dropout=0.1
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    trainer = TransformerTrainer(model, tokenizer)
    trainer.train(dataloader, num_epochs=20)
    
    # 创建生成器
    generator = TextGenerator(model, tokenizer)
    
    # 测试生成
    print("\n=== 模型回答测试 ===")
    test_questions = [
        "你好",
        "什么是人工智能",
        "如何学习编程",
        "Python是什么",
        "你能做什么",
        "谢谢"
    ]
    
    for question in test_questions:
        answer = generator.generate(question, max_length=20, temperature=0.7)
        print(f"问题: {question}")
        print(f"回答: {answer}")
        print("-" * 50)
    
    # 交互式问答
    print("\n=== 交互式问答 (输入'quit'退出) ===")
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'quit':
            break
        
        answer = generator.generate(user_input, max_length=30, temperature=0.8)
        print(f"AI: {answer}")

# 简化版测试函数
def simple_test():
    """简化版测试，用于快速验证"""
    print("=== 简化版测试 ===")
    
    # 创建最简单的训练数据
    simple_qa = [
        ("你好", "你好"),
        ("谢谢", "不客气"),
        ("再见", "再见"),
        ("天气", "很好"),
        ("吃饭", "好的"),
        ("学习", "努力"),
        ("工作", "加油"),
        ("休息", "好的"),
    ]
    
    # 构建tokenizer
    tokenizer = SimpleTokenizer()
    all_texts = []
    for q, a in simple_qa:
        all_texts.extend([q, a])
    
    tokenizer.build_vocab(all_texts, min_freq=1)
    print(f"词汇表内容: {list(tokenizer.vocab.keys())}")
    
    # 创建小模型
    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=64,
        max_len=32,
        dropout=0.1
    )
    
    print(f"简化模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 简单训练
    dataset = QADataset(simple_qa, tokenizer, max_len=32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    trainer = TransformerTrainer(model, tokenizer)
    trainer.train(dataloader, num_epochs=10)
    
    # 测试
    print("\n=== 测试生成 ===")
    generator = TextGenerator(model, tokenizer)
    test_questions = ["你好", "谢谢", "再见", "天气", "学习"]
    
    for question in test_questions:
        answer = generator.generate(question, max_length=5, temperature=0.5, top_k=5)
        print(f"问题: {question} -> 回答: {answer}")
    
    # 简单交互
    print("\n=== 简单交互 (输入'quit'退出) ===")
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'quit':
            break
        
        answer = generator.generate(user_input, max_length=10, temperature=0.7, top_k=5)
        print(f"AI: {answer}")

if __name__ == "__main__":
    # 可以选择运行完整版本或简化版本
    choice = input("选择运行模式 (1: 完整版本, 2: 简化测试): ")
    
    if choice == "2":
        simple_test()
    else:
        main()
