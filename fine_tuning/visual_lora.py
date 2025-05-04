import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 加载数据集
dataset = load_dataset("imdb", split="train[:5000]")
dataset = dataset.train_test_split(test_size=0.2)

# 2. 分词处理
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize)

# 3. 加载预训练模型
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. 应用LoRA配置
lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=16,
    target_modules=["query", "value"],  # 对 attention 模块应用LoRA
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# 5. 训练配置
training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    save_strategy="no"
)

# 6. 可视化训练损失
losses = []

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    accuracy = (preds == labels).astype(float).mean()
    return {"accuracy": accuracy}

def compute_loss(model, inputs, return_outputs=False):
    outputs = model(**inputs)
    loss = outputs.loss
    losses.append(loss.item())
    return (loss, outputs) if return_outputs else loss

# 提取LoRA层的权重（训练前）
def get_lora_weights(model):
    lora_weights = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_weights[name] = param.detach().cpu().clone()
    return lora_weights

# 训练前提取
lora_weights_before = get_lora_weights(model)

Trainer.compute_loss = compute_loss  # Override默认的loss方法

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# 7. 可视化损失变化
plt.plot(losses)
plt.title("Training Loss with LoRA Fine-Tuning")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# 训练后提取
lora_weights_after = get_lora_weights(model)

# 计算并绘制差异热力图
for name in lora_weights_before:
    before = lora_weights_before[name]
    after = lora_weights_after[name]
    delta = after - before

    # 限制维度为二维的权重可视化（跳过非矩阵权重）
    if delta.ndim == 2:
        plt.figure(figsize=(6, 4))
        sns.heatmap(delta.numpy(), cmap="coolwarm", center=0)
        plt.title(f"ΔWeights Heatmap: {name}")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 0")
        plt.tight_layout()
        plt.show()
