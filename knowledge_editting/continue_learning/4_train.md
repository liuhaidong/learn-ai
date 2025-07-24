# 下面给出两套最小可运行的代码示例，分别演示  

1) **Adapter**（单任务独立文件，磁盘即插即用）  
2) **LoRA-MoE**（token 级路由，共享底座权重）  

场景：在 LLaMA-7B 上持续学习两个问答任务（SQuAD → Natural Questions）。  
硬件：单张 A100 40G 即可跑通；如显存不足可把 `batch_size`、`lora_r` 再调小。

-------------------------------------------------

一、Adapter（参数隔离，任务切换靠文件名）
-------------------------------------------------

1. 数据准备  

```python
# prepare_adapter_data.py
import json, random, os
from datasets import load_dataset

def dump_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# 任务1：SQuAD
squad = load_dataset("squad", split="train").select(range(10000))
task1 = [{"text": f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer: {s['answers']['text'][0]}"}
         for s in squad]
random.shuffle(task1)
dump_jsonl(task1, "data/task1_squad.jsonl")

# 任务2：Natural Questions (nq_open)
nq = load_dataset("nq_open", split="train").select(range(10000))
task2 = [{"text": f"Question: {s['question']}\nAnswer: {s['answer'][0]}"}
         for s in nq]
random.shuffle(task2)
dump_jsonl(task2, "data/task2_nq.jsonl")
```

2. 训练脚本  

```python
# train_adapter.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, TaskType, AdaLoraConfig
import json, torch, os

model_name = "decapoda-research/llama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l)["text"] for l in f]

def encode(batch):
    return tokenizer(batch, truncation=True, max_length=512)

def build_dataset(path):
    texts = read_jsonl(path)
    data = encode(texts)
    return torch.utils.data.Dataset.from_dict(data)

# 每个任务单独保存一个 adapter
for task_id, data_path in enumerate(["data/task1_squad.jsonl", "data/task2_nq.jsonl"], 1):
    peft_config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        adapter_name=f"task{task_id}"
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = build_dataset(data_path)
    args = TrainingArguments(
        output_dir=f"ckpt_adapter_task{task_id}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        fp16=True,
        save_total_limit=1,
        logging_steps=50,
        remove_unused_columns=False
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(f"ckpt_adapter_task{task_id}")
```

切换任务：  

```python
from peft import PeftModel, AutoPeftModelForCausalLM
base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "ckpt_adapter_task1")  # 或 task2
```

-------------------------------------------------

二、LoRA-MoE（token 级路由，共享底座权重）
-------------------------------------------------

1. 数据准备  
与上面相同，只是把所有任务拼成一个文件，训练时动态给 `task_id` 作为标签即可。

```python
# prepare_moe_data.py
import json, random
from datasets import load_dataset

def build_sample(text, task_id):
    return {"text": text, "task_id": task_id}

all_data = []
squad = load_dataset("squad", split="train").select(range(10000))
all_data += [build_sample(f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer: {s['answers']['text'][0]}", 0)
             for s in squad]

nq = load_dataset("nq_open", split="train").select(range(10000))
all_data += [build_sample(f"Question: {s['question']}\nAnswer: {s['answer'][0]}", 1)
             for s in nq]

random.shuffle(all_data)
with open("data/mixed_tasks.jsonl", "w") as f:
    for sample in all_data:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
```

2. 训练脚本  
LoRA-MoE 需用社区实现（目前 HuggingFace PEFT 尚未官方支持）。这里用开源库 `moetify` 为例，也可换成 `MixLoRA`、`OpenMoE` 等实现。  
安装：

```bash
pip install git+https://github.com/TUDB-Labs/moetify.git
```

```python
# train_lora_moe.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from moetify import MoELoraConfig, get_moe_peft_model
import json, torch

model_name = "decapoda-research/llama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

class MoeDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.data = [json.loads(l) for l in f]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        task_id = self.data[idx]["task_id"]
        enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        enc["task_id"] = torch.tensor(task_id, dtype=torch.long)
        return {k: v.squeeze(0) for k, v in enc.items()}

config = MoELoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    num_experts=4,      # 4 个 LoRA 专家
    top_k=2,            # 每 token 激活 2 个专家
    task_type="CAUSAL_LM"
)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model = get_moe_peft_model(model, config)
model.print_trainable_parameters()

train_dataset = MoeDataset("data/mixed_tasks.jsonl")
args = TrainingArguments(
    output_dir="ckpt_lora_moe",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    fp16=True,
    save_total_limit=1,
    logging_steps=50,
    remove_unused_columns=False
)
trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
trainer.train()
model.save_pretrained("ckpt_lora_moe")
```

3. 推理示例（token 级路由）  

```python
from transformers import AutoTokenizer
from moetify import PeftModel
base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "ckpt_lora_moe")

inputs = tokenizer("Question: who wrote Harry Potter?\nAnswer:", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

-------------------------------------------------
小结  

- **Adapter**：每个任务一个独立文件，磁盘即插即用，顺序结构，简单可控。  
- **LoRA-MoE**：所有任务共用底座，通过门控网络按 token 路由专家，参数增长慢但实现更复杂。
