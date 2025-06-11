#

## 🌟 AutoModel 系列常见类及适用场景对比

| 类名                                     | 任务类型      | 输出形式               | 典型应用场景           | 说明                 |
| -------------------------------------- | --------- | ------------------ | ---------------- | ------------------ |
| **AutoModel**                          | 基础模型      | 隐层特征（不含任务头）        | 特征抽取、微调预训练       | 通用BERT、GPT等编码器或解码器 |
| **AutoModelForSequenceClassification** | 文本分类      | logits（每类一个）       | 情感分析、新闻分类、多标签任务  | 加了一个分类头（linear）    |
| **AutoModelForTokenClassification**    | 序列标注      | 每个token的分类logits   | 命名实体识别（NER）、词性标注 | Token-level分类头     |
| **AutoModelForQuestionAnswering**      | 问答        | 起始+结束位置 logits     | 抽取式问答（如SQuAD）    | 两个线性头输出起始和结束位置     |
| **AutoModelForMaskedLM**               | 掩码语言建模    | 每个token的词预测 logits | BERT预训练、MLM微调    | 典型于 BERT、RoBERTa   |
| **AutoModelForCausalLM**               | 自回归语言建模   | 下一个token概率分布       | 文本生成（如GPT）       | GPT系列使用            |
| **AutoModelForSeq2SeqLM**              | 编码器-解码器生成 | 解码序列               | 翻译、摘要            | 典型如T5、BART等        |
| **AutoModelForMultipleChoice**         | 多选分类      | 每个选项的score         | SWAG、RACE等任务     | 输入拼接多个选项，输出每个选项得分  |
| **AutoModelForImageClassification**    | 图像分类      | logits             | 图像识别             | 用于ViT、CLIP等视觉模型    |
| **AutoModelForVision2Seq**             | 图文生成      | 文本输出               | 图像描述、VQA         | 图文多模态任务如BLIP       |
| **AutoModelForAudioClassification**    | 音频分类      | logits             | 声音事件识别           | 如 Wav2Vec2         |
| **AutoModelForCTC**                    | CTC 解码    | 时间序列转文本            | 语音识别             | 常用于Wav2Vec2等语音模型   |
| **AutoModelForSpeechSeq2Seq**          | 语音转文本     | 文本序列               | Whisper模型        | Whisper音频翻译        |

---

## 🔍 使用建议（按任务类型分类）

| 任务类型   | 推荐类                                              | 推荐模型例子            | 说明                    |
| ------ | ------------------------------------------------ | ----------------- | --------------------- |
| 文本分类   | `AutoModelForSequenceClassification`             | BERT, RoBERTa     | 输出固定标签数               |
| 命名实体识别 | `AutoModelForTokenClassification`                | BERT, DeBERTa     | token级别分类             |
| 抽取式问答  | `AutoModelForQuestionAnswering`                  | BERT, ALBERT      | 起始/结束位置               |
| 多选题    | `AutoModelForMultipleChoice`                     | RoBERTa, BERT     | 输入拼接选项                |
| 文本生成   | `AutoModelForCausalLM` / `AutoModelForSeq2SeqLM` | GPT-2, T5         | GPT为自回归，T5为seq2seq    |
| 掩码预测   | `AutoModelForMaskedLM`                           | BERT, RoBERTa     | 预训练常用                 |
| 图像分类   | `AutoModelForImageClassification`                | ViT, DeiT         | HuggingFace Vision 模型 |
| 图文生成   | `AutoModelForVision2Seq`                         | BLIP, OFA         | 图文输入输出                |
| 音频识别   | `AutoModelForCTC` / `AutoModelForSpeechSeq2Seq`  | Wav2Vec2, Whisper | 语音识别转文本               |

---

## ⚙️ 使用举例（以文本分类为例）

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

##  总结对比

| 类别                | 支持的输入类型                     | 支持的输出     | 是否含下游任务头     |
| ----------------- | --------------------------- | --------- | ------------ |
| `AutoModel`       | input\_ids, attention\_mask | 隐层特征      | ❌ 无任务头       |
| `AutoModelFor...` | 通常相同输入                      | logits/预测 | ✅ 含分类/生成/抽取头 |

---
