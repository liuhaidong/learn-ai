from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 使用文泉驿微米黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载模型和分词器
model_name = "/home/liuhaidong/workspace/aha_agent/learn-ai/models/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, output_attentions=True)

inputs = tokenizer("A major power outage affects most of the Iberian Peninsula.", return_tensors="pt")
outputs = model(**inputs)
attentions = outputs.attentions

def plot_attention(attention, layer=0, head=0):
    """绘制指定层和头的注意力热力图"""
    attn = attention[layer][0, head].detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    plt.figure(figsize=(13, 8))
    ax = sns.heatmap(attn, cmap="viridis", annot=True, fmt=".2f",
                    xticklabels=tokens,
                    yticklabels=tokens)
    
    # 设置字体大小和旋转角度
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    plt.title(f"第 {layer} 层, 第 {head} 头的注意力权重")
    plt.tight_layout()  # 防止标签被截断
    plt.show()

# 示例：绘制第0层第0头的注意力
plot_attention(attentions, layer=0, head=0)

# 计算所有头在某层的平均注意力
mean_attn = torch.mean(attentions[0], dim=1)[0]

# 可视化平均注意力
plt.figure(figsize=(13, 8))
ax = sns.heatmap(mean_attn.detach().numpy(), cmap="viridis")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(tokens, rotation=0, fontsize=10)
plt.title("第 0 层所有头的平均注意力权重")
plt.tight_layout()
plt.show()

# model.config.output_attentions = True  # 确保生成时输出注意力
generated = model.generate(inputs["input_ids"], max_length=33, num_return_sequences=1)

# # 提取生成过程中的注意力
# for step, attn in enumerate(outputs.attentions):
#     print(f"Step {step}:")
#     plot_attention([attn], layer=0, head=0)  # 绘制每一步的注意力

# 对比注意力权重与生成token的关系
generated_text = tokenizer.decode(generated[0])
print("Generated:", generated_text)

for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])):
    print(f"Token '{token}' attended by:")
    for layer in range(len(attentions)):
        print(f"  Layer {layer}: max attention from {attentions[layer][0, :, -1, i].argmax().item()}")  # 查看生成最后一个token时对输入token的关注