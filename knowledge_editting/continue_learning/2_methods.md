
----------------------------------

一、Replay-based（回放/排练）方法
----------------------------------

1. 基本思想  
   用一个小规模「记忆缓存」保存旧任务的样本（或特征/梯度/生成样本），在训练新任务时把缓存数据混入当前 batch，以“复习”方式抑制灾难性遗忘。

2. 典型算法  
   • Experience Replay / Naïve Rehearsal：随机或按类别选旧样本重训。  
   • GEM / A-GEM：用缓存样本构造梯度约束，GEM 需解二次规划，A-GEM 用正交投影近似。  
   • iCaRL / TEM：结合 herding 或 k-means 选代表性样本，用知识蒸馏对齐旧类原型。  
   • Generative Replay：用 VAE/GAN 生成伪样本，无需存储原数据；代表工作 DGR、CL-GAN。  
   • RationaleCL、SLM：在 LLM 中把旧样本压缩成「理由」或检索向量，降低存储量。

3. LLM 场景落地要点  
   • 缓存内容：原始 token 太长 → 存中间特征（embedding）、或指令-回答对、或检索向量。  
   • 缓存大小：通常 <1% 原始数据，可用水库抽样、Ring Buffer、梯度相似度筛选。  
   • 训练策略：混合 batch 比例（旧:新 ≈ 1:1~1:4）+ 知识蒸馏（旧模型当 teacher）。

4. 开源实现  
   • Huggingface `trl` 的 `ReplayTrainer`（简版 rehearsal）  
   • Continual-T0、LAMOL（Generative Replay 代码）

----------------------------------

二、Regularization-based（正则化）方法
----------------------------------

1. 基本思想  
   在损失函数里增加额外项，限制“对旧任务重要参数”的大幅变动；或蒸馏旧模型的输出/中间表示以保持行为一致。

2. 典型算法  
   • EWC / SI / MAS：估计 Fisher/敏感度矩阵 Σ，对参数位移加二次惩罚  
     L_total = L_new + λ(θ−θ*)ᵀΣ(θ−θ*)。  
   • L2 / LWF / LWM：直接对输出 logits 做知识蒸馏，把旧模型当 teacher。  
   • 函数正则：对 Transformer 的 attention map、hidden states 做蒸馏（Pelosin 等）。  
   • 任务自适应校准（Task-Adaptive Calibration）：仅训练轻量校准模块，正则化旧任务输出。

3. LLM 场景落地要点  
   • Σ 估计开销大 → 常用近似：仅对最后 k 层或 LoRA 子矩阵估计。  
   • 输出蒸馏在长序列上显存高 → 改用「摘要 token」或「最后一层 hidden」蒸馏。  
   • 与 LoRA/Adapter 结合：正则化 adapter 权重即可，无需动原模型。

4. 开源实现  
   • `transformers` + `peft`：在 LoRA 训练脚本里加自定义正则项即可复现 EWC/蒸馏。  
   • TRL 库的 `KDTrainer` 支持 logit 蒸馏。

----------------------------------

三、Dynamic Architecture（动态结构）方法
----------------------------------

1. 基本思想  
   通过「加参数」而不是「改参数」把新知识隔离出来，从根本上避免覆盖旧权重；参数可按任务/专家/模块动态激活。

2. 典型算法  
   • Progressive Networks (PNN)：每任务新增一列子网络，横向扩展。  
   • Adapter / LoRA-MoE / Expert Expansion：冻结原模型，为每个任务或专家插拔小模块；推理时按 task-id 或路由函数选择专家。  
   • Lifelong Vision Transformer：在注意力层加跨任务注意力头，整合旧任务信息。  
   • CALM / DyLoRA：动态调整 LoRA 秩并做剪枝，实现参数高效扩张。  
   • 超网络（Hypernetwork）：用一个网络生成任务特定权重，避免显式存储多份参数。

3. LLM 场景落地要点  
   • 参数量控制：对 7B 模型，每层插 0.1% 参数的 Adapter，100 个任务也仅增加 ~7M 参数。  
   • 推理延迟：可用 task-id 路由，无额外计算；若用 MoE 路由需做专家并行。  
   • 与缓存回放结合：动态结构负责「容量」，回放负责「细粒度修正」。

4. 开源实现  
   • PEFT 已内置 `AdaLoRA`, `LoRA-MoE` 实验分支  
   • `transformers-adapters` 库支持 Progressive Adapters 和任务路由

----------------------------------

快速选型建议
----------------------------------

| 资源约束 | 推荐组合 |
|---|---|
| 仅 API 可用 | 正则化蒸馏（黑箱 logit KD） |
| GPU 有限、任务多 | LoRA-MoE（动态结构）+ 轻量 rehearsal |
| 可全参微调 | Progressive Adapters + 经验回放 |
| 数据极度隐私 | Generative Replay（本地 GAN 生成伪数据） |

一句话总结  
Replay 靠“复习”旧数据，Regularization 靠“限制”参数漂移，Dynamic Architecture 靠“加空间”隔离知识；LLM 场景下最实用的是「低秩 Adapter + 小缓存回放 + 蒸馏正则」的三合一方案。
