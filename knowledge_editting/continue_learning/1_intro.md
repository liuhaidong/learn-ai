
----------------------------------

一、按技术路线梳理
----------------------------------

1. 结构扩展（Architecture Expansion）
   • Progressive Neural Networks (PNN) – 横向扩展子网络，零遗忘  
   • LoRA/AdaLoRA/ MoE-LoRA – 为每个任务新增低秩适配器或专家，推理时按任务 ID 或聚类结果路由  
   • CL-MoE – 双动量混合专家框架，用任务级/实例级专家缓解多模态 LLM 遗忘

2. 参数隔离（Parameter Isolation）
   • Adapter Tuning – 冻结原模型，只训任务特定 Adapter（如 LLaVA-Adapter）  
   • Prefix / Prompt Tuning – Progressive Prompts 每任务学习少量 soft prompt tokens，计算量小

3. 回放/排练（Replay / Rehearsal）
   • Experience Replay / Continual-T0 – 维护一个记忆缓存，混合历史样本重训  
   • RationaleCL – 用对比式“理由回放”降低存储量  
   • SLM – 把向量检索引入语言模型，实现可扩展知识库式回放

4. 正则化（Regularization）
   • EWC / L2 惩罚 – 限制重要参数漂移（传统方法，在 LLM 上效果有限）  
   • TIR（Task-similarity-Informed Regularization）– 根据任务相似度自适应正则项，专为 MLLM 设计

5. 因果/鲁棒学习
   • 因果导向 CL – 用因果推断降低伪相关依赖，缓解数据稀缺场景下的遗忘

6. 黑箱提示学习（API 场景）
   • CLOB 范式 – 只通过 prompt 做增量学习，不更新参数  
   • CIS – 对历史样本做「增量摘要」放入 prompt，解决长度限制

----------------------------------

二、按训练阶段梳理
----------------------------------

1. Continual Pre-training (CPT)  
   目标：在新领域语料上继续自监督预训练，同时不丢通用能力。  
   典型工作：TAPT（任务相关数据自动检索 + 重训）

2. Continual Instruction Tuning (CIT)  
   目标：持续学习多轮指令数据，保持遵循指令的能力。  
   典型工作：ConTinTin 的 InstructionSpeak，通过“负样本再学习 + 指令回顾”提升前后向迁移

3. Continual Alignment (CA)  
   目标：随人类价值观演化持续做 RLHF/RLAIF，不遗忘上一轮对齐结果。  
   典型工作：DynaInst 将动态指令回放与局部极小正则结合，减少计算量

----------------------------------

三、快速选型建议
----------------------------------

• **GPU 资源充足、可全参训练** → 结构扩展 + 回放：PNN / MoE-LoRA + 小缓存 replay  
• **只支持 API 调用** → 黑箱提示：CIS/CLOB，无需梯度  
• **中等资源、需快速适配多任务** → 参数隔离：LoRA / Progressive Prompts  
• **数据极少且易分布漂移** → 因果导向 CL 或 TIR 正则化

----------------------------------

四、常用开源工具/代码片段
----------------------------------

• PEFT（HuggingFace）已内置 LoRA、AdaLoRA、Prefix Tuning  
• LLaMA-Adapter、LLaVA-Adapter 提供任务增量脚本  
• Continual-T0、RationaleCL 代码公开在 GitHub（搜索即得）

----------------------------------
一句话总结  
“大模型持续学习 = 任务/阶段 × 技术路线矩阵”。先确定你处于 CPT / CIT / CA 哪一阶段，再按资源约束从“结构扩展—参数隔离—回放—正则—黑箱提示”五类技术中选一到两种组合，即可快速落地。
