# RAG- retrieval-augmented generation

## 2005.11401v4

Memory

* Pre-trained neural language models : pre-trained parametric memory

* access : non-parametric memory;latent documents

## 2312.10997v5

### RAG

This enhances the accuracy and credibility of the
generation, particularly for knowledge-intensive tasks, and allows
for continuous knowledge updates and integration of domain specific information.

### Three paradigms

#### Naive RAG

mainly consists of three parts: indexing, retrieval and generation
using sparse retrieval techniques like BM25

#### Advanced RAG

dense semantic matching, re-ranking, and multi-hop query-
ing, while also introducing refined indexing strategies like
fine-grained chunking and metadata-aware retrieval.

proposes multiple optimization strategies around pre-retrieval (rewrite) and post-retrieval(rerank), with a process similar to the Naive RAG, still following a chain-like structure

#### Modular RAG

inherits and develops from the previous paradigm, showcasing greater flexibility overall. This is evident in the
introduction of multiple specific functional modules and the replacement of existing modules. The overall process is not limited to sequential retrieval and
generation; it includes methods such as iterative and adaptive retrieval.

In the CRAG workflow, a
lightweight retrieval evaluator assigning the confidence scores
about the quality of the retrieved chunks/documents — cate-
gorized as correct, incorrect, or ambiguous. When retrieval
quality is deemed suboptimal, the system activates corrective
strategies such as query rewriting or external web search to
gather better evidence. The system refines the retrieved con-
tent into a focused context and iteratively improves retrieval
until a satisfactory output is generated.

RAPTOR [Sarthi et al., 2024] introduces a recursive
tree structure from documents, allowing for more efficient and
context-aware information retrieval. This approach enhances
RAG by creating a summary tree from text chunks, providing
deeper insights and overcoming limitations of short, contigu-
ous text retrieval.

MCTS-RAG [Hu et al., 2025] integrates
a Monte Carlo Tree Search loop into the RAG process for
complex reasoning tasks. MCTS-RAG dynamically integrates
retrieval and reasoning through an iterative decision-making
process.

RAGate
[Wang et al., 2024a] uses the conversation context and model
confidence to route only those dialogue turns that truly require external knowledge to a RAG process. This ensures the system can bypass retrieval for straightforward prompts while invoking it for knowledge-intensive queries, exemplifying conditional RAG in dialogue.

### RETRIEVAL

#### Retrieval Source

* Unstructured Data, such as text, is the most widely used
retrieval source, which are mainly gathered from corpus.
* Semi-structured data. typically refers to data that contains a
combination of text and table information, such as PDF.
* Structured data, such as knowledge graphs (KGs)

In text, retrieval granularity ranges from fine to coarse,
including Token, Phrase, Sentence, Proposition, Chunks, Document.

#### Indexing Optimization

* Chunking Strategy:fixed number of tokens
* Metadata Attachments: Chunks can be enriched with metadata information such as page number, file name, author,category timestamp.

#### Query Optimization

Formulating a precise and clear question

* Query Expansion: Expanding a single query into multiple queries enriches the content of the query, providingfurther context to address any lack of specific nuances.The implementation of the query rewrite method in the Taobao, known as BEQUE [9] has notably enhanced recall effectiveness for
long-tail queries, resulting in a rise in GMV.

* Query Routing: Based on varying queries, routing to distinct RAG pipeline

#### Embedding

比较RAG中的稀疏嵌入(Sparse Embedding)和稠密嵌入(Dense Embedding)

* 基本概念对比

| 特性                | 稀疏嵌入(Sparse Embedding)       | 稠密嵌入(Dense Embedding)        |
|---------------------|----------------------------------|----------------------------------|
| **表示形式**        | 高维稀疏向量(大部分为0)          | 低维稠密向量(所有维度都有值)      |
| **典型代表**        | TF-IDF, BM25, SPLADE             | BERT, Sentence-BERT, DPR         |
| **维度**           | 通常50k-1M维                     | 通常128-1024维                   |
| **语义捕捉能力**    | 主要捕捉词汇匹配                 | 能捕捉深层语义关系               |

## 2503.10677v2

https://github.com/USTCAGI/Awesome-Papers-Retrieval-Augmented-Generation

generate more accurate and contextually relevant outputs

--

## Agentic

### TTS

**Test-Time Scaling**:  has emerged as a potent paradigm for boosting the reasoning and
agentic capabilities of LLMs (Snell et al., 2024). It assigns additional computation during inference,enabling deeper problem-solving (Zou et al., 2025a; Gu et al., 2025)

#### Agentic RAG

* Search-R1
It extends RL-based reasoning frameworks (like DeepSeek-R1) by integrating search engine interaction directly into the learning loop. In the Search-R1 framework, the search engine is modeled as part of the RL environment. The LLM agent learns a policy to generate a sequence of tokens that includes both internal reasoning steps (often enclosed in <think> tags) and explicit triggers for search actions. These triggers are special tokens, <search> and </search>, which encapsulate the generated search query.

* ReZero
aims to teach the agent the value of “trying one more time.” The framework operates within a standard RL setup (using GRPO is mentioned) where the LLM interacts with a search environment.

* DeepRetrieval
employs RL algorithms like Proximal Policy Optimization (PPO) [Schulman et al., 2017] to train this query generation process.
DeepRetrieval uses the performance of the generated query in the actual retrieval system as the
reward

* DeepResearcher
 The framework employs RL (specifically GRPO with an F1 score-based reward for answer accuracy) to train agents that interact with live web search APIs and browse actual webpages. DeepResearcher utilizes a specialized multi-agent architecture to handle the complexities of web interaction.
