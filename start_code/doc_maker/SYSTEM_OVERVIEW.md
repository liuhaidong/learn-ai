# 自动化工程标书生成系统 - 完整实现

## 系统概述

完整的自动化工程标书生成系统，支持 IT/弱电/智能化工程领域的标书自动生成。

## 已实现功能

### ✅ 核心模块

1. **文档解析模块** (`src/parsers/`)
   - Word 文档解析（段落、表格）
   - Excel 文档解析

2. **单价数据库** (`src/pricing/`)
   - SQLite 数据库存储
   - 增删查改操作
   - 支持多种来源（供应商、市场价）

3. **查重模块** (`src/duplicate/`)
   - Sentence Transformers Embedding
   - 余弦相似度计算
   - 高相似段落来源标注

4. **合规规则引擎** (`src/compliance/`)
   - JSON 规则配置
   - 章节级合规检查
   - 缺项自动补充

5. **LLM Agent 核心** (`src/agents/`)
   - 分层 Prompt（Plan → Draft → Refine → Finalize）
   - 对比式重写（防幻觉）
   - OpenAI GPT-4o-mini 集成

6. **多代理协作** (`src/agents/`)
   - 商务 Agent（CommercialAgent）
   - 技术 Agent（TechnicalAgent）
   - 造价 Agent（CostAgent）
   - 合规 Agent（ComplianceAgent）

7. **审计日志** (`src/utils/`)
   - JSONL 格式日志
   - 操作记录追踪
   - 时间戳标记

8. **主 Agent Pipeline** (`src/pipeline.py`)
   - 端到端生成流程
   - 章节自动化处理
   - Word 文档导出

### ✅ 工具脚本

- `bid_agent.py` - CLI 主入口
- `init_data.py` - 初始化示例数据
- `tests/test_modules.py` - 模块测试
- `install.sh` / `install.bat` - 自动安装脚本

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
```

### 3. 初始化数据

```bash
python init_data.py
```

### 4. 运行测试

```bash
python tests/test_modules.py
```

### 5. 生成标书

```bash
python bid_agent.py --input 招标文件.docx --history 历史标书.docx
```

## 项目结构

```
doc_maker/
├── src/
│   ├── parsers/              # 文档解析
│   ├── agents/               # LLM Agent
│   ├── compliance/          # 合规检查
│   ├── duplicate/            # 查重
│   ├── pricing/              # 单价数据库
│   └── utils/                # 工具函数
├── data/
│   ├── rules/                # 合规规则
│   ├── history/              # 历史标书
│   ├── bids/                 # 招标文件
│   ├── prices/               # 单价数据库
│   └── logs/                 # 审计日志
├── tests/                    # 测试脚本
├── output/                   # 输出文件
├── bid_agent.py             # CLI 入口
├── init_data.py             # 数据初始化
└── requirements.txt         # 依赖包
```

## 使用示例

### 基本使用

```bash
python bid_agent.py --input data/bids/招标文件.docx
```

### 使用历史标书

```bash
python bid_agent.py \
  --input data/bids/招标文件.docx \
  --history data/history/历史标书1.docx \
  --history data/history/历史标书2.docx
```

### 自定义输出

```bash
python bid_agent.py \
  --input 招标文件.docx \
  --output 输出路径/标书.docx
```

## 系统架构

```
Input Layer
    ├── 招标文件（Word）
    ├── 历史标书（Word）
    ├── Excel 报价表
    └── 企业单价库

Preprocessing Layer
    ├── Word 表格解析
    ├── Excel 表格解析
    └── 文本提取

Knowledge Base
    ├── 行业标准规范
    ├── 历史标书案例
    ├── 单价数据库
    └── 合规规则库

LLM Agent Core
    ├── Plan（规划）
    ├── Draft（草稿）
    ├── Refine（精炼）
    ├── Finalize（定稿）
    └── 对比式重写

Multi-Agent System
    ├── 商务 Agent
    ├── 技术 Agent
    ├── 造价 Agent
    └── 合规 Agent

Quality Control
    ├── 合规规则检查
    ├── 查重系统
    └── 审计日志

Output Layer
    ├── Word 文档生成
    ├── BoQ 估算表
    └── 质量报告
```

## 技术栈

- **语言**: Python 3.8+
- **LLM**: OpenAI GPT-4o-mini
- **Embedding**: Sentence Transformers (all-mpnet-base-v2)
- **文档处理**: python-docx, openpyxl
- **数据库**: SQLite
- **CLI**: Click
- **日志**: loguru

## 扩展功能

### 添加新的合规规则

编辑 `data/rules/compliance_rules.json`

### 添加单价数据

```python
from src.pricing import PriceDatabase
db = PriceDatabase()
db.add_price("材料名", "规格", "单位", 价格, "来源")
```

### 自定义章节

在 `src/pipeline.py` 中修改 `chapters` 列表

## 注意事项

1. 首次运行会自动下载 sentence-transformers 模型
2. 需要配置 OpenAI API Key
3. 建议使用虚拟环境
4. 不要将 .env 文件提交到版本控制

## 故障排除

### API 调用失败
检查 `.env` 文件中的 `OPENAI_API_KEY`

### 模型下载慢
使用国内镜像或预下载模型

### Word 解析失败
确保使用 `.docx` 格式

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request
