# Bid Generation System

自动化工程标书生成系统（IT/弱电/智能化工程领域）

## 目录结构

```
doc_maker/
├── src/
│   ├── parsers/         # 文档解析模块
│   ├── agents/          # LLM Agent 模块
│   ├── compliance/      # 合规检查模块
│   ├── duplicate/       # 查重模块
│   ├── pricing/         # 单价数据库模块
│   └── utils/           # 工具函数
├── data/
│   ├── rules/           # 合规规则库
│   ├── history/         # 历史标书
│   ├── bids/            # 招标文件
│   ├── prices/          # 单价数据库
│   └── logs/            # 审计日志
├── tests/               # 测试脚本
├── bid_agent.py         # 主程序入口
└── requirements.txt    # 依赖包
```

## 安装

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 文件，填入你的 OpenAI API Key
```

## 使用

```bash
python bid_agent.py --input 招标文件.docx --history 历史标书.docx
```

## 功能模块

- Word/Excel 文档解析
- LLM 多阶段生成
- 单价数据库查询
- 查重系统
- 合规规则检查
- 审计日志
