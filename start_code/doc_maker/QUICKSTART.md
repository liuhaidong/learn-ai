# 快速开始指南

## 安装步骤

### 方法 1: 使用自动安装脚本

**Linux/Mac:**
```bash
bash install.sh
```

**Windows:**
```cmd
install.bat
```

### 方法 2: 手动安装

1. **安装 Python 依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   ```
   然后编辑 `.env` 文件，填入你的 OpenAI API Key。

3. **初始化示例数据**
   ```bash
   python init_data.py
   ```

## 使用方法

### 1. 准备招标文件

将招标文件（.docx 格式）放到 `data/bids/` 目录。

### 2. 准备历史标书（可选）

如果有历史标书，将它们放到 `data/history/` 目录。

### 3. 运行生成

**基本用法:**
```bash
python bid_agent.py --input data/bids/招标文件.docx
```

**使用历史标书:**
```bash
python bid_agent.py \
  --input data/bids/招标文件.docx \
  --history data/history/历史标书1.docx \
  --history data/history/历史标书2.docx
```

**自定义输出路径:**
```bash
python bid_agent.py \
  --input data/bids/招标文件.docx \
  --output my_bid.docx
```

## 输出文件

标书生成完成后，会输出以下文件：

- `output/final_bid.docx` - 最终标书文档
- `output/compliance_report.json` - 合规检查报告
- `output/dup_report.json` - 查重报告
- `data/logs/audit_YYYYMMDD.jsonl` - 审计日志

## 测试

运行测试确保所有模块正常工作：

```bash
python tests/test_modules.py
```

## 常见问题

### Q: OpenAI API 调用失败
**A:** 检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确配置。

### Q: 首次运行很慢
**A:** 首次运行会自动下载 sentence-transformers 模型（约 400MB），请耐心等待。

### Q: Word 文档解析失败
**A:** 确保使用 `.docx` 格式（不是 `.doc`），且文档格式正确。

### Q: 如何添加新的单价数据
**A:** 运行以下代码：
```python
from src.pricing import PriceDatabase
db = PriceDatabase()
db.add_price("材料名称", "规格", "单位", 价格, "来源")
```

### Q: 如何自定义合规规则
**A:** 编辑 `data/rules/compliance_rules.json` 文件，添加新的规则。

## 项目结构说明

```
data/
├── bids/           # 招标文件
├── history/        # 历史标书
├── rules/          # 合规规则
├── prices/         # 单价数据库
└── logs/           # 审计日志

output/             # 输出文件

src/
├── parsers/        # 文档解析
├── agents/         # LLM Agent
├── compliance/     # 合规检查
├── duplicate/      # 查重
├── pricing/        # 单价数据库
└── utils/          # 工具函数
```

## 下一步

- 阅读 `SYSTEM_OVERVIEW.md` 了解系统架构
- 阅读 `DEPLOYMENT.md` 了解部署指南
- 阅读 `README.md` 了解详细功能说明

## 技术支持

如有问题，请提交 Issue 或联系开发团队。
