# 开发和部署指南

## 开发环境设置

### 1. 克隆项目

```bash
git clone <repository_url>
cd doc_maker
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 OPENAI_API_KEY
```

### 4. 初始化示例数据

```bash
python init_data.py
```

### 5. 运行测试

```bash
python tests/test_modules.py
```

## 使用方法

### 基本用法

```bash
python bid_agent.py --input 招标文件.docx
```

### 使用历史标书

```bash
python bid_agent.py --input 招标文件.docx --history 历史标书1.docx --history 历史标书2.docx
```

### 自定义输出路径

```bash
python bid_agent.py --input 招标文件.docx --output 自定义路径/output.docx
```

## 项目结构

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
├── output/              # 输出文件
├── bid_agent.py         # 主程序入口
└── requirements.txt    # 依赖包
```

## 扩展开发

### 添加新的合规规则

编辑 `data/rules/compliance_rules.json`，添加新的规则：

```json
{
  "name": "新章节名称",
  "rules": [
    {
      "id": "RULE-001",
      "must_include": ["必须包含项1", "必须包含项2"],
      "reference": "参考标准"
    }
  ]
}
```

### 添加单价数据

```python
from src.pricing import PriceDatabase

db = PriceDatabase()
db.add_price("材料名称", "规格", "单位", 价格, "来源")
```

### 自定义 Agent

在 `src/agents/` 目录下创建新的 Agent 类，继承自基础 Agent。

## 部署

### Docker 部署

```bash
# 构建镜像
docker build -t bid-agent:latest .

# 运行容器
docker run -d -v $(pwd)/data:/app/data bid-agent:latest
```

### 生产环境配置

1. 设置环境变量
2. 配置日志级别
3. 设置 API 速率限制
4. 配置文件存储路径

## 故障排除

### 问题 1: OpenAI API 调用失败

检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确配置。

### 问题 2: Word 文档解析失败

确保 Word 文档格式正确，使用 `.docx` 格式。

### 问题 3: 查重模型下载慢

首次运行会自动下载 sentence-transformers 模型，请耐心等待。

## 性能优化

1. 使用本地 Embedding 模型缓存
2. 并行处理多个章节
3. 减少合规规则检查频率
4. 优化历史标书数据量

## 安全建议

1. 不要将 `.env` 文件提交到版本控制
2. 使用环境变量存储敏感信息
3. 定期更新依赖包
4. 使用 HTTPS 调用 API
