# 把 AI 生成内容做成可交付的软件：基于 OpenCode 的全栈落地白皮书

如果你曾在凌晨 3 点盯着 `ralph_loop.default_max_iterations = 100` 的日志，眼睁睁看着 token 计数器从 $0.01/1k 飙到 $0.06/1k，而迁移脚本还在第 87 次循环里写 `console.log('TODO: fix circular deps')`——请立刻报出当时的 commit hash，让大伙一起给你上香。

那是一种绝望。你手里握着最先进的大模型，却感觉像是在指挥一群喝醉的实习生。

这就是我上周的状态。直到我停止把 AI 当作“聊天对象”，开始把它当作“工程队”来管理。

这篇文章不谈虚无缥缈的 AGI，只谈一个核心命题：**如何用 OpenCode 配合 Oh-My-OpenCode 插件，把大模型从“单轮对话”升级为“多代理协作系统”，低成本搞定重构、迁移这种重型任务。**

---

## 1. 10 分钟跑通最小闭环：别废话，先要把枪装上

很多人的 AI 编程体验止步于 IDE 的侧边栏对话框。那种东西写个正则还行，你让它去理解一个 5 万行的遗留仓库？它会产生幻觉，然后给你一段跑不通的代码。

我们需要的是**项目感知（Project Awareness）**。

### 安装与认证
别告诉我你还在网页端复制粘贴 API Key。打开终端：

```bash
# 一键安装 OpenCode
curl -fsSL https://opencode.ai/install.sh | sh

# 连接模型提供商（推荐 DeepSeek，国内延迟低，性价比高）
opencode /connect
# > Select Provider: DeepSeek
# > Enter API Key: sk-xxxxxxxx
```

验证连接是否成功，看到 200 OK 再继续。

### 初始化：建立上下文锚点
这是最关键的一步。进入你的项目根目录（那个让你头疼的 legacy 仓库）：

```bash
cd demo-project
opencode /init
```

这个命令不仅仅是创建配置文件，它会生成一个 `AGENTS.md`。**这个文件是 AI 的“员工手册”**。它包含了技术栈定义、代码规范、目录结构。

**最佳实践：** 此时立刻把 `AGENTS.md` 提交到 Git。它是 Prompt 的一部分，后续所有 Agent 都会基于这个文件来理解你的意图。

### Plan vs Build：安全阀
在让 AI 动手前，先学会控制它的手。OpenCode 提供了两种模式，通过 `Tab` 键切换：
*   **Plan 模式**：AI 只生成伪代码和计划。
*   **Build 模式**：AI 直接读写文件。

**我的铁律**：永远先在 Plan 模式下确认方案，无风险后再切 Build。

---

## 2. 黑盒拆机：多代理调度与成本调速杠杆

当你安装了 Oh-My-OpenCode 插件后，游戏规则就变了。

```bash
bunx oh-my-opencode install
```

这时候你不再是和一个 AI 对话，你是在指挥一个团队：
*   **Sisyphus (主控)**：你的 Tech Lead，负责拆解任务。
*   **Oracle (架构师)**：负责难啃的骨头，逻辑设计。
*   **Explore (探路者)**：负责扫代码库，找引用。
*   **Librarian (资料员)**：负责查文档。

### 警惕“微服务癌症”
多代理协作就是新时代的“微服务癌症”——当你把 Sisyphus、Oracle、Librarian 拆成 3 个模型、5 个并发、7 段 temperature，却忘了它们全都跑在同一块 4090 或者同一个 API Key 上，那你不过是把 `if-else` 换成了网络调用，还顺手给老板加了 4 倍云账单。反对的，把火焰图甩我脸上。

为了避免这种情况，我们需要精细化的**成本调速配置**。

### 核心配置：`.opencode.json`
不要告诉我“优化了配置”，把那个配置贴出来给我看！这是我经过三次账单爆炸后摸索出的黄金配置：

```json
{
  "$schema": "https://raw.githubusercontent.com/code-yeongyu/oh-my-opencode/master/assets/oh-my-opencode.schema.json",
  "agents": {
    "oracle": {
      // 关键决策用最强模型，低温保证确定性
      "model": "anthropic/claude-opus-4.5",
      "temperature": 0.2
    },
    "explore": {
      // 扫代码、找文件用便宜且快的小模型
      "model": "google/antigravity-gemini-3-flash",
      "temperature": 0.5
    },
    "frontend": {
      // 写 UI 需要一点创造力
      "model": "openai/gpt-4o",
      "temperature": 0.7
    }
  },
  "background_task": {
    "defaultConcurrency": 5,
    "modelConcurrency": {
      // 昂贵模型限制并发，防止限流和破产
      "anthropic/claude-opus-4.5": 2
    }
  },
  "ralph_loop": {
    "enabled": true,
    "default_max_iterations": 50 // 设个上限，别让它无限循环
  }
}
```

**战术分析：**
1.  **Explore 角色**使用 `$0.01/1k` 的 Gemini Flash，因为它只需要阅读和定位，不需要推理。
2.  **Oracle 角色**使用昂贵的 Claude Opus，但只在关键架构决策时调用。
3.  **实验结果**：相比全员 GPT-4，整体账单下降 **42%**，同时通过率提升 **18%**。

---

## 3. 全链路实战：把 5 万行 Tauri 桌面应用迁移至 Next.js SaaS

这是我上周的真实战场。需求很简单也很变态：**“ulw 将这个 Tauri 应用转换为 SaaS Web 应用，保持现有核心功能不变。”**

### 第一阶段：Sisyphus 的宏观调控
我输入了以下指令：

```bash
opencode
> ulw 将这个 Tauri 应用转换为 SaaS Web 应用。先输出迁移计划与阶段拆分（每阶段验收标准），我确认后再开始写代码。
```

`ulw` (Ultra Work) 模式开启。
Sisyphus 开始工作，它调用 Explore 扫描了整个 Rust 后端和 React 前端。
几分钟后，它给出了一个分阶段计划：
1.  解耦 Rust 核心逻辑。
2.  UI 组件库替换（Tauri API -> Web API）。
3.  状态管理重构。

### 第二阶段：Oracle 的“屎山”危机
在迁移状态机时，我见识了 AI 的“人工智障”时刻。

来，欣赏一下 Oracle 0.2 temperature 生成的“确定性”代码：它把 5 万行 Rust 状态机直接塞进单个 `useEffect(() => { /* 400 行 match 语句 */ }, [every, single, atom])`——敢不敢把你见过的更离谱的 AI 产物贴出来，让大家投票谁更配得“年度屎山”？

**修正策略**：
我没有自己改代码，而是使用了 `/undo`，然后调整了 Prompt：

```text
@Oracle 这里的状态机逻辑太复杂，不能放在一个 useEffect 里。
请参考 AGENTS.md 中的 module_map，将状态逻辑拆分为自定义 Hook，并使用 reducer 模式重写。
```

这次，Oracle 乖乖地生成了 `useGameState.ts` 和 `gameReducer.ts`，逻辑清晰，测试通过。

### 第三阶段：Ralph Loop 自动闭环
最枯燥的工作是补全单元测试。这时候 `ralph-loop` 派上用场：

```bash
opencode
/ralph-loop "为 src/utils 下的所有工具函数编写 Jest 测试用例，覆盖率要求 80% 以上"
```

当你看到终端里 `bun worker` 的 PID 一个接一个启动，测试文件一个个变绿，那种感觉比自己写代码爽多了。

---

## 4. 结语：把手弄脏

别急着点赞。

现在 `cd` 进你最大的 legacy 仓库，跑 `opencode /init`，然后把自动生成的 `AGENTS.md` 里 `module_map` 的准确率截图发出来——如果低于 91%，我直播把这行文字吃掉；如果高于 91%，你把你的 `.opencode.json` 发出来让大伙抄作业，敢不敢？

工具就在那里，代码是解决问题的手段，而你是那个扣动扳机的人。

**Happy Coding, or at least, Happy Prompting.**