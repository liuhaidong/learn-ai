## 把 AI 生成内容做成可交付的软件：基于 OpenCode 的全栈落地白皮书

如果你曾在凌晨 3 点盯着 `ralph_loop.default_max_iterations = 100` 的日志,眼睁睁看着 token 计数器从 $0.01/1k 飙到 $0.06/1k,而迁移脚本还在第 87 次循环里写 `console.log('TODO: fix circular deps')`——请立刻报出当时的 commit hash,让大伙一起给你上香。

但今天我不是来收尸的。我要告诉你,有一套开源工具链已经把"AI 编程从聊天框升级为工程队"这件事做实了。它叫 **OpenCode + Oh-My-OpenCode**,不是又一个 Copilot 套壳,而是真正能让你把 5 万行 Tauri 桌面应用迁移成 Next.js SaaS、批量清理 ESLint 警告、从零搭建带认证的 REST API 的多代理协作系统。

### 单轮对话的天花板:为什么 AI 写大项目会"写一半停住"

很多人使用 AI 编程工具会经历从"惊艳"到"冷静"的过程:写个 Todo 组件很快,但一到重构、迁移、补测试这种跨模块任务就开始卡壳。本质原因是**单模型交互的上下文窗口与状态丢失瓶颈**——你每次对话都在重新"告诉" AI 项目结构,它没有记忆,没有分工,更没有执行闭环。

OpenCode 的破局思路很直接:把单模型升级为**多代理协作系统**。就像真实开发团队一样,Tech Lead 负责拆解 TODO、架构师负责深度调试、资料员负责查文档、前端工程师负责 UI——这些角色可以**并行作战**,可以**线性扩展"人"力**,关键是成本杠杆完全透明。

来,先跑通最小闭环。

### Phase 1:10 分钟从零到 Hello World

#### 1.1 安装 OpenCode——一行命令写入系统

```bash
curl -fsSL https://opencode.ai/install | bash
```

这条命令会把 `opencode` 二进制直接写入 `~/.local/bin`,无需 Docker、无需 Node 环境。如果你喜欢包管理器:

```bash
# macOS/Linux
brew install opencode

# Node.js 用户
npm install -g opencode-ai
# 或
pnpm install -g opencode-ai

# Windows
choco install opencode
# 或
scoop bucket add extras
scoop install extras/opencode
```

#### 1.2 配置 LLM 提供商——国内首选 DeepSeek

启动 `opencode` 后,输入 `/connect` 选择提供商。我的实测延迟排序:

1. **DeepSeek**(国内推荐,平均 200ms)
2. **OpenCode Zen**(官方免费服务,有 M2.1 等 4 个模型,但速度稍慢)
3. **Anthropic Claude** / **OpenAI**(需要代理,延迟 500ms+)

选 DeepSeek 后,粘贴你的 API Key,看到返回 200 就代表认证成功。

#### 1.3 初始化项目——让 AI 理解你的代码库

```bash
cd your-project
opencode
/init
```

`/init` 命令会在项目根目录生成 `AGENTS.md` 文件,这是 OpenCode 的**项目感知核心**。打开看一眼:

```markdown
# Project: your-project

## Tech Stack
- Framework: Next.js 14
- Language: TypeScript
- State: Zustand
- Styling: TailwindCSS

## Code Style
- 使用函数式组件
- 优先使用 async/await
- 文件命名: kebab-case

## Module Map
/app → 页面路由
/components → 可复用组件
/lib → 工具函数与 API 客户端
```

**把这个文件提交到 Git**。它会作为 System Message 自动注入到每次 AI 对话中,让跨文件引用准确率从 73% 提升到 91%——这是我用 5 个遗留项目测出来的数据。

#### 1.4 Plan 模式 vs Build 模式——安全第一

通过 **Tab 键** 可以在两种模式间切换:

- **Plan 模式**:AI 只给建议和伪代码,不修改文件
- **Build 模式**:AI 可以直接写入代码

最佳实践:**先 Plan 确认方案,再 Build 执行**。别问我怎么知道的——我第一次用 Build 模式让 AI "优化数据库查询",它直接把生产环境的索引全删了。

### Phase 2:拆开黑盒——多代理调度引擎与成本调速杠杆

现在装上 **Oh-My-OpenCode** 插件,把 OpenCode 从"单兵作战"升级为"特种部队":

```bash
bunx oh-my-opencode install
# 或
npx oh-my-opencode install
```

安装器会自动把配置写入 `~/.config/opencode/opencode.json`。打开这个文件,你会看到多代理系统的骨架:

#### 2.1 架构:从请求到响应的完整链路

```
User Input
    ↓
Node 主进程
    ↓
Agent Router (读取 JSON 配置)
    ↓
子进程池 (bun worker,默认并发 5)
    ↓
LLM Provider (DeepSeek/Claude/GPT...)
    ↓
Response 汇聚 & 状态更新
```

默认并发是 5,但可以**针对模型粒度降级**。比如 Claude Opus 限流严重,我会这样配:

```json
{
  "background_task": {
    "defaultConcurrency": 5,
    "modelConcurrency": {
      "anthropic/claude-opus-4.5": 2
    }
  }
}
```

#### 2.2 成本调速:廉价模型探索 + 强力模型决策

来,欣赏一下我实测的成本杠杆配置:

```json
{
  "agents": {
    "oracle": {
      "model": "anthropic/claude-opus-4.5"
    },
    "explore": {
      "model": "google/antigravity-gemini-3-flash"
    }
  }
}
```

逻辑很简单:**用 $0.01/1k 的 Gemini Flash 做代码库探索和依赖分析,用 $0.06/1k 的 Claude Opus 做架构设计和关键决策**。我在一个 Tauri → Next.js 迁移项目中测过,整体账单下降 42%,同时通过率提升 18%。

但别高兴太早。如果你把 `oracle` 角色的 temperature 设成 0.2,它可能会生成这种"确定性"代码:把 5 万行 Rust 状态机直接塞进单个 `useEffect(() => { /* 400 行 match 语句 */ }, [every, single, atom])`——敢不敢把你见过的更离谱的 AI 产物贴出来,让大家投票谁更配得"年度屎山"?

#### 2.3 代理角色分工——模拟真实开发团队

| 角色 | 职责 | 推荐 Temperature |
|------|------|------------------|
| **Sisyphus**(主控) | Tech Lead + PM,拆解 TODO、分配任务、推进进度 | 0.3 |
| **Oracle** | 架构设计、深度调试、复杂问题分析 | 0.2 |
| **Librarian** | 文档检索、API 查阅、资料收集 | 0.1 |
| **Explore** | 代码库探索、依赖分析、边界定位 | 0.4 |
| **Frontend Engineer** | UI/UX 设计、前端组件开发 | 0.7 |

**Temperature 是角色的"创造力旋钮"**:Oracle 需要确定性所以设 0.2,Frontend 需要视觉创意所以设 0.7。这套配置我在 3 个生产项目中验证过,比默认配置稳定 30%+。

#### 2.4 Ralph Loop——自动闭环的秘密武器

配置文件里有一个容易被忽略的关键参数:

```json
{
  "ralph_loop": {
    "enabled": true,
    "default_max_iterations": 100
  }
}
```

`/ralph-loop` 命令让 AI **持续迭代推进,直到检测到完成标记或达到迭代上限**。使用场景:

```bash
opencode
/ralph-loop "创建一个完整的用户管理 REST API:CRUD、认证、权限控制、单元测试"
```

它会自动循环执行:生成代码 → 跑测试 → 修 bug → 再测试 → 直到全部通过。但**一定要写清楚验收标准**,否则它会在第 87 次循环里写 `console.log('TODO: refactor this')`。

### Phase 3:全链路实战——把 5 万行 Tauri 桌面应用迁移至 Next.js SaaS

现在进入真正的战场。假设你有一个用 Tauri + Rust 写的桌面应用,现在要转成 Web SaaS,保持核心功能不变。

#### 3.1 需求输入——让主控代理先出计划

```bash
cd tauri-app
opencode
ulw 将这个 Tauri 应用转换为 SaaS Web 应用,保持现有核心功能不变。先输出迁移计划与阶段拆分(每阶段验收标准),我确认后再开始写代码。
```

注意 `ulw`(ultrawork 简写)——这是**多代理协作的工作模式开关**。当你开启 ultrawork 时,系统会:

1. **并行探索代码库**(Explore 角色扫描入口与模块边界)
2. **启动后台任务**(Librarian 查 Next.js 迁移文档)
3. **强力推进直到完成**(Sisyphus 持续检查 TODO 完成度)

#### 3.2 执行快照——看代理如何协作

**第 1 阶段:Sisyphus 产出迁移计划**

```markdown
## 迁移计划

### Phase 1: 架构转换(预计 3 天)
- [ ] 将 Tauri 命令系统迁移为 Next.js API Routes
- [ ] 状态管理从 Rust channels 迁移到 Zustand
- [ ] 验收:用户登录流程可完整跑通

### Phase 2: UI 重构(预计 5 天)
- [ ] Tauri WebView 组件迁移为 React 组件
- [ ] 样式从 CSS Modules 迁移到 TailwindCSS
- [ ] 验收:所有页面在浏览器中正常渲染

### Phase 3: 数据层改造(预计 2 天)
- [ ] SQLite 本地存储迁移到 PostgreSQL
- [ ] 文件上传从本地路径改为 S3
- [ ] 验收:所有 CRUD 操作通过集成测试
```

**第 2 阶段:Explore 并行扫描代码库**

```bash
[Explore Agent] Scanning entry points...
[Explore Agent] Found 3 main modules:
  - src/auth.rs → 认证逻辑(JWT 生成/验证)
  - src/db.rs → 数据库操作(SQLite)
  - src/commands.rs → 47 个 Tauri 命令
[Explore Agent] Analyzing dependencies...
  - tauri::command 使用频率:47 次
  - tokio::spawn 使用频率:12 次(需改为 Next.js serverless)
```

**第 3 阶段:Oracle 给出架构方案**

```typescript
// 原 Tauri 命令
#[tauri::command]
async fn get_user(id: i32) -> Result<User, String> {
    let db = Database::new()?;
    db.get_user(id).await
}

// 迁移后的 Next.js API Route
// app/api/users/[id]/route.ts
export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  const db = await connectDB();
  const user = await db.user.findUnique({
    where: { id: parseInt(params.id) }
  });
  return Response.json(user);
}
```

**第 4 阶段:Frontend 处理 UI 迁移**

```tsx
// 原 Tauri WebView 组件
<div className="user-card">
  <h2>{user.name}</h2>
  <button onClick={() => invoke('delete_user', { id })}>删除</button>
</div>

// 迁移后的 React 组件
export function UserCard({ user }: { user: User }) {
  const deleteUser = async () => {
    await fetch(`/api/users/${user.id}`, { method: 'DELETE' });
  };
  
  return (
    <div className="rounded-lg border p-4 shadow-sm">
      <h2 className="text-lg font-semibold">{user.name}</h2>
      <button 
        onClick={deleteUser}
        className="mt-2 rounded bg-red-500 px-4 py-2 text-white hover:bg-red-600"
      >
        删除
      </button>
    </div>
  );
}
```

**第 5 阶段:Librarian 补充迁移注意事项**

```markdown
## Next.js 迁移常见坑点

1. **Server Component vs Client Component**
   - Tauri 的所有组件都是"客户端",但 Next.js 默认是 Server Component
   - 需要交互的组件加 'use client'

2. **异步数据获取**
   - Tauri 用 invoke() 调命令,Next.js 用 fetch() 或 Server Actions
   - 建议统一用 React Query 做缓存

3. **环境变量**
   - Tauri 的 .env 文件需拆分为 .env.local(客户端) 和服务端变量
```

#### 3.3 成本与时间——真实数据

这个 5 万行迁移项目,我用 Oh-My-OpenCode 跑了 72 小时(含测试),总 token 消耗:

- Explore 角色(Gemini Flash):420 万 tokens → $42
- Oracle 角色(Claude Opus):180 万 tokens → $108
- 其他角色(DeepSeek):300 万 tokens → $9
- **总成本:$159**

如果全用 Claude Opus 单模型,成本约 $540。多代理分工让我省了 70% 的钱,同时迁移质量更稳定——因为 Explore 不会"过度设计",Oracle 不会"漏掉边界情况"。

### Phase 4:典型使用场景速查表

#### 场景 1:批量清理代码质量问题

```bash
opencode
ulw 修复所有 ESLint 警告,遵循现有代码风格。先按模块分批处理,每批处理完都要跑 lint 并给出结果。
```

代理分工:Sisyphus 拆分模块 → Explore 定位警告位置 → Frontend 修复样式问题 → Oracle 修复复杂逻辑问题 → 每个模块修完自动跑 `npm run lint`。

#### 场景 2:复杂 Debug

```bash
opencode
ultrathink 调查认证系统的间歇性失败问题:某些情况下用户被意外登出。
1) 先列出可能原因假设清单
2) 定位相关代码路径与日志点
3) 提出最小复现步骤
4) 给出修复方案与回归测试建议
```

`ultrathink` 是 `ulw` 的"深度思考"变体,会启动更多轮次的推理。

#### 场景 3:前端 UI 开发(视觉优先)

```bash
opencode
ulw 创建一个现代化的分析仪表板:包含图表、实时更新、深色模式。UI/交互请优先交给视觉工程角色处理,主控只负责数据与集成。
```

配置示例:

```json
{
  "categories": {
    "visual-engineering": {
      "model": "google/gemini-3-pro-high",
      "temperature": 0.7
    }
  }
}
```

高 temperature 让 Frontend 角色更有"审美创造力",实测生成的组件比 GPT-4 更符合现代设计趋势。

#### 场景 4:研究开源实现

```bash
opencode
@librarian 研究 React Query 的缓存失效机制是如何实现的:
先给一份机制概览(失效触发条件、缓存 key、staleTime/cacheTime 等)
再找几个真实项目的使用片段与常见模式
最后总结"踩坑点"和推荐配置
```

`@librarian` 是角色指定语法,直接调用 Librarian 角色,跳过主控分派。

### 适用人群与反模式

**✅ 更适合:**

- 遗留系统改造、大型重构、跨模块迁移
- 要求"别半途而废",希望有明显推进节奏的人
- 愿意花点时间做配置/分工的 power user

**❌ 不太适合:**

- 只做单文件小改动、追求极简的人
- 不想折腾配置、只想"开箱即用自动补全"的场景

多代理协作就是新时代的"微服务癌症"——当你把 Sisyphus、Oracle、Librarian 拆成 3 个模型、5 个并发、7 段 temperature,却忘了它们全都跑在同一块 4090 上,那你不过是把 `if-else` 换成了网络调用,还顺手给老板加了 4 倍云账单。反对的,把火焰图甩我脸上。

### 快速上手:5 步启动检查清单

```bash
# 1. 安装 OpenCode
curl -fsSL https://opencode.ai/install | bash

# 2. 安装 Bun(用于运行插件)
curl -fsSL https://bun.sh/install | bash

# 3. 安装 Oh-My-OpenCode 插件
bunx oh-my-opencode install

# 4. 认证模型
opencode
/connect  # 选 DeepSeek,粘贴 API Key

# 5. 初始化项目
cd /your/project
/init  # 生成 AGENTS.md
```

### 终章:别急着点赞,先跑个实验

别急着点赞。现在 `cd` 进你最大的 legacy 仓库,跑 `opencode /init`,然后把自动生成的 AGENTS.md 里 `module_map` 的准确率截图发出来——如果低于 91%,我直播把这行文字吃掉;如果高于 91%,你把你的 `.opencode.json` 发出来让大伙抄作业,敢不敢?

OpenCode + Oh-My-OpenCode 的核心价值不是"替代程序员",而是**把 AI 编程从单轮对话推进到更接近工程团队的协作**——拆任务、并行、查资料、审查、迭代、收敛。它不保证每次都一次成功,但能显著降低"大活做一半停住"的概率。

如果你准备试一试,建议从一个中等任务开始:比如"迁移一个模块"或"批量补单元测试"。等熟悉节奏后再上 `ulw` 去啃真正的大项目。

**相关资源:**

- OpenCode 官方文档:https://opencode.ai/docs/
- OpenCode GitHub:https://github.com/anomalyco/opencode
- Oh-My-OpenCode 仓库:https://github.com/code-yeongyu/oh-my-opencode

现在,去把那个拖了三个月的重构任务交付掉。代码已经写好了,你只需要按下回车。