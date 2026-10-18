# `.memory/` 架构：把对话和事实外置成真源（2026-10-18）

> 借鉴 `interactive-novel` 的「状态外置」，用来解决我们三个问题：
> **上下文太短 / 版本漂移 / 记忆不可读**。

## 一、目录结构

    .memory/
    ├── facts.md        # 稳定事实（唯一真源，人可读）
    ├── episodes.md     # 对话片段（唯一真源，每轮一条）
    ├── index.json      # 向量索引（派生，随时可 rebuild）
    └── meta.json       # 索引的构建信息（源文件 sha + 时间）

**关键：markdown 是真源，索引是派生。** 索引坏了不用愁，从 md 重跑即可。

## 二、facts.md 的格式（必须结构化）

    - `cascade_pass_at_1` = `0.738`  <-- measured_at: 2026-10-17, n: 164

为什么必须这样写：

| 字段 | 作用 |
|---|---|
| `key` | 唯一标识，检索和查重都靠它 |
| `value` | 实际值 |
| **`measured_at`** | **判断新旧，治版本漂移的关键** |
| `n` | 样本量，用来判断可信度 |

## 三、episodes.md 的格式（也必须有结构）

    ## 轮次 12 | 2026-10-18 16:20
    - 用户：灰度比例初始设多少
    - 回答：5%（依据：轮次 5 的记录）
    - 事实：灰度比例=5%

**散文格式不行** —— embedding 抓不到可检索的锚点。

## 四、三个工具

| 命令 | 作用 |
|---|---|
| `python memory_store.py rebuild` | 从真源重建索引 |
| `python memory_store.py search 问题` | 检索 |
| `python memory_store.py add-fact k v` | 追加一条事实 |
| **`python memory_check.py`** | **一致性体检** |
| `python memory_check.py --fix` | 自动修（只做安全的重建索引） |

## 五、体检能抓到什么（实测）

故意注入一条冲突的旧值后：

    [ERROR] index-stale      facts.md 改了但索引没重建
    [WARN ] fact-conflict    `cascade_pass_at_1` 有 2 个版本 ['0.738', '0.883']，最新是 0.738
    [WARN ] fact-no-date     `cascade_pass_at_1` 缺 measured_at

**这正是昨天真实文档只到 70% 的病根**——同一指标多个值、看不出哪个新。
现在它是一条**可检出的告警**，而不是一个静默的错误答案。

## 六、治好了什么

| 原来的问题 | 现在 |
|---|---|
| 上下文 20k 装不下长对话 | 对话外置成 episodes.md，按需检索（实测 100% @ 15% 注入量） |
| 真实文档 70%，因版本漂移 | `fact-conflict` 告警 + `measured_at` 定新旧 |
| 记忆是不可读的黑盒向量 | **全部是 markdown，能直接看、直接改、能 git diff** |
| 索引脏了没法修 | `rebuild` 从真源重跑 |

## 七、和 interactive-novel 的对应

| 它的做法 | 我们的做法 |
|---|---|
| `.novel/outline.md` 唯一真源 | `.memory/facts.md` |
| `rebuild-notes` 从真源重建笔记 | `rebuild` 从真源重建索引 |
| `check` 报 `notes-stale` | `memory_check.py` 报 `index-stale` |
| `due` 列到期伏笔 | **未实现**（可扩展：事实的时效提醒） |

## 八、诚实边界

1. **还没接进 `serve.py`** —— 目前是独立工具，服务仍在用 `qa_bench.json`。
2. **事实冲突只报不修** —— 保留哪个版本需要人决定（这正是对的，不该自动猜）。
3. **`due` 到期提醒没做**（证书有效期这类）。
4. **`episodes.md` 的写入还没自动化** —— 现在要手动加或调 `add_episode`。
5. **索引是全量重建**，规模大了需要增量（现在 23 条，无所谓）。
