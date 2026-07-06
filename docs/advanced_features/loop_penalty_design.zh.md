# SGLang 循环检测与抑制设计说明

## 1. 背景与目标

在推理服务中，模型有时会进入一种“自我强化的重复生成”状态：它不再推进答案、工具调用或正常结束，而是持续输出重复结构，直到耗尽 `max_tokens`。这种现象会造成大量 token 浪费，也会拖慢请求完成时间。

本文档讨论的是一个 **SGLang 推理期防护机制**：在 decode 过程中检测重复循环，并在确认循环持续存在后逐步干预，尽量让生成正常收尾。

这个机制的目标是：

- 阻止明显失控的重复循环。
- 尽量不影响正常推理、代码、表格、JSON、URL 等合法重复内容。
- 支持按请求配置，默认关闭。
- 作为推理期兜底，不要求修改模型权重或重新训练模型。

这个机制不做两件事：

- 不“修复模型本身”。它只是推理时的 guardrail。
- 不理解文本语义。它主要看重复结构，而不是判断内容含义是否合理。

## 2. 为什么要区分两类循环

生产环境里的循环大致可以分成两类。它们的重复形态不同，所以不能只用一种检测方式。

| 类型 | 重复形态 | 常见场景 | 例子 | 适合的检测方式 |
| --- | --- | --- | --- | --- |
| 刚性循环 | 同一段 token 原样重复 | 短周期重复、分隔符、URL、JSON/tool 参数、数字序列 | `==== ====`, `0,0,0,0`, 重复 URL | DRY、n-gram |
| 模板化循环 | 固定前缀重复，但后面的内容变化 | 推理标记、角色标记、代码评审、`prefix: value` 结构 | `Need maybe maybe include/use/ensure`, `Assistant: User: ...`, `Potential bug: X` | 段前缀检测、未来的 hidden-state probe |

这个区分是整个设计的核心。

DRY 和 n-gram 这类方法擅长抓“逐字重复”。例如同一段 URL 或同一串符号反复出现，它们很快就能识别出来。

但模板化循环不一定有长段逐字重复。比如：

```text
Need maybe maybe include ...
Need maybe maybe use ...
Need maybe maybe ensure ...
```

这些行的开头高度一致，但尾部每次都变。它们不会形成很长的完全相同 token span，因此逐字重复检测可能漏掉。这类循环需要看“重复的结构前缀”，而不是只看整段是否完全复现。

## 3. 总体方案

该机制作为一个 SGLang logits penalizer 接入 decode 流程，在每个 decode step、采样之前运行。

整体流程如下：

```text
已生成 tokens
   │
   ├── 刚性循环检测：DRY
   │
   └── 模板化循环检测：段前缀检测，未来可升级为 hidden-state probe
          │
          ▼
   持续性门控：重复是否连续出现、是否已经足够持久
          │
          ▼
   分级干预：observe → soft → hard stop
```

各模块职责：

| 模块 | 作用 |
| --- | --- |
| DRY | 检测并抑制逐字重复的刚性循环 |
| 段前缀检测 | 检测“前缀重复、尾部变化”的模板化循环 |
| 持续性门控 | 避免把短暂、合法的重复误判为循环 |
| 分级干预 | 先观察，再软性引导结束，最后必要时强制停止 |

## 4. 刚性循环检测：DRY

DRY 的思想是：如果某个候选 token 会继续延长一段已经出现过的逐字重复后缀，就降低它的概率。

它不是简单地惩罚“出现过的 token”。普通的 repetition penalty 会把所有用过的 token 都压低，这很粗糙，因为正常语言本来就允许多次出现 `the`、标点、缩进或常见词。

DRY 更精确。它只惩罚“正在继续复读一段旧序列”的 token。

惩罚形式可以理解为：

```text
penalty(next_token) = dry_multiplier * dry_base ^ (match_len - dry_allowed_length)
```

其中：

- `match_len`：当前生成后缀在更早上下文中能匹配到的最长逐字长度。
- `dry_allowed_length`：允许的短重复长度，低于这个长度不惩罚。
- `dry_multiplier`：惩罚强度。
- `dry_base`：指数增长的底数。

重复越长，惩罚越强。这样可以有效阻止模型继续沿着同一段文本复读。

同时，DRY 会使用 sequence breakers，例如换行、句号、问号、冒号、分号等，阻止匹配跨过自然边界。这样可以减少对正常短重复的影响。

示例（刚性循环，DRY 能很好处理，均来自生产数据）：

```text
" " " " " " " "              ← 单 token 复读，一路烧到约 29000 token
━ ━ ━ ━ ━ ━ ━ ━              ← 分隔符复读，烧到约 37000 token（本批最长）
https://nitter.net/ https://nitter.net/ ...   ← URL 复读，约 17000 token
```

这些都是同一小段 token 原样反复，后缀逐字匹配长度会快速增长，DRY 的指数惩罚随之变大，能及时压住。需要注意：`====`、`- - -` 这类分隔符在正常表格里也合法，区分“失控的分隔符复读”和“正常表格”靠的是第 7 节的持续性门控，而不是 DRY 本身。

建议初始参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `dry_multiplier` | `0.8` | DRY 惩罚强度，设为 0 可关闭刚性检测 |
| `dry_base` | `1.75` | 惩罚随重复长度增长的指数底 |
| `dry_allowed_length` | `2` | 低于该长度的重复不惩罚 |
| `dry_range` | `0` | 检测范围，0 表示整段输出 |

## 5. 模板化循环检测 v1：段前缀检测

模板化循环的特征是：句子或行的开头反复出现，但后面的内容变化。因此检测器关注“段前缀”。

处理流程：

1. 用 breaker token 把输出切成段。breaker 包括换行、句号、问号、冒号、分号等，具体 token id 从 tokenizer 中一次性推导。
2. 每当一个段结束，取该段开头的 `seg_prefix_len` 个 token 作为段前缀。
3. 保存最近 `seg_window` 个段前缀。
4. 统计其中出现最多的前缀占比：

```text
dominant_share = max_count / seg_window
```

5. 如果窗口已满，并且 `dominant_share >= seg_frac`，这次段结束就标记为 hot。

示例：

```text
Need maybe maybe include ...
Need maybe maybe use ...
Need maybe maybe ensure ...
```

如果 `seg_prefix_len = 3`，这三段的前缀都是：

```text
Need maybe maybe
```

即使尾部不同，也会被识别为同一种模板化重复。

该方法不是硬编码某个短语，而是看结构。因此它也可以覆盖：

- `Assistant: User: ...` 这类角色标记循环。
- `Potential bug: ...` 这类代码评审模板循环。
- `prefix: value` 这类通用键值结构循环。
- 某些推理标记或步骤开头反复出现的循环。

建议初始参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `seg_prefix_len` | `3` | 每段取多少个 token 作为前缀 |
| `seg_window` | `12` | 比较最近多少个段前缀 |
| `seg_frac` | `0.5` | 主导前缀占比达到多少算 hot |

## 6. 模板化循环检测 v2：Hidden-State Probe

段前缀检测依赖已经生成出来的表层文本，因此它天然是滞后的。模型可能在语义上已经进入循环，但文本形态还没有明显重复。

更长期的方向是在模型 hidden states 上训练一个轻量 probe，用来判断当前状态是否已经进入“易循环态”。

设想流程：

1. 在 decode 过程中提取模型 hidden states。
2. 用一个小型线性分类器判断当前状态是正常推理还是 loop-prone。
3. 用 CUSUM 这类变点检测方法判断循环状态是否持续出现。
4. 一旦确认进入循环风险区，再交给后续的持续性门控和干预模块。

相关研究显示，语义循环可能比可见文本重复提前约 1500 tokens 出现。因此 hidden-state probe 有潜力显著提升召回率。

不过这属于后续工作，不是 v1 必须完成的内容。上线前需要先验证：

- SGLang / 目标模型是否方便稳定拿到 hidden states。
- 额外开销是否可接受。
- probe 在真实线上流量中的误报和漏报表现。

## 7. 持续性门控：避免误伤正常重复

检测到重复结构并不等于必须干预。很多正常输出本身就有重复结构，例如：

- Markdown 表格。
- bullet list。
- 缩进代码。
- JSON 或工具调用参数。
- 长 URL。
- 枚举列表。

这些内容可能局部重复，但通常是有边界的，会自然结束。真正的问题是“持续重复并且不收尾”。

一个具体对比，说明为什么“重复”不等于“循环”：

正常但重复（应放过）——一个 markdown 表格，行首都是 `|`：

```text
| 名称 | 状态 |
| --- | --- |
| A | 完成 |
| B | 进行中 |
```

行首 `|`、分隔行 `---` 反复出现，段前缀也高度一致，但它很快结束，然后继续输出别的内容。连续 hot 段数撑不到门槛。

真正的循环（应干预）——同样行首一致，但一直不收尾：

```text
Need maybe maybe include reasons.
Need maybe maybe use short reason.
Need maybe maybe ensure names.
Need maybe maybe ...（持续数千 token，直到 max_tokens）
```

区别不在“像不像重复”，而在“是否持续到结尾、是否还在产出新的实质内容”。这正是持续性门控要卡住的地方。

因此每条检测路径都需要维护连续 hot 计数：

| 计数器 | 含义 |
| --- | --- |
| `rigid_run` | 刚性循环检测连续 hot 的 decode step 数 |
| `seg_hot_run` | 段前缀检测连续 hot 的段结束次数 |

只要出现一次非重复，就把对应计数器清零。

干预门槛：

| 阶段 | 触发条件 |
| --- | --- |
| soft | `rigid_run >= persist_soft` 或 `seg_hot_run >= seg_soft` |
| hard | `rigid_run >= persist_hard` 或 `seg_hot_run >= seg_hard` |

建议初始参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `persist_soft` | `24` | 刚性路径进入 soft 的连续 hot step 数 |
| `persist_hard` | `64` | 刚性路径进入 hard 的连续 hot step 数 |
| `seg_soft` | `4` | 模板化路径进入 soft 的连续 hot 段数 |
| `seg_hard` | `10` | 模板化路径进入 hard 的连续 hot 段数 |

这些值只是初始建议，必须用标注集和线上 shadow 结果调优。

## 8. 分级干预策略

干预不是一上来就强制停止，而是按风险程度逐级升级。

### 8.1 Observe：只观测

observe 模式只记录信息，不改变 logits。

记录内容可以包括：

- 哪条路径触发：rigid 或 segment。
- 当前 `rigid_run` / `seg_hot_run`。
- 是否达到 soft / hard 门槛。
- 命中的重复块或段前缀。
- 当前输出长度。

这个模式适合上线初期做 shadow validation，用来评估检测器本身的 precision 和 recall。

### 8.2 Soft：软性干预

soft 模式开始影响 logits，但不直接强制停止。

它做两件事：

1. 对会继续重复块的 continuation tokens 减去 `loop_penalty`。
2. 给 EOS 增加一个逐步变强的 bias。

EOS bias 的形式：

```text
eos_bias * (1 + over)
```

其中 `over` 表示已经超过 soft 门槛多少步。循环持续越久，越鼓励模型结束。

这样做的意图是：先给模型一个“自己收尾”的机会，而不是立刻截断。

### 8.3 Hard：强制收尾

如果 soft 之后循环仍然持续，就进入 hard。

hard 模式把除 EOS 外的 logits 全部设为 `-inf`，强制模型输出 EOS。

如果当前请求无法确定 EOS token，则退化为 hard-block continuation tokens，也就是强行屏蔽会继续重复的 token。

hard 模式风险最高，必须验证它不会破坏 parser。尤其是：

- qwen3 reasoning parser。
- qwen3 coder tool-call parser。
- 需要结构化输出的场景。

## 9. SGLang 集成方式

建议把该机制实现为一个 `_BatchedPenalizer`，注册到 `BatchedPenalizerOrchestrator` 中，与已有的 frequency、presence、repetition penalizer 并列。

关键实现点：

### 9.1 每请求状态

每个请求维护一个 `_ReqLoopState` 对象，里面保存：

- 最近输出 token 窗口。
- DRY / 刚性路径状态。
- 段前缀窗口。
- `rigid_run` / `seg_hot_run`。
- 上一次同步到的 `output_ids` 长度。
- EOS ids、breaker ids 等缓存结果。

这样 batch filter / merge 时，只需要重排 `_ReqLoopState` 列表，而不是维护多组并行数组。

### 9.2 从 `req.output_ids` 同步

检测器应在 `_apply` 开头从 `req.output_ids` 同步状态，而不是依赖逐 token 的 `_cumulate_output_tokens` hook。

原因是 speculative decoding，尤其是 NEXTN，可能绕过某些 per-token hook。但 `req.output_ids` 是权威输出，最终一定反映实际已经接受的 token。

同步策略：

- 首次同步时，只 seed 最近 `loop_window_size` 个 token。
- 后续同步时，只摄入新增 delta。
- 如果请求输出被重排或截断，需要重建对应状态。

### 9.3 Tokenizer 相关缓存

breaker token ids 和 EOS ids 应该一次性计算并缓存：

- breaker ids 从 tokenizer vocab 中推导。
- EOS ids 来自 `req.eos_token_ids` 和 tokenizer。
- 按 tokenizer identity 做 memoization。

这样 decode 热路径中不需要反复扫描 tokenizer。

### 9.4 默认关闭

默认情况下功能关闭。关闭时，penalizer 只做一次很便宜的 `_is_required` 检查，不应给正常请求带来明显开销。

## 10. 参数汇总

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `loop_enable` | `false` | 总开关 |
| `loop_window_size` | `256` | 最近输出 token 的滑动窗口 |
| `loop_min_output_tokens` | `64` | 输出少于该值时不检测 |
| `dry_multiplier` | `0.8` | DRY 强度，0 表示关闭刚性检测 |
| `dry_base` | `1.75` | DRY 惩罚指数底 |
| `dry_allowed_length` | `2` | 低于该匹配长度不惩罚 |
| `seg_prefix_len` | `3` | 每段取多少 token 作为前缀 |
| `seg_window` | `12` | 比较最近多少个段 |
| `seg_frac` | `0.5` | 主导前缀占比达到多少算 hot |
| `persist_soft` | `24` | 刚性路径 soft 门槛 |
| `persist_hard` | `64` | 刚性路径 hard 门槛 |
| `seg_soft` | `4` | 模板化路径 soft 门槛 |
| `seg_hard` | `10` | 模板化路径 hard 门槛 |
| `loop_penalty` | `0.0` | soft 阶段对 continuation tokens 的惩罚 |
| `loop_eos_bias` | `8.0` | soft 阶段 EOS bias 基值 |
| `loop_mode` | `observe` | `observe` / `soft` / `hard` |

## 11. 上线前验证计划

非 observe 模式上线前，必须先完成验证。检测到循环不等于能安全抑制循环。

建议验证顺序如下：

### 11.1 检测质量验证

在 loop 高发样本和正常流量样本上做检测评估，产出 PR 表：

- Precision：触发的样本中有多少真的是 loop。
- Recall：真实 loop 中有多少被检测到。
- 按 loop 类型拆分：刚性、模板化、其他。
- 按内容类型拆分：表格、代码、JSON、URL、普通自然语言、工具调用。

重点关注：

- 模板化检测的漏检率。
- Markdown 表格、列表、缩进代码上的误报率。

### 11.2 阈值调优

当前阈值只是初始值，必须结合标注集调优：

- `persist_soft` / `persist_hard`
- `seg_soft` / `seg_hard`
- `seg_frac`
- `loop_min_output_tokens`
- `loop_eos_bias`

ground truth 应尽量独立于检测算法本身，避免用同一个规则既打标签又评估。

### 11.3 DRY 线上 A/B

先针对刚性循环做 DRY 的 observe 或低风险 A/B：

- 是否降低刚性 runaway rate。
- 是否影响正常输出质量。
- 是否增加过早停止。
- 对代码、JSON、URL 是否有负面影响。

### 11.4 干预 A/B

检测质量可接受后，再从 observe 进入 soft：

- runaway rate 是否下降。
- 平均生成长度是否下降。
- 正常任务成功率是否下降。
- parser error 是否增加。
- 用户可见输出是否出现异常截断。

hard force-stop 应最后验证，且只在 soft 无法解决的样本上评估。

### 11.5 Speculative Decoding 验证

需要专门验证 NEXTN / speculative decoding 下的 active path：

- observe 是否能正确计数。
- soft penalty 是否真的作用在采样前 logits 上。
- hard force-stop 是否能稳定输出 EOS。
- `req.output_ids` 同步是否在 spec decode 下不漏 token、不重复 ingest token。

### 11.6 Parser 安全性

强制 EOS 可能打断 reasoning 或 tool call，因此需要验证：

- qwen3 reasoning parser 是否能处理被提前收尾的输出。
- qwen3_coder tool-call parser 是否会产生格式错误。
- 结构化输出场景是否出现不可解析结果。

## 12. 推荐落地路线

建议分阶段推进：

1. 实现 observe-only 的 SGLang penalizer 框架，接入 `req.output_ids` 同步和 per-request state。
2. 接入 DRY，用于刚性循环检测和 shadow 统计。
3. 接入段前缀检测，用于模板化循环检测和 shadow 统计。
4. 在真实 replay 和正常流量上产出检测 PR 表。
5. 调整阈值，确认误报可控后，对刚性循环做 DRY A/B。
6. 对高置信循环开启 soft 干预，观察 runaway rate 和正常质量。
7. 谨慎评估 hard force-stop，只作为最后兜底。
8. 如果模板化循环召回仍然不足，再启动 hidden-state probe 可行性研究。

## 13. 简短总结

这套设计是在 SGLang 推理过程中增加一个默认关闭的循环防护器。

它把重复循环分成两类：

- 刚性循环：同一段 token 逐字复读，用 DRY 处理。
- 模板化循环：固定前缀反复出现但尾部变化，用段前缀检测处理，未来可升级 hidden-state probe。

检测结果不会立即触发干预，而是先经过持续性门控，确认重复确实在持续。干预也分级进行：先只观察，再软性惩罚重复 continuation tokens 并提高 EOS 概率，最后在必要时强制输出 EOS。

设计重点不是禁止所有重复，而是识别并抑制那些不会自行结束、会烧到 `max_tokens` 的失控生成。

## 附录 A：真实案例库（来自 OR 生产数据，2026-06-17）

下面的例子来自对生产 reasoning 的扫描（取最高频 8-gram 重复 ≥ 30 的 258 条样本）。每条给出重复形态、真实生成长度、归属的检测器。`ct` 指 completion_tokens。

一个重要前提：**高重复 ≠ 一定失控**。同一种形态，有的烧到几万 token（真失控），有的重复一阵就自己收住了（ct 很小）。这正是为什么检测之外还要有持续性门控——光看“有没有重复”会把大量正常内容也算进来。

### A.1 刚性循环（DRY 负责）

| 形态 | 真实样本（重复片段） | 是否失控 |
| --- | --- | --- |
| 单 token / 短周期 | `" " " " " " " "` | 是，ct≈29506 |
| 单 token / 短周期 | `- ne - ne - ne - ne` | 否，ct≈218（重复但收住） |
| 单 token / 短周期 | `0 , 0 , 0 , 0 ,` | 否，ct≈167 |
| 分隔符 | `━ ━ ━ ━ ━ ━ ━ ━` | 是，ct≈37351（本批最长） |
| 分隔符 | `= = = = = = = =` | 否，ct≈405 |
| 分隔符 | `- - - - - - - -` | 否，ct≈67 |
| URL / 路径 | `https : / / nitter . net / ...` | 是，ct≈17605 |
| URL / 路径 | `https://interactivebrokers.github/ ...` | 是，ct≈24900 |
| JSON / tool 参数 | `< / tool_results > < / tool_results >` | 部分，ct≈5633 |
| JSON / tool 参数 | `{ "head" : "Clear Channel ...` | 是，ct≈9207 |
| 数值 | `100 . 118 . 196 . 66 . ...`（IP 样式） | 否，ct≈136 |

共性：逐字重复，DRY 的后缀匹配能直接吃住。`====` / `- - -` 这种在正常表格里也合法，靠持续性门控区分。

### A.2 模板化循环（段前缀负责；DRY 看不见）

| 家族 | 真实样本（重复片段） | 是否失控 |
| --- | --- | --- |
| 推理标记 | `Need maybe maybe include / use / ensure ...` | 是，可达 ct≈47000 |
| 推理标记（口吃变体） | `Need maybe maybe maybe maybe create ...` | 是，ct≈6473 |
| 推理标记 | `Need maybe maybe maybe think about whether ...` | 是，ct≈4489 |
| 代码评审 | `. Potential bug : Tether mini game ...` | 否，ct≈171 |
| 代码评审 | `Need maybe " Potential issue : no ...` | 是，ct≈3689 |
| 角色标记 / 对话幻觉 | `Assistant : User : [ OpenClaw heartbeat poll ...` | 否，ct≈293 |
| 通用 prefix: value | `<某固定前缀> : <每次不同的值>` 反复 | 视情况 |

共性：句首前缀高度一致，尾部每次都变，**没有长段逐字重复**，所以 n-gram / DRY 容易漏。

### A.3 容易误伤的“正常重复”（必须放过）

这些不是循环，但形态上很像，是误报的主要来源。检测器必须靠持续性门控放过它们：

- markdown 表格：行首 `|`、分隔行 `---` 反复。
- bullet / 枚举列表：行首 `-`、`1.`、`*` 反复。
- 缩进代码：相同缩进 + 相同关键字反复。
- 长 URL 列表、大 JSON 对象：字段名 / 结构反复。

它们的共性是**有界、会结束、之后继续产出新内容**；真正的循环是**持续到 max_tokens、不再产出新实质内容**。

### A.4 需要特别注意的边界情况

- **口吃 / 词内退化**：`maybe maybe maybe maybe` 这种单个 token 连续复读，既是刚性（DRY 能抓），又常作为模板化循环的一部分出现。两个检测器同时命中属正常。
- **对话幻觉**：模型自己编造 `Assistant: / User:` 轮次，在“扮演”整段对话。这是和纯重复不同的失败模式，值得单独监控。
- **混合循环**：同一段里行级是模板化、行内又夹着刚性重复。设计上两条信号是 OR 关系，任一命中即可，因此能覆盖混合情况。
- **重复但自愈**：很多带 `Need maybe` 的样本只重复几句就恢复并正常收尾（ct 很小）。这类不该干预，是持续性门控存在的根本原因，也是 recall/precision 调阈值时最棘手的边界。
