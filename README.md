# AgentRecall

AgentRecall 是一个构建在 LayerRecall 之上的、面向长视频生成的推理期语义记忆控制器。当前主方案**不需要训练新的 DiT、路由器或多模态模型**：冻结的本地 Qwen3-VL 负责理解历史事件与状态，LayerRecall 继续负责保存和注入完整精度的历史 K/V。

## 核心思路：事件状态驱动的语义检索

LayerRecall 提供底层视觉记忆机制：它保存历史 chunk 的 K/V，并检索相关 chunk 注入部分注意力层。但单纯依赖隐藏状态相似度可能出现以下问题：

- 找到语义相似但主体错误的历史片段；
- 偏向较近但无关的 chunk；
- 人物换装后错误恢复旧衣服；
- 主体离开很久再出现时，无法准确定位最初外观。

AgentRecall 在 LayerRecall 之上加入高层语义控制器：

```text
历史视频 chunk / caption
          ↓
    Qwen3-VL 观察器
          ↓
带版本的事件与实体状态记忆
          ↓
  RECALL / UPDATE / IGNORE
          ↓
   相关历史 chunk ID
          ↓
限制或重排 LayerRecall 候选
          ↓
注入对应历史 chunk 的完整 K/V
```

Agent 负责判断**哪个事件、哪个状态版本仍然有效**；LayerRecall 负责从对应 K/V 中恢复人物身份、服装、物体和场景等视觉细节。Agent 不直接选择 DiT 层，也不会用文本摘要代替底层 K/V。

## 事件和状态版本

每个事件节点对应一个或多个历史 chunk。每个实体可以拥有多个带有效区间的状态版本，例如：

```text
woman_A:v1 = 红色外套，持续到 chunk 4
woman_A:v2 = 黑色外套，从 chunk 5 开始生效
```

Agent 输出三类动作：

- `RECALL`：主体或物体重新出现，读取有效历史事件；
- `UPDATE`：状态发生明确变化，创建新版本并使旧版本过期；
- `IGNORE`：历史无关或不可靠，不注入远期记忆。

例如，人物穿红衣离开后以相同装束返回，Agent 应检索第一次出场的 chunk；如果 prompt 明确要求人物换成黑衣，则应更新状态，而不是恢复已经过期的红衣 K/V。

当前验证版本采用两阶段流程：先观察视频并生成计划，再依据计划进行生成。它还不是每个去噪步骤同步调用一次 VLM 的在线系统。

## 核心代码路径

### Agent 与事件记忆

- `exp/11-eventstate-recall/observe_video.py`
  - 加载本地 Qwen3-VL；
  - 从视频 chunk 构造缩略图或事件 montage；
  - 要求模型输出结构化 JSON；
  - 校验和修复 JSON，生成事件决策与 LayerRecall 计划。
- `exp/11-eventstate-recall/event_memory.py`
  - 实现 `EventNode`、`StateVersion` 和 `EventStateMemory`；
  - 维护事件、实体状态版本及其到历史 chunk 的映射。
- `exp/11-eventstate-recall/replan_trace.py`
  - 基于已经审核的视觉 trace 重新规划，无需再次执行视频观察。
- `exp/11-eventstate-recall/schemas/event_observation.schema.json`
  - 定义 Qwen 输出的 JSON 约束。

### LayerRecall 接入

- `utils/layer_recall.py`
  - 解析 `layer_recall_agent_plan_path` 或内联计划；
  - 实现 `agent_plan`、`agent_event` 和 `agent_event_diverse`；
  - 根据 Agent 计划限制或重排历史候选 chunk。
- `wan_5b/modules/causal_model.py`
  - 在注意力检索前传入当前 chunk 和 Agent 指定的历史 chunk；
  - 记录 Agent 偏好及最终检索结果，便于审计。
- `inference.py`
  - 使用外部 Agent 计划运行常规 LayerRecall 推理流程。
- `exp/11-eventstate-recall/config_qwen_demo.yaml`
  - 启用 `agent_event_diverse`，通过 `LR_AGENT_PLAN_PATH` 加载计划。
- `exp/11-eventstate-recall/run_qwen_demo.sh`
  - Qwen 规划后重新生成视频的示例入口。

Agent 计划格式很小，不包含任何 K/V 张量：

```json
{
  "layer_recall_agent_plan": [
    {
      "current_chunks": [2, 2],
      "preferred_chunks": [0],
      "action": "RECALL"
    }
  ]
}
```

## 调用链

```text
observe_video.py
      ↓
Qwen3-VL 生成事件与状态 JSON
      ↓
event_memory.py 建立 event/state → chunk 映射
      ↓
layer_recall_plan.json
      ↓
utils/layer_recall.py 读取并执行 agent policy
      ↓
causal_model.py 在注意力中检索和注入历史 K/V
      ↓
inference.py 输出视频
```

## 复现 Agent 方案

另一台服务器需要准备：

- Wan2.2-TI2V-5B；
- LongLive/LayerRecall 发布的 checkpoint；
- Qwen3-VL-4B-Instruct；
- LayerRecall 推理环境和 Qwen3-VL 观察环境。

原开发服务器中的绝对路径仅用于实验，请在新机器上配置对应路径。生成阶段最重要的环境变量是：

```bash
export LR_AGENT_PLAN_PATH=/path/to/layer_recall_plan.json
```

推荐复现步骤：

1. 使用 `observe_video.py` 观察历史视频并生成 trace 和 plan；
2. 人工检查 JSON 中的事件、状态版本和 `preferred_chunks`；
3. 使用 `run_qwen_demo.sh` 或 `inference.py` 运行 AgentRecall；
4. 同时保存 Agent trace 和 LayerRecall 检索日志，以区分语义规划错误与底层 K/V 注入错误。

## 实验目录说明

`exp/` 记录了方案从 LayerRecall 基线到 AgentRecall、再到动态层路由探索的开发过程。生成视频、checkpoint、机器日志和本地环境文件不会提交到 Git。

| 目录 | 实验目的 | 当前结论 |
|---|---|---|
| `00-smoke` | 最小环境和模型 smoke test。 | 用于确认推理链路可运行。 |
| `01-routing-baselines` | 比较 local-only、固定 10 层和所有层注入。 | 建立 LayerRecall 路由基线。 |
| `02-temporal-layer-demand` | 分析不同时间和网络层对远期记忆的需求。 | 为选择性记忆注入提供诊断依据。 |
| `03-recall-update-conflict` | 比较人物/物体重现与显式状态更新。 | 研究“恢复旧状态”和“接受新状态”的冲突。 |
| `04-multi-prototype-diagnostic` | 多原型或多摘要诊断。 | 检查单向量检索中的语义混叠。 |
| `05-cache-horizon-stress` | 扩大相关事件之间的时间距离。 | 测试长间隔和 cache horizon 下的退化。 |
| `06-agentic-retrieval` | 第一版 Agent 检索原型。 | 早期可行性验证，结果见目录内 `RESULTS.md`。 |
| `07-agentic-within-event` | 在同一事件内部选择不同历史位置。 | 比较 early、late 和 mixed 策略。 |
| `08-agentic-seed1-replication` | 更换随机种子复现实验。 | 检查收益是否由特定 seed 导致。 |
| `09-demo-reappearance` | 人物离开后重新出现的展示案例。 | 面向定性展示。 |
| `10-demo-long-transition` | 更长过渡和干扰事件后的主体返回。 | 测试更长时间间隔。 |
| `11-eventstate-recall` | 当前主要的免训练 AgentRecall。 | Qwen 事件状态规划与 LayerRecall K/V 检索。 |
| `12-dynamic-state-chpm` | 可训练动态层路由器的 smoke test。 | 训练拓扑、梯度与 checkpoint 兼容性正常。 |
| `13-dynamic-router-train` | prior scale 为 4.0 的 500-step 训练。 | 数值健康，但 Top-10 没有脱离原固定层。 |
| `14-dynamic-router-prior05` | prior scale 降到 0.5 的 500-step 训练。 | 仍选择同一组固定层，不能证明动态选层成功。 |

## 当前结果如何理解

当前最可靠的方向是 `exp/11-eventstate-recall` 中的免训练语义检索方案。在已有的小规模审核案例中，Qwen 能正确判断动作和目标事件，并在 DINO return-consistency 筛选指标上接近人工 Oracle 事件计划。但这些结果属于检索隔离验证，不能替代完整的大规模 benchmark。

`exp/12–14` 证明了动态层 CHPM 训练链路可以运行，但两次 500-step 训练的硬 Top-10 都退化为 LayerRecall 原来的固定层集合。因此它们应当作为工程验证和消融记录，而不是动态选层有效性的证据。后续需要加入 soft gate、温度退火或 Gumbel exploration，再重新验证。

## 仓库管理

模型权重、checkpoint、日志、生成视频以及机器相关的 `exp/env.sh` 已通过 `.gitignore` 排除。建议仅提交可复现的源码、配置、schema、小规模 trace/指标和文档，大文件保存在仓库之外。
