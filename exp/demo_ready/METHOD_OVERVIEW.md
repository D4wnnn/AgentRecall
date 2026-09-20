# 基于 LayerRecall 的事件级 Agentic Memory Routing

## 1. 我们要解决什么问题

自回归长视频模型通常按 chunk 逐段生成视频，并且只保留一个有限大小的局部
K/V cache。这种机制能够维持相邻画面的连续性，但较早的内容会逐渐被滑动窗口
淘汰。

因此，当一个人物、物体或属性暂时离开画面，经过若干个 chunk 后再次出现时，
模型可能已经失去对应的历史信息。例如：

1. 第一镜头中，一位白发老人穿着黄色雨衣、戴红色眼镜、拿着紫色雨伞；
2. 第二镜头转向花摊，老人完全离开画面；
3. 第三镜头返回时，模型需要恢复**同一个人及其原有属性**。

普通局部模型可能在第三镜头生成另一张脸、改变衣服颜色、丢失雨伞，或者只恢复
场景而没有恢复人物。这类问题本质上是：

> 当重要内容离开局部上下文后，模型如何从很长的生成历史中找到真正相关的状态，
> 并在不破坏当前运动和场景的情况下使用它？

---

## 2. LayerRecall 如何解决这个问题

LayerRecall 在自回归视频模型的局部注意力之外增加了一个稀疏的长期记忆通路。
它主要回答两个问题：

- **What to retrieve：**应该从哪些历史 chunk 中读取记忆；
- **Where to use：**应该在哪些 DiT 层中注入这些历史记忆。

### 2.1 历史记忆的保存

每生成完一个历史 chunk，LayerRecall 在每个 DiT 层中：

1. 对该 chunk 的 pre-RoPE keys 做归一化和平均池化，得到一个轻量的历史摘要；
2. 保存摘要对应的 chunk 编号和物理 cache 位置；
3. 在物理 K/V cache 中保留该 chunk 完整的、已经应用 RoPE 的 K/V。

摘要只用于低成本检索，真正进入注意力计算的仍然是对应 chunk 的完整 K/V，因而
可以恢复人物外观、衣服纹理和具体物体等细节。

### 2.2 当前状态条件化检索

对于当前 chunk，LayerRecall 从当前层的隐藏状态生成查询向量，并与所有仍在物理
cache 中的历史摘要计算 cosine similarity：

$$
s_{t,i}^{l}=\cos(q_t^l,m_i^l).
$$

在每个启用记忆的层中，模型选择得分最高的两个历史 chunk，将它们的完整 K/V
加入当前 self-attention。

### 2.3 选择性层注入

LayerRecall 不在全部 30 个 DiT 层中使用长期记忆，而只在 profiling 得到的 10 个
memory-sensitive layers 中注入：

```text
[4, 9, 10, 12, 13, 15, 16, 17, 18, 26]
```

其他 20 层继续使用原始局部注意力。这样可以利用远期信息，同时减少全层注入造成
的运动抑制、画面跳变和历史状态干扰。

---

## 3. 我们发现了什么问题

我们首先比较了 local-only、固定 10 层 LayerRecall 和全部 30 层注入。实验显示，
固定 10 层通常比全层注入更稳定，说明“在哪些层使用记忆”确实重要。

随后我们分析 LayerRecall 的历史检索，发现：

1. 不同候选 chunk 的 cosine scores 非常接近，Top-1 margin 经常只有
   `0.0002–0.0024`；
2. 用 recent 或随机策略替换 cosine 后，生成质量明显下降，说明选择哪段历史确实
   会产生因果影响；
3. 受控 agent 把第一镜头的目标事件命中率从 cosine 的 `33.75%` 提高到了 `100%`；
4. 但是让 agent 在所有层中强制使用同样两个 chunk 并不可靠——即使选中了正确
   事件，如果恰好选择的是事件末尾两个 chunk，也可能出现物体颜色漂移或身份改变；
5. LayerRecall 自己有时会在不同层选择事件早期和末期的互补 chunk，这反而比统一
   强制某两个 chunk 更稳定。

因此，我们得到的关键判断是：

> Agent 擅长判断“应该回忆哪一个历史事件”，但不适合直接替代神经 router，
> 为所有层硬指定完全相同的 K/V chunk。

---

## 4. 我们提出的方案

我们提出一个 **Hierarchical Agent–Router Memory**，即“事件级 Agent + 层内神经
检索 + 有效性门控”的层次化记忆系统。

```text
每个新生成的 chunk
          │
          ▼
Chunk Observer
提取轻量语义摘要和实体状态变化
          │
          ▼
Event Memory Manager
把连续 chunk 合并为事件，并维护实体的当前有效状态
          │
          ▼
当前 chunk 与当前 prompt
          │
          ▼
Event Agent
判断当前是在继续当前状态、恢复旧实体，还是发生了状态更新
并选出相关历史事件 / 屏蔽无关或过期事件
          │
          ▼
LayerRecall Router
仅在 Agent 允许的事件范围内，每个敏感层独立计算 cosine score
并选择两个互补的历史 chunk
          │
          ▼
Uncertainty / Validity Gate
根据检索 margin、事件有效性和状态冲突决定是否注入以及注入强度
          │
          ▼
在 10 个 memory-sensitive layers 中注入完整 K/V
其他层保留 local attention
```

### 4.1 生成过程中如何记录记忆

要把受控实验扩展成真正在线工作的 Agent，生成过程中确实需要为历史 chunk 建立
语义索引。这里需要区分两种摘要：

1. **LayerRecall 数值摘要：**原方法已经为每层、每个 chunk 保存 pooled pre-RoPE
   key。它适合与当前隐藏状态计算 cosine similarity，但不直接表达“画面中是谁、
   穿什么、发生了什么”；
2. **Agent 语义摘要：**我们额外维护一份跨层共享的结构化记录，用于判断历史
   chunk 属于哪个事件、包含哪些实体和状态，以及这些状态后来是否被更新。

一个 chunk 的语义记录可以表示为：

```json
{
  "chunk_id": 17,
  "event_id": "rooftop_artist_appearance_1",
  "entities": {
    "artist_A": {
      "visible": true,
      "identity": "platinum pixie haircut, violet beret",
      "appearance": "burnt-orange jacket, striped overalls",
      "action": "painting",
      "relations": ["holding turquoise brush", "standing beside yellow toolbox"]
    },
    "painting_A": {
      "appearance": "red sailboat on white canvas",
      "relation": "on easel in front of artist_A"
    }
  },
  "camera": "medium-wide rooftop view",
  "state_delta": ["painting_A becomes more complete"],
  "confidence": 0.91
}
```

但是，这不意味着每 8 个 latent frames 都必须调用一次昂贵的大型 VLM/LLM。更
合理的实现是两级观察：

- **每个 chunk：**使用轻量视觉/latent projector，根据当前隐藏状态、生成 prompt
  和低分辨率预览产生 embedding、实体可见性及 state delta；
- **事件边界或异常时：**当镜头切换、实体出现/消失、检索不确定或状态冲突时，
  才调用较强的 VLM/Agent，把多个 chunk 合并成事件摘要并更新实体状态表。

因此，记忆系统同时维护：

```text
Layer Memory Bank
  └─ 每层、每 chunk 的数值摘要 + 完整 K/V 指针

Semantic Event Memory
  └─ 跨层共享的 chunk 摘要、事件边界、实体状态和新旧状态版本
```

语义记忆负责“找到哪个事件”，Layer Memory Bank 负责“取出哪些层的完整 K/V”。
二者通过 `chunk_id / event_id` 对齐，而不是让 Agent 自己保存或生成 K/V。

### 4.2 Event Agent：决定回忆哪个事件

Agent 不直接输出某个层必须使用的精确 chunk，而是输出事件级约束，例如：

```json
{
  "action": "RECALL",
  "target_entity": "the flower-market woman",
  "allowed_event": "shot_1",
  "forbidden_events": ["shot_2"],
  "expected_state": ["same identity", "same clothing", "same umbrella"]
}
```

它可以根据分镜 prompt、人物是否重新出现以及“换装、损坏、移除”等状态更新语义，
决定某段历史是有效记忆还是已经过期的状态。

### 4.3 LayerRecall Router：事件内部选择关键状态

Agent 选定第一镜头后，不强制所有层都使用同样的两个 chunk。对每个敏感层 $l$，
仍然保留 LayerRecall 的当前状态条件化评分，但把候选集合限制在 Agent 允许的事件
$E_t$ 内：

$$
i_l^*=\operatorname{TopK}_{i\in E_t}
\cos(q_t^l,m_i^l).
$$

因此，不同层可以选择不同的历史状态。例如，一个层可能选择人物正脸最清晰的
chunk，另一个层可能选择衣服、雨伞或场景布局更完整的 chunk。

后续可以进一步增加 diversity-aware selection：在选择第二个 chunk 时惩罚与第一
个 chunk 过于相似的摘要，使两个 memory slots 更倾向于覆盖互补信息，而不是选择
两个几乎重复的相邻 chunk。

### 4.4 Uncertainty / Validity Gate：决定是否相信历史

如果候选得分几乎相同，说明 router 并不确定；如果 prompt 明确表达人物换装或物体
状态改变，则旧记忆可能已经失效。因此可为每层增加门控：

$$
\tilde A_t^l=A_{\mathrm{local}}^l+
\lambda_t^l A_{\mathrm{memory}}^l,
$$

其中 $\lambda_t^l$ 由以下信息决定：

- Top-1/Top-2 检索 margin；
- 当前状态和历史状态的相似度；
- Agent 给出的 `RECALL / UPDATE / IGNORE` 动作；
- 该历史事件是否包含已经过期的属性。

这可以避免在检索不确定时强行注入历史，也可以避免人物已经换衣服后又被旧记忆
强制“换回去”。

---

## 5. 一个具体例子：屋顶画家重新出现

### 视频内容

- **Shot 1：**一位金色短发、戴紫色贝雷帽、穿橙色外套的画家，在屋顶绘制红色
  帆船，旁边有黄色工具箱；
- **Shot 2：**镜头绕过砖墙并停留在空旷城市天际线，人物和画架完全离开画面；
- **Shot 3：**镜头返回，prompt 只要求“同一位画家继续作画”，不重新提供全部
  外观答案。

### 三种方法的结果

- **Baseline：**镜头停留在城市天际线，没有恢复画家；
- **LayerRecall：**恢复了画架，但画作内容发生改变，而且人物没有稳定返回；
- **Agent + LayerRecall：**Agent 判断 Shot 3 是人物重现，先把检索范围约束到
  Shot 1；LayerRecall 再在 Shot 1 内为不同层选择相关 K/V，最终恢复了金发、紫帽、
  橙色外套的原画家以及红色帆船画。

这个案例说明，普通 cosine 检索虽然拥有历史访问能力，但可能从多个历史阶段中
选择错误或不完整的线索。事件级 Agent 提供了语义上的“检索方向”，LayerRecall
则保留细粒度、逐层的 K/V 选择能力，两者结合比任何一方单独硬控制更合理。

演示视频：[`demo_01_rooftop_artist.mp4`](demo_01_rooftop_artist.mp4)

---

## 6. 当前结论与下一步

当前实验中的 Agent 是一个**受控事件级规划器**，用已知分镜把 Shot 1 标记为合法
历史事件，还不是已经训练完成的 LLM/VLM Agent。因此，目前可以支持的结论是：

> 事件级语义约束能够显著改变实际 K/V 检索，并在多个案例中改善人物和属性恢复；
> 但精确的 chunk 和 layer 内选择仍应交给状态条件化神经 router。

下一步建议按以下顺序实现：

1. 在线事件分割和实体状态表；
2. 自动输出 `RECALL / UPDATE / IGNORE` 的轻量 VLM 或规则 Agent；
3. Agent 约束下的事件内 cosine 检索；
4. diversity-aware 双 slot 选择；
5. 基于检索不确定性和状态冲突的注入门控；
6. 最后再评估是否需要动态选择注入层。

这样形成的创新点不是简单扩大 Top-k，而是把长期记忆分成三个层次：

> **语义层决定回忆哪个事件，神经路由层决定每层读取哪些关键状态，门控层决定
> 当前是否应该相信并注入这段历史。**
