# Pass Network — Design Notes

> 记录 pass_network 模块的设计思路、技术决策和演进计划。

---

## 核心问题

Pass network 回答的是与 Pass Score 不同的问题：

| 模块 | 问题 | 粒度 |
|------|------|------|
| `pass_value` | **这次传球该传给谁？** (单次决策质量) | 单个 snapshot |
| `pass_network` | **这支球队的传球结构是什么？** (整体传球模式) | 全场/半场聚合 |

---

## 数据来源

使用 **Event Data**（不用 Tracking Data）。

理由：
- Pass network 只需要 passer → receiver 关系 + 球员位置
- Event data 的每个传球事件都嵌入了 22 人的 snapshot（freeze-frame）
- Tracking data 是 overkill（25fps 连续帧，没有必要）

关键字段（来自 `PassEvent`）：
- `passer_id` / `target_id` → 有向边
- `outcome == "C"` → 是否完成
- `snapshot.all_players` → 每次传球时刻的球员位置
- `passer_name` / `target_name` → 节点标签

---

## Level 1 — 基础 Pass Network（当前实现）

### 数据结构

```
PassNetwork
├── nodes: dict[player_id → PlayerNode]
│   ├── avg_x, avg_y       ← 平均站位（从所有相关 snapshot 聚合）
│   ├── pass_count         ← 发出传球次数
│   └── receive_count      ← 接球次数（被传到的次数）
│
└── edges: dict[(passer_id, receiver_id) → PassEdge]
    ├── count              ← 传球次数
    ├── completed          ← 成功次数
    └── completion_rate    ← 完成率（属性）
```

### 节点位置策略

**聚合所有参与该球队传球事件的 snapshot 中的球员位置**，取平均值。

这样做的原因：
- 比只用传球者/接球者位置更稳定（样本更多）
- 反映球员在整场比赛中的**典型站位区域**
- 与 tracking data 的平均位置近似（而不需要加载 tracking 数据）

### 过滤参数

- `team_id`: 只分析一支球队的传球（按传球者所在队过滤）
- `period`: 可按半场过滤，None = 全场
- `completed_only`: 默认 True（只统计成功传球）

---

## Level 2 — 结合 Pass Score（待实现）

将 `pass_value` 模块的评分整合进 pass network：

```python
@dataclass
class PassEdge:
    ...
    avg_pass_score: float | None  # 该传球路线的平均 Pass Score
```

这是 Threader 区别于普通 pass network 的独特之处：
- 普通工具只问"谁传给谁传了多少次"
- Threader 额外问"这条传球路线的**质量**怎么样？"

应用场景示例：
> 某球队 A→B 这条路线出现了 15 次，但平均 Pass Score 只有 35/100。
> 说明这是一条**习惯性的低质量传球**，值得战术分析。

实现思路：
- `builder.py` 调用 `analyze_pass_event()` 给每次传球打分
- 按边聚合平均分
- 注意：Pass Score 计算较慢（每次 ~0.1s），全场 500+ 传球可能需要优化

---

## Level 3 — 网络指标（✅ 已实现，`metrics.py`）

基于图论的结构指标，纯 Python 实现（不引入 networkx）：

| 指标 | 含义 | 用途 | 实现 |
|------|------|------|------|
| **Density** | 边数 / 最大可能边数 | 传球分布有多均匀？ | `\|E\| / (n×(n−1))` |
| **Degree centrality** | 有多少独特传球伙伴 | 谁是核心枢纽？ | `(out+in) / 2(n−1)` |
| **Betweenness centrality** | 经过该球员的最短路径比例 | 谁是传球"中间人"？ | Brandes 算法，距离=1/count |
| **PageRank** | 被重要球员传球的重要性 | 谁是接球焦点？ | 幂迭代，damping=0.85 |

> Clustering coefficient 暂缓：足球场景里意义不如前三个直观，后续视需求添加。

### 关键算法决策

**Betweenness 的边权重**：用 `1/edge.count` 作为路径距离。
- 传球次数多的路线 = 距离短 = 更"主干"的连接
- 这样找到的是"主要传球干道上的中间人"，而不只是"跳数少的中转站"

**PageRank 的边权重**：直接用 `edge.count`（传球越多 = 影响力越强）。

**归一化**：所有球员级指标输出 `[0, 1]`；PageRank 额外除以最大值（最重要的球员 = 1.0）。

---

## 可视化方向（待实现）

在 pitch 上绘制 pass network：
- 节点 = 球员平均站位（圆圈大小 = 传球次数）
- 边 = 传球路线（线宽 = 传球次数，颜色 = 完成率）
- 方向箭头 = 传球方向

可放入现有 Dash app 作为新的分析标签页。

---

## 技术债与待讨论

- [ ] `completed_only=True` 时，边的 `completion_rate` 恒为 1.0（无意义）
      → 考虑始终记录所有尝试，但只在可视化时过滤
- [ ] 球员名字在 snapshot 中为 None（PFF 不嵌入名字）
      → 从 `PassEvent.passer_name` / `target_name` 补充
      → 更完整的方案：传入 roster 数据（`load_roster()`）
- [ ] 替补球员导致 >11 个节点 → 后续考虑按上场时间加权
- [ ] 加时赛 period 3/4 的处理（目前按普通 period 处理）

---

*模块状态: Level 1 + Level 3 已实现；Level 2（Pass Score 集成）待定*
*最后更新: 2026-02-22*
