# -*- coding: utf-8 -*-
"""三层信任记忆：cold / warm / hot + 验证门控。

设计原则（和你最初的构想一致）
------------------------------------------------
  hot   : 照单全收，允许脏，高可塑，随时可清空
  warm  : 只有通过独立验证的条目才能进，可复现的稳定记忆
  cold  : 只读锚点，只能由外部显式写入，门控永远不许碰它

相对那份 AI 生成演示的三处关键修正（每一条都有 selftest 兜着）
------------------------------------------------
1. 验证器必须同时看到 key 和 value。
   只拿 value 去比对"已知答案"的所谓验证，等价于拿答案对答案表，
   结果必然是"错误率 0.00%、全部通过"——原演示就是这么假的。
2. 无法验证 ≠ 通过。邻居支撑不足的条目一律拒绝（unverifiable），
   而不是"没有冷备就返回 True"。
3. 读优先级、阈值、容量全部是显式参数，可以被实验扫。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------- 基础

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


@dataclass
class MemoryEntry:
    """一条记忆：key 是"什么时候该用它"，value 是"该输出什么"。"""

    key: np.ndarray
    value: Any
    tag: str = ""
    score: float = 0.0          # 验证得分，由 validator 给出
    use_count: int = 0          # 被读中几次
    step: int = -1              # 诞生时刻
    origin: str = "hot"         # 来源层
    provisional: bool = False   # 标签还没到、先收下等事后裁判

    def flat(self) -> np.ndarray:
        return np.asarray(self.key, dtype=np.float64).ravel()


@dataclass
class GateReport:
    """一轮记忆巩固的账本。"""

    step: int = -1
    n_candidates: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_unverifiable: int = 0
    n_provisional: int = 0
    accepted_score_mean: float = 0.0
    # 下面四项只用于"事后算账"，门控本身看不到，保证不舞弊
    accepted_dirty: int = -1
    accepted_clean: int = -1
    rejected_dirty: int = -1
    rejected_clean: int = -1
    cold_after: int = 0
    warm_after: int = 0
    hot_after: int = 0

    @property
    def accept_rate(self) -> float:
        return self.n_accepted / self.n_candidates if self.n_candidates else 0.0

    @property
    def false_accept_rate(self) -> float:
        """漏放率：脏条目里有多少被放进了温备。"""
        tot = self.accepted_dirty + self.rejected_dirty
        return self.accepted_dirty / tot if tot > 0 else float("nan")

    @property
    def false_reject_rate(self) -> float:
        """误拒率：干净条目里有多少被拒了。"""
        tot = self.accepted_clean + self.rejected_clean
        return self.rejected_clean / tot if tot > 0 else float("nan")

    def as_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["accept_rate"] = self.accept_rate
        return d


# ---------------------------------------------------------------- 验证器

class Validator:
    """验证器接口。

    约定：score() 必须同时使用 key 和 value。
    返回 [0, 1] 的可信度；调用方按阈值 theta 决定收不收。
    """

    name = "validator"

    def score(self, key: np.ndarray, value: Any, step: int = -1) -> float:
        raise NotImplementedError

    def extend_pool(self, x: np.ndarray, y: np.ndarray) -> None:
        """环境送来新的带标签留出样本时的钩子；不需要 oracle 的验证器可以不实现。"""
        return None


class NeighborValidator(Validator):
    """正确的做法：到独立留出池里看这条记忆的"key 落点"上，真标签是什么。

    一条脏记忆（标签被翻转）的 key 仍然落在真实类簇里，
    它的邻居真标签 != 它自称的 value —— 于是得分接近 0，被拒。
    一条干净记忆则得分接近 1，被收。

    代价是真实的：需要持续获得带标签的留出样本（这就是 oracle 的账单）。
    """

    name = "neighbor"

    def __init__(self, pool_x: np.ndarray, pool_y: np.ndarray,
                 sim_floor: float = 0.5, min_support: int = 3,
                 topk: int = 8):
        self.pool_x = np.asarray(pool_x, dtype=np.float64)
        self.pool_y = np.asarray(pool_y)
        self.sim_floor = sim_floor
        self.min_support = min_support
        self.topk = topk
        self.n_pool = len(self.pool_x)

    def extend_pool(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if len(x) == 0:
            return
        self.pool_x = np.vstack([self.pool_x, x]) if self.n_pool else x
        self.pool_y = np.concatenate([self.pool_y, y]) if self.n_pool else y
        self.n_pool = len(self.pool_x)

    def score(self, key: np.ndarray, value: Any, step: int = -1) -> float:
        if self.n_pool == 0:
            return 0.0                      # 无法验证 -> 拒绝，不是通过
        k = np.asarray(key, dtype=np.float64).ravel()
        P = self.pool_x
        if P.shape[1] != k.shape[0]:
            raise ValueError("维度不匹配: key=%d pool=%d" % (k.shape[0], P.shape[1]))
        nk = float(np.linalg.norm(k))
        np_ = np.linalg.norm(P, axis=1)
        denom = np_ * nk
        sims = np.where(denom > 0, (P @ k) / np.maximum(denom, 1e-12), 0.0)
        order = np.argsort(-sims)[: self.topk]
        idx = [i for i in order if sims[i] >= self.sim_floor]
        if len(idx) < self.min_support:
            return 0.0                      # 邻居支撑不足 -> 不可验证 -> 拒绝
        agree = np.mean([1.0 if self.pool_y[i] == value else 0.0 for i in idx])
        return float(agree)


class ValueOnlyValidator(Validator):
    """那份 AI 生成演示里的验证器，作为反例保留。

    它只问"value 长得像不像库里已有的某个答案"，完全不看 key。
    之所以能通过，是因为同类任务的答案总是同一个标签 —— 猜标签就能全中。
    用它跑出来的曲线，就是"门控形同虚设"的曲线。
    """

    name = "value_only"

    def __init__(self, known_values: Sequence[Any]):
        self.known = list(known_values)

    def score(self, key: np.ndarray, value: Any, step: int = -1) -> float:
        # 只要 value 出现在"已知答案集合"里就算通过；连相似度都不用算
        return 1.0 if value in self.known else 0.0


class NoisyOracleValidator(Validator):
    """包住一个真验证器，按概率 p 翻转它的判断（模拟不可靠 oracle）。

    阶段 2 用它扫"验证器有多可靠"这条曲线：
    p=0 完美 oracle，p=0.5 等于抛硬币，p=1 等于完全反着来。
    """

    name = "noisy_oracle"

    def __init__(self, inner: Validator, noise: float = 0.0, seed: int = 0):
        self.inner = inner
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)

    def extend_pool(self, x: np.ndarray, y: np.ndarray) -> None:
        self.inner.extend_pool(x, y)

    def score(self, key: np.ndarray, value: Any, step: int = -1) -> float:
        s = self.inner.score(key, value, step)
        if self.noise > 0 and self.rng.random() < self.noise:
            # 打了个嗝：把判断反着给
            return 1.0 - s
        return s


class DelayedLabelValidator(Validator):
    """把标签"迟到 k 个任务"这件事建模进来 —— 现实里没人同步给你标好 holdout。

    lag=0 就是阶段 1~4 用的"金色 holdout"（标签立刻可用）；
    lag=1 表示任务 t 的标签要等到任务 t+1 才到，于是第一个任务完全没有 oracle。
    """

    name = "delayed_label"

    def __init__(self, inner: Validator, lag: int = 0):
        self.inner = inner
        self.lag = int(lag)
        self.queue: List[Tuple[int, np.ndarray, np.ndarray]] = []
        self.n_calls = 0

    def extend_pool(self, x: np.ndarray, y: np.ndarray) -> None:
        self.n_calls += 1
        self.queue.append((self.n_calls, x, y))
        while self.queue and self.n_calls - self.queue[0][0] >= self.lag:
            _, xq, yq = self.queue.pop(0)
            self.inner.extend_pool(xq, yq)

    def score(self, key: np.ndarray, value: Any, step: int = -1) -> float:
        return self.inner.score(key, value, step)


class ObservationDelayValidator(Validator):
    """标签按**观测条数**延迟到达（比"按任务延迟"细一级）。

    delay=40 表示第 i 条观测的标签在第 i+40 条观测到达时才可用。
    现实里反馈就是一条一条回来的，不是按"任务"整批回来的 —— 这个验证器让
    "待裁判窗口"变成一个可以连续调节的旋钮。
    """

    name = "observation_delay"

    def __init__(self, inner: Validator, delay: int = 100):
        self.inner = inner
        self.delay = int(delay)
        self.queue: List[Tuple[int, np.ndarray, np.ndarray]] = []
        self.n_seen = 0

    def push(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) == 0:
            return
        self.queue.append((self.n_seen, x, y))
        self.n_seen += len(x)
        while self.queue and self.n_seen - self.queue[0][0] >= self.delay:
            _, xq, yq = self.queue.pop(0)
            self.inner.extend_pool(xq, yq)

    def extend_pool(self, x: np.ndarray, y: np.ndarray) -> None:
        self.push(x, y)

    def score(self, key: np.ndarray, value: Any, step: int = -1) -> float:
        return self.inner.score(key, value, step)


class ConsensusValidator(Validator):
    """零标注方案：一条候选记忆，只有当它和"已经进温备的邻居"多数一致时才被收下。

    它不需要任何带标签数据，所以理论上是最便宜的落地方式。
    但它只能保证**自洽**，不能保证**正确** —— 第一任务是自举进来的，
    如果第一批知识本身就错了，或者错误是自洽的，它会把错误一路放大。
    这个实验就是要量出这个代价有多大。
    """

    name = "consensus"

    def __init__(self, k: int = 8, bootstrap_tasks: int = 1, sim_floor: float = 0.5):
        self.k = k
        self.bootstrap_tasks = bootstrap_tasks
        self.sim_floor = sim_floor
        self.mem = None
        self.task_calls = 0

    def bind(self, mem) -> None:
        self.mem = mem

    def extend_pool(self, x: np.ndarray, y: np.ndarray) -> None:
        self.task_calls += 1

    def score(self, key: np.ndarray, value: Any, step: int = -1) -> float:
        if self.task_calls <= self.bootstrap_tasks or self.mem is None:
            return 1.0                     # 自举：头几个任务无条件信任
        pool = list(self.mem.cold) + list(self.mem.warm)
        if not pool:
            return 0.0
        k = np.asarray(key, dtype=np.float64).ravel()
        sims = np.array([cosine(k, e.key) for e in pool])
        order = np.argsort(-sims)[: self.k]
        idx = [i for i in order if sims[i] >= self.sim_floor]
        if len(idx) < 3:
            return 0.0
        agree = np.mean([1.0 if pool[i].value == value else 0.0 for i in idx])
        return float(agree)


# ---------------------------------------------------------------- 三层记忆

class TieredMemory:
    """冷/温/热三层 + 门控。这是整个项目的核心可复用件。"""

    def __init__(
        self,
        validator: Validator,
        theta: float = 0.7,
        sim_floor: float = 0.5,
        capacity_hot: Optional[int] = 512,
        capacity_warm: Optional[int] = 2000,
        read_order: Sequence[str] = ("warm", "hot", "cold"),
        retention: str = "reservoir",
        seed: int = 0,
    ):
        self.validator = validator
        self.theta = float(theta)
        self.sim_floor = float(sim_floor)
        self.capacity_hot = capacity_hot
        self.capacity_warm = capacity_warm
        self.read_order = tuple(read_order)
        self.retention = retention          # reservoir | newest | score
        self._rng = np.random.default_rng(seed)
        self._n_offered_warm = 0
        self.cold: List[MemoryEntry] = []
        self.warm: List[MemoryEntry] = []
        self.hot: List[MemoryEntry] = []
        self.journal: List[GateReport] = []

    # -------------------------------------------------- 写

    def observe(self, key: np.ndarray, value: Any, tag: str = "", step: int = -1) -> MemoryEntry:
        e = MemoryEntry(key=np.asarray(key, dtype=np.float64).copy(), value=value,
                        tag=tag, step=step, origin="hot")
        self.hot.append(e)
        if self.capacity_hot is not None and len(self.hot) > self.capacity_hot:
            self.hot.pop(0)                 # FIFO 丢最旧的
        return e

    def load_cold(self, entries: Sequence[Tuple[np.ndarray, Any]], tag: str = "cold") -> None:
        """显式写入冷备。门控不许调用这个 —— 这是人工/上游的权限。"""
        for k, v in entries:
            self.cold.append(MemoryEntry(key=np.asarray(k, dtype=np.float64).copy(),
                                         value=v, tag=tag, score=1.0, origin="cold"))

    def consolidate(self, step: int = -1,
                    truth_fn: Optional[Callable[[MemoryEntry], bool]] = None,
                    admit_unverifiable: bool = False) -> GateReport:
        """把 hot 过一遍门控：通过的进 warm，没通过的丢掉，hot 清空。

        truth_fn 仅供事后统计脏/净，门控本身拿不到它。

        admit_unverifiable=True 时，**"验不了"的条目会先被收下（provisional）**，
        等标签到了再由 revalidate() 事后裁判 —— 这是给"延迟标签"用的通道。
        默认 False：验不了就拒（阶段 0~5 的行为，逐位不变）。
        """
        rep = GateReport(step=step, n_candidates=len(self.hot))
        scores: List[float] = []
        survivors: List[MemoryEntry] = []
        rejected: List[MemoryEntry] = []
        for e in self.hot:
            s = float(self.validator.score(e.key, e.value, step))
            e.score = s
            if s <= 0.0:
                rep.n_unverifiable += 1
            if s >= self.theta:
                rep.n_accepted += 1
                scores.append(s)
                survivors.append(e)
            elif admit_unverifiable and s <= 0.0:
                # 标签还没到，先收下等事后裁判
                e.provisional = True
                rep.n_provisional += 1
                survivors.append(e)
            else:
                rep.n_rejected += 1
                rejected.append(e)
        if truth_fn is not None:
            # 事后算账：门控自己的混淆矩阵（门控看不到 truth_fn，纯事后统计）
            rep.accepted_dirty = sum(1 for e in survivors if truth_fn(e))
            rep.accepted_clean = len(survivors) - rep.accepted_dirty
            rep.rejected_dirty = sum(1 for e in rejected if truth_fn(e))
            rep.rejected_clean = len(rejected) - rep.rejected_dirty
        rep.accepted_score_mean = float(np.mean(scores)) if scores else 0.0

        for e in survivors:
            self._offer_warm(e)
        self.hot = []

        rep.cold_after, rep.warm_after, rep.hot_after = len(self.cold), len(self.warm), len(self.hot)
        self.journal.append(rep)
        return rep

    def revalidate(self) -> Dict[str, int]:
        """标签到齐之后重新裁判：把"先收下"的临时条目该撤的撤掉。

        冻结成一条规则：**只有能验证了、并且没通过，才撤回**；
        仍然验不了的继续挂着（下一轮再看）。
        """
        kept: List[MemoryEntry] = []
        evicted = 0
        for e in self.warm:
            if not e.provisional:
                kept.append(e)
                continue
            s = float(self.validator.score(e.key, e.value, e.step))
            if s <= 0.0:
                kept.append(e)                  # 还是验不了 -> 继续等
            elif s >= self.theta:
                e.provisional = False           # 验过了，转正
                e.score = s
                kept.append(e)
            else:
                evicted += 1                    # 能验了、且没过 -> 撤回
        self.warm = kept
        return {"evicted": evicted, "kept": len(kept)}

    def _offer_warm(self, e: MemoryEntry) -> None:
        """把一条经验证的记忆放进温备，并按 retention 策略维持容量。

        默认 reservoir：对"历史上所有通过验证的条目"做均匀采样，
        这样温备和 B 的回放缓冲在容量与分布上完全可比。
        """
        cap = self.capacity_warm
        self._n_offered_warm += 1
        if cap is None or len(self.warm) < cap:
            self.warm.append(e)
            return
        if self.retention == "reservoir":
            j = int(self._rng.integers(0, self._n_offered_warm))
            if j < cap:
                self.warm[j] = e
        elif self.retention == "score":
            worst = min(range(len(self.warm)), key=lambda i: self.warm[i].score)
            if e.score > self.warm[worst].score:
                self.warm[worst] = e
        else:                                # newest
            self.warm.pop(0)
            self.warm.append(e)

    # -------------------------------------------------- 读

    def read(self, key: np.ndarray) -> Optional[Tuple[Any, str, float]]:
        """按 read_order 逐层检索，返回 (value, 层名, 相似度)。"""
        k = np.asarray(key, dtype=np.float64).ravel()
        for tier in self.read_order:
            store = getattr(self, tier)
            best, best_sim = None, -1.0
            for e in store:
                s = cosine(k, e.key)
                if s > best_sim:
                    best, best_sim = e, s
            if best is not None and best_sim >= self.sim_floor:
                best.use_count += 1
                return best.value, tier, best_sim
        return None

    def replay_pool(self, tiers: Sequence[str] = ("warm", "cold")) -> List[MemoryEntry]:
        """可以被拿去回放训练的条目（默认只有经验证的部分）。"""
        out: List[MemoryEntry] = []
        for t in tiers:
            out.extend(getattr(self, t))
        return out

    # -------------------------------------------------- 账

    def stats(self) -> Dict[str, Any]:
        return {
            "cold": len(self.cold),
            "warm": len(self.warm),
            "hot": len(self.hot),
            "rounds": len(self.journal),
            "accepted_total": sum(r.n_accepted for r in self.journal),
            "rejected_total": sum(r.n_rejected for r in self.journal),
        }
