# 稳定事实（唯一真源）

格式：- `key` = `value`  <-- measured_at: YYYY-MM-DD, n: 样本数, 说明: 中文描述

中文说明是必须的：查询是中文，只靠英文 key 检索不到。

- `qa130_accuracy` = `0.800`  <-- measured_at: 2026-10-18, n: 110, 说明: 知识问答基准的可答准确率 正确率
- `qa130_abstain` = `0.85`  <-- measured_at: 2026-10-18, n: 20, 说明: 不可答问题时的弃答率 说不知道的比例
- `cascade_pass_at_1` = `0.738`  <-- measured_at: 2026-10-17, n: 164, 说明: 3B 到 9B 级联后的 pass@1 代码通过率
- `base_3b_humaneval` = `0.677`  <-- measured_at: 2026-10-17, n: 164, 说明: 3B 单跑的 HumanEval 代码通过率 基线
- `big_9b_q4_humaneval` = `0.768`  <-- measured_at: 2026-10-17, n: 164, 说明: 9B Q4 单跑 HumanEval 代码通过率
- `big_9b_iq3_humaneval` = `0.756`  <-- measured_at: 2026-10-17, n: 164, 说明: 9B IQ3 量化版单跑 HumanEval 代码通过率
- `co_resident_vram_mib` = `7481`  <-- measured_at: 2026-10-17, n: 1, 说明: 两个模型共驻留时的显存占用 MiB
- `soak_6h_p50_sec` = `0.1250`  <-- measured_at: 2026-10-18, n: 121428, 说明: 6 小时压测的 P50 延迟 中位延迟 秒
- `soak_6h_p95_sec` = `0.1745`  <-- measured_at: 2026-10-18, n: 121428, 说明: 6 小时压测的 P95 延迟 秒
- `soak_6h_requests` = `121428`  <-- measured_at: 2026-10-18, n: 1, 说明: 6 小时压测的请求总数 条数
- `soak_6h_errors` = `4`  <-- measured_at: 2026-10-18, n: 121428, 说明: 6 小时压测的异常次数 错误数
- `moe_max_tps` = `3.4`  <-- measured_at: 2026-10-17, n: 1, 说明: MoE 模型用满显存后的每秒 token 数 速度
- `dense_3b_tps` = `66`  <-- measured_at: 2026-10-17, n: 1, 说明: 3B 稠密模型的每秒 token 数 速度
- `ctx_ok_tokens` = `20003`  <-- measured_at: 2026-10-17, n: 1, 说明: 3B 实测正常的上下文长度 token 数
- `bge_recall` = `0.87`  <-- measured_at: 2026-10-18, n: 15, 说明: bge 向量检索在改写查询上的召回率
- `doc_vec_recall` = `0.50`  <-- measured_at: 2026-10-18, n: 10, 说明: 真实文档纯向量检索的召回率
- `doc_hybrid_recall` = `0.70`  <-- measured_at: 2026-10-18, n: 10, 说明: 真实文档混合检索（向量加关键词）的召回率
- `gate_precision_real_noise` = `0.000`  <-- measured_at: 2026-10-18, n: 30, 说明: 真实噪声下门控记忆的精度 precision
- `intent_map_accuracy` = `0.81`  <-- measured_at: 2026-10-18, n: 16, 说明: 意图映射（问题到指标）的准确率
- `concurrency_max_ok` = `8`  <-- measured_at: 2026-10-18, n: 24, 说明: 服务能承受的最大并发数 零失败
