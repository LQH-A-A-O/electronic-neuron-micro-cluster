# -*- coding: utf-8 -*-
"""MoE 吞吐的带宽解释：把实测数字折成"有效权重带宽"，看瓶颈到底在哪。"""
G = lambda v: v / 1e9
print("%-30s %-10s %-12s %-14s %s" % ("方案", "激活", "每token权重", "有效带宽", "显存? "))
rows = [
    ("1.5B 稠密 Q4_K_M",       1.5, 4.5, 107.0, "全进显存"),
    ("3B 稠密 Q4_K_M",         3.0, 4.5,  66.0, "全进显存"),
    ("35B MoE A3B IQ3_XXS",    3.0, 3.5,  11.7, "专家在内存"),
    ("7B MoE A1.3B Q4 (推算)", 1.3, 4.5,  None, "全进显存"),
]
for name, act, bpw, tps, where in rows:
    b = act * 1e9 * bpw / 8
    bw = ("%.1f GB/s" % G(b * tps)) if tps else "—"
    tps_s = ("%.1f" % tps) if tps else "待测"
    print("%-30s %-10s %-12s %-14s %s" % (name, "%.1fB" % act, "%.2f GB" % G(b), bw, where))
print()
print("参考带宽：RTX 4070 Laptop GDDR6 ≈ 256 GB/s（理论）；笔记本 DDR5 双通道 ≈ 76 GB/s（理论）")
print()
print("要 100 tok/s 且激活 3B，需要 3.0e9*4.5/8*100 = %.0f GB/s —— 超出内存带宽，必须全部在显存里。" % (3.0e9*4.5/8*100/1e9))
