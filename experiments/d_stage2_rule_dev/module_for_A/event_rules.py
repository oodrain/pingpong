# -*- coding: utf-8 -*-
"""D 组触台事件模块 v1（stage-2 候选栈定稿版，交接给 A 集成）。

把 D 阶段二的规则改进封装为独立模块：输入逐帧球轨迹，输出触台事件。

输入接口（与 raw/ 缓存格式一致，逐输入帧一条）：
    trajectory = [
        {"input_frame_id": int,        # 检测器/热图帧号（含官方 1 帧滞后语义）
         "detected": bool,
         "x": float | None, "y": float | None,
         "confidence": float | None},
        ...
    ]
    轨迹点达标条件：detected 且 confidence >= CONF_TH(0.75)。

输出接口：
    [{"frame_id": int, "x": float, "y": float}, ...]   # 按时间序
    frame_id 已含官方 fid-1 补偿（= input_frame_id - 1），不要再减 1。

可选桌面几何（有标注时启用投影门 + 图像四边形双判 + 固定范围门）：
    table = {"corners": [[x,y]*4],         # 按标注工具的规范化顺序（最左上起顺时针）
             "net": [[x,y],[x,y]] | None}  # 用于长短轴自动定向；缺省时用默认定向
    无 table 或配置关闭 gate 时，门相关逻辑自动跳过（纯规则部分）。

依赖：仅标准库；启用 table 时需 cv2 + numpy（懒加载）。

已验证：输出与 rule_v2_dev/m12_eval.py 的 replay_m12（候选栈全配置）逐条一致
（公开 15：F1 0.744；154 候选口径：0.658）。见同目录 verify_module.py。
"""
from __future__ import annotations

import math

CONF_TH = 0.75
MIN_GAP = 10

DEFAULT_CONFIG = {
    # —— M1.1 上下文特征 ——
    "F1": True,           # 强回升恢复：rise1>=8 且后续持续上升
    "F2": True,           # 宽窗幅度：±2 窗 fall2>=2 且 rise2>=2
    "F3": True,           # 强候选破间隔（被 MIN_GAP 拦截时按守卫放行）
    # —— M1.2 急减速回溯修正 ——
    "D2": True, "D2_back": 5, "D_smin": 8.0, "D_ratio": 0.40,
    "D_smax": 50.0, "D_dmin": 2,
    # —— 几何门（需 table 才生效）——
    "gate": 900.0,        # 投影出桌距离上限（mm）
    "gate_quad": 4.0,     # 图像四边形外扩像素（双判通道）
    # —— 固定范围门（2026-10-17，复现 C 的固定球桌范围实验）——
    "range_frac": 0.07,   # 事件点离桌面四边形 ≤ max(frac×对角线, min_px) 才保留
    "range_min_px": 4.0,  # 容差像素下限；range_frac=0 关闭该门
    # —— M1.3' 远侧掩蔽触台（上升速度浅坑）——
    "D3": True, "D3_ratio": 0.45, "D3_min": 3.0,
}


def _eligible(trajectory):
    return [(r["input_frame_id"], float(r["x"]), float(r["y"])) for r in trajectory
            if r.get("detected") and r.get("confidence") is not None and r["confidence"] >= CONF_TH]


def _build_H(corners, net=None):
    """桌面单应：图像四角 -> 桌面 mm（2740x1525）。长/宽轴方向用网段映射后 x 跨度最小自动判定。"""
    import cv2
    import numpy as np
    TABLE_L, TABLE_W = 2740.0, 1525.0
    c = np.array(corners, np.float32)
    dst_a = np.array([[0, 0], [TABLE_L, 0], [TABLE_L, TABLE_W], [0, TABLE_W]], np.float32)
    dst_b = np.array([[0, 0], [TABLE_W, 0], [TABLE_W, TABLE_L], [0, TABLE_L]], np.float32)
    best = None
    for dst in (dst_a, dst_b):
        H = cv2.getPerspectiveTransform(c, dst)
        score = 0.0
        if net and len(net) == 2:
            pts = cv2.perspectiveTransform(np.array(net, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
            score = float(abs(pts[0][0] - pts[1][0]))
        if best is None or score < best[0]:
            best = (score, H)
    return best[1]


def detect_events(trajectory, config=None, table=None):
    """核心入口：逐帧轨迹 -> 触台事件列表。逻辑说明见模块 docstring 与交接文档。"""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    traj = _eligible(trajectory)
    n = len(traj)
    out = []
    if n < 3:
        return out

    # 桌面几何（可选）
    H = quad = None
    if table and cfg.get("gate"):
        corners = table.get("corners")
        if not corners or len(corners) != 4:
            raise ValueError("table['corners'] 需要 4 个点")
        H = _build_H(corners, table.get("net"))
        quad = corners

    last_fid, last_xy, last_d3_fid = -10_000, None, -10_000

    def gate_pass(x, y):
        if not cfg.get("gate") or H is None:
            return True
        if quad is not None and cfg.get("gate_quad", 0.0) > 0:
            import cv2
            import numpy as np
            dist = cv2.pointPolygonTest(np.array(quad, np.float32).reshape(-1, 1, 2),
                                        (float(x), float(y)), True)
            if dist >= -cfg["gate_quad"]:
                return True
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)[0, 0]
        u, v = float(p[0]), float(p[1])
        g = cfg["gate"]
        return (-g <= u <= 2740.0 + g and -g <= v <= 1525.0 + g)

    def d3_trigger(k):
        if k < 2 or k + 1 >= n:
            return False
        if traj[k + 1][0] - traj[k][0] > 2 or traj[k][0] - traj[k - 1][0] > 2 \
                or traj[k - 1][0] - traj[k - 2][0] > 2:
            return False
        if not (traj[k][2] < traj[k - 1][2] and traj[k + 1][2] < traj[k][2]):
            return False
        rho_in = traj[k - 1][2] - traj[k][2]
        rho_prev = traj[k - 2][2] - traj[k - 1][2]
        rho_out = traj[k][2] - traj[k + 1][2]
        r = cfg.get("D3_ratio", 0.45)
        return (rho_prev >= cfg.get("D3_min", 3.0) and rho_out >= cfg.get("D3_min", 3.0)
                and rho_in <= r * rho_prev and rho_in <= r * rho_out)

    def scan(j):
        nonlocal last_fid, last_xy, last_d3_fid
        fid, x, y = traj[j]
        # D3 附加：远侧掩蔽触台（不改主扫描 MIN_GAP 状态）
        if cfg.get("D3") and d3_trigger(j) and fid - last_fid >= MIN_GAP and fid - last_d3_fid >= MIN_GAP:
            if gate_pass(x, y):
                out.append({"frame_id": max(0, fid - 1), "x": x, "y": y})
                last_d3_fid = fid
        if not (y > traj[j - 1][2] and y >= traj[j + 1][2]):
            return
        fall1 = y - traj[j - 1][2]
        rise1 = y - traj[j + 1][2]
        fall2 = y - traj[j - 2][2] if j >= 2 else fall1
        rise2 = y - traj[j + 2][2] if j + 2 < n else rise1
        ampl_base = (fall1 >= 2.0 and rise1 >= 1.0)
        ampl = ampl_base
        if not ampl and cfg.get("F2") and fall2 >= 2.0 and rise2 >= 2.0:
            ampl = True
        if not ampl and cfg.get("F1") and rise1 >= 8.0 and (j + 2 >= n or traj[j + 2][2] <= traj[j + 1][2]):
            ampl = True
        if not ampl:
            return
        # D2 回溯修正
        j_emit = j
        if cfg.get("D2"):
            for k in range(j - 1, max(0, j - 1 - cfg.get("D2_back", 5)), -1):
                if j - k < cfg.get("D_dmin", 1):
                    continue
                if k < 1 or k + 1 >= n:
                    continue
                if traj[k][2] <= traj[k - 1][2]:
                    continue
                s_prev = traj[k][2] - traj[k - 1][2]
                s_at = traj[k + 1][2] - traj[k][2]
                if (cfg.get("D_smin", 8.0) <= s_prev <= cfg.get("D_smax", 1e9) and s_at > 0
                        and s_at <= cfg.get("D_ratio", 0.45) * s_prev
                        and traj[k + 1][0] - traj[k][0] <= 2):
                    j_emit = k
                    break
        efid, ex, ey = traj[j_emit]
        if not gate_pass(ex, ey):
            return
        blocked = efid - last_fid < MIN_GAP
        if blocked:
            if not cfg.get("F3"):
                return
            sf1 = ey - traj[j_emit - 1][2] if j_emit >= 1 else fall1
            sr1 = ey - traj[j_emit + 1][2] if j_emit + 1 < n else rise1
            sf2 = ey - traj[j_emit - 2][2] if j_emit >= 2 else sf1
            sr2 = ey - traj[j_emit + 2][2] if j_emit + 2 < n else sr1
            strong = ((sf1 >= 3.0 and sr1 >= 3.0) or (sf2 >= 3.0 and sr2 >= 5.0)
                      or (sr1 >= 8.0 and sf1 >= 0.5))
            far_ok = last_xy is not None and math.dist((ex, ey), last_xy) >= 100.0
            jump_ok = True
            for k in range(max(1, j_emit - 2), min(n - 1, j_emit + 2)):
                if math.dist((traj[k][1], traj[k][2]), (traj[k + 1][1], traj[k + 1][2])) > 120.0:
                    jump_ok = False
                    break
            gap_ok = ((j_emit < 2 or traj[j_emit][0] - traj[j_emit - 1][0] <= 2)
                      and (traj[j_emit + 1][0] - traj[j_emit][0] <= 2)
                      and (j_emit + 2 >= n or traj[j_emit + 2][0] - traj[j_emit + 1][0] <= 2))
            if not (strong and far_ok and jump_ok and gap_ok):
                return
        if j_emit != j:
            out.append({"frame_id": max(0, efid - 1), "x": ex, "y": ey})
        elif ampl_base and not blocked:
            out.append({"frame_id": max(0, fid - 1), "x": x, "y": y})
        else:
            denom = fall1 + rise1
            frac = fall1 / denom if denom > 0.5 else 0.5
            frac = min(1.0, max(0.0, frac))
            x2, y2 = traj[j + 1][1], traj[j + 1][2]
            out.append({"frame_id": max(0, int(round((fid - 1) + frac))),
                        "x": x + frac * (x2 - x), "y": y + frac * (y2 - y)})
        last_fid, last_xy = efid, (ex, ey)

    # 两遍扫描（等价官方逐帧回放：pass1 扫 j<=n-3，pass2 只扫 j=n-2）
    for j in range(1, n - 2):
        scan(j)
    scan(n - 2)

    # 固定范围门（后处理）：C 的 TT3D-TCN 分支同款，清除"悬在台面上方、投影仍在台内"的
    # 击球类 FP；失效放行——无 table / range_frac=0 时不生效。
    if table and quad is not None and cfg.get("gate") and cfg.get("range_frac", 0) > 0:
        import cv2
        import numpy as np
        qa = np.array(quad, np.float32).reshape(-1, 1, 2)
        p4 = np.array(quad, float)
        diag = max(math.dist(p4[i], p4[j]) for i in range(4) for j in range(i + 1, 4))
        tol = max(cfg["range_frac"] * diag, cfg.get("range_min_px", 4.0))
        out = [e for e in out
               if cv2.pointPolygonTest(qa, (float(e["x"]), float(e["y"])), True) >= -tol]
    return out


if __name__ == "__main__":
    # 简易自测：python event_rules.py <raw.jsonl> [annotations.json]
    import json
    import sys
    raw_path = sys.argv[1]
    rows = [json.loads(l) for l in open(raw_path, encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["input_frame_id"])
    table = None
    if len(sys.argv) > 2:
        doc = json.loads(open(sys.argv[2], encoding="utf-8").read())
        seg = doc["segments"][0]
        table = {"corners": seg["corners"], "net": seg.get("net")}
    events = detect_events(rows, table=table)
    print(json.dumps(events, ensure_ascii=False))
    print(f"# 共 {len(events)} 个事件", file=sys.stderr)
