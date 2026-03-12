"""
plot12_er_evolution_video.py
============================
Cinematic Manim animation of the Erdős–Rényi random graph evolution
and phase transition.

Render:
    manim -ql  plot12_er_evolution_video.py Plot12ErdosRenyiEvolution
    manim -qm  plot12_er_evolution_video.py Plot12ErdosRenyiEvolution
    manim -qh  plot12_er_evolution_video.py Plot12ErdosRenyiEvolution

What this scene shows:
    Act 1 – Isolated nodes  (p ≈ 0)
    Act 2 – Fragmented clusters form
    Act 3 – Critical window  (p ≈ 1/n), giant component emerges
    Act 4 – Post-transition growth up to p ≈ log(n)/n  (connectivity)

Visual language:
    - Spring-layout (organic, well-spaced)
    - Tiny node dots (r=0.03) with faint halo (r=0.07)
    - New edges: gold flash via Create(), then cool to slate/teal
    - Giant component: teal glow
    - Live HUD with p, ⟨k⟩, S  + mini-plot of S(p)
    - Subtle camera drift during Acts 3–4
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import networkx as nx
import numpy as np
from manim import *

# ── Palette ─────────────────────────────────────────────────────────────────
BG        = "#020617"
NAVY      = "#1A3A5C"
TEAL      = "#0E7490"
RED       = "#DC2626"
GOLD      = "#D97706"
SLATE     = "#64748B"
LIGHT     = "#F1F5F9"
PURPLE    = "#7C3AED"
GREEN     = "#059669"
ROSE      = "#BE185D"
GREY_NODE = "#94A3B8"
GREY_EDGE = "#475569"
GREY_DIM  = "#94A3B8"
ACT2_COLORS = [TEAL, PURPLE, GREEN, ROSE, RED, GOLD]


# ── Snapshot dataclass ──────────────────────────────────────────────────────
@dataclass
class Snapshot:
    act: int
    p: float
    mean_degree: float
    giant_fraction: float
    edges: list[tuple[int, int]]
    new_edges: list[tuple[int, int]]
    components: list[list[int]]
    giant_nodes: set[int]
    is_connected: bool


# ── Network helpers (pre-computation) ───────────────────────────────────────
def _edge_schedule(n: int, seed: int) -> list[tuple[int, int, float]]:
    """Assign a U(0,1) threshold to every possible edge for monotone coupling."""
    rng = random.Random(seed)
    sched = [(u, v, rng.random()) for u in range(n) for v in range(u + 1, n)]
    sched.sort(key=lambda x: x[2])
    return sched


def _pick_good_seed(
    n: int, target_p: float, base: int = 42
) -> tuple[list[tuple[int, int, float]], int]:
    """Find a seed whose graph is connected by *target_p*."""
    for seed in range(base, base + 500):
        sched = _edge_schedule(n, seed)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, t in sched:
            if t > target_p:
                break
            G.add_edge(u, v)
        if nx.is_connected(G):
            return sched, seed
    return _edge_schedule(n, base), base


def build_snapshots(
    n: int, p_plan: list[tuple[int, float]], seed: int = 42
) -> tuple[list[Snapshot], int]:
    conn_p = math.log(n) / n
    sched, actual_seed = _pick_good_seed(n, conn_p, base=seed)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    snaps: list[Snapshot] = []
    ptr = 0
    prev_edges: set[tuple[int, int]] = set()

    for act, p in p_plan:
        while ptr < len(sched) and sched[ptr][2] <= p:
            G.add_edge(sched[ptr][0], sched[ptr][1])
            ptr += 1

        edges_now = {tuple(sorted(e)) for e in G.edges()}
        new = sorted(edges_now - prev_edges)
        prev_edges = set(edges_now)

        comps = sorted(
            (sorted(c) for c in nx.connected_components(G)),
            key=len,
            reverse=True,
        )
        giant = set(comps[0]) if comps else set()

        snaps.append(
            Snapshot(
                act=act,
                p=p,
                mean_degree=2.0 * len(edges_now) / n,
                giant_fraction=len(giant) / n,
                edges=sorted(edges_now),
                new_edges=new,
                components=comps,
                giant_nodes=giant,
                is_connected=(len(comps) == 1),
            )
        )
    return snaps, actual_seed


def compute_spring_positions(
    n: int,
    final_edges: list[tuple[int, int]],
    seed: int,
    graph_center: np.ndarray,
) -> dict[int, np.ndarray]:
    """nx.spring_layout on the *final* connected graph, aggressively spaced."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in final_edges:
        G.add_edge(*e)

    pos2d = nx.spring_layout(G, k=0.25, iterations=200, seed=seed, scale=3.5)

    return {
        nd: np.array([x + graph_center[0], y + graph_center[1], 0.0])
        for nd, (x, y) in pos2d.items()
    }


# ── Color / style helpers ──────────────────────────────────────────────────
def _node_colors(snap: Snapshot) -> dict[int, str]:
    cmap: dict[int, str] = {}
    if snap.act <= 2:
        ci = 0
        for comp in snap.components:
            if len(comp) == 1:
                for nd in comp:
                    cmap[nd] = GREY_NODE
            else:
                col = ACT2_COLORS[ci % len(ACT2_COLORS)]
                ci += 1
                for nd in comp:
                    cmap[nd] = col
    else:
        for comp in snap.components:
            col = TEAL if set(comp) == snap.giant_nodes else GREY_NODE
            for nd in comp:
                cmap[nd] = col
    return cmap


def _edge_style(snap: Snapshot, u: int, v: int) -> tuple[str, float, float]:
    """Return (color, stroke_width, stroke_opacity) for a settled edge."""
    if snap.act <= 2:
        return SLATE, 1.0, 0.40
    if u in snap.giant_nodes and v in snap.giant_nodes:
        return TEAL, 1.5, 0.65
    return GREY_EDGE, 0.8, 0.25


# ════════════════════════════════════════════════════════════════════════════
# SCENE
# ════════════════════════════════════════════════════════════════════════════
class Plot12ErdosRenyiEvolution(MovingCameraScene):
    def construct(self) -> None:
        self.camera.background_color = BG

        n = 100
        pc = 1.0 / n
        conn_p = math.log(n) / n

        # ── Evolution plan ──────────────────────────────────────────────
        act1 = [(1, p) for p in [0.0, 0.0002, 0.0005, 0.0008, 0.0010]]
        act2 = [(2, float(p)) for p in np.linspace(0.0015, 0.009, 14)]
        act3 = [(3, float(p)) for p in np.linspace(0.0095, 0.020, 16)]
        act4 = [(4, float(p)) for p in np.linspace(0.022, 0.050, 14)]
        p_plan = act1 + act2 + act3 + act4

        snapshots, seed = build_snapshots(n, p_plan, seed=42)

        # ── Organic spring layout from the FINAL graph ──────────────────
        GRAPH_CENTER = np.array([-1.6, 0.0, 0.0])
        pos = compute_spring_positions(n, snapshots[-1].edges, seed, GRAPH_CENTER)

        # ── Value trackers ──────────────────────────────────────────────
        p_vt = ValueTracker(0.0)
        k_vt = ValueTracker(0.0)
        s_vt = ValueTracker(0.0)

        # ── Title ───────────────────────────────────────────────────────
        title = Text("Erdős–Rényi Evolution", color=LIGHT, weight=BOLD, font_size=32)
        subtitle = Text(
            "Giant component emerging in real time",
            color=GREY_DIM,
            font_size=17,
        )
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.12)
        header.to_edge(UP, buff=0.25)
        seed_info = Text(
            f"n = {n}   seed = {seed}   p_c = 1/n ≈ {pc:.4f}",
            color=SLATE,
            font_size=12,
        ).next_to(header, DOWN, buff=0.08)

        # ── Nodes (tiny cores + faint halos) ────────────────────────────
        node_mobs: dict[int, VGroup] = {}
        nodes_vg = VGroup()
        for i in range(n):
            halo = Dot(
                pos[i], radius=0.07, color=GREY_NODE,
                fill_opacity=0.10, stroke_width=0,
            )
            core = Dot(
                pos[i], radius=0.03, color=GREY_NODE,
                fill_opacity=1.0, stroke_width=0,
            )
            grp = VGroup(halo, core)
            grp.set_z_index(10)
            node_mobs[i] = grp
            nodes_vg.add(grp)

        # ── Edge bookkeeping ────────────────────────────────────────────
        edge_mobs: dict[tuple[int, int], Line] = {}

        # ── HUD (upper-right, dark semi-transparent) ────────────────────
        hud_bg = RoundedRectangle(
            corner_radius=0.14, width=4.5, height=2.0,
            stroke_color="#1E293B", stroke_width=0.8,
        ).set_fill(BG, opacity=0.90)
        hud_bg.to_corner(UR, buff=0.25).shift(DOWN * 0.35)

        act_label = Text(
            "Act 1 – Isolated Nodes",
            color=LIGHT, font_size=16, weight=BOLD,
        )
        act_label.move_to(hud_bg.get_top() + DOWN * 0.25)

        def _metric_row(
            label: str, tracker: ValueTracker, color: str, dec: int
        ) -> VGroup:
            lab = Text(label, color=GREY_DIM, font_size=16)
            val = DecimalNumber(0, num_decimal_places=dec, color=color, font_size=18)
            val.add_updater(lambda m, t=tracker: m.set_value(t.get_value()))
            return VGroup(lab, val).arrange(RIGHT, buff=0.10, aligned_edge=DOWN)

        metrics = VGroup(
            _metric_row("p =", p_vt, GOLD, 4),
            _metric_row("⟨k⟩ =", k_vt, LIGHT, 2),
            _metric_row("S =", s_vt, TEAL, 3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.09)
        metrics.next_to(act_label, DOWN, buff=0.15)
        metrics.align_to(hud_bg, LEFT).shift(RIGHT * 0.28)

        # ── Mini-plot panel ─────────────────────────────────────────────
        plot_bg = RoundedRectangle(
            corner_radius=0.14, width=4.5, height=2.8,
            stroke_color="#1E293B", stroke_width=0.8,
        ).set_fill(BG, opacity=0.90)
        plot_bg.next_to(hud_bg, DOWN, buff=0.15).align_to(hud_bg, RIGHT)

        plot_title = Text(
            "Giant fraction  S(p)", color=LIGHT, font_size=14, weight=BOLD,
        )
        plot_title.move_to(plot_bg.get_top() + DOWN * 0.20)

        axes = Axes(
            x_range=[0, 0.055, 0.01],
            y_range=[0, 1.0, 0.2],
            x_length=3.6,
            y_length=2.0,
            axis_config={
                "color": GREY_DIM,
                "stroke_width": 1.0,
                "include_tip": False,
            },
            tips=False,
        )
        axes.move_to(plot_bg.get_center() + DOWN * 0.20)

        ax_labels = VGroup(
            Text("p", color=GREY_DIM, font_size=11).next_to(
                axes.x_axis, RIGHT, buff=0.06
            ),
            Text("S", color=GREY_DIM, font_size=11).next_to(
                axes.y_axis, UP, buff=0.06
            ),
        )

        crit_dash = DashedLine(
            axes.c2p(pc, 0), axes.c2p(pc, 1),
            color=GOLD, stroke_width=1.2,
        )
        crit_lbl = Text("1/n", color=GOLD, font_size=10).next_to(
            crit_dash, UP, buff=0.05
        )
        conn_dash = DashedLine(
            axes.c2p(conn_p, 0), axes.c2p(conn_p, 1),
            color=RED, stroke_width=1.0,
        )
        conn_lbl = Text("ln n/n", color=RED, font_size=10).next_to(
            conn_dash, UP, buff=0.05
        )

        # Live S(p) curve data
        curve_ps: list[float] = [0.0]
        curve_ss: list[float] = [0.0]

        def _build_curve() -> VMobject:
            if len(curve_ps) < 2:
                return VMobject()
            pts = [axes.c2p(p, s) for p, s in zip(curve_ps, curve_ss)]
            c = VMobject(color=TEAL, stroke_width=2.4)
            c.set_points_smoothly(pts)
            return c

        live_curve = _build_curve()
        live_dot = always_redraw(
            lambda: Dot(
                axes.c2p(p_vt.get_value(), s_vt.get_value()),
                radius=0.04, color=TEAL,
            )
        )

        # ── Bottom banner ───────────────────────────────────────────────
        banner_bg = RoundedRectangle(
            corner_radius=0.16, width=5.8, height=0.65, stroke_width=0,
        ).set_fill("#0F172A", opacity=0.90)
        banner_bg.to_edge(DOWN, buff=0.15)
        banner_txt = Text(
            "Act 1 – Isolated Nodes",
            color=WHITE, font_size=22, weight=BOLD,
        )
        banner_txt.move_to(banner_bg)

        # ── Gather HUD into a group for camera tracking ─────────────────
        hud_group = VGroup(
            hud_bg, act_label, metrics,
            plot_bg, plot_title, axes, ax_labels,
            crit_dash, crit_lbl, conn_dash, conn_lbl,
            live_curve, live_dot,
        )
        banner_group = VGroup(banner_bg, banner_txt)
        header_group = VGroup(header, seed_info)

        # Pin UI to camera frame so camera drift doesn't displace them
        _initial_frame_center = self.camera.frame.get_center().copy()
        _initial_frame_width = self.camera.frame.width

        def _pin_to_frame(grp: VGroup, original_center: np.ndarray):
            """Updater: translate *grp* by whatever the camera has drifted."""
            delta = self.camera.frame.get_center() - _initial_frame_center
            scale = self.camera.frame.width / _initial_frame_width
            grp.move_to(original_center * scale + delta)

        hud_center = hud_group.get_center().copy()
        banner_center = banner_group.get_center().copy()
        header_center = header_group.get_center().copy()

        hud_group.add_updater(lambda m: _pin_to_frame(m, hud_center))
        banner_group.add_updater(lambda m: _pin_to_frame(m, banner_center))
        header_group.add_updater(lambda m: _pin_to_frame(m, header_center))

        # ════════════════════════════════════════════════════════════════
        # ANIMATION
        # ════════════════════════════════════════════════════════════════

        # -- Intro: appear nodes --
        self.add(header_group)
        self.play(
            LaggedStart(
                *[FadeIn(node_mobs[i], scale=0.5) for i in range(n)],
                lag_ratio=0.008,
            ),
            run_time=3.5,
        )
        self.wait(1.0)

        # -- Show HUD + banner --
        self.play(
            FadeIn(banner_group),
            FadeIn(hud_group),
            run_time=1.8,
        )
        self.play(FadeOut(seed_info), run_time=0.4)
        self.wait(0.5)

        # ── Act config ──────────────────────────────────────────────────
        ACT_NAMES = {
            1: "Act 1 – Isolated Nodes",
            2: "Act 2 – Small Clusters",
            3: "Act 3 – Phase Transition",
            4: "Act 4 – Supercritical Growth",
        }
        STEP_RT = {1: 2.8, 2: 2.2, 3: 2.0, 4: 1.8}
        HOLD_T = {1: 0.5, 2: 0.3, 3: 0.3, 4: 0.3}

        prev_snap = snapshots[0]
        prev_new_set: set[tuple[int, int]] = set()
        prev_ncols = _node_colors(snapshots[0])

        for snap in snapshots[1:]:
            # ── Act transition ──────────────────────────────────────────
            if snap.act != prev_snap.act:
                nb = Text(
                    ACT_NAMES[snap.act], color=WHITE,
                    font_size=22, weight=BOLD,
                )
                nb.move_to(banner_txt)
                na = Text(
                    ACT_NAMES[snap.act], color=LIGHT,
                    font_size=16, weight=BOLD,
                )
                na.move_to(act_label)
                self.play(
                    Transform(banner_txt, nb),
                    Transform(act_label, na),
                    run_time=0.8,
                )

                if snap.act == 3:
                    badge = VGroup(
                        RoundedRectangle(
                            corner_radius=0.14, width=3.8, height=0.6,
                            stroke_width=0,
                        ).set_fill(GOLD, 0.88),
                        Text(
                            "CRITICAL POINT: p = 1/n",
                            color=WHITE, font_size=16, weight=BOLD,
                        ),
                    )
                    badge[1].move_to(badge[0])
                    badge.move_to(GRAPH_CENTER + UP * 3.0)
                    self.play(FadeIn(badge, scale=0.9), run_time=0.8)
                    self.wait(1.2)
                    self.play(FadeOut(badge), run_time=0.5)
                    # Begin gentle camera zoom-out
                    self.play(
                        self.camera.frame.animate.scale(1.06),
                        run_time=2.0,
                        rate_func=smooth,
                    )

                if snap.act == 4:
                    badge = VGroup(
                        RoundedRectangle(
                            corner_radius=0.14, width=4.2, height=0.6,
                            stroke_width=0,
                        ).set_fill(RED, 0.88),
                        Text(
                            "CONNECTIVITY: p ≈ ln(n)/n",
                            color=WHITE, font_size=16, weight=BOLD,
                        ),
                    )
                    badge[1].move_to(badge[0])
                    badge.move_to(GRAPH_CENTER + UP * 3.0)
                    self.play(FadeIn(badge, scale=0.9), run_time=0.7)
                    self.wait(1.0)
                    self.play(FadeOut(badge), run_time=0.5)
                    # Second camera drift
                    self.play(
                        self.camera.frame.animate.scale(1.04),
                        run_time=1.5,
                        rate_func=smooth,
                    )

                self.wait(0.3)

            # ── Build animation list ────────────────────────────────────
            anims: list[Animation] = [
                p_vt.animate.set_value(snap.p),
                k_vt.animate.set_value(snap.mean_degree),
                s_vt.animate.set_value(snap.giant_fraction),
            ]

            new_set = set(snap.new_edges)

            # New edges → gold Create
            for u, v in snap.new_edges:
                ln = Line(
                    pos[u], pos[v],
                    color=GOLD, stroke_width=2.0, stroke_opacity=0.90,
                )
                ln.set_z_index(1)
                edge_mobs[(u, v)] = ln
                anims.append(Create(ln, rate_func=rush_from))

            # Cool previous step's gold edges → resting palette
            for u, v in prev_new_set:
                ln = edge_mobs.get((u, v))
                if ln is not None:
                    c, w, o = _edge_style(snap, u, v)
                    anims.append(
                        ln.animate.set_stroke(color=c, width=w, opacity=o)
                    )

            # Edges whose style changed (e.g. absorbed into giant)
            for u, v in snap.edges:
                if (u, v) in new_set or (u, v) in prev_new_set:
                    continue
                c_new, w_new, o_new = _edge_style(snap, u, v)
                c_old, _, _ = _edge_style(prev_snap, u, v)
                if c_new != c_old:
                    ln = edge_mobs.get((u, v))
                    if ln is not None:
                        anims.append(
                            ln.animate.set_stroke(
                                color=c_new, width=w_new, opacity=o_new
                            )
                        )

            # Node color deltas only
            ncols = _node_colors(snap)
            for nd in range(n):
                nc = ncols.get(nd, GREY_NODE)
                oc = prev_ncols.get(nd, GREY_NODE)
                if nc != oc:
                    anims.append(node_mobs[nd].animate.set_color(nc))

            # Live mini-plot curve
            curve_ps.append(snap.p)
            curve_ss.append(snap.giant_fraction)
            anims.append(Transform(live_curve, _build_curve()))

            # Subtle continuous camera drift in Acts 3–4
            if snap.act >= 3:
                anims.append(self.camera.frame.animate.scale(1.002))

            # ── Play the step ───────────────────────────────────────────
            self.play(*anims, run_time=STEP_RT[snap.act], rate_func=smooth)

            # ── Milestone callouts ──────────────────────────────────────
            if prev_snap.giant_fraction < 0.40 <= snap.giant_fraction:
                lbl = Text(
                    "GIANT COMPONENT", color=TEAL,
                    font_size=20, weight=BOLD,
                )
                lbl.move_to(GRAPH_CENTER + UP * 3.2)
                self.play(FadeIn(lbl, shift=UP * 0.1), run_time=0.4)
                self.wait(1.0)
                self.play(FadeOut(lbl), run_time=0.3)

            if (
                not prev_snap.is_connected
                and snap.is_connected
                and snap.p >= conn_p * 0.9
            ):
                lbl = Text(
                    "Graph is connected!", color=RED,
                    font_size=20, weight=BOLD,
                )
                lbl.move_to(GRAPH_CENTER + UP * 3.2)
                self.play(FadeIn(lbl, shift=UP * 0.1), run_time=0.4)
                self.wait(1.0)
                self.play(FadeOut(lbl), run_time=0.3)

            self.wait(HOLD_T[snap.act])

            prev_snap = snap
            prev_new_set = new_set
            prev_ncols = ncols

        # ── Ending ──────────────────────────────────────────────────────
        end_bg = RoundedRectangle(
            corner_radius=0.16, width=6.2, height=0.8, stroke_width=0,
        ).set_fill(TEAL, opacity=0.90)
        end_bg.to_edge(DOWN, buff=0.15)
        end_txt = Text(
            "Random local edges → global phase transition",
            color=WHITE, font_size=20, weight=BOLD,
        )
        end_txt.move_to(end_bg)

        self.play(
            Transform(banner_bg, end_bg),
            Transform(banner_txt, end_txt),
            run_time=1.0,
        )
        self.wait(4.0)