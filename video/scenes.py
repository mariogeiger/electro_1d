"""Manim scenes of the video "Les ondes dans un milieu : personne ne ralentit", one per narration section.

Each scene is timed to its sentences (out/video/audio/timings.json, from tts.py) and draws the figures of
figures.py. Render one from the repository root:
    uv run --group video manim --media_dir out/video/media -r 1920,1080 --frame_rate 30 video/scenes.py <Scene>
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import typst
from manim import *

from narration import timing

config.background_color = "#0e1116"
GREY_C, ORANGE_C, GOLD_C, BLUE_C = "#9aa0a6", "#ff9f40", "#ffd28a", "#7fd3ff"
FONT = "DejaVu Sans"
TIMINGS = json.loads(Path("out/video/audio/timings.json").read_text())
SCHEDULE = timing({k: v["speech"] for k, v in TIMINGS.items()})
BOXES = json.loads(Path("out/video/figures/boxes.json").read_text())


def typst_mobject(source, size=48, color="white"):
    fill = f'rgb("{color}")' if color.startswith("#") else color
    document = ("#set page(width: auto, height: auto, margin: 0pt, fill: none)\n"
                f"#set text(size: {size}pt, fill: {fill})\n{source}")
    name = hashlib.sha256(document.encode()).hexdigest()[:16]
    path = Path("out/video/typst") / f"{name}.typ"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)
    svg = path.with_suffix(".svg")
    svg.write_bytes(typst.compile(str(path), format="svg"))
    return SVGMobject(str(svg), height=None).scale(0.7 / 48)


def label(text, size=30, color=WHITE, **kw):
    return Text(text, font=FONT, font_size=size, color=color, **kw)


class Section(Scene):
    section = ""

    def setup(self):
        self.lead, self.speech, self.total = SCHEDULE[self.section]
        self.marks = [self.lead + s for s, _, _ in TIMINGS[self.section]["sentences"]]

    def now(self):
        return self.renderer.time

    def until(self, t):
        if t > self.now() + 1e-3:
            self.wait(t - self.now())

    def at(self, k, offset=0.0):
        """Wait until sentence k starts (+ offset)."""
        self.until(self.marks[k] + offset)

    def finish(self):
        self.until(self.total)

    def figure(self, name):
        img = ImageMobject(f"out/video/figures/{name}.png")
        img.scale_to_fit_width(config.frame_width)
        return img

    def box(self, name, k, pad=0.06, color=GOLD_C):
        l, b, w, h = BOXES[name][k]
        W, H = config.frame_width, config.frame_height
        r = Rectangle(width=w * W + 2 * pad, height=h * H + 2 * pad, color=color, stroke_width=5)
        return r.move_to([-W / 2 + (l + w / 2) * W, -H / 2 + (b + h / 2) * H, 0])


class Field1D:
    """Curves E(x, t) of the thick-slab simulation drawn on a horizontal axis."""

    def __init__(self, x_lo=-5.0, x_hi=50.0, width=13.0, height=1.5, y=-0.5):
        S = np.load("out/data/spacetime_thick.npz")
        sel = (S["x"] >= x_lo) & (S["x"] <= x_hi)
        self.x, self.t, self.xs, self.edges = S["x"][sel], S["t"], S["xs"], S["edges"]
        self.F = {k: S[k][:, sel] for k in ("E", "Ei", "Em")}
        self.X = (self.x - x_lo) / (x_hi - x_lo) * width - width / 2
        self.height, self.y = height, y
        self.x_lo, self.x_hi, self.width = x_lo, x_hi, width

    def sx(self, x):
        return (x - self.x_lo) / (self.x_hi - self.x_lo) * self.width - self.width / 2

    def curve(self, key, time, color, width=4, opacity=1.0):
        i = int(np.clip(round(time / (self.t[1] - self.t[0])), 0, len(self.t) - 1))
        pts = np.stack([self.X, self.y + self.height * self.F[key][i], 0 * self.X], 1)
        return VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity).set_points_as_corners(pts)

    def slab(self):
        a, b = (self.sx(e) for e in self.edges)
        rect = Rectangle(width=b - a, height=2 * self.height + 0.6, fill_color="#2a2f36", fill_opacity=1,
                         stroke_width=0).move_to([(a + b) / 2, self.y, 0])
        lines = VGroup(*[Line([self.sx(xk), self.y - self.height - 0.3, 0],
                              [self.sx(xk), self.y + self.height + 0.3, 0], stroke_width=1,
                              color="#4a515a") for xk in self.xs])
        return VGroup(rect, lines)


class Intro(Section):
    section = "intro"

    def construct(self):
        title = label("Personne ne ralentit", 72)
        sub = label("Les ondes dans un milieu", 40, GREY_C).next_to(title, DOWN, 0.4)
        by = label("une vidéo de Claude", 26, GREY_C).next_to(sub, DOWN, 0.6)
        self.play(FadeIn(title, shift=UP * 0.3), run_time=1.2)
        self.play(FadeIn(sub), FadeIn(by), run_time=0.8)
        self.until(self.lead - 0.6)
        self.play(FadeOut(by), VGroup(title, sub).animate.scale(0.5).to_edge(UP, buff=0.3), run_time=0.6)
        f = Field1D()
        slab = f.slab()
        glass = label("verre", 26, GREY_C).next_to(slab, UP, 0.15)
        tr = ValueTracker(0.0)
        total = always_redraw(lambda: f.curve("E", tr.get_value(), WHITE))
        ghost = always_redraw(lambda: Dot([f.sx(5 + tr.get_value()), f.y - f.height - 0.55, 0], color=GREY_C,
                                          radius=0.07))
        ghost_lbl = always_redraw(lambda: label("à c", 22, GREY_C).next_to(ghost, DOWN, 0.1))
        self.play(FadeIn(slab), FadeIn(glass), Create(total), run_time=1.0)
        self.add(ghost, ghost_lbl)
        self.play(tr.animate.set_value(40.0), run_time=self.marks[3] - self.now(), rate_func=linear)
        self.at(1)
        q = label("Alors, qui ralentit ?", 44, GOLD_C).to_edge(DOWN, buff=0.5)
        self.at(3)
        self.play(FadeIn(q))
        self.at(4, 1.5)
        ans = label("Personne.", 60, GOLD_C)
        self.play(FadeOut(VGroup(total, ghost, ghost_lbl, slab, glass, q)), run_time=0.5)
        self.remove(total, ghost, ghost_lbl)
        self.play(FadeIn(ans, scale=1.2))
        self.at(5)
        last = label("Ce qui est lent, c'est une somme.", 44).next_to(ans, DOWN, 0.6)
        self.play(FadeIn(last))
        self.finish()


class Model(Section):
    section = "model"

    def construct(self):
        head = label("Le modèle", 40, GREY_C).to_corner(UL)
        self.play(FadeIn(head))
        axis = Arrow([-6.8, -2.2, 0], [6.8, -2.2, 0], buff=0, stroke_width=3, color=GREY_C, tip_length=0.25)
        xl = label("x", 30, GREY_C).next_to(axis, RIGHT, 0.1).shift(LEFT * 0.4 + UP * 0.3)
        self.at(1)
        tr = ValueTracker(-6.5)
        packet = always_redraw(lambda: ParametricFunction(
            lambda u: [u, 0.9 * np.exp(-((u - tr.get_value()) / 0.8) ** 2) * np.sin(6 * (u - tr.get_value())), 0],
            t_range=[-6.8, 6.8, 0.02], color=BLUE_C, stroke_width=4))
        txt1 = label("une onde plane ne dépend que de x", 30, BLUE_C).to_edge(UP, buff=0.8)
        self.play(Create(axis), FadeIn(xl), FadeIn(txt1), FadeOut(head))
        self.add(packet)
        self.play(tr.animate.set_value(-2.8), run_time=self.marks[2] - self.now(), rate_func=linear)
        self.remove(packet)
        xs = np.linspace(-1.2, 3.6, 9)
        sheets = VGroup(*[Polygon([x0, -1.6, 0], [x0 + 0.35, -1.0, 0], [x0 + 0.35, 1.9, 0], [x0, 1.3, 0],
                                  fill_color="#3a414b", fill_opacity=0.85, stroke_color="#8a929c", stroke_width=1.5)
                          for x0 in xs])
        springs = VGroup(*[ParametricFunction(lambda u, x0=x0: [x0 + 0.06 * np.sin(14 * u), -1.6 - 0.6 * u, 0],
                                              t_range=[0, 1, 0.01], stroke_width=2, color=GREY_C) for x0 in xs])
        txt2 = label("chaque couche d'atomes = un feuillet de charges sur ressorts", 30).to_edge(UP, buff=0.8)
        self.play(FadeOut(txt1), FadeIn(sheets, lag_ratio=0.1), Create(springs), FadeIn(txt2), run_time=1.5)
        self.at(3)
        k0 = 4
        xk = xs[k0] + 0.17
        osc = ValueTracker(0.0)
        charges = always_redraw(lambda: VGroup(*[Dot([xk, yy + 0.25 * np.sin(osc.get_value()), 0], radius=0.07,
                                                     color=GOLD_C) for yy in np.linspace(-0.9, 1.2, 5)]))
        sheets[k0].set_fill("#5a4a2a", 0.95)

        def emitted(sign):
            def f(u):
                d = abs(u - xk)
                ph = osc.get_value() - 6 * d
                return [u, 0.6 * np.sin(ph) * (d <= osc.get_value() / 6), 0]
            lo, hi = (xk, 6.8) if sign > 0 else (-6.8, xk)
            return ParametricFunction(f, t_range=[lo, hi, 0.02], color=ORANGE_C, stroke_width=4)

        waves = always_redraw(lambda: VGroup(emitted(1), emitted(-1)))
        txt3 = label("les charges oscillent et rayonnent, des deux côtés, à c", 30, ORANGE_C).to_edge(UP, buff=0.8)
        arrows = VGroup(Arrow([xk + 1.2, 2.2, 0], [xk + 2.6, 2.2, 0], color=ORANGE_C, buff=0),
                        Arrow([xk - 1.2, 2.2, 0], [xk - 2.6, 2.2, 0], color=ORANGE_C, buff=0))
        cl = VGroup(label("c", 28, ORANGE_C).next_to(arrows[0], UP, 0.1), label("c", 28, ORANGE_C).next_to(arrows[1], UP, 0.1))
        self.play(FadeOut(txt2), FadeIn(txt3), run_time=0.6)
        self.add(charges, waves)
        self.play(FadeIn(arrows), FadeIn(cl), osc.animate.set_value(30), run_time=self.marks[4] - self.now(),
                  rate_func=linear)
        formula = typst_mobject('$ E_k (x, t) = - 1/2 K_k (t - (|x - x_k|)/c), quad K_k = q dot(y)_k $', 40,
                                ORANGE_C).to_edge(DOWN, buff=0.35)
        bg = BackgroundRectangle(formula, color=config.background_color, fill_opacity=0.9, buff=0.15)
        rest = self.total - self.now() - 0.05
        self.play(FadeIn(bg, run_time=0.5), Write(formula, run_time=2.0),
                  osc.animate(run_time=rest, rate_func=linear).set_value(30 + 2.5 * rest))
        self.finish()


class Sum(Section):
    section = "sum"

    def construct(self):
        head = label("La somme", 40, GREY_C).to_corner(UL)
        lhs = typst_mobject("$ E(x, t) = $", 42)
        inc = typst_mobject('$ E_"inc" (x - c t) $', 42, GREY_C)
        mat = typst_mobject("$ - 1/2 sum_k K_k (t - (|x - x_k|)/c) $", 42, ORANGE_C)
        eq = VGroup(lhs, inc, mat).arrange(RIGHT, buff=0.3).shift(UP * 0.6)
        self.play(FadeIn(head), Write(lhs))
        self.at(1)
        self.play(Write(inc))
        self.play(Write(mat))
        self.at(2)
        b1, b2 = Brace(inc, DOWN, color=GREY_C), Brace(mat, DOWN, color=ORANGE_C)
        l1 = label("onde incidente : à c", 28, GREY_C).next_to(b1, DOWN, 0.15)
        l2 = label("tous les feuillets : chacun à c", 28, ORANGE_C).next_to(b2, DOWN, 0.15)
        l1.set_y(l2.get_y())
        self.play(GrowFromCenter(b1), GrowFromCenter(b2), FadeIn(l1), FadeIn(l2))
        self.at(3)
        none = label("aucun terme ne ralentit", 40, GOLD_C).to_edge(DOWN, buff=0.9)
        self.play(FadeIn(none, shift=UP * 0.2))
        self.at(4)
        self.play(Create(SurroundingRectangle(VGroup(eq, b1, b2, l1, l2), color=GOLD_C, buff=0.25,
                                              corner_radius=0.1)))
        self.finish()


class OneSheet(Section):
    section = "sheet"

    def construct(self):
        head = label("Un seul feuillet", 40, GREY_C).to_corner(UL)
        ax = Axes(x_range=[-3.5, 3.5, 1], y_range=[-0.5, 1.2, 0.5], x_length=8.5, y_length=4.2, tips=False,
                  axis_config={"include_numbers": False, "color": GREY_C}).shift(RIGHT * 1.9 + DOWN * 0.3)
        tl = label("temps", 24, GREY_C).next_to(ax.x_axis, DOWN, 0.15).align_to(ax.x_axis, RIGHT)
        tau = 0.35
        g = lambda u: np.exp(-u * u)
        dg = lambda u: -2 * u * np.exp(-u * u)
        inc = ax.plot(g, x_range=[-3.5, 3.5], color=GREY_C, stroke_width=4)
        rad = ax.plot(lambda u: -tau * dg(u), x_range=[-3.5, 3.5], color=ORANGE_C, stroke_width=4)
        tot = ax.plot(lambda u: g(u) - tau * dg(u), x_range=[-3.5, 3.5], color=WHITE, stroke_width=5)
        shifted = DashedVMobject(ax.plot(lambda u: g(u - tau), x_range=[-3.5, 3.5], color=BLUE_C, stroke_width=3),
                                 num_dashes=60)
        title = label("derrière le feuillet", 28).next_to(ax, UP, 0.1).align_to(ax, LEFT)
        self.play(FadeIn(head), Create(ax), FadeIn(tl), FadeIn(title))
        self.play(Create(inc))
        li = label("onde incidente E", 26, GREY_C).next_to(ax.c2p(-1.0, 1.0), LEFT, 0.2)
        self.add(li)
        self.at(2)
        lr = label("rayonnée : − τ × dérivée", 26, ORANGE_C).next_to(ax.c2p(-1.0, 0.72), LEFT, 0.2)
        self.play(Create(rad), FadeIn(lr))
        self.at(3)
        f1 = typst_mobject('$ E(t) - tau thin (dif E)/(dif t) approx E(t - tau) $', 44, GOLD_C).to_edge(DOWN, buff=0.3)
        self.play(Create(tot), Write(f1))
        self.play(Create(shifted))
        self.at(4)
        arrow = Arrow(ax.c2p(-0.05, 1.08), ax.c2p(tau + 0.45, 1.08), buff=0, color=GOLD_C, stroke_width=4)
        lt = label("même impulsion, en retard de τ", 26, GOLD_C).next_to(arrow, UP, 0.1).shift(RIGHT * 0.9)
        self.play(GrowArrow(arrow), FadeIn(lt), FadeOut(title))
        self.at(5)
        ax2 = Axes(x_range=[-3, 3, 1], y_range=[-0.5, 0.5, 0.5], x_length=3.6, y_length=2.0, tips=False,
                   axis_config={"include_numbers": False, "color": GREY_C}).to_corner(DL, buff=0.9).shift(UP * 1.1)
        echo = ax2.plot(lambda u: -tau * dg(u) * 1.2, x_range=[-3, 3], color=ORANGE_C, stroke_width=4)
        le = label("devant : l'écho (réflexion)", 24, ORANGE_C).next_to(ax2, DOWN, 0.12)
        self.play(Create(ax2), Create(echo), FadeIn(le))
        self.finish()


class Slab(Section):
    section = "slab"

    def construct(self):
        head = label("Soixante feuillets", 40, GREY_C).to_corner(UL)
        f = Field1D(y=-0.6, height=1.6)
        slab = f.slab()
        self.play(FadeIn(head), FadeIn(slab))
        self.at(1)
        add = label("chaque feuillet ajoute son petit retard", 30).to_edge(UP, buff=1.1)
        self.play(FadeIn(add))
        self.at(2)
        tr = ValueTracker(0.0)
        show = {"Ei": ValueTracker(0.0), "Em": ValueTracker(0.0)}
        total = always_redraw(lambda: f.curve("E", tr.get_value(), WHITE, 5))
        inc = always_redraw(lambda: f.curve("Ei", tr.get_value(), GREY_C, 3, show["Ei"].get_value()))
        mat = always_redraw(lambda: f.curve("Em", tr.get_value(), ORANGE_C, 3, show["Em"].get_value()))
        legend = VGroup(label("champ total", 26), label("onde incidente seule (à c)", 26, GREY_C),
                        label("rayonnement des charges", 26, ORANGE_C)).arrange(RIGHT, buff=0.8).to_edge(DOWN, 0.35)
        self.play(FadeOut(add), run_time=0.4)
        self.add(inc, mat, total)
        run_end = self.total - 0.8
        self.play(tr.animate(run_time=run_end - self.now(), rate_func=linear).set_value(36.0),
                  Succession(Wait(max(0.01, self.marks[3] - self.now())), FadeIn(legend[0]),
                             Wait(max(0.01, self.marks[4] - self.marks[3] - 1.0)),
                             AnimationGroup(show["Ei"].animate.set_value(1.0), FadeIn(legend[1])),
                             Wait(max(0.01, self.marks[5] - self.marks[4] - 1.0)),
                             AnimationGroup(show["Em"].animate.set_value(1.0), FadeIn(legend[2])),
                             Wait(max(0.01, self.marks[6] - self.marks[5] - 1.0)),
                             FadeIn(label("Ewald–Oseen : la matière éteint l'onde incidente", 30, GOLD_C)
                                    .to_edge(UP, buff=1.1))))
        self.finish()


class Spacetime(Section):
    section = "spacetime"

    def construct(self):
        img = self.figure("spacetime")
        self.play(FadeIn(img))
        for k in range(3):
            self.at(k + 1)
            r = self.box("spacetime", k)
            self.play(Create(r), run_time=0.6)
            self.wait(1.2)
            if k < 2:
                self.play(FadeOut(r), run_time=0.4)
        self.finish()


class Front(Section):
    section = "front"

    def construct(self):
        img = self.figure("front")
        self.play(FadeIn(img))
        self.at(2)
        r = self.box("front", 2)
        self.play(Create(r))
        self.at(3)
        img2 = self.figure("delays")
        self.play(FadeOut(r), FadeOut(img), FadeIn(img2))
        self.at(4)
        note = label("retard de l'enveloppe ≈ 2 × retard des crêtes", 32, GOLD_C).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(note))
        self.finish()


class Colors(Section):
    section = "colors"

    def construct(self):
        img = self.figure("colors")
        self.play(FadeIn(img))
        r = None
        for k in range(4):
            self.at(k + 1)
            new = self.box("colors", k, pad=0.08)
            self.play(Transform(r, new) if r else Create(new), run_time=0.6)
            r = r or new
        self.at(5)
        l, b, w, h = BOXES["colors"][4]
        W, H = config.frame_width, config.frame_height
        x6 = -W / 2 + (l + w * 6.2 / 12) * W
        x12 = -W / 2 + (l + w) * W
        y0, y1 = -H / 2 + b * H, -H / 2 + (b + h * 0.5) * H
        band = Rectangle(width=x12 - x6, height=y1 - y0, color=ORANGE_C, stroke_width=5).move_to(
            [(x6 + x12) / 2, (y0 + y1) / 2, 0])
        self.play(Transform(r, band))
        self.at(6)
        note = label("aucun signal ne dépasse c", 30, GOLD_C).move_to([2.6, -0.33, 0])
        bg = BackgroundRectangle(note, color=config.background_color, fill_opacity=0.85, buff=0.15)
        self.play(FadeIn(bg), FadeIn(note))
        self.finish()


class SlowLight(Section):
    section = "slowlight"

    def construct(self):
        img = self.figure("slowlight")
        self.play(FadeIn(img))
        l, b, w, h = BOXES["slowlight"][0]
        W, H = config.frame_width, config.frame_height
        self.at(2)
        tail = Rectangle(width=w * W * 0.78, height=h * H * 0.45, color=GOLD_C, stroke_width=5).move_to(
            [-W / 2 + (l + w * 0.6) * W, -H / 2 + (b + h * 0.38) * H, 0])
        self.play(Create(tail))
        self.at(3)
        note = label("bord de bande : vitesse de groupe → 0", 30, GOLD_C).next_to(tail, UP, 0.15)
        self.play(FadeIn(note))
        self.finish()


class Summary(Section):
    section = "summary"

    def construct(self):
        head = label("En résumé", 44, GREY_C).to_edge(UP, buff=0.5)
        self.play(FadeIn(head))
        lines = [("1.  Rien ne va moins vite que la lumière.", WHITE),
                 ("2.  L'onde incidente traverse tout, à c.", GREY_C),
                 ("3.  Les charges rayonnent, à c : une onde qui l'annule,\n     et une autre, en retard.", ORANGE_C),
                 ("4.  Indice, lumière lente, réfraction :\n     l'interférence d'un chœur d'oscillateurs.", GOLD_C)]
        items = VGroup(*[label(s, 30, c) for s, c in lines]).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        items.next_to(head, DOWN, 0.55).set_x(0)
        eq = typst_mobject('$ E = E_"inc" (x - c t) - 1/2 sum_k K_k (t - (|x - x_k|)/c) $', 44).to_edge(DOWN, buff=0.6)
        for k in range(4):
            self.at(k + 1)
            self.play(FadeIn(items[k], shift=RIGHT * 0.2))
        self.play(Write(eq))
        self.finish()


class Credits(Section):
    section = "credits"

    def construct(self):
        t1 = label("Les ondes dans un milieu — personne ne ralentit", 40)
        t2 = label("Écrit et réalisé par Claude (Anthropic)", 34, GOLD_C)
        t3 = label("Simulation : Mario Geiger   ·   Idée : Quentin Talon", 30, GREY_C)
        t4 = label("Calcul : solveur exact des feuillets couplés (JAX)   ·   Animation : Manim, matplotlib", 22,
                   GREY_C)
        t5 = label("Voix de synthèse : Piper (fr_FR siwis)   ·   novembre 2026", 22, GREY_C)
        g = VGroup(t1, t2, t3, t4, t5).arrange(DOWN, buff=0.45)
        self.play(FadeIn(g, lag_ratio=0.2), run_time=2)
        self.until(self.total - 1.0)
        self.play(FadeOut(g), run_time=1.0)
        self.finish()
