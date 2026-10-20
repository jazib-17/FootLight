#!/usr/bin/env python3
"""
football_synth.py - Synthetic football match event generator
============================================================
Built for the "Inside the Game" style hackathon brief: produces football-realistic,
fully synthetic (fictional clubs / players) match data that downstream AI agents can
ingest, interpret, explain, render and personalize.

What it generates
-----------------
* Match metadata, rosters, formations, team "style" parameters
* A possession-driven event stream:
    pass (distance, accuracy/target error, difficulty rating, ball speed, pass type,
          line-breaking, progressive, under pressure), carry, dribble, shot (xG, shot speed,
          body part, outcome), save, goal (with assist + win-probability swing), tackle,
          interception, clearance, pressure, possession_change, foul, yellow/red card,
          offside, corner, throw-in, goal-kick, free-kick, penalty, substitution
* Player identification tags on every event (name, jersey, position, display tag)
* Speed / distance indicators: sprint events above a threshold, distance milestones
* Player milestones (brace, hat-trick, 50th/100th pass, ...)
* Rolling "match_state" snapshots every minute: possession, pass accuracy, xG, tempo,
  control index, chaos index, momentum, field tilt, win probabilities, pattern tags
  (ground truth for explainability: control vs chaos, pressure shifts, rhythm)
* Per-event `importance` score + `tags` (key_pass, big_chance, ...) for key-moment detection
* Video-aligned timestamps (UTC + frame + timecode @25fps) for overlay sync
* Optional 5 Hz ball tracking traces (for "auto-eventing" model experiments)
* Multi-language event-type label dictionary (en, es, fr, pt, de, ja)
* Synthetic fan profiles (club, player focus, language, metric focus, analyst vs casual)
* Optional live replay mode that streams events in (accelerated) real time

Usage
-----
    python football_synth.py --matches 3 --seed 7 --out data --tracking
    python football_synth.py --matches 1 --live --speed 60

Coordinates: absolute pitch metres, 105 x 68, origin bottom-left, x along the length.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------- constants
L, W = 105.0, 68.0
SPRINT_KMH = 25.0                                  # speed alert threshold
DISTANCE_MILESTONES_KM = [2, 4, 6, 8, 10, 12]      # distance alert thresholds
FPS = 25
HALF = 2700
WINDOW_S = 300                                     # rolling window for context metrics


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def logistic(z):
    return 1.0 / (1.0 + math.exp(-z))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def flip(p):
    return (L - p[0], W - p[1])


# --------------------------------------------------------------------------- reference data
FORMATIONS = {
    "4-3-3": [("GK", .04, .5), ("LB", .22, .9), ("CB", .20, .65), ("CB", .20, .35), ("RB", .22, .1),
              ("CM", .42, .75), ("DM", .36, .5), ("CM", .42, .25),
              ("LW", .72, .88), ("ST", .82, .5), ("RW", .72, .12)],
    "4-4-2": [("GK", .04, .5), ("LB", .22, .9), ("CB", .20, .65), ("CB", .20, .35), ("RB", .22, .1),
              ("LM", .50, .9), ("CM", .44, .62), ("CM", .44, .38), ("RM", .50, .1),
              ("ST", .80, .62), ("ST", .80, .38)],
    "3-5-2": [("GK", .04, .5), ("CB", .20, .75), ("CB", .18, .5), ("CB", .20, .25),
              ("LWB", .52, .93), ("CM", .46, .68), ("DM", .38, .5), ("CM", .46, .32), ("RWB", .52, .07),
              ("ST", .80, .62), ("ST", .80, .38)],
}
BENCH_POS = [("GK", .04, .5), ("CB", .2, .5), ("RB", .22, .1), ("CM", .44, .5),
             ("CM", .44, .6), ("LW", .72, .88), ("ST", .8, .5)]
GROUP = {"GK": "gk", "CB": "def", "LB": "def", "RB": "def", "LWB": "def", "RWB": "def",
         "DM": "mid", "CM": "mid", "LM": "mid", "RM": "mid", "LW": "fwd", "RW": "fwd", "ST": "fwd"}
SKILLS = {
    "gk":  dict(passing=60, shooting=25, tackling=30, pace=50, stamina=65, keeping=76),
    "def": dict(passing=66, shooting=44, tackling=76, pace=69, stamina=76, keeping=10),
    "mid": dict(passing=77, shooting=62, tackling=66, pace=69, stamina=80, keeping=10),
    "fwd": dict(passing=66, shooting=78, tackling=45, pace=79, stamina=72, keeping=10),
}
WORK = {"gk": 0.28, "def": 1.75, "mid": 2.05, "fwd": 1.85}   # mean m/s over the whole match

FIRST = ["Luca", "Mateo", "Jonas", "Kofi", "Rafael", "Idris", "Tomas", "Yuki", "Andre", "Samir",
         "Nikolai", "Diego", "Felix", "Omar", "Elias", "Bruno", "Kenji", "Marek", "Tariq", "Callum",
         "Joao", "Stefan", "Mikael", "Arjun", "Hugo", "Emeka", "Dario", "Leif", "Anton", "Rui"]
LAST = ["Alvarez", "Brandt", "Cole", "Dumont", "Eriksen", "Fofana", "Greco", "Haruna", "Ibarra",
        "Jovic", "Kovac", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Petrov", "Quinn", "Rossi",
        "Santos", "Takeda", "Ulrich", "Varga", "Weston", "Xavier", "Yilmaz", "Zielinski", "Abara",
        "Bennett", "Castellano", "Dalgaard", "Esposito", "Fraser", "Gallo", "Holt", "Ito"]
CLUBS = [("Northgate United", "NOR"), ("Riverside Rovers", "RIV"), ("Harbor City", "HAR"),
         ("Ashford Athletic", "ASH"), ("Redmont FC", "RED"), ("Silverlake Town", "SIL"),
         ("Kingsbridge Wanderers", "KIN"), ("Oakfield Albion", "OAK"), ("Brightwater Park", "BRI"),
         ("Stonehaven Rangers", "STO")]
VENUES = ["Northgate Park", "Riverside Arena", "Harbor Stadium", "Ashford Ground", "Redmont Park"]

EVENT_TYPES = ["kickoff", "period_end", "pass", "carry", "dribble", "shot", "save", "goal", "tackle",
               "interception", "clearance", "pressure", "possession_change", "foul", "yellow_card",
               "red_card", "offside", "corner_kick", "throw_in", "goal_kick", "free_kick",
               "substitution", "sprint", "distance_milestone", "player_milestone", "match_state"]

I18N = {  # multi-language labels for event types (global studio / streaming scenarios)
    "en": dict(pass_="Pass", shot="Shot", goal="Goal", tackle="Tackle", interception="Interception",
               foul="Foul", yellow_card="Yellow card", red_card="Red card", corner_kick="Corner",
               throw_in="Throw-in", free_kick="Free kick", substitution="Substitution", sprint="Sprint",
               pressure="Pressure", save="Save", offside="Offside"),
    "es": dict(pass_="Pase", shot="Disparo", goal="Gol", tackle="Entrada", interception="Intercepción",
               foul="Falta", yellow_card="Tarjeta amarilla", red_card="Tarjeta roja", corner_kick="Córner",
               throw_in="Saque de banda", free_kick="Tiro libre", substitution="Cambio", sprint="Sprint",
               pressure="Presión", save="Parada", offside="Fuera de juego"),
    "fr": dict(pass_="Passe", shot="Tir", goal="But", tackle="Tacle", interception="Interception",
               foul="Faute", yellow_card="Carton jaune", red_card="Carton rouge", corner_kick="Corner",
               throw_in="Touche", free_kick="Coup franc", substitution="Remplacement", sprint="Sprint",
               pressure="Pressing", save="Arrêt", offside="Hors-jeu"),
    "pt": dict(pass_="Passe", shot="Chute", goal="Gol", tackle="Desarme", interception="Interceptação",
               foul="Falta", yellow_card="Cartão amarelo", red_card="Cartão vermelho", corner_kick="Escanteio",
               throw_in="Lateral", free_kick="Falta direta", substitution="Substituição", sprint="Arrancada",
               pressure="Pressão", save="Defesa", offside="Impedimento"),
    "de": dict(pass_="Pass", shot="Schuss", goal="Tor", tackle="Zweikampf", interception="Ballgewinn",
               foul="Foulspiel", yellow_card="Gelbe Karte", red_card="Rote Karte", corner_kick="Eckball",
               throw_in="Einwurf", free_kick="Freistoß", substitution="Auswechslung", sprint="Sprint",
               pressure="Pressing", save="Parade", offside="Abseits"),
    "ja": dict(pass_="パス", shot="シュート", goal="ゴール", tackle="タックル", interception="インターセプト",
               foul="ファウル", yellow_card="イエローカード", red_card="レッドカード", corner_kick="コーナーキック",
               throw_in="スローイン", free_kick="フリーキック", substitution="選手交代", sprint="スプリント",
               pressure="プレス", save="セーブ", offside="オフサイド"),
}

IMPORTANCE = {"goal": 1.0, "red_card": 0.85, "penalty": 0.9, "save": 0.45, "yellow_card": 0.3,
              "substitution": 0.2, "player_milestone": 0.5, "sprint": 0.25, "corner_kick": 0.2,
              "offside": 0.15, "foul": 0.15, "interception": 0.1, "tackle": 0.1}


# --------------------------------------------------------------------------- entities
@dataclass
class Player:
    pid: str
    name: str
    surname: str
    jersey: int
    pos: str
    group: str
    bx: float
    by: float
    passing: float
    shooting: float
    tackling: float
    pace: float
    stamina: float
    keeping: float
    work: float
    stats: dict = field(default_factory=lambda: defaultdict(float))
    carded: int = 0

    def tag(self, code):
        return f"#{self.jersey} {self.name[0]}. {self.surname} ({code})"


@dataclass
class Team:
    code: str
    name: str
    side: str
    formation: str
    strength: float
    style: dict
    players: list
    bench: list
    score: int = 0
    attacks_right: bool = True
    subs_used: int = 0
    stats: dict = field(default_factory=lambda: defaultdict(float))

    def gk(self):
        for p in self.players:
            if p.pos == "GK":
                return p
        return self.players[0]


def make_player(rng, code, pos, bx, by, jersey, strength, used):
    while True:
        first, last = rng.choice(FIRST), rng.choice(LAST)
        if (first, last) not in used:
            used.add((first, last))
            break
    g = GROUP[pos]
    sk = {k: clamp(rng.gauss(v + (strength - 1) * 60, 6), 35, 97) for k, v in SKILLS[g].items()}
    return Player(f"{code}{jersey:02d}", first, last, jersey, pos, g, bx, by, work=WORK[g] * rng.uniform(.9, 1.1), **sk)


def build_team(rng, name, code, side):
    formation = rng.choice(list(FORMATIONS))
    strength = clamp(rng.gauss(1.0, 0.07), 0.85, 1.15)
    style = {"tempo": rng.uniform(.4, .9), "directness": rng.uniform(.25, .85), "press": rng.uniform(.3, .9)}
    jerseys = rng.sample(range(1, 40), 18)
    used = set()
    players = [make_player(rng, code, pos, bx, by, jerseys[i], strength, used)
               for i, (pos, bx, by) in enumerate(FORMATIONS[formation])]
    bench = [make_player(rng, code, pos, bx, by, jerseys[11 + i], strength, used)
             for i, (pos, bx, by) in enumerate(BENCH_POS)]
    return Team(code, name, side, formation, strength, style, players, bench)


# --------------------------------------------------------------------------- simulation
class MatchSim:
    def __init__(self, seed, idx, home: Team, away: Team, kickoff: datetime, tracking=False):
        self.rng = random.Random(seed)
        self.match_id = f"M{idx + 1:03d}"
        self.home, self.away = home, away
        self.kickoff = kickoff
        self.tracking = tracking
        self.events, self.tracks = [], []
        self.seq = 0
        self.clock = 0.0
        self.period = 1
        self.p1_extra = 0.0
        self.poss: Team = home
        self.carrier: Player = home.players[8]
        self.ball = (L / 2, W / 2)
        self.next_snap = 60.0
        self.win = deque()
        self.last_pass = None
        self.prev_momentum = 0.0
        self.prev_wp = None
        self.lay = None
        self.lay_team = None
        self.sub_times = {t.code: sorted(self.rng.sample(range(3400, 5000, 60), 3)) for t in (home, away)}

    # ---- helpers
    def opp(self, t):
        return self.away if t is self.home else self.home

    def A(self, team, p):
        """team-relative frame -> absolute pitch coordinates"""
        if p is None:
            return None
        x, y = p if team.attacks_right else flip(p)
        return (round(x, 2), round(y, 2))

    def wall(self, t=None):
        t = self.clock if t is None else t
        return t + (self.p1_extra + 900 if self.period == 2 else 0)

    def minute_label(self, t):
        m = int(t // 60)
        if self.period == 1 and t >= HALF:
            return f"45+{m - 45}"
        if self.period == 2 and t >= 2 * HALF:
            return f"90+{m - 90}"
        return str(m)

    def log(self, kind, team, val=1.0):
        self.win.append((self.clock, team.code, kind, val))

    def emit(self, etype, team=None, player=None, pos=None, end=None, t=None, importance=None, tags=None, **kw):
        t = self.clock if t is None else t
        wall = self.wall(t)
        ts = self.kickoff + timedelta(seconds=wall)
        frames = int(wall * FPS)
        tc = f"{frames // (FPS * 3600):02d}:{(frames // (FPS * 60)) % 60:02d}:{(frames // FPS) % 60:02d}:{frames % FPS:02d}"
        a = self.A(team, pos) if (team and pos) else None
        b = self.A(team, end) if (team and end) else None
        ev = {
            "_seq": self.seq, "match_id": self.match_id, "period": self.period,
            "match_clock_s": round(t, 2), "minute": int(t // 60), "minute_label": self.minute_label(t),
            "timestamp_utc": ts.isoformat(timespec="milliseconds"), "video_frame": frames, "video_timecode": tc,
            "type": etype, "team": team.code if team else None,
            "player_id": player.pid if player else None, "player_name": f"{player.name} {player.surname}" if player else None,
            "jersey": player.jersey if player else None, "position": player.pos if player else None,
            "display_tag": player.tag(team.code) if (player and team) else None,
            "x": a[0] if a else None, "y": a[1] if a else None,
            "end_x": b[0] if b else None, "end_y": b[1] if b else None,
            "score_home": self.home.score, "score_away": self.away.score,
            "importance": round(IMPORTANCE.get(etype, 0.05) if importance is None else importance, 3),
            "tags": list(tags or []),
        }
        ev.update(kw)
        self.seq += 1
        self.events.append(ev)
        return ev

    def add_trace(self, ev, team, start, end, dur, loft=0.0, hz=5):
        if not self.tracking:
            return
        n = max(2, int(dur * hz))
        pts = []
        for i in range(n + 1):
            f = i / n
            x = start[0] + (end[0] - start[0]) * f
            y = start[1] + (end[1] - start[1]) * f
            z = 4 * loft * f * (1 - f)
            ax, ay = self.A(team, (x, y))
            pts.append([round(f * dur, 2), ax, ay, round(z, 2)])
        self.tracks.append({"_seq": ev["_seq"], "match_id": self.match_id, "hz": hz, "samples": pts})

    # ---- player layout (approximate positions around the ball)
    def _pos(self, team, ball):
        out = {}
        for p in team.players:
            x = 0.55 * p.bx * L + 0.45 * ball[0] + self.rng.gauss(0, 3)
            y = 0.70 * p.by * W + 0.30 * ball[1] + self.rng.gauss(0, 3)
            out[p.pid] = (clamp(x, 1, L - 1), clamp(y, 1, W - 1))
        return out

    def layout(self):
        t, o = self.poss, self.opp(self.poss)
        mine = self._pos(t, self.ball)
        theirs = {k: flip(v) for k, v in self._pos(o, flip(self.ball)).items()}
        self.lay, self.lay_team = (mine, theirs), t
        return mine, theirs

    def pressure(self):
        mine, theirs = self.lay
        o = self.opp(self.poss)
        best, who = 99, None
        for p in o.players:
            if p.pos == "GK":
                continue
            d = dist(theirs[p.pid], self.ball)
            if d < best:
                best, who = d, p
        pr = clamp(1 - best / 10, 0, 1) * (0.7 + 0.6 * o.style["press"])
        return clamp(pr, 0, 1), who

    # ---- possession handling
    def switch(self, team, player, ball_old_frame):
        self.ball = flip(ball_old_frame) if team is not self.poss else ball_old_frame
        self.poss, self.carrier = team, player

    def change_possession(self, new_team, new_player, ball_old, reason, at=None):
        old = self.poss
        self.emit("possession_change", old, None, pos=ball_old, from_team=old.code, to_team=new_team.code,
                  reason=reason, new_carrier_id=new_player.pid, importance=0.08)
        self.log("change", new_team)
        self.switch(new_team, new_player, ball_old)

    def nearest(self, team, loc, goalkeeper=False):
        pos = self._pos(team, loc)
        pool = [p for p in team.players if (p.pos == "GK") == goalkeeper] or team.players
        return min(pool, key=lambda p: dist(pos[p.pid], loc))

    def restart(self, team, kind, loc, dt=8.0, **kw):
        """team restarts play at loc (team frame). Records possession change if needed."""
        taker = self.gk_or_nearest(team, kind, loc)
        if team is not self.poss:
            self.log("change", team)
        self.poss, self.carrier, self.ball = team, taker, loc
        self.emit(kind, team, taker, pos=loc, **kw)
        self.advance(dt, active=False)

    def gk_or_nearest(self, team, kind, loc):
        return team.gk() if kind == "goal_kick" else self.nearest(team, loc)

    # ---- time progression
    def advance(self, dt, active=True):
        t0 = self.clock
        self.clock += dt
        self.poss.stats["poss_s"] += dt
        self.log("poss", self.poss, dt)
        fatigue = 1 - 0.12 * clamp(self.clock / 5400, 0, 1.1)
        for team in (self.home, self.away):
            for p in team.players:
                spd = p.work * fatigue * (p.stamina / 78) * (1.0 if active else 0.35) * self.rng.uniform(.8, 1.2)
                old = p.stats["distance_m"]
                p.stats["distance_m"] += spd * dt
                for km in DISTANCE_MILESTONES_KM:
                    if old < km * 1000 <= p.stats["distance_m"]:
                        self.emit("distance_milestone", team, p, t=t0 + dt * self.rng.random(), km=km,
                                  avg_speed_kmh=round(p.stats["distance_m"] / max(self.clock, 1) * 3.6, 1),
                                  importance=0.12)
        if active and self.lay and self.rng.random() < dt / 45:
            team = self.rng.choice([self.home, self.away])
            cand = [p for p in team.players if p.pos != "GK"]
            p = self.rng.choices(cand, [c.pace for c in cand])[0]
            spd = self.rng.gauss(27.5, 2.0) + (p.pace - 70) * 0.08
            if spd >= SPRINT_KMH:
                dur = self.rng.uniform(2, 6)
                d = spd / 3.6 * dur
                p.stats["sprints"] += 1
                p.stats["distance_m"] += d
                p.stats["top_speed_kmh"] = max(p.stats["top_speed_kmh"], spd)
                mine, theirs = self.lay
                pp = mine.get(p.pid) or theirs.get(p.pid)
                self.emit("sprint", team if team.code != self.lay_team.code else team, p,
                          pos=pp if team is self.lay_team else flip(pp) if pp else None,
                          t=t0 + dt * self.rng.random(), speed_kmh=round(spd, 1), duration_s=round(dur, 1),
                          distance_m=round(d, 1), threshold_kmh=SPRINT_KMH, speed_alert=True)
        while self.next_snap <= self.clock and (self.period == 2 or self.next_snap <= HALF + 600):
            self.snapshot(self.next_snap)
            self.next_snap += 60

    # ---- win probability (Poisson model on remaining time)
    def win_probs(self):
        h, a = self.home, self.away
        rem = clamp((2 * HALF - min(self.clock, 2 * HALF)) / (2 * HALF), 0.01, 1)
        lh = 1.35 * h.strength / a.strength * rem
        la = 1.35 * a.strength / h.strength * rem
        pm = lambda k, lam: math.exp(-lam) * lam ** k / math.factorial(k)
        diff = h.score - a.score
        ph = pd = pa = 0.0
        for i in range(9):
            for j in range(9):
                p = pm(i, lh) * pm(j, la)
                d = diff + i - j
                if d > 0:
                    ph += p
                elif d == 0:
                    pd += p
                else:
                    pa += p
        s = ph + pd + pa
        return ph / s, pd / s, pa / s

    # ---- rolling context snapshot (ground truth for explainability)
    def snapshot(self, t):
        while self.win and self.win[0][0] < t - WINDOW_S:
            self.win.popleft()
        agg = {self.home.code: defaultdict(float), self.away.code: defaultdict(float)}
        for _, tc, kind, val in self.win:
            agg[tc][kind] += val
        H, A_ = agg[self.home.code], agg[self.away.code]

        def share(k):
            s = H[k] + A_[k]
            return H[k] / s if s else 0.5

        def acc(d):
            n = d["pass_ok"] + d["pass_fail"]
            return d["pass_ok"] / n if n else 0.8

        ph, pa = self.home.stats["poss_s"], self.away.stats["poss_s"]
        poss_h = ph / (ph + pa) if ph + pa else 0.5
        ctrl_h = 0.6 * share("poss") + 0.4 * acc(H)
        ctrl_a = 0.6 * (1 - share("poss")) + 0.4 * acc(A_)
        tilt_h = share("f3")
        mom = clamp(2 * (H["xg"] - A_["xg"]) + (tilt_h - 0.5) * 1.2 + (ctrl_h - ctrl_a) * 0.8, -1, 1)
        changes = H["change"] + A_["change"]
        chaos = min(1.0, changes / 22)
        win_min = max(min(WINDOW_S, t) / 60, 1)
        tempo = (H["action"] + A_["action"]) / win_min
        tags = []
        for tm, d, tilt in ((self.home, H, tilt_h), (self.away, A_, 1 - tilt_h)):
            if tilt > 0.62 and (d["xg"] > 0.25 or d["shots"] >= 3):
                tags.append(f"sustained_pressure:{tm.code}")
        if chaos > 0.55 and tempo > 14:
            tags.append("end_to_end")
        if max(ctrl_h, ctrl_a) > 0.62:
            tags.append(f"control:{self.home.code if ctrl_h > ctrl_a else self.away.code}")
        if abs(mom - self.prev_momentum) > 0.5:
            tags.append("momentum_shift")
        if tempo < 9:
            tags.append("game_slowing")
        wh, wd, wa = self.win_probs()
        self.emit("match_state", t=t, window_s=WINDOW_S, possession_home_pct=round(poss_h * 100, 1),
                  pass_acc_home_pct=round(100 * self.home.stats["pass_ok"] / max(self.home.stats["passes"], 1), 1),
                  pass_acc_away_pct=round(100 * self.away.stats["pass_ok"] / max(self.away.stats["passes"], 1), 1),
                  shots_home=int(self.home.stats["shots"]), shots_away=int(self.away.stats["shots"]),
                  xg_home=round(self.home.stats["xg"], 2), xg_away=round(self.away.stats["xg"], 2),
                  control_index_home=round(ctrl_h, 3), control_index_away=round(ctrl_a, 3),
                  chaos_index=round(chaos, 3), tempo_actions_per_min=round(tempo, 1),
                  field_tilt_home=round(tilt_h, 3), momentum_home=round(mom, 3),
                  win_prob_home=round(wh, 3), win_prob_draw=round(wd, 3), win_prob_away=round(wa, 3),
                  pattern_tags=tags, importance=0.0)
        self.prev_momentum = mom

    # ---- milestones
    def milestone(self, team, p, stat, label):
        self.emit("player_milestone", team, p, stat=stat, milestone=label, value=int(p.stats[stat]))

    # ---- actions
    def pass_action(self, pressure):
        rng = self.rng
        t, o, c = self.poss, self.opp(self.poss), self.carrier
        mine, theirs = self.lay
        bx, by = self.ball
        cands = []
        for p in t.players:
            if p is c or (p.pos == "GK" and bx > 30):
                continue
            tp = mine[p.pid]
            d = dist(tp, self.ball)
            if d < 4 or d > 58:
                continue
            fwd = tp[0] - bx
            near_opp = min(dist(tp, theirs[q.pid]) for q in o.players)
            openness = clamp(near_opp / 8, 0, 1)
            w = math.exp(-d / 16) * clamp(1 + t.style["directness"] * fwd / 20, 0.15, 3) * (0.4 + openness)
            cands.append((p, tp, d, fwd, openness, w))
        r, tp, d, fwd, openness, _ = rng.choices(cands, [c_[5] for c_ in cands])[0]
        # pass type
        if d > 35:
            ptype = "long_ball"
        elif bx > L * .72 and (by < 18 or by > 50) and tp[0] > L * .8 and 20 < tp[1] < 48:
            ptype = "cross"
        elif fwd >= 12 and rng.random() < 0.35:
            ptype = "through_ball"
        elif fwd < -5:
            ptype = "back_pass"
        elif abs(fwd) <= 5:
            ptype = "lateral"
        else:
            ptype = "short_pass"
        lines = sum(1 for q in o.players if q.pos != "GK" and bx < theirs[q.pid][0] < tp[0]
                    and min(by, tp[1]) - 6 < theirs[q.pid][1] < max(by, tp[1]) + 6)
        bonus = {"long_ball": 12, "cross": 14, "through_ball": 16, "back_pass": -6, "lateral": -2}.get(ptype, 0)
        diff = clamp(10 + d * 0.85 + pressure * 22 + bonus + min(lines, 4) * 4 + (1 - openness) * 10, 1, 100)
        p_ok = clamp(0.98 - diff / 100 * 0.40 + (c.passing - 70) / 350 + (t.strength - 1) * 0.2, 0.08, 0.98)
        ok = rng.random() < p_ok
        speed = {"long_ball": rng.uniform(20, 28), "cross": rng.uniform(20, 26),
                 "through_ball": rng.uniform(14, 22)}.get(ptype, rng.uniform(11, 18))
        dur = d / speed + 0.4
        loft = {"long_ball": 7, "cross": 5}.get(ptype, 0.0)
        if ok:
            err = abs(rng.gauss(0, 1.2))
            ang = rng.uniform(0, 6.28)
            end = (clamp(tp[0] + err * math.cos(ang), 0, L), clamp(tp[1] + err * math.sin(ang), 0, W))
        else:
            err = abs(rng.gauss(7, 4)) + 3
            ang = rng.uniform(0, 6.28)
            end = (tp[0] + err * math.cos(ang), tp[1] + err * math.sin(ang))
        prog = (end[0] - bx) >= 10 and ok
        t.stats["passes"] += 1
        t.stats["pass_ok"] += ok
        c.stats["passes"] += 1
        c.stats["pass_ok"] += ok
        self.log("pass_ok" if ok else "pass_fail", t)
        self.log("action", t)
        offside = ok and ptype == "through_ball" and rng.random() < 0.10
        outcome = "offside" if offside else "complete" if ok else "incomplete"
        ev = self.emit("pass", t, c, pos=(bx, by), end=end, receiver_id=r.pid, receiver_name=f"{r.name} {r.surname}",
                       receiver_jersey=r.jersey, pass_type=ptype, outcome=outcome, distance_m=round(d, 1),
                       direction_deg=round(math.degrees(math.atan2(tp[1] - by, tp[0] - bx)), 1),
                       difficulty_rating=round(diff, 1), completion_prob=round(p_ok, 3),
                       target_error_m=round(err, 2), ball_speed_ms=round(speed, 1),
                       ball_speed_kmh=round(speed * 3.6, 1), duration_s=round(dur, 2),
                       progressive=bool(prog), breaks_lines=lines >= 2, lines_broken=min(lines, 4),
                       under_pressure=pressure > 0.4, importance=0.05 + (0.1 if prog else 0) + diff / 1000)
        self.add_trace(ev, t, (bx, by), end, dur, loft)
        if c.stats["passes"] in (50, 100):
            self.milestone(t, c, "passes", f"{int(c.stats['passes'])}_passes")
        if ok and bx < L * 2 / 3 <= end[0]:
            t.stats["f3_entries"] += 1
            self.log("f3", t)
        self.last_pass = dict(team=t, passer=c, t=self.clock, ev=ev, ptype=ptype)
        self.advance(dur + rng.uniform(0.8, 2.2))
        if offside:
            self.emit("offside", t, r, pos=tp, importance=0.15)
            self.restart(o, "free_kick", flip(tp), dt=10)
            return
        if ok:
            self.carrier, self.ball = r, end
            return
        roll = rng.random()
        if roll < 0.62:
            q = min((p for p in o.players if p.pos != "GK"), key=lambda p: dist(theirs[p.pid], end))
            q.stats["interceptions"] += 1
            self.emit("interception", self.opp(t) if False else o, q, pos=flip(end), importance=0.12)
            self.change_possession(o, q, end, "interception")
        elif roll < 0.88:
            ex, ey = clamp(end[0], 0, L), clamp(end[1], 0, W)
            if end[0] > L - 1 or (end[0] > L * .8 and rng.random() < .3):
                self.emit("out_of_play", t, None, pos=(L, clamp(end[1], 0, W)), importance=0.03)
                self.restart(o, "goal_kick", (5.5, W / 2), dt=12)
            else:
                sy = 0.0 if end[1] < W / 2 else W
                self.emit("out_of_play", t, None, pos=(ex, sy), importance=0.03)
                self.restart(o, "throw_in", flip((ex, sy)), dt=8)
        else:
            q = min((p for p in o.players if p.pos != "GK"), key=lambda p: dist(theirs[p.pid], end))
            self.change_possession(o, q, end, "loose_ball")

    def carry_action(self, pressure, presser):
        rng = self.rng
        t, o, c = self.poss, self.opp(self.poss), self.carrier
        x, y = self.ball
        d = rng.uniform(4, 16)
        nx = clamp(x + d * rng.uniform(.4, 1.0), 0, L - 1)
        ny = clamp(y + rng.gauss(0, d * .4), 1, W - 1)
        d = dist((x, y), (nx, ny))
        speed = clamp(rng.gauss(5.5, 1.5) + (c.pace - 70) * .05, 2.5, 9.5)
        dur = d / speed
        self.log("action", t)
        if presser and rng.random() < 0.10 + 0.40 * pressure:
            p_win = clamp(.45 + (presser.tackling - c.pace) / 250 + (o.strength - t.strength) * .2, .15, .85)
            if rng.random() < 0.14:
                presser.stats["fouls"] += 1
                self.emit("tackle", o, presser, pos=flip((x, y)), outcome="foul", victim_id=c.pid, importance=0.12)
                self.advance(1.5)
                self.foul(o, presser, c, (x, y))
                return
            if rng.random() < p_win:
                presser.stats["tackles_won"] += 1
                self.emit("tackle", o, presser, pos=flip((x, y)), outcome="won", victim_id=c.pid, importance=0.12)
                self.advance(1.5)
                self.change_possession(o, presser, (x, y), "tackle")
                return
            presser.stats["tackles_lost"] += 1
            self.emit("tackle", o, presser, pos=flip((x, y)), outcome="lost", victim_id=c.pid, importance=0.08)
            ev = self.emit("dribble", t, c, pos=(x, y), end=(nx, ny), outcome="beat_defender",
                           beaten_player_id=presser.pid, distance_m=round(d, 1), importance=0.18)
        else:
            kmh = speed * 3.6
            ev = self.emit("carry", t, c, pos=(x, y), end=(nx, ny), distance_m=round(d, 1),
                           speed_ms=round(speed, 1), speed_kmh=round(kmh, 1), duration_s=round(dur, 1),
                           speed_alert=kmh >= SPRINT_KMH, under_pressure=pressure > .4, importance=0.06)
        self.add_trace(ev, t, (x, y), (nx, ny), max(dur, .4))
        if x < L * 2 / 3 <= nx:
            t.stats["f3_entries"] += 1
            self.log("f3", t)
        c.stats["top_speed_kmh"] = max(c.stats["top_speed_kmh"], speed * 3.6)
        self.ball = (nx, ny)
        self.advance(dur + 0.5)

    def shot_action(self, pressure, kind="open_play", body=None, fixed_xg=None, pos=None):
        rng = self.rng
        t, o, c = self.poss, self.opp(self.poss), self.carrier
        x, y = pos or self.ball
        dg = math.hypot(L - x, y - W / 2)
        a1 = math.atan2(W / 2 - 3.66 - y, L - x)
        a2 = math.atan2(W / 2 + 3.66 - y, L - x)
        ang = abs(a2 - a1)
        if body is None:
            lp = self.last_pass
            crossed = lp and lp["ptype"] == "cross" and self.clock - lp["t"] < 6 and lp["team"] is t
            body = "head" if crossed and rng.random() < .55 else "foot"
        xg = fixed_xg if fixed_xg is not None else logistic(-1.2 - 0.12 * dg + 2.2 * ang)
        if fixed_xg is None:
            xg *= clamp(0.85 + (c.shooting - 60) / 200, .6, 1.15) * (1 - 0.3 * pressure) * (0.8 if body == "head" else 1.0)
        xg = clamp(xg, 0.005, 0.95)
        speed = clamp(rng.gauss(85, 12) + (c.shooting - 70) * .3 + min(dg, 30) * .4, 55, 135)
        if body == "head":
            speed = clamp(speed * .55, 30, 80)
        dur = dg / (speed / 3.6) + .4
        p_block = 0 if kind == "penalty" else .12 + .25 * pressure
        blocked = rng.random() < p_block
        p_on = clamp(.38 + (c.shooting - 60) / 250 + (.2 if xg > .12 else 0), .2, .88)
        on_target = (not blocked) and rng.random() < p_on
        p_goal = clamp(xg / (p_on * (1 - p_block)), .02, .93)
        goal = on_target and rng.random() < p_goal
        gk = o.gk()
        if blocked:
            outcome, endp = "blocked", (x + rng.uniform(1, 4), y + rng.gauss(0, 2))
        elif goal:
            outcome, endp = "goal", (L, W / 2 + rng.uniform(-3.2, 3.2))
        elif on_target:
            outcome, endp = "saved", (L, W / 2 + rng.uniform(-3.2, 3.2))
        else:
            outcome = "post" if rng.random() < .06 else "off_target"
            endp = (L, W / 2 + rng.choice([-1, 1]) * rng.uniform(3.7, 9))
        assist = None
        lp = self.last_pass
        if lp and lp["team"] is t and self.clock - lp["t"] < 8 and lp["passer"] is not c and kind in ("open_play", "corner"):
            assist = lp["passer"]
            lp["ev"]["tags"].append("key_pass")
        t.stats["shots"] += 1
        t.stats["xg"] += xg
        c.stats["shots"] += 1
        c.stats["xg"] += xg
        if on_target or goal:
            t.stats["shots_on_target"] += 1
        self.log("xg", t, xg)
        self.log("shots", t)
        self.log("action", t)
        tags = ["big_chance"] if xg >= .3 else []
        ev = self.emit("shot", t, c, pos=(x, y), end=endp, kind=kind, body_part=body, outcome=outcome,
                       on_target=bool(on_target or goal), xg=round(xg, 3), distance_to_goal_m=round(dg, 1),
                       shot_angle_deg=round(math.degrees(ang), 1), shot_speed_ms=round(speed / 3.6, 1),
                       shot_speed_kmh=round(speed, 1), end_z_m=round(rng.uniform(0, 2.4), 2),
                       assist_player_id=assist.pid if assist else None, under_pressure=pressure > .4,
                       goalkeeper_id=gk.pid, importance=clamp(.2 + xg, 0, .95), tags=tags)
        self.add_trace(ev, t, (x, y), endp, dur, loft=1.5 if body == "head" else .8)
        if c.stats["shots"] == 5:
            self.milestone(t, c, "shots", "5_shots")
        self.advance(dur)
        if goal:
            self.goal(t, c, assist, xg, ev)
        elif outcome == "saved":
            kind_ = rng.choice(["catch", "dive_parry", "catch", "tip_over"])
            gk.stats["saves"] += 1
            self.emit("save", o, gk, pos=(5, W / 2), save_type=kind_, shot_xg=round(xg, 3), shooter_id=c.pid,
                      importance=.35 + xg)
            r = rng.random()
            if kind_ == "catch" or r < .55:
                self.change_possession(o, gk, (x, y), "keeper_claim")
                self.ball = (5.5, W / 2)
                self.advance(6, active=False)
            elif r < .85:
                self.corner(t)
            else:
                self.rebound(t)
        elif outcome == "blocked":
            if rng.random() < .45:
                self.corner(t)
            else:
                q = self.nearest(o, flip(endp))
                self.change_possession(o, q, endp, "blocked_shot")
        elif outcome == "post":
            if rng.random() < .5:
                self.rebound(t)
            else:
                self.restart(o, "goal_kick", (5.5, W / 2), dt=12)
        else:
            self.restart(o, "goal_kick", (5.5, W / 2), dt=12)

    def rebound(self, t):
        mine = self._pos(t, (L - 8, W / 2))
        cand = [p for p in t.players if p.pos != "GK"]
        p = min(cand, key=lambda q: dist(mine[q.pid], (L - 8, W / 2)))
        self.poss, self.carrier, self.ball = t, p, (L - 9, W / 2 + self.rng.gauss(0, 4))
        self.advance(1.5)

    def goal(self, t, scorer, assist, xg, shot_ev):
        o = self.opp(t)
        before = self.win_probs()
        t.score += 1
        scorer.stats["goals"] += 1
        if assist:
            assist.stats["assists"] += 1
        after = self.win_probs()
        i = 0 if t is self.home else 2
        swing = after[i] - before[i]
        self.emit("goal", t, scorer, pos=(L, W / 2), assist_player_id=assist.pid if assist else None,
                  assist_name=f"{assist.name} {assist.surname}" if assist else None,
                  xg=round(xg, 3), scoreline=f"{self.home.score}-{self.away.score}",
                  win_prob_before=round(before[i], 3), win_prob_after=round(after[i], 3),
                  win_prob_swing=round(swing, 3), importance=clamp(.8 + abs(swing), 0, 1.0),
                  tags=["goal"] + (["equaliser"] if self.home.score == self.away.score else [])
                  + (["go_ahead"] if abs(self.home.score - self.away.score) == 1 and t.score > o.score else []))
        g = int(scorer.stats["goals"])
        if g in (1, 2, 3):
            self.milestone(t, scorer, "goals", {1: "first_goal", 2: "brace", 3: "hat_trick"}[g])
        if assist and assist.stats["assists"] == 2:
            self.milestone(t, assist, "assists", "two_assists")
        self.advance(35, active=False)
        k = self.nearest(o, (L / 2, W / 2))
        self.poss, self.carrier, self.ball = o, k, (L / 2, W / 2)
        self.log("change", o)
        self.emit("kickoff", o, k, pos=(L / 2, W / 2), reason="after_goal", importance=.05)

    def corner(self, t):
        rng = self.rng
        o = self.opp(t)
        side = rng.choice([0.0, W])
        taker = max((p for p in t.players if p.pos != "GK"), key=lambda p: p.passing + rng.gauss(0, 5))
        t.stats["corners"] += 1
        self.poss, self.carrier, self.ball = t, taker, (L, side)
        self.emit("corner_kick", t, taker, pos=(L, side), end=(L - 8, W / 2), importance=.2)
        self.advance(8, active=False)
        if rng.random() < .5:
            att = rng.choice([p for p in t.players if p.pos != "GK"])
            self.carrier, self.ball = att, (L - 8, W / 2 + rng.gauss(0, 5))
            self.last_pass = dict(team=t, passer=taker, t=self.clock, ev=self.events[-1], ptype="cross")
            self.shot_action(0.4, kind="corner", body="head", fixed_xg=rng.uniform(.03, .14))
        else:
            d = rng.choice([p for p in o.players if p.pos in ("CB", "LB", "RB", "DM")] or o.players)
            self.emit("clearance", o, d, pos=(8, W / 2), importance=.1)
            self.poss, self.carrier = o, d
            self.ball = (rng.uniform(18, 32), rng.uniform(5, 63))
            self.log("change", o)
            self.advance(3)

    def foul(self, ft, fouler, victim, loc):
        """ft = fouling team, loc = location in the victim's (current possession) frame"""
        rng = self.rng
        vt = self.opp(ft)
        self.emit("foul", ft, fouler, pos=flip(loc), victim_id=victim.pid, importance=.15)
        r = rng.random()
        if r < .01:
            self.emit("red_card", ft, fouler, pos=flip(loc))
            ft.players.remove(fouler)
        elif r < .13:
            fouler.carded += 1
            self.emit("yellow_card", ft, fouler, pos=flip(loc), second_yellow=fouler.carded >= 2)
            if fouler.carded >= 2:
                self.emit("red_card", ft, fouler, pos=flip(loc), reason="second_yellow")
                ft.players.remove(fouler)
        self.advance(10, active=False)
        if loc[0] > L - 16.5 and 13.84 < loc[1] < 54.16:
            taker = max((p for p in vt.players if p.pos != "GK"), key=lambda p: p.shooting)
            self.poss, self.carrier = vt, taker
            self.emit("penalty_awarded", vt, taker, pos=(L - 11, W / 2), importance=.7)
            self.advance(20, active=False)
            self.shot_action(0, kind="penalty", fixed_xg=.76, pos=(L - 11, W / 2))
            return
        taker = max((p for p in vt.players if p.pos != "GK"), key=lambda p: p.passing)
        if vt is not self.poss:
            self.log("change", vt)
        self.poss, self.carrier, self.ball = vt, taker, loc
        self.emit("free_kick", vt, taker, pos=loc, importance=.1)
        self.advance(8, active=False)
        if loc[0] > L * .72 and math.hypot(L - loc[0], loc[1] - W / 2) < 28 and rng.random() < .4:
            self.shot_action(0, kind="free_kick", fixed_xg=rng.uniform(.03, .1))

    def maybe_sub(self):
        for tm in (self.home, self.away):
            sched = self.sub_times[tm.code]
            if sched and self.clock >= sched[0] and tm.subs_used < 3 and tm.bench:
                outs = [p for p in tm.players if p is not self.carrier and p.pos != "GK"]
                if not outs:
                    continue
                off = min(outs, key=lambda p: p.stamina * self.rng.uniform(.7, 1.3))
                cand = [b for b in tm.bench if b.group == off.group and b.pos != "GK"] or [b for b in tm.bench if b.pos != "GK"]
                if not cand:
                    continue
                on = cand[0]
                tm.bench.remove(on)
                on.bx, on.by = off.bx, off.by
                tm.players[tm.players.index(off)] = on
                tm.subs_used += 1
                sched.pop(0)
                self.emit("substitution", tm, on, player_off_id=off.pid, player_off_name=f"{off.name} {off.surname}",
                          player_on_id=on.pid, importance=.2)

    # ---- main loop
    def step(self):
        rng = self.rng
        self.maybe_sub()
        t, c = self.poss, self.carrier
        self.layout()
        pressure, presser = self.pressure()
        x, y = self.ball
        dg = math.hypot(L - x, y - W / 2)
        if presser and pressure > .4 and rng.random() < .45:
            ppos = self.lay[1][presser.pid]
            self.emit("pressure", self.opp(t), presser, pos=ppos, target_player_id=c.pid,
                      intensity=round(pressure, 2), duration_s=round(rng.uniform(1.5, 5), 1), importance=.08)
            self.opp(t).stats["pressures"] += 1
        w_shoot = (0.04 + 0.30 * (1 - dg / 32)) * (c.shooting / 75) if (x > L * .66 and dg < 32) else 0.0
        w_carry = 0.20 + 0.05 * (c.group == "fwd") + 0.1 * t.style["tempo"] * (x < L * .5)
        action = rng.choices(["pass", "carry", "shot"], [0.70, w_carry, w_shoot])[0]
        c.stats["touches"] += 1
        if action == "pass":
            self.pass_action(pressure)
        elif action == "carry":
            self.carry_action(pressure, presser)
        else:
            self.shot_action(pressure)

    def start_period(self, p):
        self.period = p
        self.home.attacks_right = (p == 1)
        self.away.attacks_right = (p == 2)
        if p == 2:
            self.clock = float(HALF)
            self.next_snap = HALF + 60.0
        kicker = self.home if p == 1 else self.away
        self.poss = kicker
        self.carrier = max((q for q in kicker.players if q.group in ("mid", "fwd")), key=lambda q: q.passing)
        self.ball = (L / 2, W / 2)
        self.emit("kickoff", kicker, self.carrier, pos=self.ball, reason="period_start", importance=.1)

    def run(self):
        stop1, stop2 = self.rng.randint(60, 240), self.rng.randint(120, 360)
        for period, limit in ((1, HALF + stop1), (2, 2 * HALF + stop2)):
            self.start_period(period)
            while self.clock < limit:
                self.step()
            self.emit("period_end", importance=.1, stoppage_s=stop1 if period == 1 else stop2)
            if period == 1:
                self.p1_extra = self.clock - HALF
        return self.finish()

    def finish(self):
        self.events.sort(key=lambda e: (e["period"], e["match_clock_s"], e["_seq"]))
        remap = {}
        for i, e in enumerate(self.events):
            remap[e["_seq"]] = f"{self.match_id}-{i + 1:05d}"
            e["event_id"] = remap[e["_seq"]]
        for e in self.events:
            e.pop("_seq")
        for tr in self.tracks:
            tr["event_id"] = remap[tr.pop("_seq")]
        ph, pa = self.home.stats["poss_s"], self.away.stats["poss_s"]

        def team_meta(tm, bench_all):
            return {
                "code": tm.code, "name": tm.name, "side": tm.side, "formation": tm.formation,
                "strength": round(tm.strength, 3), "style": {k: round(v, 2) for k, v in tm.style.items()},
                "score": tm.score,
                "team_stats": {k: round(v, 2) for k, v in tm.stats.items()},
                "roster": [{"player_id": p.pid, "name": f"{p.name} {p.surname}", "jersey": p.jersey, "position": p.pos,
                            "skills": {k: round(getattr(p, k)) for k in ("passing", "shooting", "tackling", "pace", "stamina")},
                            "stats": {k: round(v, 1) for k, v in p.stats.items()}} for p in bench_all],
            }

        rosters = {t.code: t.players + t.bench for t in (self.home, self.away)}
        # include players subbed off: collect from events
        seen = {}
        for t in (self.home, self.away):
            seen[t.code] = list(rosters[t.code])
        meta = {
            "match_id": self.match_id, "kickoff_utc": self.kickoff.isoformat(), "competition": "Synthetic Premier Series",
            "venue": self.rng.choice(VENUES), "attendance": self.rng.randint(18000, 60000),
            "weather": self.rng.choice(["clear", "overcast", "light_rain", "windy"]),
            "referee": f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}",
            "final_score": f"{self.home.score}-{self.away.score}",
            "possession_home_pct": round(100 * ph / (ph + pa), 1),
            "home": team_meta(self.home, self.home.players + self.home.bench),
            "away": team_meta(self.away, self.away.players + self.away.bench),
            "event_count": len(self.events),
        }
        return meta, self.events, self.tracks


# --------------------------------------------------------------------------- fan profiles
def make_fans(rng, n, metas):
    clubs = [m[s]["code"] for m in metas for s in ("home", "away")]
    players = [(p["player_id"], p["name"]) for m in metas for s in ("home", "away") for p in m[s]["roster"]]
    fans = []
    for i in range(n):
        mode = rng.choice(["analyst", "casual", "storyteller", "player_focus", "tactical"])
        pid, pname = rng.choice(players)
        fans.append({
            "fan_id": f"F{i + 1:04d}", "mode": mode, "language": rng.choice(list(I18N)),
            "favorite_club": rng.choice(clubs), "favorite_player_id": pid if mode == "player_focus" or rng.random() < .6 else None,
            "metric_focus": rng.choice(["pass_difficulty", "shot_speed", "distance_covered", "xg", "pressure", "sprints"]),
            "detail_level": {"analyst": "high", "tactical": "high", "casual": "low"}.get(mode, "medium"),
            "overlay_density": rng.choice(["minimal", "standard", "rich"]),
            "spoiler_free": rng.random() < .15,
        })
    return fans


# --------------------------------------------------------------------------- IO
def write_outputs(out, metas, events, tracks, fans):
    out.mkdir(parents=True, exist_ok=True)
    schema = {
        "coordinate_system": {"length_m": L, "width_m": W, "origin": "bottom-left", "unit": "metres"},
        "thresholds": {"sprint_kmh": SPRINT_KMH, "distance_milestones_km": DISTANCE_MILESTONES_KM},
        "video": {"fps": FPS, "timecode": "HH:MM:SS:FF relative to kickoff"},
        "event_types": EVENT_TYPES, "i18n_labels": I18N,
        "notes": "Fictional clubs/players. All data is synthetic.",
    }
    (out / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "matches.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "fans.json").write_text(json.dumps(fans, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "events.jsonl", "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    cols = []
    for e in events:
        for k in e:
            if k not in cols:
                cols.append(k)
    with open(out / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in events:
            row = {k: ("|".join(v) if isinstance(v, list) else v) for k, v in e.items()}
            w.writerow(row)
    if tracks:
        with open(out / "tracking.jsonl", "w", encoding="utf-8") as f:
            for tr in tracks:
                f.write(json.dumps(tr) + "\n")


def live_replay(events, speed):
    last = events[0]["match_clock_s"]
    for e in events:
        if e["type"] == "match_state":
            continue
        time.sleep(max(0, (e["match_clock_s"] - last) / speed))
        last = e["match_clock_s"]
        print(json.dumps(e, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Synthetic football event generator")
    ap.add_argument("--matches", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="synthetic_data")
    ap.add_argument("--tracking", action="store_true", help="also write 5 Hz ball traces")
    ap.add_argument("--fans", type=int, default=20, help="number of synthetic fan profiles")
    ap.add_argument("--live", action="store_true", help="stream the first match to stdout in accelerated real time")
    ap.add_argument("--speed", type=float, default=60.0, help="live replay speed multiplier")
    args = ap.parse_args()

    base_kick = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)
    metas, events, tracks = [], [], []
    for i in range(args.matches):
        rng = random.Random(args.seed * 1000 + i)
        (hn, hc), (an, ac) = rng.sample(CLUBS, 2)
        home, away = build_team(rng, hn, hc, "home"), build_team(rng, an, ac, "away")
        sim = MatchSim(args.seed * 1000 + i, i, home, away, base_kick + timedelta(days=i), tracking=args.tracking)
        meta, ev, tr = sim.run()
        metas.append(meta)
        events += ev
        tracks += tr
        print(f"{meta['match_id']}: {hn} {meta['final_score']} {an} | {len(ev)} events | "
              f"possession {meta['possession_home_pct']}%", file=sys.stderr)
    fans = make_fans(random.Random(args.seed), args.fans, metas)
    out = Path(args.out)
    write_outputs(out, metas, events, tracks, fans)
    print(f"Wrote {len(events)} events to {out}/", file=sys.stderr)
    if args.live:
        live_replay([e for e in events if e["match_id"] == metas[0]["match_id"]], args.speed)


if __name__ == "__main__":
    main()
