"""
Gravity Dam Stability Calculator — Engine
==========================================
All calculation logic. No __main__ block.
Called by main.py (FastAPI).
"""

import io, base64, math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — required for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# =============================================================================
# REGULATORY LIMITS
# =============================================================================

MAX_DAM_HEIGHT_FOR_ROCK_BOLTS = 7.0

LOAD_CASES_USE_L6 = {"MFV", "DFV (no rock bolts)", "HRV+EQ (X-dom)", "HRV+EQ (Y-dom)"}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DamGeometry:
    coordinates:        List[Tuple[float, float]]
    upstream_top_point: Tuple[float, float]

    downstream_toe:         Tuple[float, float]       = field(init=False, default_factory=lambda: (0.0, 0.0))
    upstream_heel:          Tuple[float, float]       = field(init=False, default_factory=lambda: (0.0, 0.0))
    toe_elevation:          float                     = field(init=False, default=0.0)
    heel_elevation:         float                     = field(init=False, default=0.0)
    dam_top_elevation_rel:  float                     = field(init=False, default=0.0)
    base_length_horizontal: float                     = field(init=False, default=0.0)
    us_face:                List[Tuple[float, float]] = field(init=False, default_factory=list)
    ds_face:                List[Tuple[float, float]] = field(init=False, default_factory=list)

    def __post_init__(self):
        coords = [(float(x), float(y)) for x, y in self.coordinates]
        utp    = (float(self.upstream_top_point[0]), float(self.upstream_top_point[1]))

        # ── Heel/toe detection ───────────────────────────────────────────────
        # Rule: vertices with x ≤ utp_x belong to the upstream face;
        #       vertices with x >  utp_x belong to the downstream face.
        # Heel = lowest y on the upstream face.
        # Toe  = lowest y on the downstream face.
        # This is unambiguous because the user always inputs the UTP at the
        # junction of the crest and the upstream face, so upstream = left of UTP.
        n = len(coords)
        if n < 3:
            raise ValueError("Dam polygon must have at least 3 vertices.")

        us_pts = [(x, y) for x, y in coords if x <= utp[0]]
        ds_pts = [(x, y) for x, y in coords if x >  utp[0]]

        if not us_pts or not ds_pts:
            raise ValueError(
                "Could not split polygon into upstream/downstream faces. "
                "Check that upstream_top_point x-coordinate separates the polygon.")

        heel = min(us_pts, key=lambda p: p[1])
        toe  = min(ds_pts, key=lambda p: p[1])

        self.heel_elevation = heel[1]
        self.toe_elevation  = toe[1]

        ox, oy = toe
        mirror = -1.0 if heel[0] < toe[0] else 1.0

        def tr(x, y):
            return (mirror * (x - ox), y - oy)

        self.coordinates        = [tr(x, y) for x, y in coords]
        self.upstream_top_point = tr(utp[0], utp[1])
        self.downstream_toe     = (0.0, 0.0)
        self.upstream_heel      = tr(heel[0], heel[1])

        if self.upstream_heel[0] <= 1e-9:
            raise ValueError(
                f"Heel x = {self.upstream_heel[0]:.4f} is not positive after normalisation. "
                "Check that upstream_top_point is on the correct (upstream) side.")

        self.base_length_horizontal = self.upstream_heel[0]
        self.dam_top_elevation_rel  = max(y for _, y in self.coordinates)
        self._build_faces()

    def _build_faces(self):
        coords = self.coordinates
        n      = len(coords)

        def nearest_idx(pt):
            return int(np.argmin([np.hypot(x - pt[0], y - pt[1]) for x, y in coords]))

        heel_idx   = nearest_idx(self.upstream_heel)
        us_top_idx = nearest_idx(self.upstream_top_point)
        toe_idx    = nearest_idx(self.downstream_toe)

        crest_pts  = [(x, y) for x, y in coords if abs(y - self.dam_top_elevation_rel) < 1e-6]
        ds_crest   = min(crest_pts, key=lambda p: abs(p[0]))
        ds_top_idx = nearest_idx(ds_crest)

        def walk(start, end, direction):
            path, i = [], start
            for _ in range(n + 1):
                path.append(coords[i])
                if i == end:
                    break
                i = (i + direction) % n
            return path

        # US face: walk from heel to us_top.
        # Pick the shorter path that ends at or above start elevation.
        def best_us_face(start, end):
            cands = [walk(start, end, d) for d in [1, -1]]
            cands = [p for p in cands if p[-1][1] >= p[0][1]]
            return min(cands, key=len) if cands else [coords[start], coords[end]]

        # DS face: walk from toe to ds_top.
        # Among valid paths (end y ≥ start y), pick the one whose intermediate
        # vertices all have x > 0 (stay on the downstream slope, never visiting
        # the heel side). This correctly excludes paths that detour through the
        # upstream face when both paths have the same length.
        def best_ds_face(start, end):
            cands = [walk(start, end, d) for d in [1, -1]]
            cands = [p for p in cands if p[-1][1] >= p[0][1]]
            if not cands:
                return [coords[start], coords[end]]
            heel_x = self.upstream_heel[0]
            # Prefer the path whose vertices stay well below heel_x
            # (i.e. don't wander across to the upstream face)
            def max_x(path): return max(p[0] for p in path)
            # DS face vertices should have x < heel_x (they're closer to the toe)
            ds_cands = [p for p in cands if max_x(p) < heel_x - 1e-6]
            if ds_cands:
                return min(ds_cands, key=len)
            # Fallback: pick shorter path
            return min(cands, key=len)

        self.us_face = best_us_face(heel_idx, us_top_idx)
        self.ds_face = best_ds_face(toe_idx,  ds_top_idx)


@dataclass
class WaterLevels:
    HRV_us: float;  DFV_us: float;  MFV_us: float
    HRV_ds: float = 0.0
    DFV_ds: float = 0.0
    MFV_ds: float = 0.0


@dataclass
class MaterialProperties:
    unit_weight_dam:   float = 24.0
    unit_weight_water: float = 10.0
    friction_coeff:    float = 0.70
    gravity:           float = 10.0


@dataclass
class DrainageConfig:
    include:            bool  = False
    distance_from_heel: float = 0.0
    reduction_factor:   float = 0.333


@dataclass
class SiltConfig:
    include:               bool  = False
    height_us:             float = 0.0
    unit_weight_submerged: float = 9.0
    phi_deg:               float = 30.0


@dataclass
class BackfillConfig:
    include_pressure:      bool  = False
    include_weight:        bool  = False
    height:                float = 0.0
    coeff_pressure:        float = 0.333
    unit_weight_dry:       float = 18.0
    unit_weight_wet:       float = 20.0
    unit_weight_submerged: float = 10.0


@dataclass
class IcePressureConfig:
    include:  bool  = False
    pressure: float = 150.0


@dataclass
class RockBoltConfig:
    include:         bool  = False
    force_per_m:     float = 0.0
    cover_from_heel: float = 0.0


@dataclass
class RockAnchorConfig:
    include:         bool  = False
    force_per_m:     float = 0.0
    cover_from_heel: float = 0.0


@dataclass
class AppliedForceConfig:
    vertical_forces:   List[Tuple[float, float]] = field(default_factory=list)
    horizontal_forces: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class EarthquakeConfig:
    include: bool  = False
    a_h:     float = 0.0   # horizontal design acceleration (m/s²)
    a_v:     float = 0.0   # vertical design acceleration (m/s²)


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def polygon_area_centroid(coords):
    n = len(coords); A = cx = cy = 0.0
    for i in range(n):
        x0, y0 = coords[i]; x1, y1 = coords[(i+1)%n]
        c = x0*y1 - x1*y0; A += c
        cx += (x0+x1)*c;   cy += (y0+y1)*c
    A /= 2.0
    if abs(A) < 1e-12:
        return 0.0, 0.0, 0.0
    return abs(A), cx/(6*A), cy/(6*A)


def x_on_face_at_y(face, y_target):
    for i in range(len(face)-1):
        x0, y0 = face[i]; x1, y1 = face[i+1]
        lo, hi = min(y0,y1), max(y0,y1)
        if lo-1e-9 <= y_target <= hi+1e-9:
            if abs(y1-y0) < 1e-9:
                return (x0+x1)/2
            return x0 + (y_target-y0)/(y1-y0)*(x1-x0)
    return face[-1][0] if y_target >= face[-1][1] else face[0][0]


def make_force(name, V=0.0, H=0.0, x=0.0, y=0.0, stabilising=True):
    return dict(name=name, V=V, H=H, x_from_toe=x, y_from_toe=y, stabilising=stabilising)


def build_uplift_pressure_polygon(L, h_us_u, h_ds_u, drainage, Lt):
    """
    Build uplift pressure polygon as a list of (x_from_toe, head).

    Coordinate system (UNCHANGED):
      x = 0.0 → downstream toe
      x = L   → upstream heel
    """

    # End of compression zone (from toe)
    x_cs = L - Lt

    pts = []

    # ─────────────────────────────────────────────
    # NO DRAINAGE
    # ─────────────────────────────────────────────
    if not drainage.include:

        if Lt > 0.0:
            pts = [
                (L,    h_us_u),
                (x_cs, h_us_u),
                (0.0,  h_ds_u),
            ]
        else:
            pts = [
                (L,   h_us_u),
                (0.0, h_ds_u),
            ]

        pts.sort(key=lambda p: p[0])
        return pts

    # ─────────────────────────────────────────────
    # WITH DRAINAGE
    # ─────────────────────────────────────────────
    d  = drainage.distance_from_heel
    rf = drainage.reduction_factor

    # Drain position (from toe)
    xd = L - d

    # Head at drain
    hd = h_ds_u + rf * (h_us_u - h_ds_u)

    # ── CASE 1: NO TENSION ───────────────────────
    if Lt <= 0.0:

        pts = [
            (L,  h_us_u),
            (xd, hd),
            (0.0, h_ds_u),
        ]

    # ── CASE 2: TENSION PRESENT ──────────────────
    else:

        # Drain in compression zone → effective
        if xd <= x_cs:
            pts = [
                (L,    h_us_u),
                (x_cs, h_us_u),
                (xd,   hd),
                (0.0,  h_ds_u),
            ]

        # Drain in tension zone → ineffective
        else:
            pts = [
                (L,    h_us_u),
                (x_cs, h_us_u),
                (0.0,  h_ds_u),
            ]

    pts.sort(key=lambda p: p[0])
    return pts

# =============================================================================
# FORCE CALCULATIONS
# =============================================================================

def compute_dam_weight(geom, mat):
    area, cx, cy = polygon_area_centroid(geom.coordinates)
    return make_force('Dam Weight', V=mat.unit_weight_dam*area, x=cx, stabilising=True)


def compute_water_weight_upstream(geom, mat, wl_us_abs):
    gw     = mat.unit_weight_water
    heel   = geom.upstream_heel
    dam_top = geom.dam_top_elevation_rel
    wl_h   = min(wl_us_abs - geom.heel_elevation, dam_top - heel[1])
    if wl_h <= 0:
        return None
    heel_x, heel_y = heel
    x_at_wl = x_on_face_at_y(geom.us_face, heel_y + wl_h)
    if abs(x_at_wl - heel_x) < 1e-6:
        return None
    poly = [(heel_x, heel_y)]
    for x, y in geom.us_face:
        if heel_y + 1e-9 < y < heel_y + wl_h - 1e-9:
            poly.append((x, y))
    poly += [(x_at_wl, heel_y + wl_h), (heel_x, heel_y + wl_h)]
    if len(poly) < 3:
        return None
    area, cx, cy = polygon_area_centroid(poly)
    if area < 1e-9:
        return None
    return make_force('Water Weight (US)', V=gw*area, x=cx, y=cy, stabilising=True)


def compute_water_weight_downstream(geom, mat, wl_ds_abs):
    gw     = mat.unit_weight_water
    dam_top = geom.dam_top_elevation_rel
    wl_h   = min(wl_ds_abs - geom.toe_elevation, dam_top)
    if wl_h <= 0:
        return None
    x_at_wl = x_on_face_at_y(geom.ds_face, wl_h)
    if abs(x_at_wl - 0.0) < 1e-6:
        return None
    poly = [(0.0, 0.0)]
    for x, y in geom.ds_face:
        if 1e-9 < y < wl_h - 1e-9:
            poly.append((x, y))
    poly += [(x_at_wl, wl_h), (0.0, wl_h)]
    if len(poly) < 3:
        return None
    area, cx, cy = polygon_area_centroid(poly)
    if area < 1e-9:
        return None
    return make_force('Water Weight (DS)', V=gw*area, x=cx, y=cy, stabilising=True)


def _trap_pressure_resultant(gw, h_face, h_above_top):
    if h_face < 1e-9:
        return 0.0, 0.0
    p_top  = gw * h_above_top
    p_base = gw * (h_above_top + h_face)
    F      = 0.5 * (p_top + p_base) * h_face
    if F < 1e-9:
        return 0.0, 0.0
    y_bar = h_face/3.0 * (p_base + 2*p_top) / (p_base + p_top)
    return F, y_bar


def compute_horizontal_water_pressure(geom, mat, wl_us_abs, wl_ds_abs):
    forces  = []
    gw      = mat.unit_weight_water
    dam_top = geom.dam_top_elevation_rel
    heel_y  = geom.upstream_heel[1]
    toe_y   = geom.downstream_toe[1]

    h_us = wl_us_abs - geom.heel_elevation
    if h_us > 0:
        h_face = dam_top - heel_y
        h_ot   = max(h_us - h_face, 0.0)
        h_act  = min(h_us, h_face)
        F, ybar = _trap_pressure_resultant(gw, h_act, h_ot)
        if F > 1e-9:
            lbl = 'Water Pressure US (trap)' if h_ot > 0 else 'Water Pressure US (tri)'
            forces.append(make_force(lbl, H=F, x=geom.base_length_horizontal,
                                     y=heel_y + ybar, stabilising=False))

    h_ds = wl_ds_abs - geom.toe_elevation
    if h_ds > 0:
        h_face = dam_top - toe_y
        h_ot   = max(h_ds - h_face, 0.0)
        h_act  = min(h_ds, h_face)
        F, ybar = _trap_pressure_resultant(gw, h_act, h_ot)
        if F > 1e-9:
            lbl = 'Water Pressure DS (trap)' if h_ot > 0 else 'Water Pressure DS (tri)'
            forces.append(make_force(lbl, H=F, x=0.0, y=toe_y + ybar, stabilising=True))

    return forces


def compute_uplift(geom, mat, wl_us_abs, wl_ds_abs, drainage, tension_length=0.0):
    gw = mat.unit_weight_water
    L  = geom.base_length_horizontal
    h_us = max(wl_us_abs - geom.heel_elevation, 0.0)
    h_ds = max(wl_ds_abs - geom.toe_elevation,  0.0)
    Lt   = min(tension_length, L)
    pts  = build_uplift_pressure_polygon(L, h_us, h_ds, drainage, Lt)
    pts  = [(x, gw * h) for x, h in pts]
    poly = [(x, 0.0) for x, _ in pts] + [(x, p) for x, p in reversed(pts)]
    area, cx, cy = polygon_area_centroid(poly)
    if area < 1e-9:
        return make_force('Uplift', V=0.0, x=L/2, stabilising=False)
    return make_force('Uplift', V=-area, x=cx, stabilising=False)


def compute_silt_pressure(geom, mat, silt):
    if not silt.include or silt.height_us <= 0:
        return []
    forces = []
    hs  = silt.height_us
    gs  = silt.unit_weight_submerged
    Ka  = (1 - np.sin(np.radians(silt.phi_deg))) / (1 + np.sin(np.radians(silt.phi_deg)))
    heel_x, heel_y = geom.upstream_heel

    F_h = 0.5 * Ka * gs * hs**2
    if F_h > 1e-9:
        forces.append(make_force('Silt Pressure (horiz)', H=F_h,
            x=geom.base_length_horizontal, y=heel_y + hs/3.0, stabilising=False))

    y_silt  = heel_y + hs
    x_at_hs = x_on_face_at_y(geom.us_face, y_silt)

    poly = [(heel_x, heel_y), (heel_x, y_silt), (x_at_hs, y_silt)]
    # FIX: use reversed() so horizontal ledges are traversed in correct face order
    for x, y in reversed(geom.us_face):
        if heel_y + 1e-9 < y < y_silt - 1e-9:
            poly.append((x, y))
    poly.append((heel_x, heel_y))

    area, cx, cy = polygon_area_centroid(poly)
    if area > 1e-9:
        forces.append(make_force('Silt Weight', V=gs*area, x=cx, y=cy, stabilising=True))
    return forces


def compute_backfill(geom, mat, bf, wl_ds_abs):
    if not (bf.include_pressure or bf.include_weight) or bf.height <= 0:
        return []
    forces  = []
    hb      = bf.height
    Ka      = bf.coeff_pressure
    wl_ds_h = max(wl_ds_abs - geom.toe_elevation, 0.0)
    h_sub   = min(hb, wl_ds_h)
    h_dry   = hb - h_sub
    if bf.include_pressure:
        if h_dry > 0:
            F   = 0.5*Ka*bf.unit_weight_dry*h_dry**2
            y_r = h_sub + h_dry/3                           # resultant height (from base)
            x_r = x_on_face_at_y(geom.ds_face, y_r)        # x on DS face at that height
            forces.append(make_force('Backfill Pressure (dry)', H=F,
                x=x_r, y=y_r, stabilising=True))
        if h_sub > 0:
            sur = Ka*bf.unit_weight_dry*h_dry
            gs  = bf.unit_weight_submerged
            Fr  = sur*h_sub; Ft = 0.5*Ka*gs*h_sub**2; Fs = Fr+Ft
            yc  = (Fr*h_sub/2+Ft*h_sub/3)/Fs if Fs>1e-9 else h_sub/3
            x_r = x_on_face_at_y(geom.ds_face, yc)         # x on DS face at resultant height
            forces.append(make_force('Backfill Pressure (sub)', H=Fs,
                x=x_r, y=yc, stabilising=True))
    if bf.include_weight:
        x_at_hb = x_on_face_at_y(geom.ds_face, hb)
        poly = [(0.0,0.0)]
        for x,y in geom.ds_face:
            if 1e-9 < y < hb-1e-9:
                poly.append((x,y))
        poly += [(x_at_hb,hb),(0.0,hb)]
        if len(poly) >= 3:
            area,cx,cy = polygon_area_centroid(poly)
            if area > 1e-9:
                g_eff = (bf.unit_weight_submerged*h_sub + bf.unit_weight_dry*h_dry)/hb
                forces.append(make_force('Backfill Weight', V=g_eff*area,
                    x=cx, y=cy, stabilising=False))
    return forces


def compute_ice(geom, ice, wl_us_abs):
    if not ice.include:
        return None
    h_us = wl_us_abs - geom.heel_elevation
    if h_us <= 0 or h_us > geom.dam_top_elevation_rel - geom.upstream_heel[1]:
        return None
    y_ice = geom.upstream_heel[1] + max(h_us - 0.25, 0.0)
    return make_force('Ice Pressure', H=ice.pressure,
        x=geom.base_length_horizontal, y=y_ice, stabilising=False)


def compute_rock_bolt(geom, rb):
    if not rb.include or rb.force_per_m < 1e-9:
        return None
    return make_force('Rock Bolt', V=rb.force_per_m,
        x=geom.upstream_heel[0] - rb.cover_from_heel, stabilising=True)


def compute_rock_anchor(geom, ra):
    if not ra.include or ra.force_per_m < 1e-9:
        return None
    return make_force('Rock Anchor', V=ra.force_per_m,
        x=geom.upstream_heel[0] - ra.cover_from_heel, stabilising=True)


def compute_applied(geom, app):
    forces = []; heel_x = geom.upstream_heel[0]
    for i,(F,dist) in enumerate(app.vertical_forces):
        forces.append(make_force(f'Applied V{i+1}', V=F, x=heel_x-dist, stabilising=(F>=0)))
    for i,(F,h) in enumerate(app.horizontal_forces):
        forces.append(make_force(f'Applied H{i+1}', H=abs(F), x=0.0, y=h, stabilising=(F<=0)))
    return forces


# =============================================================================
# STABILITY
# =============================================================================

def moments_and_stability(forces, geom, mat):
    L = geom.base_length_horizontal
    rows = []
    for f in forces:
        V,H  = f['V'],f['H']
        x,y  = f['x_from_toe'],f['y_from_toe']
        stab = f['stabilising']
        Mv_r = V*x if (V>0 and stab)  else 0.0
        Mv_o = V*x if (V>0 and not stab) else (abs(V)*x if V<0 else 0.0)
        Mh_r = H*y if (H>0 and stab)  else 0.0
        Mh_o = H*y if (H>0 and not stab) else 0.0
        rows.append({**f, 'M_res':Mv_r+Mh_r, 'M_ov':Mv_o+Mh_o})

    sum_V     = sum(r['V'] for r in rows)
    sum_M_res = sum(r['M_res'] for r in rows)
    sum_M_ov  = sum(r['M_ov']  for r in rows)
    net_M     = sum_M_res - sum_M_ov
    H_dest    = sum(r['H'] for r in rows if not r['stabilising'])
    H_stab    = sum(r['H'] for r in rows if     r['stabilising'])
    H_net_horiz = H_dest - H_stab   # net horizontal (downstream +ve)

    x_res = net_M/sum_V if abs(sum_V)>1e-6 else L/2
    e     = L/2 - x_res

    if abs(sum_V)>1e-6:
        sigma_toe  = sum_V/L*(1 + 6*e/L)
        sigma_heel = sum_V/L*(1 - 6*e/L)
    else:
        sigma_toe = sigma_heel = 0.0

    # ── Sliding on inclined base ─────────────────────────────────────────────
    # Foundation slope: positive alpha means heel is higher than toe
    heel_x, heel_y = geom.upstream_heel
    toe_y          = geom.downstream_toe[1]
    dx       = heel_x            # horizontal projection of base
    dy       = heel_y - toe_y    # rise from toe to heel (+ve = heel higher)
    base_len = np.hypot(dx, dy)
    sin_a    = dy / base_len if base_len > 1e-9 else 0.0
    cos_a    = dx / base_len if base_len > 1e-9 else 1.0

    # Resolve total resultant onto base plane
    # Internal coords: x points upstream (+ve toward heel), y points upward.
    # Force vector in internal coords: F = (-F_H, -F_V)
    #   (H is downstream = -x; V is downward = -y)
    # Outward normal to base (points away from rock, toward dam body):
    #   n_out = (-sin_a, cos_a)  in internal coords
    # Downstream unit vector along base:
    #   e_down = (-cos_a, -sin_a) in internal coords
    #
    # N = compression (force INTO base) = -F · n_out
    #   = -[(-F_H)(-sin_a) + (-F_V)(cos_a)]
    #   = F_V*cos_a - F_H*sin_a
    #
    # T = shear downstream (driving) = F · e_down
    #   = (-F_H)(-cos_a) + (-F_V)(-sin_a)
    #   = F_H*cos_a + F_V*sin_a
    #
    # Note: when alpha=0 (flat base): N=V, T=H ✓
    # When heel is higher than toe (alpha>0, base slopes down to DS):
    #   weight component adds to T (both gravity and water drive sliding downstream)
    F_V =  sum_V         # net vertical, downward +ve
    F_H =  H_net_horiz   # net horizontal, downstream +ve
    N   =  F_V * cos_a - F_H * sin_a
    T   =  F_H * cos_a + F_V * sin_a

    mu = mat.friction_coeff
    if T > 1e-6:
        FS_slide = mu * max(N, 0.0) / T
    elif T <= 1e-6:
        FS_slide = float('inf')   # shear towards heel or negligible

    H_net = max(H_net_horiz, 0.0)   # keep for display / overturning
    FS_ov = sum_M_res/sum_M_ov if sum_M_ov>1e-6 else float('inf')

    return dict(rows=rows, sum_V=sum_V, H_net=H_net,
                sum_M_res=sum_M_res, sum_M_ov=sum_M_ov, net_M=net_M,
                x_resultant=x_res, eccentricity=e,
                in_middle_third=abs(e)<=L/6,
                heel_tension=sigma_heel<-1e-4,
                sigma_toe=sigma_toe, sigma_heel=sigma_heel,
                FS_sliding=FS_slide, FS_overturning=FS_ov,
                foundation_angle_deg=float(np.degrees(np.arctan2(dy, dx))),
                N_foundation=float(N), T_foundation=float(T))

def compute_earthquake_forces(geom, mat, wl_us, eq):
    """
    Compute earthquake inertia and hydrodynamic forces.

    Inertia horizontal: F_ih = (a_h/g) * W   at dam centroid
    Inertia vertical:   F_iv = (a_v/g) * W   at dam centroid (upward = destabilising)
    Hydrodynamic (Westergaard with cos²θ correction and integrated moment arm):
        p(d) = (7/8) * (a_h/g) * γ_w * √(H*d) * cos²θ   [kPa at depth d below WL]
        F_hd = ∫₀ᴴ p(d) dd  = (7/12) * (a_h/g) * γ_w * H² * cos²θ
        Moment arm from base: ȳ = ∫ p(d)*(H-d) dd / F_hd  = 0.4*H (exact for parabola)
    Returns list of force dicts (may be empty if EQ not included or no water).
    """
    if not eq.include:
        return []
    g = 9.81
    forces = []

    # Dam weight (reuse existing area/centroid)
    area, cx, cy = polygon_area_centroid(geom.coordinates)
    W = mat.unit_weight_dam * area   # kN/m

    # Inertia horizontal (toward downstream = destabilising)
    F_ih = (eq.a_h / g) * W
    if F_ih > 1e-6:
        forces.append(make_force('Inertia (horiz, EQ)',
                                 H=F_ih, x=cx, y=cy, stabilising=False))

    # Inertia vertical (upward = destabilising → negative V)
    F_iv = (eq.a_v / g) * W
    if F_iv > 1e-6:
        forces.append(make_force('Inertia (vert, EQ)',
                                 V=-F_iv, x=cx, y=cy, stabilising=False))

    # Hydrodynamic (Westergaard) — only if upstream water present
    H = wl_us - geom.heel_elevation
    if H > 1e-6:
        # Upstream face angle from vertical
        us_face = geom.us_face
        if len(us_face) >= 2:
            dx = us_face[-1][0] - us_face[0][0]
            dy = us_face[-1][1] - us_face[0][1]
            theta = math.atan2(abs(dx), abs(dy)) if abs(dy) > 1e-9 else math.pi/2
        else:
            theta = 0.0
        cos2_theta = math.cos(theta)**2

        gw = mat.unit_weight_water
        F_hd = (7/12) * (eq.a_h / g) * gw * H**2 * cos2_theta
        # Integrated moment arm for parabolic distribution = 0.4H above base
        y_arm = geom.heel_elevation + 0.4 * H
        if F_hd > 1e-6:
            forces.append(make_force('Hydrodynamic (Westergaard, EQ)',
                                     H=F_hd,
                                     x=geom.base_length_horizontal,
                                     y=y_arm - geom.toe_elevation,
                                     stabilising=False))
    return forces


def assemble_forces(geom, mat, wl_us, wl_ds,
                    drainage, silt, backfill, ice,
                    rock_bolt, rock_anchor, applied,
                    include_rock_bolts, tension_length=0.0,
                    rb_depth_limit_apply=True,
                    rb_depth_limit=MAX_DAM_HEIGHT_FOR_ROCK_BOLTS,
                    earthquake=None):
    forces = []
    forces.append(compute_dam_weight(geom, mat))
    f = compute_water_weight_upstream(geom, mat, wl_us)
    if f: forces.append(f)
    f = compute_water_weight_downstream(geom, mat, wl_ds)
    if f: forces.append(f)
    forces.extend(compute_horizontal_water_pressure(geom, mat, wl_us, wl_ds))
    forces.append(compute_uplift(geom, mat, wl_us, wl_ds, drainage, tension_length))
    forces.extend(compute_silt_pressure(geom, mat, silt))
    forces.extend(compute_backfill(geom, mat, backfill, wl_ds))
    f = compute_ice(geom, ice, wl_us)
    if f: forces.append(f)
    # Earthquake forces (ice excluded per Eurocode 8)
    if earthquake and earthquake.include:
        forces.extend(compute_earthquake_forces(geom, mat, wl_us, earthquake))

    # Depth limit check: uses HRV head (depth of water above heel).
    # If rb_depth_limit_apply is False the limit is ignored — bolts always included.
    _rb_head = wl_us - geom.heel_elevation
    _limit_ok = (not rb_depth_limit_apply) or (_rb_head <= rb_depth_limit)

    if (include_rock_bolts and rock_bolt.include and _limit_ok):
        f = compute_rock_bolt(geom, rock_bolt)
        if f: forces.append(f)

    f = compute_rock_anchor(geom, rock_anchor)
    if f: forces.append(f)
    forces.extend(compute_applied(geom, applied))
    return forces


def generate_messages(case_name, geom, mat, wl_us, wl_ds,
                      drainage, silt, rock_bolt, ice,
                      res, include_rock_bolts, tension_L=0.0,
                      rb_depth_limit_apply=True,
                      rb_depth_limit=MAX_DAM_HEIGHT_FOR_ROCK_BOLTS):
    """
    Produce a concise list of engineering messages.
    Each message: { "type": "info"|"warning"|"alert", "text": "..." }
    """
    msgs = []
    L    = geom.base_length_horizontal
    tL   = tension_L
    h_us = wl_us - geom.heel_elevation
    fs   = res['FS_sliding']

    # ── 1. Rock bolts disabled by depth limit ────────────────────────────────
    hrv_head = wl_us - geom.heel_elevation
    if (rock_bolt.include and include_rock_bolts and rb_depth_limit_apply
            and hrv_head > rb_depth_limit):
        msgs.append({"type": "warning",
                     "text": (f"Rock bolts NOT included: HRV water depth above heel "
                              f"({hrv_head:.2f} m) exceeds the user-set "
                              f"{rb_depth_limit:.1f} m depth limit for rock bolt use.")})

    # ── 2. Inclined base ─────────────────────────────────────────────────────
    if abs(geom.heel_elevation - geom.toe_elevation) > 0.05:
        msgs.append({"type": "info",
                     "text": (f"Inclined base: heel elevation "
                              f"{geom.heel_elevation:.2f} m, toe elevation "
                              f"{geom.toe_elevation:.2f} m. Uplift and base "
                              f"stress are computed on the horizontal projection "
                              f"(L = {L:.3f} m).")})

    # ── 3. Drainage in tension zone → ineffective ────────────────────────────
    if drainage.include and tL > 1e-3:
        xd   = L - drainage.distance_from_heel   # drain position from toe
        x_cs = L - tL                            # start of compression zone from toe
        if xd > x_cs:
            msgs.append({"type": "warning",
                         "text": (f"Drainage curtain is within the tension zone "
                                  f"(drain at {xd:.2f} m from toe; compression "
                                  f"zone starts at {x_cs:.3f} m from toe). "
                                  f"Drain is ineffective — uplift calculated "
                                  f"without drainage reduction.")})

    # ── 4. Silt height above water level ────────────────────────────────────
    if silt.include and silt.height_us > 0 and silt.height_us > h_us + 1e-3:
        msgs.append({"type": "warning",
                     "text": (f"Silt height ({silt.height_us:.2f} m) exceeds "
                              f"upstream water depth ({max(h_us, 0):.2f} m). "
                              f"Only submerged silt is calculated — "
                              f"dry silt above the water level is ignored.")})

    # ── 5. FS_sliding infinite — explain why ────────────────────────────────
    if math.isinf(fs):
        T = res.get('T_foundation', 0.0)
        alpha_deg = res.get('foundation_angle_deg', 0.0)
        if T <= 1e-6 and alpha_deg < -0.5:
            reason = (f"The base slopes upward toward the downstream toe "
                      f"(toe is {abs(geom.toe_elevation - geom.heel_elevation):.2f} m "
                      f"higher than the heel). The dam's weight component along the "
                      f"inclined base acts upstream, fully counteracting the downstream "
                      f"driving forces. Net shear on the base plane = {T:.3f} kN/m ≤ 0 "
                      f"— no sliding tendency in the downstream direction.")
        elif T <= 1e-6 and res.get('H_net', 0.0) <= 0:
            reason = (f"Net horizontal force is zero or acts upstream "
                      f"(H_net = {res.get('H_net',0):.3f} kN/m). "
                      f"No downstream sliding tendency.")
        else:
            reason = (f"Net shear on the base plane = {T:.3f} kN/m ≤ 0 "
                      f"— no downstream sliding tendency under this load combination.")
        msgs.append({"type": "info",
                     "text": f"FS Sliding = ∞: {reason}"})

    # ── 6. FS_sliding below 1.0 ─────────────────────────────────────────────
    if fs < 1.0:
        msgs.append({"type": "alert",
                     "text": (f"CRITICAL: Sliding factor of safety ({fs:.3f}) "
                              f"is below 1.0 — the dam will slide under "
                              f"this load case.")})

    return msgs


def run_load_case(case_name, geom, mat, wl_us, wl_ds,
                  drainage, silt, backfill, ice,
                  rock_bolt, rock_anchor, applied,
                  include_rock_bolts=True,
                  rb_depth_limit_apply=True,
                  rb_depth_limit=MAX_DAM_HEIGHT_FOR_ROCK_BOLTS,
                  earthquake=None,
                  fs_uls=1.5, fs_als=1.1,
                  res_uls='middle_third', res_als='l6'):
    """
    fs_uls / fs_als : user-defined FS sliding thresholds
    res_uls / fs_als: 'middle_third' (L/3–2L/3) or 'l6' (L/6–5L/6)
    """
    L = geom.base_length_horizontal
    tension_L = 0.0
    for _ in range(30):
        forces = assemble_forces(geom, mat, wl_us, wl_ds, drainage, silt,
                                 backfill, ice, rock_bolt, rock_anchor,
                                 applied, include_rock_bolts, tension_L,
                                 rb_depth_limit_apply, rb_depth_limit,
                                 earthquake)
        res = moments_and_stability(forces, geom, mat)
        if not res['heel_tension']:
            break
        st, sh = res['sigma_toe'], res['sigma_heel']
        if st <= 0.0:
            tension_L = L
            forces = assemble_forces(geom, mat, wl_us, wl_ds, drainage, silt,
                                     backfill, ice, rock_bolt, rock_anchor,
                                     applied, include_rock_bolts, tension_L,
                                     rb_depth_limit_apply, rb_depth_limit,
                                     earthquake)
            res = moments_and_stability(forces, geom, mat)
            break
        new_tL = L - st/(st-sh)*L
        if abs(new_tL - tension_L) < 0.001:
            break
        tension_L = new_tL

    xr = res['x_resultant']
    # Determine which resultant criterion applies for this case
    use_l6 = (case_name in LOAD_CASES_USE_L6)
    criterion = res_als if use_l6 else res_uls
    if criterion == 'l6':
        res['in_middle_third']      = (L/6 <= xr <= 5*L/6)
        res['resultant_check_type'] = "L/6–5L/6"
    else:
        res['in_middle_third']      = abs(res['eccentricity']) <= L/6
        res['resultant_check_type'] = "Middle third"

    messages = generate_messages(
        case_name, geom, mat, wl_us, wl_ds,
        drainage, silt, rock_bolt, ice, res,
        include_rock_bolts, tension_L,
        rb_depth_limit_apply, rb_depth_limit)

    res.update(case_name=case_name, tension_length=tension_L,
               wl_us=wl_us, wl_ds=wl_ds, forces=forces,
               messages=messages,
               fs_threshold=(fs_uls if not use_l6 else fs_als))
    return res


# =============================================================================
# PLOT → base64 PNG  (no files written to disk)
# =============================================================================

def plot_to_base64(res, geom, mat, drainage, silt, backfill=None):
    """Render the dam figure and return a base64-encoded PNG string.

    Fixed display scales:
      Pressures  : 1/10   — 1 kPa  = 0.1 m  (e.g. 60 kPa → 6 m polygon width)
      Forces     : 1/10   — 1 kN/m = 0.1 m  (e.g. 200 kN/m → 20 m arrow)
      Resultant  : 1/100  — 1 kN/m = 0.01 m (overall resultant arrow)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import io, base64, math
    import numpy as np

    p_scale         = 1.0   # pressure:  p2w(p)  = p / 10
    f_scale         = 1.0   # force:     f2l(F)  = F / 10
    resultant_scale = 10.0  # resultant: r2l(F)  = F / 100
    kN2t = 1.0 / 10.0

    def p2w(p): return (p * kN2t) / p_scale
    def f2l(F): return (abs(F) * kN2t) / f_scale
    def r2l(F): return (abs(F) * kN2t) / resultant_scale

    L           = geom.base_length_horizontal
    dam_top     = geom.dam_top_elevation_rel
    heel_x, heel_y = geom.upstream_heel
    toe_y       = geom.downstream_toe[1]
    gw          = mat.unit_weight_water
    y_base_ref  = min(y for _, y in geom.coordinates)
    utp_x       = geom.upstream_top_point[0]

    h_us_full = res['wl_us'] - geom.heel_elevation
    h_ds_full = res['wl_ds'] - geom.toe_elevation
    h_us_act  = max(min(h_us_full, dam_top - heel_y), 0.0)
    h_ds_act  = max(min(h_ds_full, dam_top - toe_y),  0.0)
    h_us_u    = max(h_us_full, 0.0)
    h_ds_u    = max(h_ds_full, 0.0)
    Ka_silt   = (math.tan(math.radians(45 - silt.phi_deg / 2))) ** 2 if silt.include else 0.0

    # Outer edges for stacked polygons (pressure, silt, Westergaard, backfill)
    us_water_outer = heel_x + p2w(gw * h_us_act)
    silt_max_w     = p2w(gw * Ka_silt * silt.height_us) if (silt.include and silt.height_us > 0) else 0.0
    wg_anchor      = us_water_outer + silt_max_w   # Westergaard starts where silt ends
    ds_water_outer = -p2w(gw * h_ds_act)           # DS pressure outer edge (negative x)

    # EQ Westergaard geometry
    eq_cfg      = res.get('earthquake')
    eq_forces   = [f for f in res['forces'] if 'Westergaard' in f['name']]
    g_val       = 9.81
    ah_g        = (eq_cfg.a_h / g_val) if eq_cfg else 0.0
    us_face_s   = sorted(geom.us_face, key=lambda p: p[1])
    dx_f = us_face_s[-1][0] - us_face_s[0][0] if len(us_face_s) >= 2 else 0
    dy_f = us_face_s[-1][1] - us_face_s[0][1] if len(us_face_s) >= 2 else 1
    theta_f  = math.atan2(abs(dx_f), abs(dy_f)) if abs(dy_f) > 1e-9 else math.pi / 2
    cos2_f   = math.cos(theta_f) ** 2
    wg_base_w = p2w((7/8) * ah_g * gw * math.sqrt(h_us_act * h_us_act) * cos2_f) if (eq_forces and h_us_act > 0) else 0.0

    # Rock bolt / anchor forces
    rb_f = next((f for f in res['forces'] if 'Rock Bolt'   in f['name']), None)
    ra_f = next((f for f in res['forces'] if 'Rock Anchor' in f['name']), None)
    rb_line_len = f2l(rb_f['V']) if rb_f else 0.0   # below base, 1/10 scale
    anchor_ext  = 5.0                                 # RA extends 5 m below base

    # Backfill
    bf_forces    = [f for f in res['forces'] if 'Backfill Pressure' in f['name']]
    bf_max_w     = max((p2w(f['H']) for f in bf_forces), default=0.0) * 1.0

    # Axis limits — include all polygon extents and RB line depth
    p_us_max_w   = p2w(gw * max(h_us_full, 0.0)) * 1.7
    p_ds_max_w   = p2w(gw * max(h_ds_full, 0.0)) * 1.7
    uplift_depth = max(p2w(gw * max(h_us_u, h_ds_u)) * 1.5,
                       rb_line_len + 0.5,
                       anchor_ext + 0.5)
    pad_us = p_us_max_w + silt_max_w + wg_base_w + 1.2
    pad_ds = p_ds_max_w + bf_max_w + 1.0

    fig_w_in = 160.0 / 25.4
    x_lo     = -pad_ds - 0.3
    x_hi     =  heel_x + pad_us + 0.3
    x_span   = x_hi - x_lo
    y_lo     = y_base_ref - uplift_depth - 0.5
    y_hi     = dam_top * 1.45
    y_span   = y_hi - y_lo
    main_h_in = (y_span / x_span) * fig_w_in

    fig = plt.figure(figsize=(fig_w_in, main_h_in))
    ax  = fig.add_subplot(1, 1, 1)

    # ── Water body fills ──────────────────────────────────────────────────────
    def water_polygon_us(wl_h):
        y_wl = heel_y + wl_h; y_crest = dam_top
        face  = sorted(geom.us_face, key=lambda p: p[1])
        if y_wl <= y_crest:
            x_us_wl = x_on_face_at_y(geom.us_face, y_wl)
            wx = [x_hi, x_hi, x_us_wl]
            wy = [heel_y, y_wl, y_wl]
            for fx, fy in reversed(face):
                if heel_y - 1e-9 <= fy <= y_crest + 1e-9: wx.append(fx); wy.append(fy)
        else:
            x_us_cr = x_on_face_at_y(geom.us_face, y_crest)
            wx = [x_hi, x_hi, x_us_cr, x_us_cr]
            wy = [heel_y, y_wl, y_wl, y_crest]
            for fx, fy in reversed(face):
                if heel_y - 1e-9 <= fy <= y_crest + 1e-9: wx.append(fx); wy.append(fy)
        return wx, wy

    def water_polygon_ds(wl_h):
        y_clip  = min(toe_y + wl_h, dam_top)
        x_face  = x_on_face_at_y(geom.ds_face, y_clip)
        face    = sorted([(x, y) for x, y in geom.ds_face if toe_y - 1e-9 <= y <= y_clip + 1e-9], key=lambda p: p[1])
        wx = [0.0]; wy = [toe_y]
        for fx, fy in face[1:]: wx.append(fx); wy.append(fy)
        if abs(face[-1][1] - y_clip) > 1e-6: wx.append(x_face); wy.append(y_clip)
        wx += [x_lo, x_lo]; wy += [y_clip, toe_y]
        return wx, wy

    if h_us_full > 0:
        wx, wy = water_polygon_us(h_us_full)
        ax.fill(wx, wy, color='#aaccff', alpha=0.35, zorder=1)
        wl_y_us = heel_y + h_us_full
        # WL line extends to dam face at WL elevation, or to UTP x-plane if overtopping
        wl_right_x = x_on_face_at_y(geom.us_face, wl_y_us) if wl_y_us <= dam_top else utp_x
        ax.plot([x_hi, wl_right_x], [wl_y_us, wl_y_us], color='#2255cc', lw=1.5, ls='--', zorder=2)
        ax.text(heel_x + (x_hi - heel_x) * 0.05, wl_y_us + dam_top * 0.025,
                f'WL_US={res["wl_us"]:.2f} m  h={h_us_full:.2f} m', fontsize=8, color='#1a3399', zorder=6)

    if h_ds_full > 0:
        wx, wy = water_polygon_ds(min(h_ds_full, dam_top - toe_y))
        ax.fill(wx, wy, color='#aaccff', alpha=0.35, zorder=1)
        wl_y_ds   = toe_y + min(h_ds_full, dam_top - toe_y)
        x_face_wl = x_on_face_at_y(geom.ds_face, wl_y_ds)
        ax.plot([x_lo, x_face_wl], [wl_y_ds, wl_y_ds], color='#2255cc', lw=1.5, ls='--', zorder=2)
        ax.text(x_lo + (x_face_wl - x_lo) * 0.05, wl_y_ds + dam_top * 0.025,
                f'WL_DS={res["wl_ds"]:.2f} m  h={h_ds_full:.2f} m', fontsize=8, color='#1a3399', zorder=6)

    # ── Silt label line (at silt top, from silt outer edge to dam face) ───────
    if silt.include and silt.height_us > 0:
        y_silt = heel_y + silt.height_us
        if y_silt <= dam_top + 1e-6:
            x_sf = x_on_face_at_y(geom.us_face, y_silt)
            ax.plot([us_water_outer + silt_max_w, x_sf], [y_silt, y_silt],
                    color='saddlebrown', lw=0.9, ls=':', zorder=4)
            ax.text(us_water_outer + silt_max_w * 0.5, y_silt + dam_top * 0.02,
                    f'Silt h={silt.height_us:.2f} m', fontsize=8,
                    color='saddlebrown', fontstyle='italic', ha='center', zorder=6)

    # ── US water pressure polygon ─────────────────────────────────────────────
    def draw_pressure_diagram(h_act, h_ot, base_y, face_x, side, color):
        if h_act < 1e-9: return
        sign   = +1.0 if side == 'us' else -1.0
        p_top  = gw * h_ot
        p_base = gw * (h_ot + h_act)
        y_top  = base_y + h_act
        w_base = sign * p2w(p_base); w_top = sign * p2w(p_top)
        ax.fill([face_x, face_x + w_base, face_x + w_top, face_x],
                [base_y, base_y, y_top, y_top], color=color, alpha=0.22, zorder=2)
        ax.plot([face_x + w_base, face_x + w_top], [base_y, y_top], color=color, lw=1.3, zorder=3)
        ax.plot([face_x, face_x], [base_y, y_top], color=color, lw=1.0, zorder=3)
        for k in range(9):
            frac = k / 8; yv = base_y + frac * h_act
            pv = gw * ((h_ot + h_act) - frac * h_act); wv = sign * p2w(pv)
            if abs(wv) > 1e-4:
                ax.annotate('', xy=(face_x, yv), xytext=(face_x + wv, yv),
                            arrowprops=dict(arrowstyle='->', color=color, lw=0.9, mutation_scale=8), zorder=4)
        # No resultant arrow — pressure diagram only

    if h_us_act > 0:
        draw_pressure_diagram(h_us_act, max(h_us_full - (dam_top - heel_y), 0.0),
                              heel_y, heel_x, 'us', 'red')
    if h_ds_act > 0:
        draw_pressure_diagram(h_ds_act, max(h_ds_full - (dam_top - toe_y), 0.0),
                              toe_y, 0.0, 'ds', 'green')

    # ── Silt pressure polygon (starts at us_water_outer, extends upstream) ────
    if silt.include and silt.height_us > 0:
        n_s     = 30
        ys_s    = np.linspace(heel_y, heel_y + silt.height_us, n_s)
        p_s_arr = gw * Ka_silt * (silt.height_us - (ys_s - heel_y))
        w_s_arr = np.array([p2w(p) for p in p_s_arr])
        # Right edge: vertical at us_water_outer; left edge: us_water_outer + width (upstream)
        sx_right = np.full(n_s, us_water_outer)
        sx_left  = us_water_outer + w_s_arr
        poly_x   = np.concatenate([sx_right, sx_left[::-1]])
        poly_y   = np.concatenate([ys_s, ys_s[::-1]])
        ax.fill(poly_x, poly_y, color='saddlebrown', alpha=0.35, zorder=5)
        ax.plot(sx_left, ys_s, color='#5d4037', lw=1.3, zorder=5)
        ax.plot([us_water_outer, us_water_outer], [heel_y, heel_y + silt.height_us],
                color='#5d4037', lw=1.0, ls='--', zorder=4)
        for i in range(0, n_s, 5):
            if w_s_arr[i] > 1e-4:
                ax.annotate('', xy=(us_water_outer, ys_s[i]), xytext=(sx_left[i], ys_s[i]),
                            arrowprops=dict(arrowstyle='->', color='#5d4037', lw=0.9, mutation_scale=7), zorder=5)
        # Silt weight resultant arrow
        sw_f = next((f for f in res['forces'] if f['name'] == 'Silt Weight'), None)
        if sw_f:
            ln = f2l(sw_f['V'])
            ax.annotate('', xy=(sw_f['x_from_toe'], sw_f['y_from_toe']),
                        xytext=(sw_f['x_from_toe'], sw_f['y_from_toe'] + ln),
                        arrowprops=dict(arrowstyle='->', color='saddlebrown', lw=2.5, mutation_scale=14), zorder=7)
            ax.text(sw_f['x_from_toe'] + 0.15, sw_f['y_from_toe'] + ln * 0.5,
                    'W_silt', fontsize=7.5, color='saddlebrown', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7), zorder=7)

    # ── Backfill pressure polygon (starts at ds_water_outer, extends downstream) ──
    if bf_forces:
        # Find height from forces
        bf_h = max((f['y_from_toe'] * 3 for f in bf_forces), default=0.0)  # approx
        # Use the backfill config directly from silt param isn't available — build from forces
        # Reconstruct polygon from force data isn't possible; draw from y=0 to max height
        # Get backfill height from the highest bf force y-arm
        bf_h_act = 0.0
        for f in bf_forces: bf_h_act = max(bf_h_act, f['y_from_toe'] * 3)  # y_from_toe = h/3
        # We can't access backfill config here directly, reconstruct from force magnitude
    # ── Backfill pressure polygon (starts at ds_water_outer, extends downstream) ──
    if backfill is not None and backfill.include_pressure and backfill.height > 0:
        Ka_bf    = backfill.coeff_pressure
        n_bf     = 25
        ys_bf    = np.linspace(toe_y, toe_y + backfill.height, n_bf)
        p_bf_arr = Ka_bf * backfill.unit_weight_dry * (backfill.height - (ys_bf - toe_y))
        w_bf_arr = np.array([p2w(p) for p in p_bf_arr])
        # Right edge: vertical at ds_water_outer (outer edge of DS water pressure)
        # Left edge:  ds_water_outer - width (further downstream = more negative x)
        bx_right = np.full(n_bf, ds_water_outer)
        bx_left  = ds_water_outer - w_bf_arr
        poly_x   = np.concatenate([bx_right, bx_left[::-1]])
        poly_y   = np.concatenate([ys_bf,    ys_bf[::-1]])
        ax.fill(poly_x, poly_y, color='#78909c', alpha=0.35, zorder=2)
        ax.plot(bx_left, ys_bf, color='#37474f', lw=1.3, zorder=3)
        ax.plot([ds_water_outer, ds_water_outer], [toe_y, toe_y + backfill.height],
                color='#37474f', lw=1.0, ls='--', zorder=3)
        for i in range(0, n_bf, 5):
            if w_bf_arr[i] > 1e-4:
                ax.annotate('', xy=(ds_water_outer, ys_bf[i]),
                            xytext=(bx_left[i], ys_bf[i]),
                            arrowprops=dict(arrowstyle='->', color='#37474f',
                                            lw=0.9, mutation_scale=7), zorder=4)
        ax.text(ds_water_outer - w_bf_arr[0] * 0.5, toe_y + backfill.height + dam_top * 0.02,
                'BF press', fontsize=7.5, color='#37474f', fontstyle='italic',
                ha='center', zorder=6)

    # ── Uplift pressure polygon ───────────────────────────────────────────────
    Lt     = min(res['tension_length'], L)
    up_pts = build_uplift_pressure_polygon(L, h_us_u, h_ds_u, drainage, Lt)
    up_xs  = [x for x, _ in up_pts]
    up_pws = [p2w(gw * h) for _, h in up_pts]
    ref_y  = y_base_ref
    poly_xu = [x for x, _ in up_pts] + [x for x, _ in reversed(up_pts)]
    poly_yu = [ref_y] * len(up_pts)  + [ref_y - p2w(gw * h) for _, h in reversed(up_pts)]
    ax.fill(poly_xu, poly_yu, color='orange', alpha=0.25, zorder=2)
    ax.plot(up_xs, [ref_y - pw for pw in up_pws], color='darkorange', lw=1.5, zorder=3)
    for xv in np.linspace(0.0, L, 10):
        pw = float(np.interp(xv, up_xs, up_pws))
        if pw > 1e-4:
            ax.annotate('', xy=(xv, ref_y), xytext=(xv, ref_y - pw),
                        arrowprops=dict(arrowstyle='->', color='darkorange', lw=0.9, mutation_scale=8), zorder=4)

    # ── Westergaard hydrodynamic pressure ────────────────────────────────────
    if eq_forces and h_us_act > 0 and eq_cfg is not None:
        n_pts  = 60
        depths = np.linspace(0, h_us_act, n_pts)
        wl_rel = res['wl_us'] - geom.toe_elevation
        y_rel_wg   = wl_rel - depths
        p_wg       = (7/8) * ah_g * gw * np.sqrt(np.maximum(h_us_act * depths, 0)) * cos2_f
        w_wg       = np.array([p2w(p) for p in p_wg])
        x_anchor   = wg_anchor
        wg_outer_x = x_anchor + w_wg
        right_edge  = np.column_stack([np.full(n_pts, x_anchor), y_rel_wg])
        left_edge   = np.column_stack([wg_outer_x, y_rel_wg])
        wg_poly_pts = np.vstack([right_edge, left_edge[::-1]])
        ax.fill(wg_poly_pts[:, 0], wg_poly_pts[:, 1],
                color='#7d3c98', alpha=0.30, hatch='////', zorder=5)
        ax.plot(wg_outer_x, y_rel_wg, color='#5b2c6f', lw=1.4, zorder=6)
        ax.plot([x_anchor, x_anchor], [y_rel_wg[0], y_rel_wg[-1]],
                color='#5b2c6f', lw=0.8, ls='--', zorder=5)
        # Resultant arrow at 0.4H
        y_arm_wg_rel = geom.upstream_heel[1] + 0.4 * h_us_act
        d_arm_wg     = res['wl_us'] - (y_arm_wg_rel + geom.toe_elevation)
        w_wg_arm     = p2w((7/8) * ah_g * gw * math.sqrt(max(h_us_act * d_arm_wg, 0)) * cos2_f)
        x_wg_outer_arm = x_anchor + w_wg_arm
        F_hd_val = eq_forces[0]['H']
        arr_len  = f2l(F_hd_val)
        ax.annotate('', xy=(x_anchor, y_arm_wg_rel),
                    xytext=(x_wg_outer_arm + arr_len, y_arm_wg_rel),
                    arrowprops=dict(arrowstyle='->', color='#5b2c6f', lw=2.5, mutation_scale=15), zorder=8)
        ax.text(x_wg_outer_arm + arr_len + 0.05, y_arm_wg_rel + dam_top * 0.02,
                'Hydrodyn', fontsize=7.5, color='#5b2c6f', ha='left', zorder=8)

    # ── EQ inertia arrows ────────────────────────────────────────────────────
    ih_forces = [f for f in res['forces'] if 'Inertia (horiz' in f['name']]
    iv_forces = [f for f in res['forces'] if 'Inertia (vert'  in f['name']]
    EQ_COL2   = '#1a5276'
    for f in ih_forces:
        cx_f, cy_f = f['x_from_toe'], f['y_from_toe']
        ln = min(f2l(f['H']), cx_f - 0.3); ln = max(ln, 0.3)
        ax.annotate('', xy=(cx_f - ln, cy_f), xytext=(cx_f, cy_f),
                    arrowprops=dict(arrowstyle='->', color=EQ_COL2, lw=2.5, mutation_scale=15), zorder=7)
        ax.text(cx_f - ln * 0.5, cy_f - dam_top * 0.04, 'F_ih',
                fontsize=7.5, color=EQ_COL2, ha='center',
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7), zorder=7)
    for f in iv_forces:
        cx_f, cy_f = f['x_from_toe'], f['y_from_toe']
        ln = min(f2l(abs(f['V'])), dam_top - cy_f - 0.2); ln = max(ln, 0.3)
        ax.annotate('', xy=(cx_f, cy_f + ln), xytext=(cx_f, cy_f),
                    arrowprops=dict(arrowstyle='->', color=EQ_COL2, lw=2.5, mutation_scale=15), zorder=7)
        ax.text(cx_f + 0.15, cy_f + ln * 0.5, 'F_iv',
                fontsize=7.5, color=EQ_COL2, va='center',
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7), zorder=7)

    # ── Dam body (drawn over pressure polygons) ───────────────────────────────
    xs = [p[0] for p in geom.coordinates] + [geom.coordinates[0][0]]
    ys = [p[1] for p in geom.coordinates] + [geom.coordinates[0][1]]
    ax.fill(xs, ys, color='lightgray', ec='black', lw=1.5, zorder=3)

    # ── Rock bolt ─────────────────────────────────────────────────────────────
    # Cap plate just ABOVE the base; dashed line extends DOWNWARD = f2l(force).
    # Downward arrow at midpoint. No connection to dam face.
    RB_COL = '#b5451b'
    if rb_f:
        rbX      = rb_f['x_from_toe']
        rb_ln    = f2l(rb_f['V'])          # line length below base (1/10 scale)
        # Base elevation at rbX: interpolate on the inclined base segment
        # Base runs from downstream toe (0, toe_y) to upstream heel (heel_x, heel_y)
        rb_base_y = toe_y + (heel_y - toe_y) * (rbX / heel_x) if heel_x > 1e-9 else y_base_ref
        rb_bot   = rb_base_y - rb_ln
        cap_half = 0.25
        cap_top  = rb_base_y + cap_half * 0.6
        # Cap plate above base
        ax.fill([rbX - cap_half, rbX + cap_half, rbX + cap_half, rbX - cap_half],
                [rb_base_y, rb_base_y, cap_top, cap_top], color=RB_COL, zorder=6)
        # Dashed line from base downward
        ax.plot([rbX, rbX], [rb_base_y, rb_bot], color=RB_COL, lw=2.2, ls='--', zorder=5)
        # Downward arrow at midpoint
        rb_mid = (rb_base_y + rb_bot) * 0.5
        ax.annotate('', xy=(rbX, rb_mid - 0.3), xytext=(rbX, rb_mid + 0.3),
                    arrowprops=dict(arrowstyle='->', color=RB_COL, lw=2.2, mutation_scale=13), zorder=6)
        ax.text(rbX + 0.15, rb_mid, 'RB', fontsize=7.5, color=RB_COL, va='center',
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7), zorder=7)

    # ── Rock anchor ───────────────────────────────────────────────────────────
    # Find highest intersection of x=raX with dam polygon (US or DS face).
    # Cap at top, dashed line to 5 m below base, downward arrow at midpoint.
    RA_COL = '#880e4f'
    if ra_f:
        raX  = ra_f['x_from_toe']
        # Find highest y on dam polygon at this x
        def dam_top_y_at_x(x_int):
            pts  = [(p[0], p[1]) for p in geom.coordinates]
            n    = len(pts); best = None
            for i in range(n):
                x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
                if min(x1, x2) - 1e-9 <= x_int <= max(x1, x2) + 1e-9:
                    y = y1 + (x_int - x1) / (x2 - x1) * (y2 - y1) if abs(x2 - x1) > 1e-9 else (y1 + y2) / 2
                    if best is None or y > best: best = y
            return best

        ra_top = dam_top_y_at_x(raX)
        if ra_top is None: ra_top = ref_y
        # Base elevation at raX: interpolate on inclined base segment
        ra_base_y = toe_y + (heel_y - toe_y) * (raX / heel_x) if heel_x > 1e-9 else y_base_ref
        ra_bot = ra_base_y - anchor_ext
        cap_half = 0.3
        # Cap plate at top
        ax.fill([raX - cap_half, raX + cap_half, raX + cap_half, raX - cap_half],
                [ra_top, ra_top, ra_top + cap_half * 0.6, ra_top + cap_half * 0.6],
                color=RA_COL, zorder=6)
        # Dashed line from dam face to 5 m below base
        ax.plot([raX, raX], [ra_top, ra_bot], color=RA_COL, lw=2.2, ls='--', zorder=5)
        # Downward arrow at midpoint
        ra_mid = (ra_top + ra_bot) * 0.5
        ax.annotate('', xy=(raX, ra_mid - 0.3), xytext=(raX, ra_mid + 0.3),
                    arrowprops=dict(arrowstyle='->', color=RA_COL, lw=2.2, mutation_scale=13), zorder=6)
        ax.text(raX + 0.15, ra_mid, 'RA', fontsize=7.5, color=RA_COL, va='center',
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7), zorder=7)

    # ── Generic force arrows ──────────────────────────────────────────────────
    NO_ARROW = {
        'Dam Weight',
        'Water Pressure US (tri)', 'Water Pressure US (trap)',
        'Water Pressure DS (tri)', 'Water Pressure DS (trap)',
        'Uplift',
        'Silt Pressure (horiz)',           # drawn as polygon above
        'Silt Weight',                     # drawn separately above
        'Backfill Pressure (dry)', 'Backfill Pressure (sub)',
        'Rock Bolt', 'Rock Anchor',        # drawn separately above
        'Inertia (horiz, EQ)', 'Inertia (vert, EQ)',
        'Hydrodynamic (Westergaard, EQ)',
    }
    FORCE_COLORS = {
        'Ice Pressure':      '#1a6bcc',
        'Water Weight (US)': '#1565c0',
        'Water Weight (DS)': '#1565c0',
        'Backfill Weight':   '#37474f',
    }
    SHORT = {
        'Water Weight (US)':         'W_w',
        'Water Weight (DS)':         'W_w(DS)',
        'Ice Pressure':              'Ice',
        'Backfill Weight':           'W_BF',
        'Backfill Pressure (dry)':   'BF',
        'Backfill Pressure (sub)':   'BF(sub)',
    }
    def short_label(name):
        if name in SHORT: return SHORT[name]
        if name.startswith('Applied V'): return f'P_v{name[-1]}'
        if name.startswith('Applied H'): return f'P_h{name[-1]}'
        return name.split('(')[0].strip()[:8]

    for f in res['forces']:
        if f['name'] in NO_ARROW: continue
        col  = FORCE_COLORS.get(f['name'], 'green' if f['stabilising'] else 'red')
        x, y = f['x_from_toe'], f['y_from_toe']
        lbl  = short_label(f['name'])
        ap   = dict(arrowstyle='->', color=col, lw=2.5, mutation_scale=16)
        txt  = dict(fontsize=7.5, color=col, va='center',
                    bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7), zorder=7)
        if abs(f['V']) > 1e-6:
            ln = f2l(abs(f['V']))
            if f['V'] > 0:
                ax.annotate('', xy=(x, y), xytext=(x, y + ln), arrowprops=ap, zorder=6)
                ax.text(x + 0.15, y + ln * 0.5, lbl, **txt)
            else:
                ax.annotate('', xy=(x, y + ln), xytext=(x, y), arrowprops=ap, zorder=6)
                ax.text(x + 0.15, y + ln * 0.5, lbl, **txt)
        if abs(f['H']) > 1e-6:
            ln = f2l(abs(f['H']))
            if 'Ice' in f['name']:
                ax.annotate('', xy=(heel_x, y), xytext=(heel_x + ln, y), arrowprops=ap, zorder=6)
                ax.text(heel_x + ln * 0.5, y + dam_top * 0.025, lbl, ha='center', **txt)
            elif 'Applied' in f['name'] and f['stabilising']:
                ax.annotate('', xy=(x + ln, y), xytext=(x, y), arrowprops=ap, zorder=6)
                ax.text(x + ln * 0.5, y + dam_top * 0.025, lbl, ha='center', **txt)
            elif not f['stabilising']:
                ln = min(ln, x - 0.3); ln = max(ln, 0.3)
                ax.annotate('', xy=(x - ln, y), xytext=(x, y), arrowprops=ap, zorder=6)
                ax.text(x - ln * 0.5, y + dam_top * 0.025, lbl, ha='center', **txt)
            else:
                ax.annotate('', xy=(x + ln, y), xytext=(x, y), arrowprops=ap, zorder=6)

    # ── Overall resultant ─────────────────────────────────────────────────────
    xr   = res['x_resultant']
    by_r = y_base_ref
    SV, HN = res['sum_V'], res['H_net']
    if abs(SV) > 1e-6 or abs(HN) > 1e-6:
        dx_r = -r2l(HN); dy_r = -r2l(abs(SV))
        ax.annotate('', xy=(xr, by_r), xytext=(xr - dx_r, by_r - dy_r),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2.5, mutation_scale=14), zorder=8)

    # Middle-third / L/6 zone markers — short ticks at the base line, label immediately above
    if res.get('resultant_check_type') == "L/6–5L/6":
        mt1, mt2 = L / 6.0, 5.0 * L / 6.0
        tl_l, tl_r = "L/6", "5L/6"
    else:
        mt1, mt2 = L / 3.0, 2.0 * L / 3.0
        tl_l, tl_r = "L/3", "2L/3"
    tick_h = dam_top * 0.08   # tick height: 8% of dam height
    for mx, ml in [(mt1, tl_l), (mt2, tl_r)]:
        ax.plot([mx, mx], [by_r, by_r + tick_h],
                color='#7d3c98', lw=1.2, ls='--', alpha=0.8, zorder=5)
        ax.text(mx, by_r + tick_h + dam_top * 0.01, ml,
                fontsize=7, color='#7d3c98', ha='center', va='bottom', zorder=6)

    # ── Axes, labels, title ───────────────────────────────────────────────────
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.invert_xaxis(); ax.set_aspect('equal')
    ax.grid(True, alpha=0.18)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{v + geom.toe_elevation:.0f}'))
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.set_xlabel('← UPSTREAM          Distance from downstream toe (m)          DOWNSTREAM →',
                  fontsize=9)
    title = 'HRV+IS' if res['case_name'] == 'HRV' else res['case_name']
    ax.set_title(f'Load Case: {title}', fontsize=11, fontweight='bold', pad=6)
    # NOTE: do NOT call set_xlim(x_lo, x_hi) here — the axis was already
    # inverted above via invert_xaxis(); re-setting xlim with lo<hi would
    # un-invert it. Only the ylim may be updated safely here.
    ax.text(0.02, 0.97, 'UPSTREAM',   transform=ax.transAxes, fontsize=10,
            color='#1a3399', ha='left', va='top', fontweight='bold')
    ax.text(0.98, 0.97, 'DOWNSTREAM', transform=ax.transAxes, fontsize=10,
            color='#1a3399', ha='right', va='top', fontweight='bold')

    LEFT, RIGHT = 0.12, 0.97
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=0.92, bottom=0.10)

    buf1 = io.BytesIO()
    fig.savefig(buf1, format='png', dpi=130)
    plt.close(fig)
    buf1.seek(0)
    dam_b64 = base64.b64encode(buf1.read()).decode('utf-8')

    return {'dam': dam_b64, 'stress': ''}


# ══════════════════════════════════════════════════════════════════════════════
# DETAILED CALCULATION TEXT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_detailed_calc(res, geom, mat, wl_us, wl_ds,
                            drainage, silt, backfill, ice,
                            rock_bolt, rock_anchor, applied,
                            earthquake=None):
    """
    Return a formatted plain-text detailed calculation for one load case.
    Shows formula → substitution → result for every force.
    """
    import math, numpy as np
    gw   = mat.unit_weight_water
    g_eq = 9.81
    L    = geom.base_length_horizontal
    toe  = geom.toe_elevation
    heel_x, heel_y = geom.upstream_heel
    toe_y = geom.downstream_toe[1]

    lines = []
    def ln(s=''): lines.append(s)
    def hdr1(s):
        ln(); ln('=' * 72)
        ln(f'  {s}')
        ln('=' * 72)
    def hdr2(s):
        ln(); ln(s)
        ln('-' * len(s))
    def row(label, formula, result, unit=''):
        lines.append(f'  {label:<38s}= {formula}')
        lines.append(f'  {"":38s}= {result} {unit}'.rstrip())

    case_display = 'HRV+IS' if res['case_name'] == 'HRV' else res['case_name']
    hdr1(f'LOAD CASE: {case_display}')

    # ── Project / geometry summary ────────────────────────────────────────────
    hdr2('GEOMETRY')
    ln(f'  Toe elevation              = {toe:.3f} m')
    ln(f'  Heel elevation             = {geom.heel_elevation:.3f} m')
    ln(f'  Base length (L)            = {L:.3f} m')
    ln(f'  Dam height                 = {geom.dam_top_elevation_rel:.3f} m')
    ln(f'  US water level (WL_US)     = {wl_us:.3f} m  (h = {wl_us - geom.heel_elevation:.3f} m above heel)')
    ln(f'  DS water level (WL_DS)     = {wl_ds:.3f} m  (h = {max(wl_ds - toe, 0):.3f} m above toe)')
    ln()
    ln(f'  Material:')
    ln(f'    Unit weight dam   γ_c    = {mat.unit_weight_dam:.1f} kN/m³')
    ln(f'    Unit weight water γ_w    = {gw:.1f} kN/m³')
    ln(f'    Friction coeff    μ      = {mat.friction_coeff:.3f}')

    # ── Forces ────────────────────────────────────────────────────────────────
    hdr2('FORCE CALCULATIONS')

    for f in res['forces']:
        name = f['name']
        V, H = f['V'], f['H']
        x, y = f['x_from_toe'], f['y_from_toe']

        ln()
        ln(f'  ┌─ {name} ─')

        # ── Dam Weight ────────────────────────────────────────────────────────
        if name == 'Dam Weight':
            area, cx, cy = polygon_area_centroid(geom.coordinates)
            ln(f'  │  Dam cross-section polygon area:')
            ln(f'  │    A                          = {area:.4f} m²')
            row('│  V = γ_c × A', f'{mat.unit_weight_dam:.1f} × {area:.4f}', f'{V:.3f}', 'kN/m')
            ln(f'  │  Centroid x from toe (cx)     = {cx:.4f} m')
            ln(f'  │  Stabilising → M_res = V × x = {V:.3f} × {x:.4f} = {V*x:.3f} kN·m/m')

        # ── Water Weight (US) ─────────────────────────────────────────────────
        elif name == 'Water Weight (US)':
            h_ww = wl_us - geom.heel_elevation
            ln(f'  │  Upstream water column above heel:')
            ln(f'  │    h                          = {wl_us:.3f} - {geom.heel_elevation:.3f} = {h_ww:.3f} m')
            area_ww = V / gw if gw > 0 else 0
            row('│  V = γ_w × A_water', f'{gw:.1f} × {area_ww:.4f}', f'{V:.3f}', 'kN/m')
            ln(f'  │  Centroid x from toe (cx)     = {x:.4f} m')
            ln(f'  │  Stabilising → M_res = V × x = {V:.3f} × {x:.4f} = {V*x:.3f} kN·m/m')

        # ── Water Weight (DS) ─────────────────────────────────────────────────
        elif name == 'Water Weight (DS)':
            h_ww = max(wl_ds - toe, 0)
            ln(f'  │  Downstream water column above toe:')
            ln(f'  │    h                          = {wl_ds:.3f} - {toe:.3f} = {h_ww:.3f} m')
            area_ww = V / gw if gw > 0 else 0
            row('│  V = γ_w × A_water', f'{gw:.1f} × {area_ww:.4f}', f'{V:.3f}', 'kN/m')
            ln(f'  │  Centroid x from toe (cx)     = {x:.4f} m')
            ln(f'  │  Stabilising → M_res = V × x = {V:.3f} × {x:.4f} = {V*x:.3f} kN·m/m')

        # ── Water Pressure US ─────────────────────────────────────────────────
        elif 'Water Pressure US' in name:
            h_us_full = wl_us - geom.heel_elevation
            dam_h_us  = geom.dam_top_elevation_rel - heel_y
            h_act = min(h_us_full, dam_h_us)
            h_ot  = max(h_us_full - dam_h_us, 0)
            p_top  = gw * h_ot
            p_base = gw * (h_ot + h_act)
            ln(f'  │  Upstream water pressure (trapezoidal if overtopping):')
            ln(f'  │    h_us (full depth)           = {wl_us:.3f} - {geom.heel_elevation:.3f} = {h_us_full:.3f} m')
            ln(f'  │    h_act (on face)             = {h_act:.3f} m')
            if h_ot > 0:
                ln(f'  │    h_ot (above crest)         = {h_ot:.3f} m')
                ln(f'  │    p_top  = γ_w × h_ot        = {gw:.1f} × {h_ot:.3f} = {p_top:.3f} kPa')
            ln(f'  │    p_base = γ_w × h_us         = {gw:.1f} × {h_us_full:.3f} = {p_base:.3f} kPa')
            row('│  F = ½(p_top + p_base) × h_act',
                f'½({p_top:.3f}+{p_base:.3f})×{h_act:.3f}', f'{H:.3f}', 'kN/m')
            F2, ybar = H, y - heel_y
            ln(f'  │  Moment arm (from heel)        = {y:.4f} m')
            ln(f'  │  Overturning → M_ov = F × y   = {H:.3f} × {y:.4f} = {H*y:.3f} kN·m/m')

        # ── Water Pressure DS ─────────────────────────────────────────────────
        elif 'Water Pressure DS' in name:
            h_ds_full = max(wl_ds - toe, 0)
            dam_h_ds  = geom.dam_top_elevation_rel - toe_y
            h_act_ds  = min(h_ds_full, dam_h_ds)
            h_ot_ds   = max(h_ds_full - dam_h_ds, 0)
            p_top_ds  = gw * h_ot_ds
            p_base_ds = gw * (h_ot_ds + h_act_ds)
            ln(f'  │  Downstream water pressure:')
            ln(f'  │    h_ds                        = {wl_ds:.3f} - {toe:.3f} = {h_ds_full:.3f} m')
            ln(f'  │    p_base = γ_w × h_ds         = {gw:.1f} × {h_ds_full:.3f} = {p_base_ds:.3f} kPa')
            row('│  F = ½ × p_base × h_act',
                f'½ × {p_base_ds:.3f} × {h_act_ds:.3f}', f'{H:.3f}', 'kN/m')
            ln(f'  │  Stabilising → M_res = F × y  = {H:.3f} × {y:.4f} = {H*y:.3f} kN·m/m')

        # ── Uplift ────────────────────────────────────────────────────────────
        elif name == 'Uplift':
            h_us_u = max(wl_us - geom.heel_elevation, 0)
            h_ds_u = max(wl_ds - toe, 0)
            p_heel = gw * h_us_u
            p_toe  = gw * h_ds_u
            ln(f'  │  Uplift pressure (trapezoidal):')
            ln(f'  │    p_heel = γ_w × h_us         = {gw:.1f} × {h_us_u:.3f} = {p_heel:.3f} kPa')
            ln(f'  │    p_toe  = γ_w × h_ds         = {gw:.1f} × {h_ds_u:.3f} = {p_toe:.3f} kPa')
            if drainage.include:
                d_x   = drainage.distance_from_heel
                rf    = drainage.reduction_factor
                p_dr  = p_toe + (p_heel - p_toe) * rf
                ln(f'  │  Drainage curtain at {d_x:.2f} m from heel:')
                ln(f'  │    reduction factor           = {rf:.3f}')
                ln(f'  │    p_drain = p_toe + (p_heel - p_toe) × rf')
                ln(f'  │           = {p_toe:.3f} + ({p_heel:.3f} - {p_toe:.3f}) × {rf:.3f} = {p_dr:.3f} kPa')
            area_u = abs(V) / gw if gw > 0 else 0
            row('│  U = γ_w × A_uplift', f'{gw:.1f} × {area_u:.4f}', f'{abs(V):.3f}', 'kN/m  (upward)')
            ln(f'  │  Centroid x from toe           = {x:.4f} m')
            ln(f'  │  Overturning → M_ov = U × x   = {abs(V):.3f} × {x:.4f} = {abs(V)*x:.3f} kN·m/m')

        # ── Silt Pressure ─────────────────────────────────────────────────────
        elif name == 'Silt Pressure (horiz)':
            hs  = silt.height_us
            gs  = silt.unit_weight_submerged
            Ka  = (1 - math.sin(math.radians(silt.phi_deg))) / (1 + math.sin(math.radians(silt.phi_deg)))
            ln(f'  │  Silt height (h_s)             = {hs:.3f} m')
            ln(f'  │  Submerged unit weight γ\'_s    = {gs:.1f} kN/m³')
            ln(f'  │  Friction angle φ              = {silt.phi_deg:.1f}°')
            row('│  Ka = (1-sinφ)/(1+sinφ)', f'(1-sin{silt.phi_deg:.0f}°)/(1+sin{silt.phi_deg:.0f}°)', f'{Ka:.4f}')
            row('│  F = ½ × Ka × γ\'_s × h_s²',
                f'½ × {Ka:.4f} × {gs:.1f} × {hs:.3f}²', f'{H:.3f}', 'kN/m')
            ln(f'  │  Moment arm y = h_s/3          = {hs:.3f}/3 = {hs/3:.4f} m')
            ln(f'  │  Overturning → M_ov = F × y   = {H:.3f} × {y:.4f} = {H*y:.3f} kN·m/m')

        # ── Silt Weight ───────────────────────────────────────────────────────
        elif name == 'Silt Weight':
            gs  = silt.unit_weight_submerged
            area_s = V / gs if gs > 0 else 0
            ln(f'  │  Silt wedge on upstream face:')
            ln(f'  │    Submerged unit weight γ\'_s  = {gs:.1f} kN/m³')
            ln(f'  │    A_silt (wedge area)         = {area_s:.4f} m²')
            row('│  V = γ\'_s × A_silt', f'{gs:.1f} × {area_s:.4f}', f'{V:.3f}', 'kN/m')
            ln(f'  │  Centroid x from toe           = {x:.4f} m')
            ln(f'  │  Stabilising → M_res = V × x  = {V:.3f} × {x:.4f} = {V*x:.3f} kN·m/m')

        # ── Backfill Pressure ─────────────────────────────────────────────────
        elif 'Backfill Pressure' in name:
            hb   = backfill.height
            Ka_b = backfill.coeff_pressure
            h_ds_u = max(wl_ds - toe, 0)
            h_sub  = min(hb, h_ds_u)
            h_dry  = hb - h_sub
            ln(f'  │  Backfill height               = {hb:.3f} m')
            ln(f'  │  K₀ (pressure coefficient)     = {Ka_b:.3f}')
            if 'dry' in name:
                gd = backfill.unit_weight_dry
                ln(f'  │  Dry unit weight γ_dry         = {gd:.1f} kN/m³')
                ln(f'  │  Dry height h_dry              = {h_dry:.3f} m')
                row('│  F = ½ × K₀ × γ_dry × h_dry²',
                    f'½ × {Ka_b:.3f} × {gd:.1f} × {h_dry:.3f}²', f'{H:.3f}', 'kN/m')
            else:
                gs_b = backfill.unit_weight_submerged
                sur  = Ka_b * backfill.unit_weight_dry * h_dry
                Fr   = sur * h_sub
                Ft   = 0.5 * Ka_b * gs_b * h_sub**2
                ln(f'  │  Submerged height h_sub        = {h_sub:.3f} m')
                ln(f'  │  Surcharge from dry layer:')
                ln(f'  │    sur = K₀ × γ_dry × h_dry   = {Ka_b:.3f} × {backfill.unit_weight_dry:.1f} × {h_dry:.3f} = {sur:.3f} kPa')
                ln(f'  │    Fr  = sur × h_sub           = {sur:.3f} × {h_sub:.3f} = {Fr:.3f} kN/m')
                ln(f'  │    Ft  = ½K₀γ\'h_sub²          = ½×{Ka_b:.3f}×{gs_b:.1f}×{h_sub:.3f}² = {Ft:.3f} kN/m')
                ln(f'  │    Fs  = Fr + Ft               = {Fr:.3f} + {Ft:.3f} = {H:.3f} kN/m')
            ln(f'  │  x on DS face (moment arm)     = {x:.4f} m')
            ln(f'  │  Stabilising → M_res = F × x  = {H:.3f} × {x:.4f} = {H*x:.3f} kN·m/m')

        # ── Backfill Weight ───────────────────────────────────────────────────
        elif name == 'Backfill Weight':
            h_ds_u = max(wl_ds - toe, 0)
            h_sub  = min(backfill.height, h_ds_u)
            h_dry  = backfill.height - h_sub
            g_eff  = (backfill.unit_weight_submerged * h_sub + backfill.unit_weight_dry * h_dry) / backfill.height if backfill.height > 0 else 0
            area_b = V / g_eff if g_eff > 0 else 0
            ln(f'  │  Backfill weight wedge on DS face:')
            ln(f'  │    Effective unit weight γ_eff = {g_eff:.3f} kN/m³')
            ln(f'  │    Wedge area A_bf             = {area_b:.4f} m²')
            row('│  V = γ_eff × A_bf', f'{g_eff:.3f} × {area_b:.4f}', f'{V:.3f}', 'kN/m')
            ln(f'  │  Centroid x from toe           = {x:.4f} m')

        # ── Ice ───────────────────────────────────────────────────────────────
        elif name == 'Ice Pressure':
            ln(f'  │  Horizontal ice pressure (user-defined):')
            ln(f'  │    F_ice                       = {H:.3f} kN/m  (applied at WL)')
            ln(f'  │  Height above base y           = {y:.4f} m')
            ln(f'  │  Overturning → M_ov = F × y   = {H:.3f} × {y:.4f} = {H*y:.3f} kN·m/m')

        # ── Rock Bolt ─────────────────────────────────────────────────────────
        elif name == 'Rock Bolt':
            ln(f'  │  Rock bolt — user-defined vertical force:')
            ln(f'  │    Force per unit width        = {V:.3f} kN/m')
            ln(f'  │    Cover from heel             = {rock_bolt.cover_from_heel:.3f} m')
            ln(f'  │    Position x from toe         = {x:.4f} m')
            ln(f'  │  Stabilising → M_res = V × x  = {V:.3f} × {x:.4f} = {V*x:.3f} kN·m/m')

        # ── Rock Anchor ───────────────────────────────────────────────────────
        elif name == 'Rock Anchor':
            ln(f'  │  Rock anchor — user-defined vertical force:')
            ln(f'  │    Force per unit width        = {V:.3f} kN/m')
            ln(f'  │    Cover from heel             = {rock_anchor.cover_from_heel:.3f} m')
            ln(f'  │    Position x from toe         = {x:.4f} m')
            ln(f'  │  Stabilising → M_res = V × x  = {V:.3f} × {x:.4f} = {V*x:.3f} kN·m/m')

        # ── Applied Forces ────────────────────────────────────────────────────
        elif name.startswith('Applied'):
            direction = 'vertical' if abs(V) > abs(H) else 'horizontal'
            magnitude = V if abs(V) > abs(H) else H
            ln(f'  │  Applied {direction} force (user-defined):')
            ln(f'  │    Force                       = {magnitude:.3f} kN/m')
            ln(f'  │    Position x from toe         = {x:.4f} m')
            ln(f'  │    Height above base           = {y:.4f} m')

        # ── EQ Inertia Horizontal ─────────────────────────────────────────────
        elif name == 'Inertia (horiz, EQ)':
            area_eq, cx_eq, cy_eq = polygon_area_centroid(geom.coordinates)
            W_eq = mat.unit_weight_dam * area_eq
            ah   = earthquake.a_h if earthquake else 0
            ln(f'  │  Horizontal earthquake inertia force:')
            ln(f'  │    Dam weight W                = γ_c × A = {mat.unit_weight_dam:.1f} × {area_eq:.4f} = {W_eq:.3f} kN/m')
            ln(f'  │    a_h/g                       = {ah:.4f} / {g_eq:.3f} = {ah/g_eq:.6f}')
            row('│  F_ih = (a_h/g) × W', f'{ah/g_eq:.6f} × {W_eq:.3f}', f'{H:.3f}', 'kN/m')
            ln(f'  │  Applied at dam centroid x     = {x:.4f} m, y = {y:.4f} m')
            ln(f'  │  Overturning → M_ov = F × y   = {H:.3f} × {y:.4f} = {H*y:.3f} kN·m/m')

        # ── EQ Inertia Vertical ───────────────────────────────────────────────
        elif name == 'Inertia (vert, EQ)':
            area_eq, cx_eq, cy_eq = polygon_area_centroid(geom.coordinates)
            W_eq = mat.unit_weight_dam * area_eq
            av   = earthquake.a_v if earthquake else 0
            ln(f'  │  Vertical earthquake inertia force (upward):')
            ln(f'  │    Dam weight W                = {W_eq:.3f} kN/m')
            ln(f'  │    a_v/g                       = {av:.4f} / {g_eq:.3f} = {av/g_eq:.6f}')
            row('│  F_iv = (a_v/g) × W', f'{av/g_eq:.6f} × {W_eq:.3f}', f'{abs(V):.3f}', 'kN/m  (upward)')
            ln(f'  │  Applied at dam centroid x     = {x:.4f} m, y = {y:.4f} m')

        # ── Westergaard ───────────────────────────────────────────────────────
        elif 'Westergaard' in name:
            H_wg = wl_us - geom.heel_elevation
            ah   = earthquake.a_h if earthquake else 0
            us_f = geom.us_face
            if len(us_f) >= 2:
                dx_f = us_f[-1][0] - us_f[0][0]; dy_f = us_f[-1][1] - us_f[0][1]
                theta = math.atan2(abs(dx_f), abs(dy_f)) if abs(dy_f) > 1e-9 else math.pi/2
            else:
                theta = 0.0
            cos2t = math.cos(theta)**2
            F_hd_check = (7/12) * (ah/g_eq) * gw * H_wg**2 * cos2t
            ln(f'  │  Westergaard hydrodynamic pressure:')
            ln(f'  │    H (depth of water)          = {wl_us:.3f} - {geom.heel_elevation:.3f} = {H_wg:.3f} m')
            ln(f'  │    a_h/g                       = {ah/g_eq:.6f}')
            ln(f'  │    θ (face inclination)        = {math.degrees(theta):.2f}°')
            ln(f'  │    cos²θ                       = {cos2t:.6f}')
            row('│  F_hd = (7/12)(a_h/g)γ_w H² cos²θ',
                f'(7/12)×{ah/g_eq:.4f}×{gw:.1f}×{H_wg:.3f}²×{cos2t:.4f}', f'{H:.3f}', 'kN/m')
            y_arm = 0.4 * H_wg
            ln(f'  │  Moment arm = 0.4 × H         = 0.4 × {H_wg:.3f} = {y_arm:.4f} m')
            ln(f'  │  Overturning → M_ov = F × y   = {H:.3f} × {y:.4f} = {H*y:.3f} kN·m/m')

        else:
            ln(f'  │  V = {V:.3f} kN/m   H = {H:.3f} kN/m')
            ln(f'  │  x = {x:.4f} m   y = {y:.4f} m')

        ln(f'  └─')

    # ── Force summary table ───────────────────────────────────────────────────
    hdr2('FORCE SUMMARY')
    ln(f'  {"Force":<38s} {"V (kN/m)":>10s} {"H (kN/m)":>10s} {"x (m)":>8s} {"y (m)":>8s} {"M_res":>10s} {"M_ov":>10s}')
    ln(f'  {"-"*38} {"-"*10} {"-"*10} {"-"*8} {"-"*8} {"-"*10} {"-"*10}')
    for f in res['forces']:
        V_f, H_f = f['V'], f['H']
        x_f, y_f = f['x_from_toe'], f['y_from_toe']
        stab = f['stabilising']
        # Replicate assemble_forces moment logic
        Mv_r = V_f * x_f if (V_f > 0 and stab)     else 0.0
        Mv_o = V_f * x_f if (V_f > 0 and not stab) else (abs(V_f) * x_f if V_f < 0 else 0.0)
        Mh_r = H_f * y_f if (H_f > 0 and stab)     else 0.0
        Mh_o = H_f * y_f if (H_f > 0 and not stab) else 0.0
        m_r = Mv_r + Mh_r; m_o = Mv_o + Mh_o
        ln(f'  {f["name"]:<38s} {V_f:>10.3f} {H_f:>10.3f} {x_f:>8.4f} {y_f:>8.4f} {m_r:>10.3f} {m_o:>10.3f}')
    ln(f'  {"":38s} {"-"*10} {"-"*10} {"":8s} {"":8s} {"-"*10} {"-"*10}')
    sv = res['sum_V']; hn = res['H_net']
    smr = res['sum_M_res']; smo = res['sum_M_ov']
    ln(f'  {"TOTAL / NET":<38s} {sv:>10.3f} {hn:>10.3f} {"":8s} {"":8s} {smr:>10.3f} {smo:>10.3f}')

    # ── Stability calculations ────────────────────────────────────────────────
    hdr2('STABILITY CALCULATIONS')
    import math as _math
    alpha_deg = res.get('foundation_angle_deg', 0.0)
    alpha     = _math.radians(alpha_deg)
    N_f       = res.get('N_foundation', res['sum_V'])
    T_f       = res.get('T_foundation', res['H_net'])
    sv        = res['sum_V']; hn = res['H_net']
    smr       = res['sum_M_res']; smo = res['sum_M_ov']

    ln(f'  Base inclination α         = {alpha_deg:.3f}°')
    if abs(alpha_deg) > 0.01:
        sin_a = _math.sin(alpha); cos_a = _math.cos(alpha)
        ln(f'  sin α = {sin_a:.6f},  cos α = {cos_a:.6f}')
        ln(f'  N = ΣV·cosα - ΣH·sinα')
        ln(f'    = {sv:.3f}·{cos_a:.6f} - {hn:.3f}·{sin_a:.6f}')
        ln(f'    = {N_f:.3f} kN/m  (normal to base)')
        ln(f'  T = ΣH·cosα + ΣV·sinα')
        ln(f'    = {hn:.3f}·{cos_a:.6f} + {sv:.3f}·{sin_a:.6f}')
        ln(f'    = {T_f:.3f} kN/m  (shear along base, downstream +ve)')
    else:
        ln(f'  N = ΣV = {N_f:.3f} kN/m')
        ln(f'  T = ΣH = {T_f:.3f} kN/m')

    fs_s = res['FS_sliding']
    fs_s_str = '∞' if fs_s >= 9998 else f'{fs_s:.4f}'
    ln()
    ln(f'  Sliding:')
    if T_f > 1e-6:
        ln(f'    FS = μ × N / T')
        ln(f'       = {mat.friction_coeff:.3f} × {N_f:.3f} / {T_f:.3f}')
        ln(f'       = {fs_s_str}')
    else:
        ln(f'    Net shear T ≤ 0 → FS = ∞  (no downstream sliding tendency)')
    thr = res.get('fs_threshold', 1.5)
    ok_s = '✓ PASS' if (fs_s >= thr or fs_s >= 9998) else '✗ FAIL'
    ln(f'    Threshold ≥ {thr:.1f}   → {ok_s}')

    ln()
    fs_o = res['FS_overturning']
    fs_o_str = '∞' if fs_o >= 9998 else f'{fs_o:.4f}'
    ln(f'  Overturning:')
    ln(f'    FS = ΣM_res / ΣM_ov')
    ln(f'       = {smr:.3f} / {smo:.3f}')
    ln(f'       = {fs_o_str}')

    ln()
    xr  = res['x_resultant']
    ecc = res['eccentricity']
    L   = geom.base_length_horizontal
    ln(f'  Resultant location:')
    ln(f'    x_res = (ΣM_res - ΣM_ov) / ΣV')
    ln(f'          = ({smr:.3f} - {smo:.3f}) / {sv:.3f}')
    ln(f'          = {xr:.4f} m from downstream toe')
    ln(f'    e     = L/2 - x_res')
    ln(f'          = {L:.3f}/2 - {xr:.4f} = {ecc:.4f} m')
    ct = res.get('resultant_check_type', 'Middle third')
    if 'L/6' in str(ct) or 'l6' in str(ct).lower():
        lo, hi = L/6, 5*L/6
        zone_str = f'L/6 to 5L/6 = {lo:.3f} m to {hi:.3f} m'
    else:
        lo, hi = L/3, 2*L/3
        zone_str = f'L/3 to 2L/3 = {lo:.3f} m to {hi:.3f} m'
    mid_ok = res['in_middle_third']
    ok_r   = '✓ PASS' if mid_ok else '✗ FAIL'
    ln(f'    Zone ({ct}): {zone_str}')
    ln(f'    x_res = {xr:.4f} m → {ok_r}')

    ln()
    s_toe  = res['sigma_toe']
    s_heel = res['sigma_heel']
    ln(f'  Base stress (linear distribution):')
    ln(f'    σ_toe  = ΣV/L × (1 + 6e/L)')
    ln(f'           = {sv:.3f}/{L:.3f} × (1 + 6×{ecc:.4f}/{L:.3f})')
    ln(f'           = {s_toe:.3f} kN/m²')
    ln(f'    σ_heel = ΣV/L × (1 - 6e/L)')
    ln(f'           = {sv:.3f}/{L:.3f} × (1 - 6×{ecc:.4f}/{L:.3f})')
    ln(f'           = {s_heel:.3f} kN/m²')
    tl = res['tension_length']
    if tl > 1e-6:
        ln(f'    Tension zone length = {tl:.4f} m (heel in tension)')

    ln()
    ln('=' * 72)
    return '\n'.join(lines)
