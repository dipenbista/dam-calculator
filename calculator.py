"""
Gravity Dam Stability Calculator — Engine
==========================================
All calculation logic. No __main__ block.
Called by main.py (FastAPI).
"""

import io, base64
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

LOAD_CASES_USE_L6 = {"MFV", "DFV (no rock bolts)"}

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

        y_vals = [y for _, y in coords]
        y_min  = min(y_vals)
        tol    = 0.1
        base_candidates = [(x, y) for x, y in coords if y <= y_min + tol]
        if len(base_candidates) < 2:
            raise ValueError("Could not identify base vertices for heel/toe.")

        toe  = max(base_candidates, key=lambda p: abs(p[0] - utp[0]))
        heel = min(base_candidates, key=lambda p: abs(p[0] - utp[0]))

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

        def best_face(start, end):
            cands = [walk(start, end, d) for d in [1, -1]]
            cands = [p for p in cands if p[-1][1] >= p[0][1]]
            return min(cands, key=len) if cands else [coords[start], coords[end]]

        self.us_face = best_face(heel_idx, us_top_idx)
        self.ds_face = best_face(toe_idx,  ds_top_idx)


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
    x_cs = L - Lt
    pts  = []
    if not drainage.include:
        if Lt > 0.0:
            pts = [(L, h_us_u), (x_cs, h_us_u), (0.0, h_ds_u)]
        else:
            pts = [(L, h_us_u), (0.0, h_ds_u)]
    else:
        d  = drainage.distance_from_heel
        rf = drainage.reduction_factor
        hd = h_ds_u + rf * (h_us_u - h_ds_u)
        xd = L - d
        if Lt == 0.0:
            pts = [(L, h_us_u), (xd, hd), (0.0, h_ds_u)]
        elif Lt < xd:
            pts = [(L, h_us_u), (x_cs, h_us_u), (0.0, h_ds_u)]
        else:
            pts = [(L, h_us_u), (x_cs, h_us_u), (xd, hd), (0.0, h_ds_u)]
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
            F = 0.5*Ka*bf.unit_weight_dry*h_dry**2
            forces.append(make_force('Backfill Pressure (dry)', H=F,
                x=0.0, y=h_sub+h_dry/3, stabilising=True))
        if h_sub > 0:
            sur = Ka*bf.unit_weight_dry*h_dry
            gs  = bf.unit_weight_submerged
            Fr  = sur*h_sub; Ft = 0.5*Ka*gs*h_sub**2; Fs = Fr+Ft
            yc  = (Fr*h_sub/2+Ft*h_sub/3)/Fs if Fs>1e-9 else h_sub/3
            forces.append(make_force('Backfill Pressure (sub)', H=Fs,
                x=0.0, y=yc, stabilising=True))
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
    if h_us <= 0 or h_us >= geom.dam_top_elevation_rel - geom.upstream_heel[1]:
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
    H_net     = max(H_dest-H_stab, 0.0)

    x_res = net_M/sum_V if abs(sum_V)>1e-6 else L/2
    e     = L/2 - x_res

    if abs(sum_V)>1e-6:
        sigma_toe  = sum_V/L*(1 + 6*e/L)
        sigma_heel = sum_V/L*(1 - 6*e/L)
    else:
        sigma_toe = sigma_heel = 0.0

    V_comp   = max(sum_V, 0.0)
    FS_slide = (mat.friction_coeff*V_comp)/H_net if H_net>1e-6 else float('inf')
    FS_ov    = sum_M_res/sum_M_ov if sum_M_ov>1e-6 else float('inf')

    return dict(rows=rows, sum_V=sum_V, H_net=H_net,
                sum_M_res=sum_M_res, sum_M_ov=sum_M_ov, net_M=net_M,
                x_resultant=x_res, eccentricity=e,
                in_middle_third=abs(e)<=L/6,
                heel_tension=sigma_heel<-1e-4,
                sigma_toe=sigma_toe, sigma_heel=sigma_heel,
                FS_sliding=FS_slide, FS_overturning=FS_ov)


def assemble_forces(geom, mat, wl_us, wl_ds,
                    drainage, silt, backfill, ice,
                    rock_bolt, rock_anchor, applied,
                    include_rock_bolts, tension_length=0.0):
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

    if geom.dam_top_elevation_rel > MAX_DAM_HEIGHT_FOR_ROCK_BOLTS and rock_bolt.include:
        pass  # silently skip — warning returned separately

    if (include_rock_bolts and rock_bolt.include
            and geom.dam_top_elevation_rel <= MAX_DAM_HEIGHT_FOR_ROCK_BOLTS):
        f = compute_rock_bolt(geom, rock_bolt)
        if f: forces.append(f)

    f = compute_rock_anchor(geom, rock_anchor)
    if f: forces.append(f)
    forces.extend(compute_applied(geom, applied))
    return forces


def run_load_case(case_name, geom, mat, wl_us, wl_ds,
                  drainage, silt, backfill, ice,
                  rock_bolt, rock_anchor, applied,
                  include_rock_bolts=True):
    L = geom.base_length_horizontal
    tension_L = 0.0
    for _ in range(30):
        forces = assemble_forces(geom, mat, wl_us, wl_ds, drainage, silt,
                                 backfill, ice, rock_bolt, rock_anchor,
                                 applied, include_rock_bolts, tension_L)
        res = moments_and_stability(forces, geom, mat)
        if not res['heel_tension']:
            break
        st, sh = res['sigma_toe'], res['sigma_heel']
        if st <= 0.0:
            tension_L = L
            forces = assemble_forces(geom, mat, wl_us, wl_ds, drainage, silt,
                                     backfill, ice, rock_bolt, rock_anchor,
                                     applied, include_rock_bolts, tension_L)
            res = moments_and_stability(forces, geom, mat)
            break
        new_tL = L - st/(st-sh)*L
        if abs(new_tL - tension_L) < 0.001:
            break
        tension_L = new_tL

    xr = res['x_resultant']
    if case_name in LOAD_CASES_USE_L6:
        res['in_middle_third']      = (L/6 <= xr <= 5*L/6)
        res['resultant_check_type'] = "L/6–5L/6"
    else:
        res['in_middle_third']      = abs(res['eccentricity']) <= L/6
        res['resultant_check_type'] = "Middle third"

    res.update(case_name=case_name, tension_length=tension_L,
               wl_us=wl_us, wl_ds=wl_ds, forces=forces)
    return res


# =============================================================================
# PLOT → base64 PNG  (no files written to disk)
# =============================================================================

def plot_to_base64(res, geom, mat, drainage, silt,
                   p_scale=1.0, f_scale=1.0, resultant_scale=30.0):
    """Render the dam figure and return a base64-encoded PNG string."""
    kN2t = 1.0 / 10.0

    def p2w(p): return (p * kN2t) / p_scale
    def f2l(F): return (F * kN2t) / f_scale
    def r2l(F): return (F * kN2t) / resultant_scale

    L           = geom.base_length_horizontal
    dam_top     = geom.dam_top_elevation_rel
    heel_x, heel_y = geom.upstream_heel
    toe_y       = geom.downstream_toe[1]
    gw          = mat.unit_weight_water
    y_base_ref  = min(y for _, y in geom.coordinates)

    h_us_full = res['wl_us'] - geom.heel_elevation
    h_ds_full = res['wl_ds'] - geom.toe_elevation
    h_us_act  = max(min(h_us_full, dam_top - heel_y), 0.0)
    h_ds_act  = max(min(h_ds_full, dam_top - toe_y),  0.0)
    h_us_u    = max(h_us_full, 0.0)
    h_ds_u    = max(h_ds_full, 0.0)

    p_us_max_w   = p2w(gw * max(h_us_full, 0.0)) * 1.7
    p_ds_max_w   = p2w(gw * max(h_ds_full, 0.0)) * 1.7
    uplift_depth = p2w(gw * max(h_us_u, h_ds_u)) * 1.5
    pad_us = p_us_max_w + 1.0
    pad_ds = p_ds_max_w + 1.0

    fig_w_in    = 160.0 / 25.4
    stress_h_in = 1.4
    x_lo = -pad_ds - 0.3
    x_hi =  heel_x + pad_us + 0.3
    x_span = x_hi - x_lo
    y_lo = y_base_ref - uplift_depth - 0.5
    y_hi = dam_top * 1.45
    y_span = y_hi - y_lo
    main_h_in = (y_span / x_span) * fig_w_in
    gap_in    = 0.55
    fig_h_in  = main_h_in + stress_h_in + gap_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    gs2 = fig.add_gridspec(2, 1, height_ratios=[main_h_in, stress_h_in],
                           hspace=gap_in / fig_h_in)
    ax  = fig.add_subplot(gs2[0])
    ax2 = fig.add_subplot(gs2[1])

    # Dam body
    xs = [p[0] for p in geom.coordinates] + [geom.coordinates[0][0]]
    ys = [p[1] for p in geom.coordinates] + [geom.coordinates[0][1]]
    ax.fill(xs, ys, color='lightgray', ec='black', lw=1.5, zorder=3)

    # Water polygons
    def water_polygon_us(wl_h):
        y_wl = heel_y + wl_h; y_crest = dam_top
        overtopping = y_wl > y_crest
        face = sorted(geom.us_face, key=lambda p: p[1])
        if not overtopping:
            x_us_wl = x_on_face_at_y(geom.us_face, y_wl)
            wx = [x_hi, x_hi, x_us_wl]
            wy = [heel_y, y_wl, y_wl]
            for fx, fy in reversed(face):
                if heel_y - 1e-9 <= fy <= y_crest + 1e-9:
                    wx.append(fx); wy.append(fy)
        else:
            x_us_crest = x_on_face_at_y(geom.us_face, y_crest)
            wx = [x_hi, x_hi, x_us_crest, x_us_crest]
            wy = [heel_y, y_wl, y_wl, y_crest]
            for fx, fy in reversed(face):
                if heel_y - 1e-9 <= fy <= y_crest + 1e-9:
                    wx.append(fx); wy.append(fy)
        return wx, wy

    def water_polygon_ds(wl_h):
        y_clip = min(toe_y + wl_h, dam_top)
        x_face = x_on_face_at_y(geom.ds_face, y_clip)
        face   = sorted([(x,y) for x,y in geom.ds_face if toe_y-1e-9 <= y <= y_clip+1e-9],
                        key=lambda p: p[1])
        wx = [0.0]; wy = [toe_y]
        for fx, fy in face[1:]: wx.append(fx); wy.append(fy)
        if abs(face[-1][1] - y_clip) > 1e-6: wx.append(x_face); wy.append(y_clip)
        wx += [x_lo, x_lo]; wy += [y_clip, toe_y]
        return wx, wy

    if h_us_full > 0:
        wx, wy = water_polygon_us(h_us_full)
        ax.fill(wx, wy, color='#aaccff', alpha=0.35, zorder=1)
        wl_y_us = heel_y + h_us_full
        ax.plot([heel_x, x_hi], [wl_y_us, wl_y_us], color='#2255cc', lw=1.5, ls='--', zorder=2)
        ax.text(heel_x+(x_hi-heel_x)*0.05, wl_y_us+dam_top*0.025,
                f'WL_US={res["wl_us"]:.2f} m  h={h_us_full:.2f} m',
                fontsize=8, color='#1a3399', zorder=6)

    if h_ds_full > 0:
        wx, wy = water_polygon_ds(min(h_ds_full, dam_top-toe_y))
        ax.fill(wx, wy, color='#aaccff', alpha=0.35, zorder=1)
        wl_y_ds = toe_y + min(h_ds_full, dam_top-toe_y)
        x_face_wl = x_on_face_at_y(geom.ds_face, wl_y_ds)
        ax.plot([x_lo, x_face_wl], [wl_y_ds, wl_y_ds], color='#2255cc', lw=1.5, ls='--', zorder=2)
        ax.text(x_lo+(x_face_wl-x_lo)*0.05, wl_y_ds+dam_top*0.025,
                f'WL_DS={res["wl_ds"]:.2f} m  h={h_ds_full:.2f} m',
                fontsize=8, color='#1a3399', zorder=6)

    if silt.include and silt.height_us > 0:
        y_silt = heel_y + silt.height_us
        if y_silt <= dam_top + 1e-6:
            x_sf = x_on_face_at_y(geom.us_face, y_silt)
            ax.plot([x_sf, x_hi], [y_silt, y_silt], color='saddlebrown', lw=1.3, ls=':', zorder=4)
            ax.text(x_sf+0.05*(x_hi-x_sf), y_silt+dam_top*0.02,
                    f'Silt h={silt.height_us:.2f} m', fontsize=8,
                    color='saddlebrown', fontstyle='italic', zorder=6)

    # Pressure diagrams
    def draw_pressure_diagram(h_act, h_ot, base_y, face_x, side, color):
        if h_act < 1e-9: return
        sign   = +1.0 if side == 'us' else -1.0
        p_top  = gw * h_ot
        p_base = gw * (h_ot + h_act)
        y_top  = base_y + h_act
        w_base = sign * p2w(p_base); w_top = sign * p2w(p_top)
        ax.fill([face_x, face_x+w_base, face_x+w_top, face_x],
                [base_y, base_y, y_top, y_top], color=color, alpha=0.22, zorder=2)
        ax.plot([face_x+w_base, face_x+w_top], [base_y, y_top], color=color, lw=1.3, zorder=3)
        ax.plot([face_x, face_x], [base_y, y_top], color=color, lw=1.0, zorder=3)
        for k in range(9):
            frac = k / 8; yv = base_y + frac*h_act
            pv = gw*((h_ot+h_act) - frac*h_act); wv = sign*p2w(pv)
            if abs(wv) > 1e-4:
                ax.annotate('', xy=(face_x, yv), xytext=(face_x+wv, yv),
                            arrowprops=dict(arrowstyle='->', color=color, lw=0.9, mutation_scale=8), zorder=4)
        F, y_bar = _trap_pressure_resultant(gw, h_act, h_ot)
        avg_w = sign * p2w((p_base+p_top)/2.0) * 1.4
        ax.annotate('', xy=(face_x, base_y+y_bar), xytext=(face_x+avg_w, base_y+y_bar),
                    arrowprops=dict(arrowstyle='->', color=color, lw=3.0, mutation_scale=17), zorder=6)

    if h_us_act > 0:
        draw_pressure_diagram(h_us_act, max(h_us_full-(dam_top-heel_y),0.0),
                              heel_y, heel_x, 'us', 'red')
    if h_ds_act > 0:
        draw_pressure_diagram(h_ds_act, max(h_ds_full-(dam_top-toe_y),0.0),
                              toe_y, 0.0, 'ds', 'green')

    # Uplift
    Lt     = min(res['tension_length'], L)
    up_pts = build_uplift_pressure_polygon(L, h_us_u, h_ds_u, drainage, Lt)
    up_xs  = [x for x,_ in up_pts]
    up_pws = [p2w(gw*h) for _,h in up_pts]
    ref_y  = y_base_ref
    ax.fill(up_xs + list(reversed(up_xs)),
            [ref_y-pw for pw in up_pws] + [ref_y]*len(up_xs),
            color='orange', alpha=0.25, zorder=2)
    ax.plot(up_xs, [ref_y-pw for pw in up_pws], color='darkorange', lw=1.5, zorder=3)
    for xv in np.linspace(0.0, L, 10):
        pw = np.interp(xv, up_xs, up_pws)
        if pw > 1e-4:
            ax.annotate('', xy=(xv, ref_y), xytext=(xv, ref_y-pw),
                        arrowprops=dict(arrowstyle='->', color='darkorange', lw=0.9, mutation_scale=8), zorder=4)
    for f in res['forces']:
        if 'Uplift' in f['name']:
            avg_pw = p2w(gw*(h_us_u+h_ds_u)/2.0)*1.4
            ax.annotate('', xy=(f['x_from_toe'], ref_y), xytext=(f['x_from_toe'], ref_y-avg_pw),
                        arrowprops=dict(arrowstyle='->', color='darkorange', lw=3.0, mutation_scale=17), zorder=6)
            break

    # Other forces
    skip = {'Water Pressure US (tri)','Water Pressure US (trap)',
            'Water Pressure DS (tri)','Water Pressure DS (trap)',
            'Dam Weight','Uplift','Silt Pressure (horiz)','Silt Weight'}
    for f in res['forces']:
        if any(s in f['name'] for s in skip): continue
        color = 'green' if f['stabilising'] else 'red'
        x, y  = f['x_from_toe'], f['y_from_toe']
        ap    = dict(arrowstyle='->', color=color, lw=2.2, mutation_scale=15)
        if abs(f['V']) > 1e-6:
            ln = f2l(abs(f['V']))
            if f['V'] > 0: ax.annotate('', xy=(x,y),    xytext=(x,y+ln), arrowprops=ap, zorder=6)
            else:          ax.annotate('', xy=(x,y+ln),  xytext=(x,y),   arrowprops=ap, zorder=6)
        if abs(f['H']) > 1e-6:
            ln = f2l(f['H'])
            if 'Ice' in f['name']:
                ax.annotate('', xy=(heel_x,y), xytext=(heel_x+ln,y), arrowprops=ap, zorder=6)
            elif not f['stabilising']:
                ax.annotate('', xy=(x-ln,y), xytext=(x,y), arrowprops=ap, zorder=6)
            else:
                ax.annotate('', xy=(x+ln,y), xytext=(x,y), arrowprops=ap, zorder=6)

    for f in res['forces']:
        if f['name'] == 'Silt Pressure (horiz)':
            ax.annotate('', xy=(f['x_from_toe'],f['y_from_toe']),
                        xytext=(f['x_from_toe']+f2l(f['H']),f['y_from_toe']),
                        arrowprops=dict(arrowstyle='->', color='saddlebrown', lw=3.0, mutation_scale=18), zorder=7)
        if f['name'] == 'Silt Weight':
            ax.annotate('', xy=(f['x_from_toe'],f['y_from_toe']),
                        xytext=(f['x_from_toe'],f['y_from_toe']+f2l(f['V'])),
                        arrowprops=dict(arrowstyle='->', color='saddlebrown', lw=3.0, mutation_scale=18), zorder=7)

    # Resultant arrow
    xr  = res['x_resultant']
    by_r = heel_y * xr / heel_x if heel_x > 1e-9 else 0.0
    SV, HN = res['sum_V'], res['H_net']
    if abs(SV) > 1e-6 or abs(HN) > 1e-6:
        dx_r = -r2l(HN); dy_r = -r2l(abs(SV))
        ax.annotate('', xy=(xr, by_r), xytext=(xr-dx_r, by_r-dy_r),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=5, mutation_scale=20), zorder=8)

    # Middle-third / L6 ticks
    if res.get('resultant_check_type') == "L/6–5L/6":
        mt1, mt2 = L/6.0, 5.0*L/6.0
        tl_l, tl_r = "L/6", "5L/6"
    else:
        mt1, mt2 = L/3.0, 2.0*L/3.0
        tl_l, tl_r = "L/3", "2L/3"

    bn = np.hypot(heel_x, heel_y)
    nx_b = -heel_y/bn if bn > 1e-9 else 0.0
    ny_b =  heel_x/bn if bn > 1e-9 else 1.0
    tl   = dam_top * 0.06
    def base_y_at(xv): return heel_y * xv / heel_x if heel_x > 1e-9 else 0.0
    for mx, lbl in [(mt1, f"{tl_l}\n{mt1:.2f} m"), (mt2, f"{tl_r}\n{mt2:.2f} m")]:
        my = base_y_at(mx)
        ax.plot([mx-nx_b*tl, mx+nx_b*tl], [my-ny_b*tl, my+ny_b*tl], color='goldenrod', lw=2.0, zorder=3)
        ax.text(mx, my-dam_top*0.08, lbl, fontsize=7, color='goldenrod', ha='center', va='top', zorder=6)

    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.invert_xaxis(); ax.set_aspect('equal')
    elev_offset = geom.toe_elevation
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_: f'{v+elev_offset:.0f}'))
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.tick_params(labelbottom=False)
    ax.grid(True, alpha=0.18)
    ax.text(heel_x+pad_us*0.5, y_hi*0.97, 'UPSTREAM',   fontsize=10, color='#1a3399', ha='center', fontweight='bold', va='top')
    ax.text(-pad_ds*0.5,       y_hi*0.97, 'DOWNSTREAM', fontsize=10, color='#1a3399', ha='center', fontweight='bold', va='top')
    check_name = res.get('resultant_check_type','Middle third')
    mt_str = f"✓ Within {check_name}" if res['in_middle_third'] else f"✗ Outside {check_name}"
    ax.set_title(f"Load Case: {res['case_name']}\n"
                 f"FS_slide={res['FS_sliding']:.3f}  |  FS_overturn={res['FS_overturning']:.3f}  |  {mt_str}",
                 fontsize=10, fontweight='bold', pad=6)

    # Stress subplot
    s_toe = res['sigma_toe']; s_heel = res['sigma_heel']
    def zero_cross(x0,y0,x1,y1):
        if (y0>=0)==(y1>=0): return None
        return x0 - y0*(x1-x0)/(y1-y0)
    xz = zero_cross(0.0, s_toe, L, s_heel)
    def fill2(xa,xb,ya,yb,c): ax2.fill_between([xa,xb],[ya,yb],0,color=c,alpha=0.30,zorder=2)
    if xz is None:
        fill2(0, L, s_toe, s_heel, 'green' if s_toe>=0 else 'red')
    else:
        fill2(0, xz, s_toe, 0,     'green' if s_toe>=0  else 'red')
        fill2(xz, L, 0, s_heel,    'green' if s_heel>=0 else 'red')
        ax2.axvline(xz, color='black', lw=1.0, ls=':', zorder=4)
        ax2.text(xz, 0, f' σ=0 x={xz:.2f}m', fontsize=7, va='bottom')
    ax2.plot([0,L], [s_toe,s_heel], 'k-', lw=2.0, zorder=5)
    ax2.axhline(0, color='black', lw=0.8, zorder=3)
    ax2.plot(0, s_toe,'ko',ms=5,zorder=6); ax2.plot(L,s_heel,'ko',ms=5,zorder=6)
    slim = max(abs(s_toe), abs(s_heel), 10.0)*1.4; off = slim*0.06
    ax2.text(0, s_toe+off,  f'{s_toe:.1f} kN/m²', fontsize=8, ha='left')
    ax2.text(L, s_heel+off, f'{s_heel:.1f} kN/m²',fontsize=8, ha='right')
    ax2.axvspan(mt1,mt2,alpha=0.10,color='gold',zorder=0)
    ax2.axvline(mt1,color='goldenrod',lw=1.5,ls='--',alpha=0.9,zorder=3)
    ax2.axvline(mt2,color='goldenrod',lw=1.5,ls='--',alpha=0.9,zorder=3)
    ax2.axvline(xr, color='purple',   lw=1.5,ls='--',alpha=0.9,zorder=5)
    ax2.text(0,  -slim*0.12,'Toe (DS)', fontsize=8,color='gray',ha='left',  va='top')
    ax2.text(L,  -slim*0.12,'Heel (US)',fontsize=8,color='gray',ha='right', va='top')
    ax2.text(xr, -slim*0.12,f'R x={xr:.2f}m',fontsize=7,color='purple',ha='center',va='top')
    ax2.set_ylim(-slim,slim); ax2.set_xlim(x_lo,x_hi)
    ax2.invert_xaxis()
    ax2.axvline(0.0,color='black',lw=1.0,ls='-',alpha=0.4,zorder=3)
    ax2.axvline(L,  color='black',lw=1.0,ls='-',alpha=0.4,zorder=3)
    ax2.set_ylabel('σ (kN/m²)', fontsize=9)
    ax2.set_xlabel('← UPSTREAM          Distance from downstream toe (m)          DOWNSTREAM →', fontsize=9)
    ax2.set_title('Base Stress Distribution  (green=compression, red=tension)', fontsize=9)
    ax2.grid(True, alpha=0.20)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.08)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
