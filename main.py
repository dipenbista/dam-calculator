"""
main.py — FastAPI backend for the Gravity Dam Stability Calculator
Run with:  uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import traceback
import io
import base64
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import (Font, PatternFill, Alignment,
                              Border, Side, numbers)
from openpyxl.utils import get_column_letter

from calculator import (
    DamGeometry, MaterialProperties, WaterLevels,
    DrainageConfig, SiltConfig, BackfillConfig,
    IcePressureConfig, RockBoltConfig, RockAnchorConfig,
    AppliedForceConfig, run_load_case, plot_to_base64
)

app = FastAPI(title="Gravity Dam Stability Calculator")

# Serve the frontend (index.html and any other static files)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class CoordPoint(BaseModel):
    x: float
    y: float

class LoadCaseRequest(BaseModel):
    # Geometry
    coordinates:         List[CoordPoint]
    upstream_top_point:  CoordPoint

    # Water levels
    HRV_us: float;  DFV_us: float;  MFV_us: float
    HRV_ds: float = 0.0
    DFV_ds: float = 0.0
    MFV_ds: float = 0.0

    # Material
    unit_weight_dam:   float = 24.0
    unit_weight_water: float = 10.0
    friction_coeff:    float = 0.70

    # Drainage
    drainage_include:            bool  = False
    drainage_distance_from_heel: float = 2.0
    drainage_reduction_factor:   float = 0.333

    # Silt
    silt_include:               bool  = False
    silt_height_us:             float = 0.0
    silt_unit_weight_submerged: float = 9.0
    silt_phi_deg:               float = 30.0

    # Backfill
    backfill_include_pressure:      bool  = False
    backfill_include_weight:        bool  = False
    backfill_height:                float = 0.0
    backfill_coeff_pressure:        float = 0.333
    backfill_unit_weight_dry:       float = 18.0
    backfill_unit_weight_wet:       float = 20.0
    backfill_unit_weight_submerged: float = 10.0

    # Ice
    ice_include:   bool  = False
    ice_pressure:  float = 150.0

    # Rock bolt
    rock_bolt_include:         bool  = False
    rock_bolt_force_per_m:     float = 0.0
    rock_bolt_cover_from_heel: float = 0.5

    # Rock anchor
    rock_anchor_include:         bool  = False
    rock_anchor_force_per_m:     float = 0.0
    rock_anchor_cover_from_heel: float = 2.0

    # Applied external forces
    # vertical_forces   : list of [force_kN_per_m, distance_from_toe_m]
    # horizontal_forces : list of [force_kN_per_m, height_above_toe_m]
    #   positive V = downward (stabilising), negative V = upward (destabilising)
    #   positive H = acts toward upstream (destabilising), negative = toward DS (stabilising)
    applied_vertical_forces:   List[List[float]] = Field(default_factory=list)
    applied_horizontal_forces: List[List[float]] = Field(default_factory=list)

    # Which load cases to run
    run_HRV:             bool = True
    run_DFV:             bool = True
    run_MFV:             bool = True
    run_DFV_no_bolts:    bool = False


class ForceRow(BaseModel):
    name:         str
    V:            float
    H:            float
    x_from_toe:   float
    y_from_toe:   float
    M_res:        float
    M_ov:         float
    stabilising:  bool

class CaseResult(BaseModel):
    case_name:            str
    sum_V:                float
    H_net:                float
    sum_M_res:            float
    sum_M_ov:             float
    x_resultant:          float
    eccentricity:         float
    in_middle_third:      bool
    resultant_check_type: str
    sigma_toe:            float
    sigma_heel:           float
    FS_sliding:           float
    FS_overturning:       float
    tension_length:       float
    forces:               List[ForceRow]
    messages:             List[dict]   # engineering messages for this case
    plot_base64:          str          # PNG image encoded as base64

class GeometryInfo(BaseModel):
    toe_elevation:          float
    heel_elevation:         float
    base_length_horizontal: float
    dam_height:             float
    warnings:               List[str]   # kept for backward compat (now empty)

class CalculationResponse(BaseModel):
    geometry: GeometryInfo
    results:  List[CaseResult]


# =============================================================================
# MAIN ENDPOINT
# =============================================================================

@app.post("/calculate", response_model=CalculationResponse)
def calculate(req: LoadCaseRequest):
    try:
        # ── Build geometry ────────────────────────────────────────────
        coords = [(p.x, p.y) for p in req.coordinates]
        utp    = (req.upstream_top_point.x, req.upstream_top_point.y)
        geom   = DamGeometry(coordinates=coords, upstream_top_point=utp)

        # ── Build configs ─────────────────────────────────────────────
        mat = MaterialProperties(
            unit_weight_dam   = req.unit_weight_dam,
            unit_weight_water = req.unit_weight_water,
            friction_coeff    = req.friction_coeff,
        )
        drainage = DrainageConfig(
            include            = req.drainage_include,
            distance_from_heel = req.drainage_distance_from_heel,
            reduction_factor   = req.drainage_reduction_factor,
        )
        silt = SiltConfig(
            include               = req.silt_include,
            height_us             = req.silt_height_us,
            unit_weight_submerged = req.silt_unit_weight_submerged,
            phi_deg               = req.silt_phi_deg,
        )
        backfill = BackfillConfig(
            include_pressure      = req.backfill_include_pressure,
            include_weight        = req.backfill_include_weight,
            height                = req.backfill_height,
            coeff_pressure        = req.backfill_coeff_pressure,
            unit_weight_dry       = req.backfill_unit_weight_dry,
            unit_weight_wet       = req.backfill_unit_weight_wet,
            unit_weight_submerged = req.backfill_unit_weight_submerged,
        )
        ice_base    = IcePressureConfig(include=req.ice_include, pressure=req.ice_pressure)
        rock_bolt   = RockBoltConfig(include=req.rock_bolt_include,
                                     force_per_m=req.rock_bolt_force_per_m,
                                     cover_from_heel=req.rock_bolt_cover_from_heel)
        rock_anchor = RockAnchorConfig(include=req.rock_anchor_include,
                                       force_per_m=req.rock_anchor_force_per_m,
                                       cover_from_heel=req.rock_anchor_cover_from_heel)
        applied     = AppliedForceConfig(
            vertical_forces   = [tuple(v) for v in req.applied_vertical_forces],
            horizontal_forces = [tuple(h) for h in req.applied_horizontal_forces],
        )

        # ── Warnings ──────────────────────────────────────────────────
        warnings = []
        if geom.dam_top_elevation_rel > 7.0 and rock_bolt.include:
            warnings.append(
                f"Rock bolts disabled by regulation: dam height "
                f"{geom.dam_top_elevation_rel:.2f} m > 7.0 m (NVE's retningslinjer for betongdammer).")

        # ── Which cases to run ────────────────────────────────────────
        cases = []
        if req.run_HRV:
            cases.append(('HRV',               req.HRV_us, req.HRV_ds, True))
        if req.run_DFV:
            cases.append(('DFV',               req.DFV_us, req.DFV_ds, True))
        if req.run_MFV:
            cases.append(('MFV',               req.MFV_us, req.MFV_ds, True))
        if req.run_DFV_no_bolts:
            cases.append(('DFV (no rock bolts)', req.DFV_us, req.DFV_ds, False))

        if not cases:
            raise HTTPException(status_code=400, detail="No load cases selected.")

        # ── Run each case ─────────────────────────────────────────────
        results_out = []
        for case_name, wl_us, wl_ds, incl_rb in cases:
            # Ice only for HRV
            ice_case = IcePressureConfig(
                include  = req.ice_include and (case_name == 'HRV'),
                pressure = req.ice_pressure,
            )
            res = run_load_case(
                case_name=case_name, geom=geom, mat=mat,
                wl_us=wl_us, wl_ds=wl_ds,
                drainage=drainage, silt=silt, backfill=backfill,
                ice=ice_case, rock_bolt=rock_bolt, rock_anchor=rock_anchor,
                applied=applied, include_rock_bolts=incl_rb,
            )
            # Generate plot
            plot_b64 = plot_to_base64(res, geom, mat, drainage, silt)

            # Serialize forces
            forces_out = [
                ForceRow(
                    name        = r['name'],
                    V           = round(r['V'],       3),
                    H           = round(r['H'],       3),
                    x_from_toe  = round(r['x_from_toe'], 4),
                    y_from_toe  = round(r['y_from_toe'], 4),
                    M_res       = round(r['M_res'],   3),
                    M_ov        = round(r['M_ov'],    3),
                    stabilising = r['stabilising'],
                )
                for r in res['rows']
            ]

            def safe(v):
                """Replace inf with a large sentinel for JSON serialisation."""
                if v == float('inf'):  return 9999.0
                if v == float('-inf'): return -9999.0
                return round(v, 4)

            results_out.append(CaseResult(
                case_name            = res['case_name'],
                sum_V                = safe(res['sum_V']),
                H_net                = safe(res['H_net']),
                sum_M_res            = safe(res['sum_M_res']),
                sum_M_ov             = safe(res['sum_M_ov']),
                x_resultant          = safe(res['x_resultant']),
                eccentricity         = safe(res['eccentricity']),
                in_middle_third      = res['in_middle_third'],
                resultant_check_type = res['resultant_check_type'],
                sigma_toe            = safe(res['sigma_toe']),
                sigma_heel           = safe(res['sigma_heel']),
                FS_sliding           = safe(res['FS_sliding']),
                FS_overturning       = safe(res['FS_overturning']),
                tension_length       = safe(res['tension_length']),
                forces               = forces_out,
                messages             = res.get('messages', []),
                plot_base64          = plot_b64,
            ))

        geom_info = GeometryInfo(
            toe_elevation          = geom.toe_elevation,
            heel_elevation         = geom.heel_elevation,
            base_length_horizontal = round(geom.base_length_horizontal, 3),
            dam_height             = round(geom.dam_top_elevation_rel,  3),
            warnings               = warnings,
        )
        return CalculationResponse(geometry=geom_info, results=results_out)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# =============================================================================
# EXCEL EXPORT ENDPOINT
# =============================================================================

class ExportRequest(BaseModel):
    """Same calculation response data sent back from the frontend for export."""
    geometry: GeometryInfo
    results:  List[CaseResult]


@app.post("/export")
def export_excel(data: ExportRequest):
    """
    Receive the already-calculated results (including base64 plots),
    build a formatted Excel workbook in memory, and stream it back
    as a downloadable .xlsx file.
    """
    try:
        wb = Workbook()
        wb.remove(wb.active)   # remove default empty sheet

        # ── Colour palette ────────────────────────────────────────────
        BLUE_DARK   = "1A3A5C"
        BLUE_MID    = "2E6DA4"
        BLUE_LIGHT  = "D6E8F7"
        GREEN_DARK  = "1A6B3A"
        GREEN_LIGHT = "D6F0E0"
        RED_DARK    = "8B1A1A"
        RED_LIGHT   = "FAD7D7"
        GOLD        = "F0C040"
        GRAY_LIGHT  = "F5F5F5"
        WHITE       = "FFFFFF"

        def hfill(color):
            return PatternFill("solid", fgColor=color)

        def side():
            return Side(style="thin", color="AAAAAA")

        thin_border = Border(left=side(), right=side(), top=side(), bottom=side())

        def header_font(bold=True, white=True, sz=10):
            return Font(bold=bold, color=WHITE if white else "000000",
                        name="Calibri", size=sz)

        def normal_font(bold=False, sz=9):
            return Font(bold=bold, name="Calibri", size=sz)

        def num_fmt(ws, row, col, value, fmt='0.00', bold=False, fill=None, align='right'):
            c = ws.cell(row=row, column=col, value=value)
            c.number_format = fmt
            c.font          = Font(bold=bold, name="Calibri", size=9)
            c.alignment     = Alignment(horizontal=align, vertical="center")
            c.border        = thin_border
            if fill: c.fill = fill
            return c

        def write_cell(ws, row, col, value, bold=False, fill=None,
                       align='left', color="000000", sz=9, wrap=False):
            c = ws.cell(row=row, column=col, value=value)
            c.font      = Font(bold=bold, name="Calibri", size=sz,
                               color=color)
            c.alignment = Alignment(horizontal=align, vertical="center",
                                    wrap_text=wrap)
            c.border    = thin_border
            if fill: c.fill = fill
            return c

        # ── 1. SUMMARY sheet ─────────────────────────────────────────
        ws_sum = wb.create_sheet("Summary")
        ws_sum.sheet_view.showGridLines = False

        # Title row
        ws_sum.merge_cells("A1:J1")
        t = ws_sum["A1"]
        t.value     = "GRAVITY DAM STABILITY — SUMMARY OF RESULTS"
        t.font      = Font(bold=True, size=14, color=WHITE, name="Calibri")
        t.fill      = hfill(BLUE_DARK)
        t.alignment = Alignment(horizontal="center", vertical="center")
        ws_sum.row_dimensions[1].height = 28

        # Geometry info row
        geo = data.geometry
        geo_txt = (f"Toe elev: {geo.toe_elevation} m  |  "
                   f"Heel elev: {geo.heel_elevation} m  |  "
                   f"Base width: {geo.base_length_horizontal} m  |  "
                   f"Dam height: {geo.dam_height} m")
        ws_sum.merge_cells("A2:J2")
        g = ws_sum["A2"]
        g.value     = geo_txt
        g.font      = Font(size=9, italic=True, name="Calibri", color=BLUE_DARK)
        g.fill      = hfill(BLUE_LIGHT)
        g.alignment = Alignment(horizontal="center", vertical="center")
        ws_sum.row_dimensions[2].height = 18

        # Table headers row 4
        headers = ["Load Case", "FS Sliding", "FS Overturning",
                   "Resultant Check", "x_resultant (m)", "Eccentricity e (m)",
                   "σ_toe (kN/m²)", "σ_heel (kN/m²)",
                   "Tension Length (m)", "ΣV (kN/m)"]
        for col, h in enumerate(headers, 1):
            c = ws_sum.cell(row=4, column=col, value=h)
            c.font      = Font(bold=True, color=WHITE, name="Calibri", size=9)
            c.fill      = hfill(BLUE_MID)
            c.alignment = Alignment(horizontal="center", vertical="center",
                                    wrap_text=True)
            c.border    = thin_border
        ws_sum.row_dimensions[4].height = 30

        # Data rows
        for r_idx, res in enumerate(data.results, 5):
            fs_ok  = res.FS_sliding >= 1.5
            fso_ok = res.FS_overturning >= 1.5
            mid_ok = res.in_middle_third
            sh_ok  = res.sigma_heel >= 0

            row_fill = hfill(GRAY_LIGHT) if r_idx % 2 == 0 else None

            write_cell(ws_sum, r_idx, 1, res.case_name,
                       bold=True, fill=row_fill, align='center')

            def fs_cell(col, val, ok):
                disp = "∞" if val >= 9998 else round(val, 3)
                c = num_fmt(ws_sum, r_idx, col, disp if isinstance(disp, str) else val,
                            fmt='0.000', fill=hfill(GREEN_LIGHT) if ok else hfill(RED_LIGHT))
                if isinstance(disp, str): c.value = disp
                c.alignment = Alignment(horizontal="center", vertical="center")

            fs_cell(2, res.FS_sliding,     fs_ok)
            fs_cell(3, res.FS_overturning, fso_ok)

            # Resultant check
            ck = ws_sum.cell(row=r_idx, column=4,
                             value="✓ OK" if mid_ok else "✗ OUTSIDE")
            ck.font      = Font(bold=True, name="Calibri", size=9,
                                color=GREEN_DARK if mid_ok else RED_DARK)
            ck.fill      = hfill(GREEN_LIGHT) if mid_ok else hfill(RED_LIGHT)
            ck.alignment = Alignment(horizontal="center", vertical="center")
            ck.border    = thin_border

            num_fmt(ws_sum, r_idx, 5,  res.x_resultant,   fmt='0.000', fill=row_fill)
            num_fmt(ws_sum, r_idx, 6,  res.eccentricity,   fmt='0.000', fill=row_fill)
            num_fmt(ws_sum, r_idx, 7,  res.sigma_toe,      fmt='0.00',  fill=row_fill)

            sh_c = num_fmt(ws_sum, r_idx, 8, res.sigma_heel, fmt='0.00',
                           fill=hfill(GREEN_LIGHT) if sh_ok else hfill(RED_LIGHT))

            tl_val = res.tension_length if res.tension_length > 0.001 else 0
            num_fmt(ws_sum, r_idx, 9,  tl_val,             fmt='0.000', fill=row_fill)
            num_fmt(ws_sum, r_idx, 10, res.sum_V,           fmt='0.00',  fill=row_fill)

            ws_sum.row_dimensions[r_idx].height = 18

        # Column widths
        col_widths = [18, 11, 15, 16, 15, 17, 14, 14, 15, 13]
        for i, w in enumerate(col_widths, 1):
            ws_sum.column_dimensions[get_column_letter(i)].width = w

        # ── 2. One sheet per load case ────────────────────────────────
        for res in data.results:
            sname = res.case_name[:31]
            ws = wb.create_sheet(sname)
            ws.sheet_view.showGridLines = False

            # Sheet title
            ws.merge_cells("A1:H1")
            t2 = ws["A1"]
            t2.value     = f"Load Case: {res.case_name}"
            t2.font      = Font(bold=True, size=13, color=WHITE, name="Calibri")
            t2.fill      = hfill(BLUE_DARK)
            t2.alignment = Alignment(horizontal="center", vertical="center")
            ws.row_dimensions[1].height = 26

            # ── KPI block (row 3–11, cols A–D) ───────────────────────
            kpis = [
                ("FS Sliding",         res.FS_sliding,       res.FS_sliding >= 1.5,   '0.000'),
                ("FS Overturning",     res.FS_overturning,   res.FS_overturning >= 1.5,'0.000'),
                ("Resultant Check",    "✓ OK" if res.in_middle_third else "✗ OUTSIDE",
                                       res.in_middle_third,  '@'),
                ("σ_toe (kN/m²)",      res.sigma_toe,        res.sigma_toe >= 0,       '0.00'),
                ("σ_heel (kN/m²)",     res.sigma_heel,       res.sigma_heel >= 0,      '0.00'),
                ("x_resultant (m)",    res.x_resultant,      None,                     '0.000'),
                ("Eccentricity e (m)", res.eccentricity,     None,                     '0.000'),
                ("Tension length (m)", res.tension_length,   res.tension_length < 0.001,'0.000'),
                ("ΣV (kN/m)",          res.sum_V,            None,                     '0.00'),
                ("ΣM_res (kNm/m)",     res.sum_M_res,        None,                     '0.00'),
                ("ΣM_ov (kNm/m)",      res.sum_M_ov,         None,                     '0.00'),
            ]

            ws.merge_cells("A3:B3")
            kh = ws["A3"]
            kh.value     = "RESULTS SUMMARY"
            kh.font      = Font(bold=True, size=10, color=WHITE, name="Calibri")
            kh.fill      = hfill(BLUE_MID)
            kh.alignment = Alignment(horizontal="center", vertical="center")
            ws.row_dimensions[3].height = 20

            for ki, (lbl, val, ok, fmt) in enumerate(kpis, 4):
                fill_c = (hfill(GREEN_LIGHT) if ok is True
                          else hfill(RED_LIGHT) if ok is False
                          else hfill(GRAY_LIGHT))
                write_cell(ws, ki, 1, lbl, bold=True,
                           fill=hfill(BLUE_LIGHT), align='left')
                c = ws.cell(row=ki, column=2, value=val)
                c.number_format = fmt
                if isinstance(val, float) and val >= 9998: c.value = "∞"
                c.font      = Font(bold=True, name="Calibri", size=9)
                c.fill      = fill_c
                c.alignment = Alignment(horizontal="center", vertical="center")
                c.border    = thin_border
                ws.row_dimensions[ki].height = 16

            ws.column_dimensions['A'].width = 22
            ws.column_dimensions['B'].width = 14

            # ── Force table (starts at row 3, cols D onwards) ─────────
            ft_start_row = 3
            ft_col_start = 4   # column D

            force_headers = ["Force", "Stab/Dest",
                             "V (kN/m)", "H (kN/m)",
                             "x_toe (m)", "y (m)",
                             "M_res (kNm/m)", "M_ov (kNm/m)"]
            for ci, h in enumerate(force_headers, ft_col_start):
                c = ws.cell(row=ft_start_row, column=ci, value=h)
                c.font      = Font(bold=True, color=WHITE, name="Calibri", size=9)
                c.fill      = hfill(BLUE_MID)
                c.alignment = Alignment(horizontal="center", vertical="center",
                                        wrap_text=True)
                c.border    = thin_border
            ws.row_dimensions[ft_start_row].height = 28

            for fi, f in enumerate(res.forces, ft_start_row + 1):
                row_fill2 = (hfill(GREEN_LIGHT) if f.stabilising
                             else hfill(RED_LIGHT))
                write_cell(ws, fi, ft_col_start,   f.name,   fill=row_fill2)
                write_cell(ws, fi, ft_col_start+1,
                           "Stabilising" if f.stabilising else "Destabilising",
                           fill=row_fill2, align='center',
                           color=GREEN_DARK if f.stabilising else RED_DARK,
                           bold=True)
                num_fmt(ws, fi, ft_col_start+2, f.V,         fmt='0.00',  fill=row_fill2)
                num_fmt(ws, fi, ft_col_start+3, f.H,         fmt='0.00',  fill=row_fill2)
                num_fmt(ws, fi, ft_col_start+4, f.x_from_toe,fmt='0.000', fill=row_fill2)
                num_fmt(ws, fi, ft_col_start+5, f.y_from_toe,fmt='0.000', fill=row_fill2)
                num_fmt(ws, fi, ft_col_start+6, f.M_res,     fmt='0.00',  fill=row_fill2)
                num_fmt(ws, fi, ft_col_start+7, f.M_ov,      fmt='0.00',  fill=row_fill2)
                ws.row_dimensions[fi].height = 15

            # Totals row
            tot_row = ft_start_row + len(res.forces) + 1
            write_cell(ws, tot_row, ft_col_start, "RESULTANT",
                       bold=True, fill=hfill(BLUE_LIGHT))
            write_cell(ws, tot_row, ft_col_start+1, "",
                       fill=hfill(BLUE_LIGHT))
            num_fmt(ws, tot_row, ft_col_start+2, res.sum_V,
                    fmt='0.00', bold=True, fill=hfill(BLUE_LIGHT))
            num_fmt(ws, tot_row, ft_col_start+3, res.H_net,
                    fmt='0.00', bold=True, fill=hfill(BLUE_LIGHT))
            write_cell(ws, tot_row, ft_col_start+4, "",
                       fill=hfill(BLUE_LIGHT))
            write_cell(ws, tot_row, ft_col_start+5, "",
                       fill=hfill(BLUE_LIGHT))
            num_fmt(ws, tot_row, ft_col_start+6, res.sum_M_res,
                    fmt='0.00', bold=True, fill=hfill(BLUE_LIGHT))
            num_fmt(ws, tot_row, ft_col_start+7, res.sum_M_ov,
                    fmt='0.00', bold=True, fill=hfill(BLUE_LIGHT))
            ws.row_dimensions[tot_row].height = 18

            # Force table column widths
            ft_widths = [28, 14, 11, 11, 11, 9, 15, 14]
            for i, w in enumerate(ft_widths, ft_col_start):
                ws.column_dimensions[get_column_letter(i)].width = w

            # ── Messages / notices block ─────────────────────────────
            msg_start = tot_row + 2
            if res.messages:
                # Header
                ws.merge_cells(f"A{msg_start}:H{msg_start}")
                mh = ws[f"A{msg_start}"]
                mh.value     = "ENGINEERING NOTICES"
                mh.font      = Font(bold=True, size=10, color=WHITE, name="Calibri")
                mh.fill      = hfill(BLUE_MID)
                mh.alignment = Alignment(horizontal="center", vertical="center")
                ws.row_dimensions[msg_start].height = 20
                msg_start += 1

                TYPE_FILL  = {"info": "D6E8F7", "warning": "FFF3CD", "alert": "FAD7D7"}
                TYPE_COLOR = {"info": "1A3A5C", "warning": "7A5000", "alert": "8B1A1A"}
                TYPE_ICON  = {"info": "ℹ",      "warning": "⚠",      "alert": "🚨"}

                for m in res.messages:
                    mtype = m.get("type", "info") if isinstance(m, dict) else "info"
                    mtext = m.get("text", str(m)) if isinstance(m, dict) else str(m)
                    icon  = TYPE_ICON.get(mtype, "ℹ")
                    fgColor = TYPE_FILL.get(mtype, "D6E8F7")
                    txtColor = TYPE_COLOR.get(mtype, "1A3A5C")

                    ws.merge_cells(f"A{msg_start}:H{msg_start}")
                    mc = ws[f"A{msg_start}"]
                    mc.value     = f"{icon}  {mtext}"
                    mc.font      = Font(name="Calibri", size=9, color=txtColor)
                    mc.fill      = hfill(fgColor)
                    mc.alignment = Alignment(horizontal="left", vertical="center",
                                             wrap_text=True)
                    mc.border    = thin_border
                    ws.row_dimensions[msg_start].height = 30
                    msg_start += 1

                msg_start += 1  # blank spacer row

            # ── Insert plot image ─────────────────────────────────────
            img_row = msg_start
            if res.plot_base64:
                img_bytes = base64.b64decode(res.plot_base64)
                img_buf   = io.BytesIO(img_bytes)
                img       = XLImage(img_buf)
                # Scale to fit nicely — keep aspect ratio
                img.width  = 700
                img.height = int(700 * img.height / img.width) if img.width else 480
                ws.add_image(img, f"A{img_row}")

        # ── Stream workbook back ──────────────────────────────────────
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type=(
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            ),
            headers={
                "Content-Disposition":
                    'attachment; filename="dam_stability_results.xlsx"'
            },
        )

    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
