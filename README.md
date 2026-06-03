# Gravity Dam Stability Calculator

A web-based tool for performing stability analysis of concrete gravity dams per unit width. The tool evaluates sliding stability, overturning, resultant location, and base stress distribution across multiple load cases, and exports a fully formatted Excel report.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Defining the Dam Geometry](#defining-the-dam-geometry)
3. [Material Properties](#material-properties)
4. [Water Levels](#water-levels)
5. [Additional Loads](#additional-loads)
6. [Acceptance Criteria](#acceptance-criteria)
7. [Load Cases](#load-cases)
8. [Earthquake Analysis](#earthquake-analysis)
9. [Understanding the Results](#understanding-the-results)
10. [Exporting to Excel](#exporting-to-excel)
11. [Theory and Sign Conventions](#theory-and-sign-conventions)
12. [Limitations](#limitations)

---

## Getting Started

Open the application in your browser. The interface is divided into three panels:

- **Left sidebar** — all input parameters
- **Centre** — live geometry preview that updates as you type
- **Right panel** — calculation results, displayed after clicking **Calculate**

Enter your inputs from top to bottom in the sidebar, then click **Calculate**. Results appear immediately in the right panel. To export, click **Download Excel**.

---

## Defining the Dam Geometry

### Polygon Coordinates

Enter the dam cross-section as a closed polygon, one vertex per line, in the format:

```
x, y
x, y
...
```

- **x** is the horizontal distance in metres (positive to the right)
- **y** is the absolute elevation in metres
- Vertices can be entered in any order (clockwise or counter-clockwise)
- The polygon must have at least 3 vertices
- The coordinate system convention is: **upstream is to the left, downstream is to the right** (i.e. upstream vertices have smaller x values)

**Example** (simple trapezoidal dam, toe at x=12, heel at x=0):
```
0,  100
0,  109
5,  109
5.4,109
12, 100
```

### Upstream Top Point (UTP)

The upstream top point is the **single most important input** — it is the corner where the dam crest meets the upstream face. The calculator uses it to split the polygon into upstream and downstream faces, from which the heel and toe are identified.

- The UTP must exactly match (or be very close to) one of the polygon vertices
- Enter the absolute x and y coordinates
- **If the UTP is wrong, the heel, toe, uplift, water pressure, and all moment arms will be wrong**

*Rule of thumb:* the UTP is typically the vertex with the highest elevation on the upstream (left) side of the crest.

---

## Material Properties

| Parameter | Description | Typical range |
|---|---|---|
| Unit weight of dam | kN/m³ | 22–26 (concrete) |
| Unit weight of water | kN/m³ | 9.81–10.0 |
| Base friction coefficient | μ, dimensionless | 0.6–1.0 |

---

## Water Levels

Enter absolute upstream and downstream water levels (in metres) for each load case. The water depth used in calculations is always relative to the heel or toe elevation.

| Field | Description |
|---|---|
| HRV+IS Upstream / Downstream | Normal high reservoir level |
| DFV Upstream / Downstream | Design flood level |
| MFV Upstream / Downstream | Maximum (extreme) flood level |

Downstream water level may be left at the toe elevation (no tailwater) by entering the toe elevation value.

---

## Additional Loads

### Drainage Curtain

A drainage curtain reduces the uplift pressure behind the heel. Uplift is modelled as trapezoidal: full head at the heel, reduced to a fraction at the drain location, then linearly reducing to tailwater head at the toe.

| Parameter | Description |
|---|---|
| Distance from heel | Position of the drain along the base (m) |
| Reduction factor | Fraction of residual head remaining at drain (e.g. 0.333 reduces to one-third) |

### Silt Pressure

Horizontal silt pressure and the weight of silt on the upstream face. Enter the absolute silt surface elevation. The calculator uses Rankine active pressure with the submerged unit weight of silt.

| Parameter | Description |
|---|---|
| Silt surface elevation | Absolute elevation of top of silt deposit (m) |
| Submerged unit weight | kN/m³ (typically 8–11) |
| Friction angle φ | Degrees (typically 20–35°) |

### Backfill

Backfill on the downstream face (e.g. earth fill against the downstream slope). Can include horizontal pressure and/or the weight of the fill wedge.

| Parameter | Description |
|---|---|
| Backfill height | Height of backfill above the toe (m) |
| Pressure coefficient K₀ | Lateral earth pressure coefficient |
| Unit weight (dry / wet / submerged) | kN/m³ |

### Ice Pressure

A horizontal force applied at the upstream water surface. Only included in HRV+IS load case. Not applied to earthquake load cases.

| Parameter | Description |
|---|---|
| Ice pressure | Horizontal pressure in kN/m (total, per unit width) |

### Rock Bolts

A vertical tensile force applied near the heel to resist uplift and improve stability.

| Parameter | Description |
|---|---|
| Force per unit width | kN/m |
| Cover from heel | Horizontal distance from heel to bolt position (m) |
| Apply depth limit | If enabled, rock bolts are excluded when HRV water depth above heel exceeds the depth limit — bolts in deep water may be impractical |

### Rock Anchors

Similar to rock bolts. A vertical stabilising force applied at a specified distance from the heel.

### Applied Forces

Additional user-defined forces, useful for modelling external loads not covered by the standard categories.

- **Vertical forces**: positive = downward (stabilising). Enter force (kN/m) and horizontal distance from toe (m).
- **Horizontal forces**: positive = toward upstream (stabilising). Enter force (kN/m) and height above base (m).

---

## Acceptance Criteria

The tool uses two sets of acceptance criteria, configurable by the user:

| Criterion | ULS (Ultimate Limit State) | ALS (Accidental Limit State) |
|---|---|---|
| Applies to | HRV+IS, DFV | MFV, DFV (no bolts), HRV+EQ |
| Default FS sliding | ≥ 1.5 | ≥ 1.1 |
| Default resultant zone | Middle third (L/3 – 2L/3) | L/6 – 5L/6 |

Both the FS threshold and the resultant zone criterion can be changed in the **Acceptance Criteria** card before running.

---

## Load Cases

| Load Case | Description | Criteria |
|---|---|---|
| **HRV+IS** | High reservoir + ice pressure | ULS |
| **DFV** | Design flood, with rock bolts | ULS |
| **MFV** | Maximum (extreme) flood, with rock bolts | ALS |
| **DFV (no bolts)** | Design flood, rock bolts excluded | ALS |
| **HRV+EQ (X-dom)** | Seismic, horizontal dominant (full aₕ, 0.3aᵥ) | ALS |
| **HRV+EQ (Y-dom)** | Seismic, vertical dominant (0.3aₕ, full aᵥ) | ALS |

Each load case can be enabled or disabled individually in the **Load Cases** card.

---

## Earthquake Analysis

The earthquake analysis follows Eurocode 8 with two load combinations:

- **X-dominant**: full horizontal acceleration (aₕ) + 30% vertical (0.3aᵥ)
- **Y-dominant**: 30% horizontal (0.3aₕ) + full vertical (aᵥ)

Both cases use the HRV water levels. Ice is excluded from earthquake cases.

### Forces calculated

| Force | Formula | Point of application |
|---|---|---|
| Inertia horizontal | `F_ih = (aₕ/g) × W` | Dam centroid |
| Inertia vertical | `F_iv = (aᵥ/g) × W` (upward) | Dam centroid |
| Hydrodynamic (Westergaard) | `F_hd = (7/12)(aₕ/g)γ_w H² cos²θ` | 0.4H above base |

Where θ is the inclination of the upstream face from vertical, applied as a cos²θ correction to the standard Westergaard formula.

**Worst-case sign convention is assumed throughout:** horizontal forces act toward the downstream side, vertical inertia acts upward (reducing effective weight).

---

## Understanding the Results

Results are shown for each load case in a tabbed panel.

### Stability checks

| Output | Description | Pass condition |
|---|---|---|
| FS Sliding | `μN / T` on the inclined base plane | ≥ threshold (ULS or ALS) |
| FS Overturning | `ΣM_res / ΣM_ov` about the toe | > 1.0 (informational) |
| Resultant location | Distance of resultant from toe | Within middle-third or L/6–5L/6 |
| σ_toe | Base stress at downstream toe (kN/m²) | Compression positive |
| σ_heel | Base stress at upstream heel (kN/m²) | Compression positive |
| Tension zone | Length of base in tension (m) | Ideally = 0 |

### Infinite FS Sliding

When FS Sliding shows **∞**, the net shear force on the base plane is zero or acts toward the upstream side — there is no tendency to slide downstream. This can occur when the base slopes steeply upward toward the toe (the dam weight component along the inclined plane acts upstream, cancelling the horizontal driving forces), or when stabilising forces exceed driving forces. An explanatory message is shown in the results.

### Inclined base

For dams with an inclined base, the normal and shear forces are computed on the actual inclined plane:

- `N = V·cosα − H·sinα`
- `T = H·cosα + V·sinα`

where α is the base inclination angle from horizontal, V is the net vertical force, and H is the net horizontal force. FS Sliding = `μN / T`.

### Tension zone iteration

If the heel stress σ_heel is tensile, the effective base length is reduced (tension zone is excluded from the base), and the calculation is iterated until a consistent tension-free solution is found.

---

## Exporting to Excel

Click **Download Excel** to export a fully formatted workbook. The workbook contains:

- **Summary sheet** — one row per load case, all key results at a glance
- **One sheet per dam section** — force table, stability results table, and the dam figure for each load case

The force table lists every force with its vertical component (V), horizontal component (H), x-arm, y-arm, restoring moment, and overturning moment. Message rows (blue = information, yellow = warning, red = alert) are included for any engineering notes relevant to that load case.

---

## Theory and Sign Conventions

### Coordinate system

- All calculations are per unit width (kN/m, kN/m², kN·m/m)
- Moments are taken about the downstream toe
- Restoring moments are positive (counterclockwise), overturning moments are negative
- Compressive stresses are positive

### Uplift

Trapezoidal distribution: full upstream head at the heel, reducing linearly to full downstream head at the toe (or reduced at the drain if a drainage curtain is active). When a tension zone exists, uplift over the tension region is applied at full upstream head.

### Silt pressure

Rankine active horizontal pressure: `p(z) = Ka × γ'_silt × z`, where `Ka = tan²(45° − φ/2)` and z is depth below the silt surface.

### Display scale

Forces and pressures in the stability diagram are drawn at the following scales:

- **Pressure polygons** (water, uplift, silt): 1 kPa = 0.1 m
- **Force arrows** (ice, water weight, rock bolt, etc.): 1 kN/m = 0.1 m
- **Overall resultant arrow**: 1 kN/m = 0.01 m (drawn smaller to distinguish from individual forces)

---

## Limitations

- **2D analysis only** — per unit width of dam. No 3D or abutment effects.
- **Rigid body assumption** — the dam is treated as a rigid block; no structural stress analysis within the dam body.
- **Linear base stress distribution** — assumes plane sections remain plane. Valid when the resultant is within the middle third; less accurate outside it.
- **Westergaard hydrodynamic pressure** — valid for nearly vertical upstream faces. Applied with a cos²θ correction for inclined faces, which is conservative for typical gravity dam slopes.
- **Single drainage line** — only one drainage curtain position is modelled. Multiple drainage lines are not supported.
- **Homogeneous base** — a single friction coefficient is used across the entire base. Variable foundation conditions are not modelled.
