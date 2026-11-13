import pyvista as pv
import numpy as np
from scipy.interpolate import interp1d, splprep, splev
import glob, os, re

# --- datasets: one entry per EYE; each has a basal-top folder and an apical-bottom folder ---
BASE = r"D:\Patricia\optic_cup_analysis-main\plane slice figure\apical vs basal\to use"
DATASETS = [
    {"name": "pt2",
     "basal":  os.path.join(BASE, "200731_pt2_basal_top_bottom"),
     "apical": os.path.join(BASE, "200731_pt2_apical_top_bottom"),
     "tint": "lightblue",  "scale_basal": 0.5, "scale_apical": 1.0},
    {"name": "pt5",
     "basal":  os.path.join(BASE, "200731_pt5_basal_top_bottom"),
     "apical": os.path.join(BASE, "200731_pt5_apical_top_bottom"),
     "tint": "lightgreen", "scale_basal": 0.8, "scale_apical": 0.8, "mirror": "y"},
    {"name": "ptA",
     "basal":  os.path.join(BASE, "220722_ptA_basal_top_bottom"),
     "apical": os.path.join(BASE, "220722_ptA_apical_top_bottom"),
     "tint": "lightpink",  "scale_basal": 0.5, "scale_apical": 0.5, "mirror": None},
]

export_dir = r"D:\Patricia\optic_cup_analysis-main\plane slice figure\apical vs basal\exports"
PROP = "contour_curv"
REF_SIGN = +1
CMAP = "turbo"
COLOR_SLICES_BY_LAYER = False
SLICE_COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "black"]
SIDE_MEAN_COLOR = {"basal": "white", "apical": "yellow"}
N_RESAMPLE = 200
CONTOUR_SPLINE_SMOOTH = 0.5   # spline smoothing for contour curvature (higher = smoother)
INVERT_SIGN = False           # flip the whole convention if convex/concave come out reversed

# --- ICP alignment options ---
ALIGN_ICP = True          # ICP-align each eye's apical to the reference eye; basal follows
ICP_REF = "ptA"           # reference eye name (others align to this one)
ICP_PER_FRAME = True     # True = align each timepoint independently; False = one transform per eye
ICP_ANCHOR_FRAME = 15     # (per-eye mode only) which frame defines the single transform

# --- export naming: True = number by viewer frame (1-based); False = number from filename ---
EXPORT_BY_VIEWER_FRAME = True

# --- interactive rotation: name of eye to adjust (set None when done) ---
ADJUST = None             # e.g. "pt5" to enable rotation keys for that eye
ROT_STEP = 5.0
adjust_state = {"rx": 0.0, "ry": 0.0, "rz": 0.0}

os.makedirs(export_dir, exist_ok=True)

def trailing_num(path):
    """Last integer in the filename: GM4_top_t12 -> 12, scaled_top_7 -> 7."""
    nums = re.findall(r'\d+', os.path.basename(path))
    return int(nums[-1]) if nums else -1

def list_meshes(folder):
    """All .ply in a folder, sorted by trailing number."""
    return sorted(glob.glob(os.path.join(folder, "*.ply")), key=trailing_num)

def load_dataset(ds):
    """Load basal + apical surfaces for one eye; scale; mirror + bake rotation in place. No centering."""
    surfaces = {"basal": ds["basal"], "apical": ds["apical"]}
    rot = ds.get("rotate", (0, 0, 0))
    mir = ds.get("mirror")

    raw = {}
    for surf, folder in surfaces.items():
        files = list_meshes(folder)
        meshes = [pv.read(f).triangulate() for f in files]
        sc = ds.get(f"scale_{surf}", 1.0)
        if sc != 1.0:
            for m in meshes:
                m.points = m.points * sc
        raw[surf] = {"files": files, "meshes": meshes}

    n = min(len(raw["basal"]["meshes"]), len(raw["apical"]["meshes"]))

    for surf in ("basal", "apical"):
        for i in range(n):
            m = raw[surf]["meshes"][i]
            if mir:
                ax = {"x": 0, "y": 1, "z": 2}[mir]
                c = m.center[ax]
                m.points[:, ax] = 2*c - m.points[:, ax]   # reflect about centre, in place
            if any(rot):
                if rot[0]: m.rotate_x(rot[0], inplace=True)
                if rot[1]: m.rotate_y(rot[1], inplace=True)
                if rot[2]: m.rotate_z(rot[2], inplace=True)

    raw["_n"] = n
    return raw

# --- load all eyes (raw), pick reference, then ICP-align + curvature ---
RAW = {ds["name"]: load_dataset(ds) for ds in DATASETS}
n_ref = RAW[ICP_REF]["_n"]

LAYERS = []
for ds in DATASETS:
    name = ds["name"]
    raw = RAW[name]
    n = raw["_n"]

    # one transform per eye (used only if ICP_PER_FRAME is False)
    eye_matrix = None
    if ALIGN_ICP and name != ICP_REF and not ICP_PER_FRAME:
        a = min(ICP_ANCHOR_FRAME, n - 1, n_ref - 1)         # guard against short sequences
        ref_a = RAW[ICP_REF]["apical"]["meshes"][a]
        _, eye_matrix = raw["apical"]["meshes"][a].align(ref_a, return_matrix=True)

    # per-frame transforms (used only if ICP_PER_FRAME is True)
    frame_matrix = [None] * n
    if ALIGN_ICP and name != ICP_REF and ICP_PER_FRAME:
        for i in range(n):
            j = min(i, n_ref - 1)
            ref_i = RAW[ICP_REF]["apical"]["meshes"][j]
            _, frame_matrix[i] = raw["apical"]["meshes"][i].align(ref_i, return_matrix=True)

    for surf in ("basal", "apical"):
        meshes = raw[surf]["meshes"][:n]

        # NOTE: curvature is now the CONTOUR (in-plane) curvature MAGNITUDE, computed
        # per slice at draw/export time (see contour_curvature). Here we only apply the
        # alignment transform to the meshes.
        out_meshes = []
        for i in range(n):
            m = meshes[i]
            mat = frame_matrix[i] if ICP_PER_FRAME else eye_matrix
            if mat is not None:
                m.transform(mat, inplace=True)            # alignment transform, both surfaces
            out_meshes.append(m)

        LAYERS.append({"name": f"{name}_{surf}", "meshes": out_meshes,
                       "files": raw[surf]["files"][:n], "tint": ds["tint"]})

if not LAYERS:
    raise SystemExit("No meshes found - check folders.")

for L in LAYERS:
    L["orig_points"] = [m.points.copy() for m in L["meshes"]]

n_frames = min(len(L["files"]) for L in LAYERS)
print(f"Loaded {len(LAYERS)} layers: {[L['name'] for L in LAYERS]}, {n_frames} timepoints")
print(f"ICP: {'per-frame' if ICP_PER_FRAME else 'single-transform'} , reference = {ICP_REF}")

# colour range: recomputed from contour curvature after current_slice is defined
CLIM = (0.0, 1.0)   # placeholder, updated below

ref = LAYERS[0]["meshes"][0]

# --- place slice plane at the centre of ALL meshes, transversal to the long axis ---
# SLICE_AXIS: "auto" = perpendicular to the longest extent; or force "x" / "y" / "z"
SLICE_AXIS = "auto"
_all_pts = np.vstack([L["meshes"][0].points for L in LAYERS])
_center_all = _all_pts.mean(axis=0)                       # true middle of everything
if SLICE_AXIS == "auto":
    _extent = _all_pts.max(axis=0) - _all_pts.min(axis=0) # size along x, y, z
    _axis = int(np.argmax(_extent))                       # longest dimension
else:
    _axis = {"x": 0, "y": 1, "z": 2}[SLICE_AXIS]
_normal = [0, 0, 0]; _normal[_axis] = 1
print(f"slice plane: centre={np.round(_center_all,2)}, normal axis={'xyz'[_axis]}")

state = {"idx": 0, "slice_origin": list(_center_all), "slice_normal": _normal,
         "mesh_visible": True, "show_mean": False}
mesh_actors = {}

p = pv.Plotter()

def _order_contour_indices(sl):
    """Return the slice's point indices walked in contour order (or None)."""
    lines = sl.lines.reshape(-1, 3)[:, 1:]
    adj = {}
    for a, b in lines:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    if not adj:
        return None
    ends = [k for k in adj if len(adj[k]) == 1]
    seed = ends[0] if ends else next(iter(adj))
    order = [seed]; seen = {seed}; prev, cur = None, seed
    while True:
        nxt = [x for x in adj[cur] if x != prev and x not in seen]
        if not nxt: break
        prev, cur = cur, nxt[0]; order.append(cur); seen.add(cur)
    return order

def contour_curvature(sl):
    """In-plane signed curvature of the slice line via a periodic smoothing spline.
    Always leaves a valid sl[PROP] array (zeros on any degenerate/early-return path)."""
    sl[PROP] = np.zeros(sl.n_points)          # ensure the array always exists
    order = _order_contour_indices(sl)
    if order is None or len(order) < 5:
        return sl
    pts = sl.points[order]

    # in-plane basis spanning the slice plane
    nrm = np.asarray(state["slice_normal"], float)
    nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
    ref = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(nrm, ref); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(nrm, e1)
    x = pts @ e1; y = pts @ e2

    # --- anchor traversal direction to a FIXED reference axis (e1) so the sign is
    #     stable across timepoints and embryos. If the arc runs "backwards" relative
    #     to e1 (its net progress along e1 is negative), reverse it before fitting. ---
    if (x[-1] - x[0]) < 0:
        x = x[::-1]; y = y[::-1]
        order = order[::-1]

    d = np.r_[0, np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))]
    if d[-1] == 0:
        return sl
    try:
        s_ = CONTOUR_SPLINE_SMOOTH * len(x)
        tck, u = splprep([x, y], u=d, s=s_, k=3)   # open arc: no per=1
        dx, dy = splev(d, tck, der=1)
        ddx, ddy = splev(d, tck, der=2)
        denom = (dx*dx + dy*dy)**1.5 + 1e-12
        k = (dx*ddy - dy*ddx) / denom
    except Exception:
        return sl

    # kill endpoint artifacts (open arcs: ends are estimated from one side only)
    TRIM = 3
    if len(k) > 2 * TRIM:
        k[:TRIM] = 0.0
        k[-TRIM:] = 0.0

    out = np.zeros(sl.n_points)
    out[order] = k
    sl[PROP] = -out if INVERT_SIGN else out
    return sl

def current_slice(L):
    m = L["meshes"][state["idx"]]
    sl = m.slice(normal=state["slice_normal"], origin=state["slice_origin"])
    if sl.n_points == 0:
        return None
    sl = contour_curvature(sl)      # overwrite PROP with contour curvature magnitude
    return sl

# recompute CLIM from contour curvature across all frames/layers
_cvals = []
for _idx in range(n_frames):
    state["idx"] = _idx
    for L in LAYERS:
        _sl = current_slice(L)
        if _sl is not None:
            _cvals.append(np.asarray(_sl[PROP]))
state["idx"] = 0
if _cvals:
    _cv = np.concatenate(_cvals)
    lim = float(np.percentile(np.abs(_cv), 95)); CLIM = (-lim, lim)
    print("contour-curvature CLIM =", CLIM)

def resample_contour(sl, n=N_RESAMPLE):
    """Order contour and resample to n points along arclength. Returns (xyz, vals) or None."""
    lines = sl.lines.reshape(-1, 3)[:, 1:]
    adj = {}
    for a, b in lines:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    if not adj:
        return None
    ends = [k for k in adj if len(adj[k]) == 1]
    seed = ends[0] if ends else next(iter(adj))
    order = [seed]; seen = {seed}; prev, cur = None, seed
    while True:
        nxt = [x for x in adj[cur] if x != prev and x not in seen]
        if not nxt: break
        prev, cur = cur, nxt[0]; order.append(cur); seen.add(cur)
    pts = sl.points[order]; vals = sl[PROP][order]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] == 0: return None
    u = d / d[-1]; t = np.linspace(0, 1, n)
    xyz = np.column_stack([interp1d(u, pts[:, k])(t) for k in range(3)])
    v = interp1d(u, vals)(t)
    return xyz, v

def mean_contours_by_side():
    """Mean resampled slice per side, with direction alignment across datasets."""
    groups = {}
    for L in LAYERS:
        side = L["name"].split("_")[-1]
        sl = current_slice(L)
        if sl is None: continue
        r = resample_contour(sl)
        if r is None: continue
        groups.setdefault(side, [[], []])
        groups[side][0].append(r[0]); groups[side][1].append(r[1])

    out = {}
    for side, (sx, sv) in groups.items():
        if not sx: continue
        ref_c = sx[0]                                  # first contour = direction reference
        aligned_xyz, aligned_v = [ref_c], [sv[0]]
        for xyz, v in zip(sx[1:], sv[1:]):
            fwd = np.sum(np.linalg.norm(xyz - ref_c, axis=1))
            rev = np.sum(np.linalg.norm(xyz[::-1] - ref_c, axis=1))
            if rev < fwd:
                xyz, v = xyz[::-1], v[::-1]
            aligned_xyz.append(xyz); aligned_v.append(v)
        poly = pv.lines_from_points(np.mean(aligned_xyz, axis=0))
        poly[PROP] = np.mean(aligned_v, axis=0)
        out[side] = poly
    return out

def redraw():
    n_drawn = 0
    for li, L in enumerate(LAYERS):
        m = L["meshes"][state["idx"]]
        a = p.add_mesh(m, name=f"frame_{L['name']}", color=L["tint"],
                       opacity=0.25, show_edges=False)
        a.visibility = state["mesh_visible"]
        mesh_actors[L["name"]] = a

        sl = current_slice(L)
        if sl is not None and PROP in sl.point_data:
            if COLOR_SLICES_BY_LAYER:
                p.add_mesh(sl, name=f"slice_{L['name']}",
                           color=SLICE_COLORS[li % len(SLICE_COLORS)],
                           line_width=6, render_lines_as_tubes=True)
            else:
                p.add_mesh(sl, name=f"slice_{L['name']}", scalars=PROP, cmap=CMAP,
                           clim=CLIM, line_width=6, render_lines_as_tubes=True,
                           scalar_bar_args={"title": PROP})
            n_drawn += 1
        else:
            p.remove_actor(f"slice_{L['name']}", reset_camera=False)

    # mean contour per side (toggle with "c")
    for side in ("basal", "apical"):
        p.remove_actor(f"mean_{side}", reset_camera=False)
    if state["show_mean"]:
        for side, mc in mean_contours_by_side().items():
            p.add_mesh(mc, name=f"mean_{side}",
                       color=SIDE_MEAN_COLOR.get(side, "white"),
                       line_width=10, render_lines_as_tubes=True)

    p.add_text(f"frame {state['idx']+1}/{n_frames}  |  {n_drawn}/{len(LAYERS)} slices",
               name="label")

def slice_cb(normal, origin):
    state["slice_normal"], state["slice_origin"] = list(normal), list(origin)
    redraw()

slice_widget = p.add_plane_widget(slice_cb, normal=state["slice_normal"],
                                  origin=state["slice_origin"], bounds=ref.bounds)

def next_frame():
    state["idx"] = (state["idx"] + 1) % n_frames; redraw()
def prev_frame():
    state["idx"] = (state["idx"] - 1) % n_frames; redraw()
def jump_fwd10():
    state["idx"] = (state["idx"] + 10) % n_frames; redraw()
def jump_back10():
    state["idx"] = (state["idx"] - 10) % n_frames; redraw()

def toggle_mesh():
    state["mesh_visible"] = not state["mesh_visible"]
    for a in mesh_actors.values():
        a.visibility = state["mesh_visible"]
    p.render()

def toggle_widget():
    slice_widget.SetEnabled(not slice_widget.GetEnabled())
    p.render()

def toggle_mean():
    state["show_mean"] = not state["show_mean"]; redraw()

# --- interactive rotation of the ADJUST dataset ---
def apply_adjust_rotation():
    if ADJUST is None:
        return
    for L in LAYERS:
        if not L["name"].startswith(ADJUST):
            continue
        for i, m in enumerate(L["meshes"]):
            m.points = L["orig_points"][i].copy()
            if adjust_state["rx"]: m.rotate_x(adjust_state["rx"], inplace=True)
            if adjust_state["ry"]: m.rotate_y(adjust_state["ry"], inplace=True)
            if adjust_state["rz"]: m.rotate_z(adjust_state["rz"], inplace=True)
    redraw()
    print(f"{ADJUST} rotate=({adjust_state['rx']:.0f}, {adjust_state['ry']:.0f}, {adjust_state['rz']:.0f})")

def rz_p(): adjust_state["rz"] += ROT_STEP; apply_adjust_rotation()
def rz_m(): adjust_state["rz"] -= ROT_STEP; apply_adjust_rotation()
def rx_p(): adjust_state["rx"] += ROT_STEP; apply_adjust_rotation()
def rx_m(): adjust_state["rx"] -= ROT_STEP; apply_adjust_rotation()
def ry_p(): adjust_state["ry"] += ROT_STEP; apply_adjust_rotation()
def ry_m(): adjust_state["ry"] -= ROT_STEP; apply_adjust_rotation()

def _tp_for_index(idx):
    return (idx + 1) if EXPORT_BY_VIEWER_FRAME else trailing_num(LAYERS[0]["files"][idx])

def _export_one(idx, hide_widget=True):
    tp = _tp_for_index(idx)
    for L in LAYERS:
        sl = current_slice(L)
        if sl is None:
            continue
        sl.save(os.path.join(export_dir, f"slice_{L['name']}_t{tp}.vtp"))
    for side, mc in mean_contours_by_side().items():
        mc.save(os.path.join(export_dir, f"slice_MEAN_{side}_t{tp}.vtp"))
    png = os.path.join(export_dir, f"view_t{tp}.png")
    if hide_widget:
        slice_widget.SetEnabled(False)
    p.render()
    p.screenshot(png, scale=4)
    if hide_widget:
        slice_widget.SetEnabled(True)
        p.render()
    print(f"  exported t{tp}")

def export_frame():
    """Export the current timepoint only (key: x)."""
    _export_one(state["idx"])

def export_all_frames():
    """Export every timepoint (key: X). Uses the current slice plane + camera for all frames."""
    original = state["idx"]
    slice_widget.SetEnabled(False)
    for idx in range(n_frames):
        state["idx"] = idx
        redraw()
        _export_one(idx, hide_widget=False)
        print(f"    ({idx+1}/{n_frames})")
    slice_widget.SetEnabled(True)
    state["idx"] = original
    redraw()
    print("done - all timepoints exported")

# keys
p.add_key_event("Right", next_frame)
p.add_key_event("Left", prev_frame)
p.add_key_event("m", toggle_mesh)
p.add_key_event("b", toggle_widget)
p.add_key_event("c", toggle_mean)
p.add_key_event("x", export_frame)
p.add_key_event("X", export_all_frames)
# rotation keys (1/2 = z, 3/4 = x, 5/6 = y)
p.add_key_event("1", rz_p)
p.add_key_event("2", rz_m)
p.add_key_event("3", rx_p)
p.add_key_event("4", rx_m)
p.add_key_event("5", ry_p)
p.add_key_event("6", ry_m)
p.add_key_event("period", jump_fwd10)   # . = +10 timepoints
p.add_key_event("comma",  jump_back10)  # , = -10 timepoints

redraw()
print("Keys: <-/-> +/-1 | ,/. +/-10 timepoint | m mesh | b widget | c mean | x export | X export ALL")
print("Rotate ADJUST set: 1/2 = z+/- , 3/4 = x+/- , 5/6 = y+/-")
print("CLIM =", CLIM)
p.show()
