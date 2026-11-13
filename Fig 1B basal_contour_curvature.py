import pyvista as pv
import numpy as np
from scipy.interpolate import interp1d
import glob, os, re

# --- datasets: one entry per EYE; each has a basal surface folder ---
BASE = r"D:\Patricia\optic_cup_analysis-main\plane slice figure\Basal only"
DATASETS = [
    {"name": "pt1", "basal": os.path.join(BASE, "decimated_meshes_230524_pt1"), "tint": "lightblue"},
    {"name": "pt2", "basal": os.path.join(BASE, "decimated_meshes_230524_pt2"), "tint": "lightgreen", "mirror": "x"},
    {"name": "pt4", "basal": os.path.join(BASE, "decimated_meshes_230524_pt4"), "tint": "lightpink"},
]

export_dir = r"D:\Patricia\optic_cup_analysis-main\plane slice figure\Basal only\exports"
PROP = "contour_curv"
REF_SIGN = +1
CMAP = "turbo"
COLOR_SLICES_BY_LAYER = False
SLICE_COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "black"]
MEAN_COLOR = "white"
N_RESAMPLE = 200

# --- curvature sign: the basal meshes are CLOSED, so we compute consistent (outward)
# normals with auto_orient_normals and take VTK mean curvature directly. This gives a
# coherent convex(+)/concave(-) sign across each whole surface, no within-mesh flips. ---
SIGN_MODE = "geometric"    # "geometric" = consistent-normals curvature; "median" = old whole-mesh flip
CENTER_MODE = "frame0"     # unused with the closed-mesh method (kept for compatibility)
INVERT_SIGN = False       # flip the whole convention if convex/concave come out reversed

# --- ICP alignment options ---
ALIGN_ICP = True          # ICP-align each eye's basal to the reference eye
ICP_REF = "pt4"           # reference eye name (others align to this one)
ICP_PER_FRAME = True      # True = align each timepoint independently; False = one transform per eye
ICP_ANCHOR_FRAME = 40     # (single-transform mode) which frame defines the transform

# --- export naming: True = number by viewer frame (1-based); False = number from filename ---
EXPORT_BY_VIEWER_FRAME = True

# --- interactive rotation: name of eye to adjust (set None when done) ---
ADJUST = None             # e.g. "pt2" to enable rotation keys for that eye
ROT_STEP = 5.0
adjust_state = {"rx": 0.0, "ry": 0.0, "rz": 0.0}

os.makedirs(export_dir, exist_ok=True)

def trailing_num(path):
    """Last integer in the filename."""
    nums = re.findall(r'\d+', os.path.basename(path))
    return int(nums[-1]) if nums else -1

def list_meshes(folder):
    """All .ply in a folder, sorted by trailing number."""
    return sorted(glob.glob(os.path.join(folder, "*.ply")), key=trailing_num)

def signed_curvature(m, center=None):
    """Mean curvature with globally consistent (outward) normals.
    Valid because the basal meshes are CLOSED, so inside/outside is well-defined
    and auto_orient_normals gives a coherent sign across the whole surface
    (no within-mesh flips). `center` is unused, kept for call-site compatibility."""
    m2 = m.compute_normals(cell_normals=False, point_normals=True,
                           consistent_normals=True, auto_orient_normals=True,
                           inplace=False)
    cv = m2.curvature(curv_type="mean")
    return -cv if INVERT_SIGN else cv

def load_dataset(ds):
    """Load basal surface for one eye; scale; mirror + bake rotation in place. No centering."""
    rot = ds.get("rotate", (0, 0, 0))
    mir = ds.get("mirror")

    files = list_meshes(ds["basal"])
    meshes = [pv.read(f).triangulate() for f in files]
    sc = ds.get("scale_basal", 1.0)
    if sc != 1.0:
        for m in meshes:
            m.points = m.points * sc

    n = len(meshes)
    for i in range(n):
        m = meshes[i]
        if mir:
            ax = {"x": 0, "y": 1, "z": 2}[mir]
            c = m.center[ax]
            m.points[:, ax] = 2*c - m.points[:, ax]   # reflect about centre, in place
        if any(rot):
            if rot[0]: m.rotate_x(rot[0], inplace=True)
            if rot[1]: m.rotate_y(rot[1], inplace=True)
            if rot[2]: m.rotate_z(rot[2], inplace=True)

    if not meshes:
        raise SystemExit(f"No meshes loaded for {ds['name']} - check its 'basal' path")
    return {"files": files, "meshes": meshes, "_n": n}

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
        a = min(ICP_ANCHOR_FRAME, n - 1, n_ref - 1)
        ref_a = RAW[ICP_REF]["meshes"][a]
        _, eye_matrix = raw["meshes"][a].align(ref_a, return_matrix=True)

    # per-frame transforms (used only if ICP_PER_FRAME is True)
    frame_matrix = [None] * n
    if ALIGN_ICP and name != ICP_REF and ICP_PER_FRAME:
        for i in range(n):
            j = min(i, n_ref - 1)
            ref_i = RAW[ICP_REF]["meshes"][j]
            _, frame_matrix[i] = raw["meshes"][i].align(ref_i, return_matrix=True)

    meshes = raw["meshes"][:n]

    # sign anchor (used only if SIGN_MODE == "median")
    if SIGN_MODE == "median":
        cv0 = meshes[0].curvature(curv_type="mean")
        flip = np.sign(np.mean(cv0)) not in (0, REF_SIGN)
    center0 = np.asarray(meshes[0].center)   # for geometric sign, frame0 mode

    out_meshes = []
    for i in range(n):
        m = meshes[i]
        mat = frame_matrix[i] if ICP_PER_FRAME else eye_matrix
        if mat is not None:
            m.transform(mat, inplace=True)

        if SIGN_MODE == "geometric":
            center = center0 if CENTER_MODE == "frame0" else np.asarray(m.center)
            m[PROP] = signed_curvature(m, center)
        else:  # "median" - old whole-mesh flip
            cv = m.curvature(curv_type="mean")
            if flip:
                cv = -cv
            m[PROP] = cv

        out_meshes.append(m)

    LAYERS.append({"name": f"{name}_basal", "meshes": out_meshes,
                   "files": raw["files"][:n], "tint": ds["tint"]})

if not LAYERS:
    raise SystemExit("No meshes found - check folders.")

for L in LAYERS:
    L["orig_points"] = [m.points.copy() for m in L["meshes"]]

n_frames = min(len(L["files"]) for L in LAYERS)
print(f"Loaded {len(LAYERS)} layers: {[L['name'] for L in LAYERS]}, {n_frames} timepoints")
print(f"ICP: {'per-frame' if ICP_PER_FRAME else 'single-transform'} , reference = {ICP_REF}")
print(f"sign mode: {SIGN_MODE}" + (f" (center={CENTER_MODE}, invert={INVERT_SIGN})" if SIGN_MODE=='geometric' else ""))

# shared colour range across everything
allvals = np.concatenate([m[PROP] for L in LAYERS for m in L["meshes"]])
CLIM = tuple(np.percentile(allvals, [2, 98]))

ref = LAYERS[0]["meshes"][0]

# --- place slice plane at the centre of ALL meshes, transversal to the long axis ---
SLICE_AXIS = "auto"
_all_pts = np.vstack([L["meshes"][0].points for L in LAYERS])
_center_all = _all_pts.mean(axis=0)
if SLICE_AXIS == "auto":
    _extent = _all_pts.max(axis=0) - _all_pts.min(axis=0)
    _axis = int(np.argmax(_extent))
else:
    _axis = {"x": 0, "y": 1, "z": 2}[SLICE_AXIS]
_normal = [0, 0, 0]; _normal[_axis] = 1
print(f"slice plane: centre={np.round(_center_all,2)}, normal axis={'xyz'[_axis]}")

state = {"idx": 0, "slice_origin": list(_center_all), "slice_normal": _normal,
         "mesh_visible": True, "show_mean": False}
mesh_actors = {}

p = pv.Plotter()

# --- curvature source for the SLICE colouring/export ---
#   "surface" = the mesh's mean curvature sampled along the cut (default so far)
#   "contour" = the signed in-plane bending of the cut LINE itself (matches how the
#               line visually turns; two same-turning regions get the same sign)
CURV_SOURCE = "contour"

# contour-mode: smoothing strength of the spline fit (higher = smoother, broader-scale
# curvature; lower = follows finer wiggles). Tune this instead of a neighbour size.
CONTOUR_SPLINE_SMOOTH = 0.5

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
    """Signed in-plane curvature of the slice line via a smoothing spline fit.
    Much less noisy than point-to-point angles. Sign is consistent around the
    line (relative to the slice-plane normal). Magnitude = 1/radius."""
    from scipy.interpolate import splprep, splev

    order = _order_contour_indices(sl)
    if order is None or len(order) < 5:
        return sl
    pts = sl.points[order]

    # in-plane basis (two axes spanning the slice plane) so we fit a 2-D curve
    nrm = np.asarray(state["slice_normal"], float)
    nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
    ref = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(nrm, ref); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(nrm, e1)
    x = pts @ e1
    y = pts @ e2

    # arclength parametrisation
    d = np.r_[0, np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))]
    if d[-1] == 0:
        return sl

    try:
        # smoothing spline: s controls smoothness (scaled by number of points)
        s = CONTOUR_SPLINE_SMOOTH * len(x)
        # per=1 -> treat as a closed periodic curve, so the loop seam is smooth
        tck, u = splprep([x, y], u=d, s=s, k=3, per=1)
        dx, dy = splev(d, tck, der=1)
        ddx, ddy = splev(d, tck, der=2)
        # signed 2-D curvature: (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        denom = (dx*dx + dy*dy)**1.5 + 1e-12
        k = (dx*ddy - dy*ddx) / denom
    except Exception:
        return sl

    # make winding consistent across embryos: signed loop area sets orientation
    # (positive = counter-clockwise in the e1-e2 plane). Flip so all match.
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area < 0:
        k = -k

    out = np.zeros(sl.n_points)
    out[order] = np.abs(k)          # magnitude only: notch and arc same sign
    sl[PROP] = out
    return sl

def current_slice(L):
    m = L["meshes"][state["idx"]]
    sl = m.slice(normal=state["slice_normal"], origin=state["slice_origin"])
    if sl.n_points == 0:
        return None
    if CURV_SOURCE == "contour":
        sl = contour_curvature(sl)                 # overwrite with in-plane bending
    return sl

# in contour mode, curvature is a different quantity than the surface values used
# for CLIM above -> recompute the colour range by sampling every frame's contours
if CURV_SOURCE == "contour":
    _cvals = []
    for _idx in range(n_frames):
        state["idx"] = _idx
        for L in LAYERS:
            sl = current_slice(L)
            if sl is not None:
                _cvals.append(np.asarray(sl[PROP]))
    state["idx"] = 0
    if _cvals:
        _cv = np.concatenate(_cvals)
        lim = np.percentile(np.abs(_cv), 98)       # symmetric range about 0
        CLIM = (-lim, lim)
        print(f"contour-curvature CLIM = {CLIM}")

def resample_contour(sl, n=N_RESAMPLE):
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

def mean_contour():
    """Mean resampled slice across all eyes (basal), with direction alignment."""
    sx, sv = [], []
    for L in LAYERS:
        sl = current_slice(L)
        if sl is None: continue
        r = resample_contour(sl)
        if r is None: continue
        sx.append(r[0]); sv.append(r[1])
    if not sx:
        return None
    ref_c = sx[0]
    aligned_xyz, aligned_v = [ref_c], [sv[0]]
    for xyz, v in zip(sx[1:], sv[1:]):
        fwd = np.sum(np.linalg.norm(xyz - ref_c, axis=1))
        rev = np.sum(np.linalg.norm(xyz[::-1] - ref_c, axis=1))
        if rev < fwd:
            xyz, v = xyz[::-1], v[::-1]
        aligned_xyz.append(xyz); aligned_v.append(v)
    poly = pv.lines_from_points(np.mean(aligned_xyz, axis=0))
    poly[PROP] = np.mean(aligned_v, axis=0)
    return poly

def redraw():
    n_drawn = 0
    for li, L in enumerate(LAYERS):
        m = L["meshes"][state["idx"]]
        a = p.add_mesh(m, name=f"frame_{L['name']}", color=L["tint"],
                       opacity=0.25, show_edges=False)
        a.visibility = state["mesh_visible"]
        mesh_actors[L["name"]] = a

        sl = current_slice(L)
        if sl is not None:
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

    # mean contour (toggle with "c")
    p.remove_actor("mean_basal", reset_camera=False)
    if state["show_mean"]:
        mc = mean_contour()
        if mc is not None:
            p.add_mesh(mc, name="mean_basal", color=MEAN_COLOR,
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
    mc = mean_contour()
    if mc is not None:
        mc.save(os.path.join(export_dir, f"slice_MEAN_basal_t{tp}.vtp"))
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
    _export_one(state["idx"])

def export_all_frames():
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
p.add_key_event("1", rz_p)
p.add_key_event("2", rz_m)
p.add_key_event("3", rx_p)
p.add_key_event("4", rx_m)
p.add_key_event("5", ry_p)
p.add_key_event("6", ry_m)
p.add_key_event("period", jump_fwd10)   # . = +10
p.add_key_event("comma",  jump_back10)  # , = -10

redraw()
print("Keys: <-/-> ±1 | ,/. ±10 timepoint | m mesh | b widget | c mean | x export | X export ALL")
print("Rotate ADJUST set: 1/2 = z+/- , 3/4 = x+/- , 5/6 = y+/-")
print("CLIM =", CLIM)
p.show()
