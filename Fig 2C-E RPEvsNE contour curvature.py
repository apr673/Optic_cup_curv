import pyvista as pv
import numpy as np
from scipy.interpolate import interp1d
import glob, os

# --- datasets: one entry per image set ---
DATASETS = [
    {"name": "set1", "folder": r"D:\Patricia\optic_cup_analysis-main\optic_cup_230524_pt1 changed\data\meshes\230524\basal_top_bottom",
     "top": "GM2_top_t*.ply", "bottom": "GM2_bottom_t*.ply", "tint": "lightblue"},
    {"name": "set2", "folder": r"D:\Patricia\optic_cup_analysis-main\optic_cup_230524_pt4\data\meshes\230524\basal_top_bottom",
     "top": "GM2_top_t*.ply", "bottom": "GM2_bottom_t*.ply", "tint": "lightgreen"},
    {"name": "set3", "folder": r"D:\Patricia\optic_cup_analysis-main\optic_cup_230524_pt2\data\meshes\230524\basal_top_bottom",
     "top": "GM2_top_t*.ply", "bottom": "GM2_bottom_t*.ply", "tint": "lightpink","mirror": "x",   },  # "x", "y", "z", or None
    



    # add more datasets here...
]
export_dir = r"D:\Patricia\Mesh_visualization\exports"
PROP = "mean_curv"
REF_SIGN = +1
ALIGN_MODE = "per_frame"      # "per_frame" = anchor each dataset to its bottom centroid; "none" = raw
CMAP = "turbo"
COLOR_SLICES_BY_LAYER = False
SLICE_COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "black"]
SIDE_MEAN_COLOR = {"top": "white", "bottom": "yellow"}
N_RESAMPLE = 200

# --- interactive rotation: name of dataset to adjust (set None when done) ---
ADJUST = "set3"
ROT_STEP = 5.0                # degrees per keypress
adjust_state = {"rx": 0.0, "ry": 0.0, "rz": 0.0}

os.makedirs(export_dir, exist_ok=True)

def tp_num(path):                       # ..._t12.ply -> 12
    return int(os.path.basename(path).split("_t")[-1].split(".")[0])

def load_dataset(ds):
    """Load top+bottom; anchor to bottom centroid; mirror; apply baked rotation; compute + sign-align curvature."""
    raw = {}
    for side in ("top", "bottom"):
        files = sorted(glob.glob(os.path.join(ds["folder"], ds[side])), key=tp_num)
        raw[side] = {"files": files, "meshes": [pv.read(f).triangulate() for f in files]}

    n = min(len(raw["top"]["meshes"]), len(raw["bottom"]["meshes"]))
    rot = ds.get("rotate", (0, 0, 0))
    mir = ds.get("mirror")
    layers = []
    for side in ("top", "bottom"):
        meshes = []
        for i in range(n):
            m = raw[side]["meshes"][i]
            if ALIGN_MODE == "per_frame":
                ref_center = np.asarray(raw["bottom"]["meshes"][i].center)  # anchor to bottom
                m.points = m.points - ref_center
            if mir:                                       # mirror first (left/right eye)
                ax = {"x": 0, "y": 1, "z": 2}[mir]
                m.points[:, ax] = -m.points[:, ax]
            if any(rot):                                  # then baked rotation (degrees)
                if rot[0]: m.rotate_x(rot[0], inplace=True)
                if rot[1]: m.rotate_y(rot[1], inplace=True)
                if rot[2]: m.rotate_z(rot[2], inplace=True)
            cv = m.curvature(curv_type="mean")
            if np.sign(np.median(cv)) not in (0, REF_SIGN):
                cv = -cv
            m[PROP] = cv
            meshes.append(m)
        layers.append({"name": f"{ds['name']}_{side}", "meshes": meshes,
                       "files": raw[side]["files"][:n], "tint": ds["tint"]})
    return layers




# build flat list of layers
LAYERS = []
for ds in DATASETS:
    LAYERS.extend(load_dataset(ds))
if not LAYERS:
    raise SystemExit("No meshes found - check folders/patterns.")

# store centered+baked-rotation points, so interactive rotation re-applies cleanly
for L in LAYERS:
    L["orig_points"] = [m.points.copy() for m in L["meshes"]]

n_frames = min(len(L["files"]) for L in LAYERS)
print(f"Loaded {len(LAYERS)} layers: {[L['name'] for L in LAYERS]}, {n_frames} timepoints")

# shared colour range across everything
allvals = np.concatenate([m[PROP] for L in LAYERS for m in L["meshes"]])
CLIM = tuple(np.percentile(allvals, [2, 98]))

ref = LAYERS[0]["meshes"][0]
state = {"idx": 0, "slice_origin": list(ref.center), "slice_normal": [0, 0, 1],
         "mesh_visible": True, "show_mean": False}
mesh_actors = {}

p = pv.Plotter()

def current_slice(L):
    m = L["meshes"][state["idx"]]
    sl = m.slice(normal=state["slice_normal"], origin=state["slice_origin"])
    return sl if sl.n_points > 0 else None

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
        ref = sx[0]                                   # first contour = direction reference
        aligned_xyz, aligned_v = [ref], [sv[0]]
        for xyz, v in zip(sx[1:], sv[1:]):
            # compare forward vs reversed against the reference; keep the closer one
            fwd = np.sum(np.linalg.norm(xyz - ref, axis=1))
            rev = np.sum(np.linalg.norm(xyz[::-1] - ref, axis=1))
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

    # mean contour per side (toggle with "c")
    for side in ("top", "bottom"):
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
    """Re-apply rotation from the stored originals (no compounding), then recompute curvature sign."""
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

def export_frame():
    tp = tp_num(LAYERS[0]["files"][state["idx"]])
    for L in LAYERS:
        sl = current_slice(L)
        if sl is None:
            continue
        vtp = os.path.join(export_dir, f"slice_{L['name']}_t{tp}.vtp")
        sl.save(vtp)
        print(f"  wrote {vtp}")
    for side, mc in mean_contours_by_side().items():
        mvtp = os.path.join(export_dir, f"slice_MEAN_{side}_t{tp}.vtp")
        mc.save(mvtp)
        print(f"  wrote {mvtp}")
    png = os.path.join(export_dir, f"view_t{tp}.png")
    slice_widget.SetEnabled(False); p.render()
    p.screenshot(png, scale=4)
    slice_widget.SetEnabled(True); p.render()
    print(f"  wrote {png}")

# keys
p.add_key_event("Right", next_frame)
p.add_key_event("Left", prev_frame)
p.add_key_event("m", toggle_mesh)
p.add_key_event("b", toggle_widget)
p.add_key_event("c", toggle_mean)
p.add_key_event("x", export_frame)
# rotation keys (1/2 = z, 3/4 = x, 5/6 = y)
p.add_key_event("1", rz_p)
p.add_key_event("2", rz_m)
p.add_key_event("3", rx_p)
p.add_key_event("4", rx_m)
p.add_key_event("5", ry_p)
p.add_key_event("6", ry_m)

redraw()
print("Keys: <-/-> timepoint | m mesh | b widget | c mean | x export")
print("Rotate ADJUST set: 1/2 = z+/- , 3/4 = x+/- , 5/6 = y+/-")
print("CLIM =", CLIM)
p.show()