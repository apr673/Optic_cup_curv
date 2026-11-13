import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob, os, re
from scipy.stats import gaussian_kde

export_dir = r"D:\Patricia\optic_cup_analysis-main\plane slice figure\apical vs basal\exports"
snap_dir   = os.path.join(export_dir, "for fig")
PROP = "contour_curv"          # contour curvature from the updated script
HIST_TRIM = 15                   # drop this many points from EACH end of the contour
                               # (excludes extremity artifacts from the histogram only)
ROLL_WINDOW = 6                 # pool +/- (ROLL_WINDOW//2) neighbouring timepoints
                               # into each histogram (1 = no pooling)

hist_dir = os.path.join(export_dir, "histograms")
os.makedirs(hist_dir, exist_ok=True)

# timepoints present as MEAN files (sorted)
tps = sorted({int(re.search(r'_t(\d+)\.vtp$', os.path.basename(f)).group(1))
              for f in glob.glob(os.path.join(export_dir, "slice_MEAN_*_t*.vtp"))})
tps_set = set(tps)

# snapshots sorted by the FIRST number in the filename (numeric, not string order)
def first_num(path):
    m = re.search(r'(\d+(?:\.\d+)?)', os.path.basename(path))
    return float(m.group(1)) if m else -1
snap_files = sorted(glob.glob(os.path.join(snap_dir, "*.*")), key=first_num)

print(f"{len(tps)} timepoints, {len(snap_files)} snapshots")
if len(tps) != len(snap_files):
    print("WARNING: counts differ - pairing by order may misalign!")
pairs = list(zip(tps, snap_files))

def load_mean(side, tp):
    f = os.path.join(export_dir, f"slice_MEAN_{side}_t{tp}.vtp")
    if not os.path.exists(f):
        return np.array([])
    vals = -pv.read(f)[PROP]          # INVERT sign for display
    if HIST_TRIM > 0 and vals.size > 2 * HIST_TRIM:
        vals = vals[HIST_TRIM:-HIST_TRIM]   # exclude endpoint artifacts
    return vals

def load_mean_pooled(side, tp, all_tps):
    """Pool contour values from a centred rolling window of timepoints around tp."""
    half = ROLL_WINDOW // 2
    pool = []
    for t in range(tp - half, tp + half + 1):
        if t in all_tps:
            v = load_mean(side, t)
            if v.size:
                pool.append(v)
    return np.concatenate(pool) if pool else np.array([])

# --- fixed axis limits across ALL timepoints ---
all_vals = []
for tp in tps:
    for side in ("basal", "apical"):
        v = load_mean_pooled(side, tp, tps_set)
        if v.size: all_vals.append(v)
all_vals = np.concatenate(all_vals)
bins = np.linspace(all_vals.min(), all_vals.max(), 40)

max_dens = 0.0
for tp in tps:
    for side in ("basal", "apical"):
        v = load_mean_pooled(side, tp, tps_set)
        if v.size:
            counts, _ = np.histogram(v, bins=bins, density=True)  # density scale
            max_dens = max(max_dens, counts.max())
y_max = max_dens * 1.15   # headroom (density; also covers KDE peaks)

for tp, sf in pairs:
    basal  = load_mean_pooled("basal",  tp, tps_set)
    apical = load_mean_pooled("apical", tp, tps_set)

    # snapshot on TOP, histogram on BOTTOM
    fig, (ax_i, ax_h) = plt.subplots(2, 1, figsize=(7, 9),
                                     gridspec_kw={"height_ratios": [1.2, 1]})

    # --- snapshot (top) ---
    ax_i.axis("off")
    if os.path.exists(sf):
        ax_i.imshow(mpimg.imread(sf))
        ax_i.set_title(f"t{tp}", fontsize=10)

    # --- histogram (bottom) ---
    if basal.size:
        ax_h.hist(basal,  bins=bins, alpha=0.5, label="basal",  color="indianred", density=True)
    if apical.size:
        ax_h.hist(apical, bins=bins, alpha=0.5, label="apical", color="steelblue", density=True)
    ax_h.axvline(0, color="gray", lw=0.6)

    # --- KDE overlay (smooth density) ---
    xs = np.linspace(bins[0], bins[-1], 400)
    peak_dens = 0.0
    for vals, col, lbl in [(basal, "indianred", "basal"),
                           (apical, "steelblue", "apical")]:
        if vals.size > 2 and np.ptp(vals) > 0:
            kde = gaussian_kde(vals)
            dens = kde(xs)                             # density, matches density=True bars
            ax_h.plot(xs, dens, color=col, lw=1.8)
            peak_dens = max(peak_dens, dens.max())

    ax_h.set_xlim(bins[0], bins[-1])
    ax_h.set_ylim(0, y_max)
    ax_h.set_xlabel("contour curvature"); ax_h.set_ylabel("density")
    ax_h.legend(frameon=False)

    fig.tight_layout()
    out = os.path.join(hist_dir, f"histogram_CONTOUR_t{tp:03d}.tiff")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  t{tp} + {os.path.basename(sf)} -> {os.path.basename(out)}")

print(f"\ndone - {len(pairs)} figures in {hist_dir}")
