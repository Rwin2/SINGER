"""Generate the V9 visuomotor policy architecture diagram with the new centroid pathway."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# colors
C_BLOCK   = "#2c2c2c"
C_TEXT    = "white"
C_NEW     = "#ff3b30"      # red highlight for the new centroid pathway
C_NEW_BG  = "#ffe5e3"
C_ARROW   = "#5b8def"
C_ARROW_NEW = "#ff3b30"
C_STATE   = "#1d1d1f"
C_BORDER  = "#1d1d1f"

def block(x, y, w, h, label, fc=C_BLOCK, tc=C_TEXT, fs=11, ec=C_BORDER, lw=1.2, radius=0.15):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad=0.02,rounding_size={radius}",
                        fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold")

def arrow(x1, y1, x2, y2, color=C_ARROW, lw=2.2, style="->"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle=style, mutation_scale=18,
                         color=color, lw=lw,
                         shrinkA=2, shrinkB=2)
    ax.add_patch(a)

def label(x, y, text, fs=10, color="black", weight="normal", ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, fontweight=weight)

# ====== Title ======
ax.text(8, 8.55, "Visuomotor Policy  —  with Centroid Features (V9)",
        ha="center", fontsize=18, fontweight="bold")

# ====== Image input ======
block(0.3, 5.2, 1.4, 1.0, "image\n$I^k$", fc="#e8e8e8", tc="black", fs=10)

# ====== CLIPSeg ======
block(2.2, 5.2, 1.6, 1.0, "CLIPSeg", fs=12)
arrow(1.7, 5.7, 2.2, 5.7)

# heatmap thumbnail (just a colored box to suggest the heatmap)
hm = Rectangle((4.2, 5.25), 0.9, 0.9, fc="#ff6b35", ec="black", lw=1)
ax.add_patch(hm)
inner = Rectangle((4.45, 5.50), 0.4, 0.4, fc="#ffd23f", ec="none")
ax.add_patch(inner)
ax.text(4.65, 6.25, "heatmap\n352×352", ha="center", fontsize=8, style="italic")
arrow(3.8, 5.7, 4.2, 5.7)

# ====== Top branch: SqueezeNet → Feature Extractor MLP ======
block(5.6, 6.6, 1.7, 0.9, "SqueezeNet", fs=11)
block(7.7, 6.6, 2.1, 0.9, "Feature\nExtractor MLP", fs=10)
arrow(5.1, 6.0, 5.6, 7.05)
arrow(7.3, 7.05, 7.7, 7.05)

# ====== NEW: Centroid Extraction branch ======
# Big red box highlighting the new pathway
nbox_x, nbox_y, nbox_w, nbox_h = 5.6, 3.8, 2.1, 1.1
block(nbox_x, nbox_y, nbox_w, nbox_h,
      "Centroid\nExtraction", fc="white", tc=C_NEW, fs=11, ec=C_NEW, lw=2.5)
# arrow from heatmap down into the new block
arrow(5.1, 5.4, 5.6, 4.5, color=C_ARROW_NEW, lw=2.5)

# c^k vector label (placed ABOVE the arrow so it doesn't collide with Action Head)
ax.text(9.15, 4.75,
        r"$c^k = (\mathrm{bearing},\ \mathrm{elevation},\ \mathrm{apparent\_size}) \in \mathbb{R}^3$",
        ha="center", va="center", fontsize=10.5, color=C_NEW, fontweight="bold")

# "NEW" badge
badge = FancyBboxPatch((4.05, 3.55), 1.45, 0.35,
                        boxstyle="round,pad=0.02,rounding_size=0.1",
                        fc=C_NEW, ec="none")
ax.add_patch(badge)
ax.text(4.775, 3.725, "NEW in V9", ha="center", va="center",
        color="white", fontsize=9, fontweight="bold")

# ====== State inputs (velocity, attitude, position) ======
state_x = 5.8
state_y = 2.6
ax.text(state_x - 0.4, state_y + 0.45,
        r"$v_{\mathcal{W}}^k,\ q_{\mathcal{BW}}^k$",
        ha="left", va="center", fontsize=12)
ax.text(state_x - 0.4, state_y + 0.05,
        r"$p_{z\mathcal{W}}^k$",
        ha="left", va="center", fontsize=12)

# ====== History block ======
hist_x = 0.4
hist_y = 0.9
ax.text(hist_x, hist_y + 1.0,
        r"$\delta t^{k-5:k-1}$" + "\n" +
        r"$\delta p_{\mathcal{W}}^{k-5:k-1}$" + "\n" +
        r"$\delta v_{\mathcal{W}}^{k-5:k-1}$" + "\n" +
        r"$\delta q_{\mathcal{BW}}^{k-5:k-1}$" + "\n" +
        r"$u^{k-5:k-1}$",
        ha="left", va="center", fontsize=10)
ax.text(hist_x + 1.4, hist_y + 2.05, "History (250 ms)", fontsize=9, style="italic")

block(3.8, 1.0, 2.0, 0.95, "History MLP\n[64,32,8]", fs=10)
arrow(3.0, 1.5, 3.8, 1.5)
ax.text(6.3, 1.5, r"$k_{th},\ m_{dr}$", fontsize=11, va="center")
arrow(5.8, 1.5, 6.25, 1.5)

# ====== Concat & Action Head ======
# concat point (dashed bracket)
concat_x = 10.6
ax.plot([concat_x, concat_x], [1.2, 7.4], color="black", lw=1.2, ls="--")
ax.plot([concat_x, concat_x + 0.15], [7.4, 7.4], color="black", lw=1.2, ls="--")
ax.plot([concat_x, concat_x + 0.15], [1.2, 1.2], color="black", lw=1.2, ls="--")
ax.text(concat_x - 0.1, 7.65, "concat", ha="right", fontsize=9, style="italic")

# arrows into concat
arrow(9.8, 7.05, concat_x, 7.05)            # SqueezeNet→FE→concat
arrow(7.7, 4.35, concat_x, 4.35, color=C_ARROW_NEW, lw=2.5)  # centroid→concat (NEW)
arrow(7.0, 3.05, concat_x, 3.05)            # state v,q→concat
arrow(7.0, 2.65, concat_x, 2.65)            # state p→concat
arrow(7.1, 1.5, concat_x, 1.5)               # history→concat

# Action Head
block(11.0, 3.4, 2.4, 1.6, "Action\nHead\n[100,100]", fs=12)
arrow(concat_x + 0.05, 4.2, 11.0, 4.2)

# Output u^k
ax.text(13.9, 4.2, r"$u^k$", fontsize=20, va="center", fontweight="bold")
ax.text(13.9, 3.75, "[4,1]", fontsize=9, va="center", color="gray")
arrow(13.4, 4.2, 13.85, 4.2)

# ====== Legend / annotation box (bottom right) ======
legend_x, legend_y, legend_w, legend_h = 11.2, 0.4, 4.5, 2.0
lp = FancyBboxPatch((legend_x, legend_y), legend_w, legend_h,
                     boxstyle="round,pad=0.05,rounding_size=0.15",
                     fc="#fff8f7", ec=C_NEW, lw=1.5)
ax.add_patch(lp)
ax.text(legend_x + legend_w/2, legend_y + legend_h - 0.25,
        "Centroid features (V9)", ha="center", fontsize=11,
        fontweight="bold", color=C_NEW)
ax.text(legend_x + 0.15, legend_y + legend_h - 0.6,
        "• Tap CLIPSeg heatmap (no new params)\n"
        "• Threshold @ P75 → weighted (x,y) mean\n"
        "• 3 scalars concatenated into Action Head\n"
        "• +44 pp BC success    (36.7% → 80.7%)",
        ha="left", va="top", fontsize=9, color="#1d1d1f")

plt.tight_layout()
out = "/data/erwinpi/SINGER/assets/v9_arch_diagram.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
