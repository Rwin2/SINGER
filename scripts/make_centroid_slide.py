"""
Generate a slide-ready overlay illustrating the centroid features
(centroid, bearing, elevation) added to SINGER on top of CLIPSeg.

Usage:
    conda run -n FiGS python /data/erwinpi/SINGER/scripts/make_centroid_slide.py \
        --frame /data/erwinpi/SINGER/assets/leafblower_frame_45.png \
        --phrase "green and pink leafblower" \
        --out  /data/erwinpi/SINGER/assets/centroid_overlay.png

If --frame is omitted it will extract frame 45 from assets/leafblower_rgb.mp4.
"""
import argparse, math, os, subprocess
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

ap = argparse.ArgumentParser()
ap.add_argument("--frame", default="/data/erwinpi/SINGER/assets/leafblower_frame_45.png")
ap.add_argument("--phrase", default="green and pink leafblower")
ap.add_argument("--out", default="/data/erwinpi/SINGER/assets/centroid_overlay.png")
ap.add_argument("--hfov", type=float, default=85.0, help="ZED Mini horizontal FOV (deg)")
ap.add_argument("--vfov", type=float, default=54.0, help="ZED Mini vertical FOV (deg)")
args = ap.parse_args()

# auto-extract frame 45 from the demo video if missing
if not os.path.exists(args.frame):
    print(f"extracting frame 45 from leafblower_rgb.mp4 -> {args.frame}")
    subprocess.run([
        "ffmpeg", "-y", "-i", "/data/erwinpi/SINGER/assets/leafblower_rgb.mp4",
        "-vf", "select=eq(n\\,45)", "-frames:v", "1", "-update", "1", args.frame
    ], check=True)

img = Image.open(args.frame).convert("RGB")
W, H = img.size

# ---- CLIPSeg, zero-shot, exactly what SINGER deploys onboard ----
device = "cuda" if torch.cuda.is_available() else "cpu"
proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()

with torch.no_grad():
    inputs = proc(text=[args.phrase], images=[img], padding=True, return_tensors="pt").to(device)
    logits = model(**inputs).logits  # (352,352) for single prompt, (N,352,352) for batch
    if logits.dim() == 3:
        logits = logits[0]
    heat = torch.sigmoid(logits).cpu().numpy()

heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
heat_arr = np.asarray(heat_img).astype(np.float32) / 255.0

# ---- centroid: intensity-weighted mean over the top 5% of activations ----
thr = np.quantile(heat_arr, 0.95)
mask = heat_arr >= thr
ys, xs = np.where(mask)
weights = heat_arr[mask]
cx = float((xs * weights).sum() / weights.sum())
cy = float((ys * weights).sum() / weights.sum())

# ---- bearing / elevation in body frame (image center = forward axis) ----
img_cx, img_cy = W / 2.0, H / 2.0
dx = cx - img_cx        # +right
dy = cy - img_cy        # +down (image coords)
fx = (W / 2.0) / math.tan(math.radians(args.hfov / 2.0))
fy = (H / 2.0) / math.tan(math.radians(args.vfov / 2.0))
bearing_rad   = math.atan2(dx, fx)
elevation_rad = math.atan2(-dy, fy)  # negate so up is positive
bearing_deg   = math.degrees(bearing_rad)
elevation_deg = math.degrees(elevation_rad)

print(f"image: {W}x{H}")
print(f"centroid pixel: ({cx:.1f}, {cy:.1f})")
print(f"bearing:   {bearing_deg:+.1f} deg")
print(f"elevation: {elevation_deg:+.1f} deg")

# ---- compose overlay ----
canvas = img.copy().convert("RGBA")
heat_rgba = np.zeros((H, W, 4), dtype=np.uint8)
heat_rgba[..., 0] = 255                                # red channel
heat_rgba[..., 3] = (heat_arr * 180).astype(np.uint8)  # alpha = similarity
canvas = Image.alpha_composite(canvas, Image.fromarray(heat_rgba, "RGBA"))

draw = ImageDraw.Draw(canvas)

# crosshair at image center (body-frame forward)
ch = 18
draw.line([(img_cx - ch, img_cy), (img_cx + ch, img_cy)], fill=(255, 255, 255, 255), width=2)
draw.line([(img_cx, img_cy - ch), (img_cx, img_cy + ch)], fill=(255, 255, 255, 255), width=2)
draw.ellipse([img_cx - 3, img_cy - 3, img_cx + 3, img_cy + 3], fill=(255, 255, 255, 255))

# bearing arrow (cyan, horizontal) and elevation arrow (green, vertical)
draw.line([(img_cx, img_cy), (cx, img_cy)], fill=(0, 200, 255, 255), width=4)
draw.line([(cx, img_cy), (cx, cy)], fill=(0, 255, 120, 255), width=4)

# centroid marker
r = 9
draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 0, 255), width=3)
draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 255, 0, 255))

try:
    font  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    sfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except Exception:
    font = ImageFont.load_default(); sfont = font

def label(xy, text, fill):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=sfont)
    draw.rectangle([bbox[0]-3, bbox[1]-3, bbox[2]+3, bbox[3]+3], fill=(0, 0, 0, 200))
    draw.text((x, y), text, font=sfont, fill=fill)

label((img_cx + 6, img_cy + 6), "image center (body forward)", (255, 255, 255, 255))
label(((img_cx + cx) / 2 - 30, img_cy - 22), f"bearing  {bearing_deg:+.1f}\u00b0", (0, 220, 255, 255))
label((cx + 8, (img_cy + cy) / 2 - 8),       f"elevation {elevation_deg:+.1f}\u00b0", (0, 255, 140, 255))
label((cx + 12, cy - 22),                    f"centroid ({int(cx)},{int(cy)})", (255, 255, 0, 255))

# title strip
title = f'CLIPSeg("{args.phrase}")  \u2192  centroid  \u2192  (bearing, elevation)'
draw.rectangle([0, 0, W, 28], fill=(0, 0, 0, 200))
draw.text((10, 6), title, font=font, fill=(255, 255, 255, 255))

canvas.convert("RGB").save(args.out)
print("saved", args.out)
