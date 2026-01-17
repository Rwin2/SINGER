import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim
from scipy.stats import mode

import torch
import torch.nn.functional as F
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")
# import onnxruntime as ort

import cv2
import imageio
from PIL import Image
import albumentations as A

from typing import Any, List, Optional, Tuple, Type, Union

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class CLIPSegHFModel:
    def __init__(
        self,
        hf_model: str = "CIDAS/clipseg-rd64-refined",
        device: Optional[str] = None,
        cmap: str = "turbo",
    ):
        """
        HuggingFace CLIPSeg wrapper for direct torch inference.

        hf_model: model ID or path
        device: torch.device string, e.g. 'cuda' or 'cpu'
        cmap: matplotlib colormap for colorization
        """
        # select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # load processor and model
        self.processor = CLIPSegProcessor.from_pretrained(hf_model, use_fast=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained(hf_model)
        self.model.to(self.device)
        self.model.eval()
        # LUT for colorization
        self.lut = get_colormap_lut(cmap_name=cmap)

        # initialize running bounds
        self.running_min = float('inf')
        self.running_max = float('-inf')
        # self.running_min = None
        # self.running_max = 1.0
        # self.eps = 1e-10

        # initialize frame cache
        self.prev_image = None
        self.prev_output = None

        # ema for smoothing
        self.segmentation_ema = None
        self.ema_alpha = 0.7  # adjust between [0.0, 1.0]; higher = more current weight
        self.last_superpixel_mask = None
        self.superpixel_every = 1  # update superpixels every N frames
        self.frame_counter = 0

    def _rescale_global(self, arr: np.ndarray) -> np.ndarray:
        """
        Rescale `arr` to [0,1] using the min/max seen so far (updated here).
        """
        cur_min = float(arr.min())
        cur_max = float(arr.max())
        self.running_min = min(self.running_min, cur_min)
        self.running_max = max(self.running_max, cur_max)
        # print(f"Running bounds: min={self.running_min}, max={self.running_max}")
        # print(f"Current bounds: min={cur_min}, max={cur_max}")
        span = self.running_max - self.running_min
        if span < 1e-9:
            scaled = np.zeros_like(arr, dtype=np.float32)
        else:
            scaled = (arr - self.running_min) / span
        return scaled

    def clipseg_hf_inference(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        resize_output_to_input: bool = True,
        use_refinement: bool = True,
        scene_change_threshold: float = 1.00,
        verbose=False
    ) -> np.ndarray:
        """
        Run CLIPSeg on a PIL image or numpy array, return colorized mask as (H,W,3) uint8.
        """
        def log(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        # --- Step 1: Normalize input to PIL + NumPy ---
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
            image_np = image
        elif isinstance(image, Image.Image):
            img = image
            image_np = np.array(image)
        else:
            raise TypeError(f"Unsupported image type {type(image)}")

        # --- Step 2: Determine whether to reuse ---
        should_reuse = False
        ssim_score = None
        if self.prev_image is not None and self.prev_output is not None:
            # Compare resized grayscale SSIM
            prev_small = cv2.resize(self.prev_image, (64, 64))
            curr_small = cv2.resize(image_np, (64, 64))
            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            ssim_score = ssim(prev_gray, curr_gray, data_range=1.0)
            should_reuse = ssim_score >= scene_change_threshold
            log(f"[DEBUG] SSIM = {ssim_score:.4f}, Threshold = {scene_change_threshold}, Reuse = {should_reuse}")

        # --- Step 3: Reuse path (warp previous mask) ---
        if should_reuse:
            mask_u8 = warp_mask(self.prev_image, image_np, self.prev_output)
            # Optional: fast filtering
            mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=75, sigmaSpace=75)
            colorized = colorize_mask_fast(mask_u8, self.lut)
            overlayed = blend_overlay_gpu(image_np, colorized)
            return overlayed

        # --- Step 4: Run inference (scene has changed) ---
        start = time.time()
        inputs = self.processor(images=img, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1,1,H,W]

        arr = logits.cpu().squeeze().numpy().astype(np.float32)
        skip_norm = prompt.strip().lower() == "null"
        if skip_norm:
            prob = 1.0 / (1.0 + np.exp(-arr))
            mask_u8 = (prob * 255).astype(np.uint8)
        else:
            scaled = self._rescale_global(arr)
            mask_u8 = (scaled * 255).astype(np.uint8)

        if resize_output_to_input:
            mask_u8 = np.array(Image.fromarray(mask_u8).resize(img.size, resample=Image.BILINEAR))

        # --- Step 5: Post-processing only on fresh inference ---
        # Temporal EMA
        mask_f = mask_u8.astype(np.float32)
        if self.segmentation_ema is None or self.segmentation_ema.shape != mask_f.shape:
            self.segmentation_ema = mask_f.copy()
        else:
            self.segmentation_ema = (
                self.ema_alpha * mask_f + (1 - self.ema_alpha) * self.segmentation_ema
            )
        mask_u8 = np.clip(self.segmentation_ema, 0, 255).astype(np.uint8)

        # Optional filtering
        mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=75, sigmaSpace=75)

        # Optional: superpixel refinement
        if use_refinement:
            self.frame_counter += 1
            if self.frame_counter % self.superpixel_every == 0:
                # self.last_superpixel_mask = superpixel_smoothing(image_np, mask_u8)
                self.last_superpixel_mask = fast_superpixel_seeds(image_np, mask_u8)
            if self.last_superpixel_mask is not None:
                mask_u8 = self.last_superpixel_mask

        # --- Step 6: Render and cache ---
        colorized = colorize_mask_fast(mask_u8, self.lut)
        overlayed = blend_overlay_gpu(image_np, colorized)

        # Store just raw mask + image for reuse
        self.prev_image = image_np.copy()
        self.prev_output = mask_u8.copy()
        # # --- Step 1: Convert to PIL and NumPy ---
        # if isinstance(image, np.ndarray):
        #     img = Image.fromarray(image)
        #     image_np = image
        # elif isinstance(image, Image.Image):
        #     img = image
        #     image_np = np.array(image)
        # else:
        #     raise TypeError(f"Unsupported image type {type(image)}")
        
        # start = time.time()
        # # --- Step 2: Decide whether to reuse previous output ---
        # should_reuse = False
        # ssim_score = None

        # if self.prev_image is not None:
        #     # Compute SSIM
        #     prev_small = cv2.resize(self.prev_image, (64, 64))
        #     curr_small = cv2.resize(image_np, (64, 64))
        #     prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY)
        #     curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_RGB2GRAY)
        #     ssim_score = ssim(prev_gray, curr_gray, data_range=1.0)

        #     if ssim_score > scene_change_threshold:
        #         mask_u8 = warp_mask(prev_gray, image_np, self.prev_output)
        #         should_reuse = True
        #     log(f"[DEBUG] SSIM = {ssim_score:.4f}, Threshold = {scene_change_threshold}, Reuse = {ssim_score >= scene_change_threshold}")

        #     # Determine reuse
        #     should_reuse = ssim_score >= scene_change_threshold

        # if should_reuse and self.prev_output is not None:
        #     log(f"[DEBUG] Reusing previous output | SSIM = {ssim_score:.4f} ≥ {scene_change_threshold}")
        #     return self.prev_output

        # # --- 3. Run CLIPSeg inference ---
        # if not should_reuse:
        #     inputs = self.processor(images=img, text=prompt, return_tensors="pt")
        #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
        #     with torch.no_grad():
        #         outputs = self.model(**inputs)
        #         logits = outputs.logits  # shape [1, 1, H, W]

        #     arr = logits.cpu().squeeze().numpy().astype(np.float32)
        #     skip_norm = prompt.strip().lower() == "null"

        #     if skip_norm:
        #         prob = 1.0 / (1.0 + np.exp(-arr))
        #         mask_u8 = (prob * 255).astype(np.uint8)
        #     else:
        #         scaled = self._rescale_global(arr)
        #         mask_u8 = (scaled * 255).astype(np.uint8)

        # # --- 4. Resize to input resolution if needed ---
        # if resize_output_to_input:
        #     target_size = img.size  # (W, H)
        #     mask_u8 = np.array(
        #         Image.fromarray(mask_u8).resize(target_size, resample=Image.BILINEAR)
        #     )

        # # --- 5. Postprocessing: guided + superpixel ---
        # if not should_reuse:
        #     if mask_u8.shape != image_np.shape[:2]:
        #         mask_u8 = cv2.resize(mask_u8, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        #     # === [ Temporal smoothing via EMA ] ===
        #     mask_f = mask_u8.astype(np.float32)
        #     if self.segmentation_ema is None or self.segmentation_ema.shape != mask_f.shape:
        #         self.segmentation_ema = mask_f.copy()
        #     else:
        #         self.segmentation_ema = (
        #             self.ema_alpha * mask_f + (1 - self.ema_alpha) * self.segmentation_ema
        #         )
        #     mask_u8 = np.clip(self.segmentation_ema, 0, 255).astype(np.uint8)
        #     # mask_u8 = guided_smoothing(image_np, mask_u8)
        #     mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=75, sigmaSpace=75)
        #     # === [ Superpixel smoothing every N frames ] ===
        #     if use_refinement:
        #         self.frame_counter += 1
        #         if self.frame_counter % self.superpixel_every == 0:
        #             self.last_superpixel_mask = superpixel_smoothing(image_np, mask_u8)
        #         if self.last_superpixel_mask is not None:
        #             mask_u8 = self.last_superpixel_mask

        # # --- 6. Colorize and overlay ---
        # colorized = colorize_mask_fast(mask_u8, self.lut)
        # overlayed = blend_overlay_gpu(image_np, colorized)

        # # --- 7. Cache for reuse ---
        # self.prev_image = image_np.copy()
        # self.prev_output = mask_u8.copy()
        # # 1) ensure PIL
        # # if isinstance(image, np.ndarray):
        # #     # handle BGR->RGB if needed
        # #     if image.ndim == 3 and image.shape[2] == 3:
        # #         img = Image.fromarray(image)
        # #     else:
        # #         img = Image.fromarray(image)
        # # elif isinstance(image, Image.Image):
        # #     img = image
        # # else:
        # #     raise TypeError(f"Unsupported image type {type(image)}")
        
        # # skip_norm = isinstance(prompt, str) and prompt.strip().lower() == "null"

        # # # 2) preprocess
        # # inputs = self.processor(images=img, text=prompt, return_tensors="pt")
        # # inputs = {k: v.to(self.device) for k,v in inputs.items()}

        # # # 3) inference
        # # with torch.no_grad():
        # #     outputs = self.model(**inputs)
        # #     logits = outputs.logits  # shape [1,1,H,W]

        # # # 4) postprocess logits to (H,W)
        # # arr = logits.cpu().squeeze().numpy().astype(np.float32)
        # # if skip_norm:
        # #     # apply sigmoid directly
        # #     prob = 1.0 / (1.0 + np.exp(-arr))
        # #     mask_u8 = (prob * 255).astype(np.uint8)
        # # else:
        # #     scaled = self._rescale_global(arr)
        # #     mask_u8 = (scaled * 255).astype(np.uint8)

        # # # 5) resize if needed
        # # if resize_output_to_input:
        # #     target_size = img.size  # (W,H)
        # #     mask_u8 = np.array(
        # #         Image.fromarray(mask_u8).resize(target_size, resample=Image.BILINEAR)
        # #     )
        
        # # # # === [ NEW ] Edge-aware + spatial consistency refinement === #
        # # # image_np = np.array(img)
        # # # # 7) Guided smoothing
        # # # mask_u8 = guided_smoothing(image_np, mask_u8)

        # # # # 8) Superpixel smoothing
        # # # mask_u8 = superpixel_smoothing(image_np, mask_u8)
        # # # # =========================================================== #

        # # # 6) colorize
        # # colorized = colorize_mask_fast(mask_u8, self.lut)
        # # overlayed = blend_overlay_gpu(image, colorized)
        end = time.time()
        log(f"CLIPSeg inference time: {end - start:.3f} seconds")
        return overlayed

################################################
# 2. Lookup Table for Semantic Probability Map #
################################################

# Precompute the LUT once
def get_colormap_lut(cmap_name="turbo", lut_size=256):
    cmap = plt.get_cmap(cmap_name, lut_size)
    lut = (cmap(np.linspace(0, 1, lut_size))[:, :3] * 255).astype(np.uint8)  # shape: (256, 3)
    return lut  # Each row is an RGB color

# Fast LUT-based colorizer
def colorize_mask_fast(mask_np, lut):
    """
    Convert a (H, W) uint8 mask into a (H, W, 3) RGB image using the provided LUT.
    """
    return lut[mask_np]  # Very fast NumPy indexing: shape (H, W, 3)

################################################
# 3. Utility Functions for Image Processing #
################################################
    

def blend_overlay_gpu(base: np.ndarray,
                      overlay: np.ndarray,
                      alpha: float = 1.00) -> np.ndarray:
    """
    Convert `base`→mono-gray on GPU (if needed), resize `overlay` on GPU,
    then blend:  result = α·overlay + (1−α)·gray_base.  Entirely on CUDA.

    Parameters:
        base (np.ndarray): 
            • If 3-channel: H×W×3 BGR/uint8. 
            • If single-channel: H×W/uint8. 
        overlay (np.ndarray): H'×W'×3 BGR/uint8 image to overlay.
        alpha (float): opacity of overlay in [0,1].

    Returns:
        np.ndarray: H×W×3 BGR/uint8 blended result (on CPU).
    """
    # 1. Upload base to GPU and convert to float
    if base.ndim == 2:
        # already gray
        base_gray = torch.from_numpy(base).float().cuda()             # shape [H, W]
    elif base.ndim == 3 and base.shape[2] == 3:
        B = torch.from_numpy(base[:, :, 0]).float().cuda()            # BGR channels
        G = torch.from_numpy(base[:, :, 1]).float().cuda()
        R = torch.from_numpy(base[:, :, 2]).float().cuda()
        # Standard Rec. 601 luma-weights for BGR→gray:
        base_gray = 0.114 * B + 0.587 * G + 0.299 * R               # shape [H, W]
    else:
        raise ValueError(
            "`base` must be H×W (gray) or H×W×3 (BGR).")

    H, W = base_gray.shape

    # 2. Upload overlay to GPU as a float tensor [3, H', W']
    if overlay.ndim != 3 or overlay.shape[2] != 3:
        raise ValueError("`overlay` must be H'×W'×3 (BGR/uint8).")
    ov = torch.from_numpy(overlay).float().permute(2, 0, 1).cuda()    # [3, H', W']
    ov = ov.unsqueeze(0)                                             # [1, 3, H', W']

    # 3. Resize overlay to match base’s H×W (bilinear on GPU)
    ov_resized = F.interpolate(
        ov, size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(0)                                                     # [3, H, W]

    # 4. Stack gray→3 channels: [3, H, W]
    gray3 = base_gray.unsqueeze(0).repeat(3, 1, 1)                    # [3, H, W]

    # 5. Blend on GPU: α·overlay + (1−α)·gray3
    blended = alpha * ov_resized + (1.0 - alpha) * gray3              # [3, H, W]

    # 6. Clamp to [0,255], cast → uint8, move to CPU, return as H×W×3
    blended = blended.clamp(0, 255).round().byte()                    # [3, H, W]
    return blended.permute(1, 2, 0).cpu().numpy()   

def guided_smoothing(rgb: np.ndarray, seg_mask: np.ndarray, radius=4, eps=1e-3) -> np.ndarray:
    rgb_float = rgb.astype(np.float32) / 255.0
    seg_float = seg_mask.astype(np.float32) / 255.0
    guided = cv2.ximgproc.guidedFilter(guide=rgb_float, src=seg_float, radius=radius, eps=eps)
    return (guided * 255).astype(np.uint8)

def superpixel_smoothing(image: np.ndarray, seg_mask: np.ndarray, n_segments=500) -> np.ndarray:
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)
    smoothed = np.zeros_like(seg_mask)
    for label in np.unique(segments):
        mask = segments == label
        values = seg_mask[mask]
        if values.size == 0:
            continue  # skip empty segment
        majority = mode(values, axis=None)[0]
        smoothed[mask] = majority.item()  # safely extract scalar
    return smoothed

def fast_superpixel_seeds(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    num_superpixels = 400  # Adjust as needed
    num_levels = 4
    prior = 2
    histogram_bins = 5
    double_step = False

    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        w, h, image.shape[2],
        num_superpixels, num_levels, prior,
        histogram_bins, double_step
    )

    seeds.iterate(image, num_iterations=2)  # Keep small
    labels = seeds.getLabels()
    out = np.zeros_like(mask)

    for label in np.unique(labels):
        region = mask[labels == label]
        if region.size > 0:
            out[labels == label] = np.bincount(region).argmax()
    return out


def scene_changed(prev: np.ndarray, curr: np.ndarray, threshold=0.02) -> bool:
    # resize if needed
    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))

    # normalize
    prev = prev.astype(np.float32) / 255.0
    curr = curr.astype(np.float32) / 255.0

    diff = np.mean(np.abs(prev - curr))  # Mean absolute difference
    return diff > threshold

def scene_changed_ssim(prev: np.ndarray, curr: np.ndarray, threshold=0.95) -> bool:
    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    score = ssim(prev_gray, curr_gray, data_range=1.0)

    return score < threshold

def compute_ssim(im1: np.ndarray, im2: np.ndarray) -> float:
    gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def warp_mask(prev_rgb, curr_rgb, prev_mask):
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    h, w = prev_mask.shape
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
    remap = flow_map + flow
    warped = cv2.remap(prev_mask, remap[..., 0], remap[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped.astype(np.uint8)


def render_rescale(self, srgb_mono):
    '''This function takes a single channel semantic similarity and rescales it globally'''
    # Maintain running min/max
    if not hasattr(self, "running_min"):
        self.running_min = -1.0
    if not hasattr(self, "running_max"):
        self.running_max = 1.0

    current_min = srgb_mono.min().item()
    current_max = srgb_mono.max().item()
    self.running_min = min(self.running_min, current_min)
    self.running_max = max(self.running_max, current_max)

    similarity_clip = (srgb_mono - self.running_min) / (self.running_max - self.running_min + 1e-10)

    return similarity_clip