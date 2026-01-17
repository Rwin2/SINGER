import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

#############################################
# 1. Initialize CLIPSeg Model and Processor #
#############################################

# class CLIPSegONNXModel:
#     def __init__(
#         self,
#         onnx_path: str,
#         hf_model: str = "CIDAS/clipseg-rd64-refined",
#         device: Optional[str] = None,
#         providers: Optional[list] = None,
#         cmap: str = "turbo",
#     ):
#         # Choose torch device
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         # HF processor
#         self.processor = CLIPSegProcessor.from_pretrained(hf_model, use_fast=True)
#         # ORT session
#         so = ort.SessionOptions()
#         so.log_severity_level = 2
#         provs = providers or (["CUDAExecutionProvider"] if "cuda" in self.device else ["CPUExecutionProvider"])
#         self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=provs)
#         # Cache output name
#         self.output_name = self.session.get_outputs()[0].name
#         # LUT for colorization
#         self.lut = get_colormap_lut(cmap_name=cmap)

#     # def _preprocess(self, image: Image.Image, prompt: str) -> dict:
#     #     inputs = self.processor(images=image, text=prompt, return_tensors="pt")
#     #     for k, v in inputs.items():
#     #         inputs[k] = v.to(self.device)
#     #     return {
#     #         "pixel_values": inputs["pixel_values"].cpu().numpy(),
#     #         "input_ids":     inputs["input_ids"].cpu().numpy(),
#     #         "attention_mask":inputs["attention_mask"].cpu().numpy(),
#     #     }
#     def _preprocess_tensors(self, image: Image.Image, prompt: str) -> dict:
#         """
#         Returns torch.Tensor dict on self.device, casting ints to int32 for ONNX.
#         """
#         inputs = self.processor(images=image, text=prompt, return_tensors="pt")
#         # move to device
#         for k, v in inputs.items():
#             inputs[k] = v.to(self.device)
#         # cast input_ids and attention_mask to int32 for compatibility
#         inputs["input_ids"] = inputs["input_ids"].to(torch.int32)
#         inputs["attention_mask"] = inputs["attention_mask"].to(torch.int32)
#         return inputs

#     def _preprocess_numpy(self, image: Image.Image, prompt: str) -> dict:
#         """
#         Returns NumPy inputs via CPU, casting ints to int32 for ONNX.
#         """
#         inputs = self.processor(images=image, text=prompt, return_tensors="np")
#         return {
#             "pixel_values": inputs["pixel_values"].astype(np.float32),
#             "input_ids":     inputs["input_ids"].astype(np.int32),
#             "attention_mask":inputs["attention_mask"].astype(np.int32),
#         }

#     # def _postprocess(
#     #     self,
#     #     logits: np.ndarray,
#     #     target_size: Optional[Tuple[int, int]] = None
#     # ) -> np.ndarray:
#     #     # logits: (1, H, W) or (1,1,H,W)
#     #     pm = logits.squeeze(0).squeeze(0)  # (H, W)
#     #     mn, mx = pm.min(), pm.max()
#     #     if mx - mn < 1e-9:
#     #         scaled = np.zeros_like(pm, dtype=np.float32)
#     #     else:
#     #         scaled = (pm - mn) / (mx - mn)
#     #     mask_u8 = (scaled * 255).astype(np.uint8)
#     #     # resize if needed
#     #     if target_size and mask_u8.shape[::-1] != target_size:
#     #         mask_u8 = np.array(
#     #             Image.fromarray(mask_u8)
#     #                  .resize(target_size, resample=Image.BILINEAR)
#     #         )
#     #     return colorize_mask_fast(mask_u8, self.lut)
#     def _postprocess(
#         self,
#         logits: np.ndarray,
#         target_size: Optional[Tuple[int, int]] = None
#     ) -> np.ndarray:
#         # Handle various output shapes: [1,1,H,W], [1,H,W], or [H,W]
#         arr = logits
#         if arr.ndim == 4:
#             pm = arr[0, 0]
#         elif arr.ndim == 3:
#             pm = arr[0]
#         elif arr.ndim == 2:
#             pm = arr
#         else:
#             raise ValueError(f"Unexpected logits shape: {arr.shape}")
#         mn, mx = pm.min(), pm.max()
#         if mx - mn < 1e-9:
#             scaled = np.zeros_like(pm, dtype=np.float32)
#         else:
#             scaled = (pm - mn) / (mx - mn)
#         mask_u8 = (scaled * 255).astype(np.uint8)
#         if target_size and mask_u8.shape[::-1] != target_size:
#             mask_u8 = np.array(
#                 Image.fromarray(mask_u8)
#                      .resize(target_size, resample=Image.BILINEAR)
#             )
#         return colorize_mask_fast(mask_u8, self.lut)

#     # def clipseg_onnx_inference(
#     #     self,
#     #     image: Image.Image,
#     #     prompt: str,
#     #     resize_output_to_input: bool = True
#     # ) -> np.ndarray:
#     #     """
#     #     Default inference using GPU I/O binding.
#     #     Returns a colorized NumPy array (H, W, 3) uint8.
#     #     """
#     #     # 1) preprocess to torch tensors
#     #     inputs = self.processor(images=image, text=prompt, return_tensors="pt")
#     #     for k, v in inputs.items():
#     #         inputs[k] = v.to(self.device)

#     #     # 2) bind I/O
#     #     io_binding = self.session.io_binding()
#     #     def bind_input(name, tensor):
#     #         io_binding.bind_input(
#     #             name=name,
#     #             device_type=self.device,
#     #             device_id=0,
#     #             element_type=np.float32 if tensor.dtype==torch.float32 else np.int64,
#     #             shape=tuple(tensor.shape),
#     #             buffer_ptr=tensor.data_ptr(),
#     #         )
#     #     # bind inputs
#     #     bind_input("pixel_values", inputs["pixel_values"])
#     #     bind_input("input_ids", inputs["input_ids"])
#     #     bind_input("attention_mask", inputs["attention_mask"])

#     #     # 3) prepare output buffer
#     #     b, _, H, W = inputs["pixel_values"].shape
#     #     out_tensor = torch.empty((b, H, W), dtype=torch.float32, device=self.device)
#     #     io_binding.bind_output(
#     #         name=self.output_name,
#     #         device_type=self.device,
#     #         device_id=0,
#     #         element_type=ort.TensorProto.FLOAT,
#     #         shape=tuple(out_tensor.shape),
#     #         buffer_ptr=out_tensor.data_ptr(),
#     #     )

#     #     # 4) run
#     #     self.session.run_with_iobinding(io_binding)

#     #     # 5) postprocess
#     #     logits = out_tensor.cpu().numpy()
#     #     size   = image.size if resize_output_to_input else None
#     #     return self._postprocess(logits, target_size=size)
#     def clipseg_onnx_inference(
#         self,
#         image: Union[Image.Image, np.ndarray],
#         prompt: str,
#         resize_output_to_input: bool = True
#     ) -> np.ndarray:
#         """
#         Returns a colorized mask (H,W,3 uint8).
#         Prefers GPU I/O binding if CUDAExecutionProvider is active; otherwise falls back to CPU run().
#         """

#         if isinstance(image, np.ndarray):
#             # If it's BGRA/BGR from OpenCV, strip alpha & convert
#             if image.ndim == 3 and image.shape[2] == 4:
#                 image = image[..., :3]
#             # assume BGR if dtype is uint8 and colors look like OpenCV;
#             # if you know it’s already RGB you can skip the cvtColor
#             image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#         elif not isinstance(image, Image.Image):
#             raise TypeError(f"Unsupported input type {type(image)}; expected PIL.Image or ndarray")
        
#         if resize_output_to_input:
#         # now that image is a PIL, image.size == (W, H)
#             target_size = image.size
#         else:
#             target_size = None


#         providers = self.session.get_providers()
#         target_size = image.size if resize_output_to_input else None


#         # GPU I/O binding path
#         if "CUDAExecutionProvider" in providers:
#             tensors = self._preprocess_tensors(image, prompt)
#             io_binding = self.session.io_binding()
#             # bind inputs
#             for name in ["pixel_values", "input_ids", "attention_mask"]:
#                 t = tensors[name]
#                 io_binding.bind_input(
#                     name=name,
#                     device_type=self.device,
#                     device_id=0,
#                     element_type=np.float32 if t.dtype==torch.float32 else np.int64,
#                     shape=tuple(t.shape),
#                     buffer_ptr=t.data_ptr(),
#                 )
#             # bind output
#             b, _, H, W = tensors["pixel_values"].shape
#             out = torch.empty((b, H, W), dtype=torch.float32, device=self.device)
#             io_binding.bind_output(
#                 name=self.output_name,
#                 device_type=self.device,
#                 device_id=0,
#                 element_type=np.float32,
#                 shape=tuple(out.shape),
#                 buffer_ptr=out.data_ptr(),
#             )
#             self.session.run_with_iobinding(io_binding)
#             logits = out.cpu().numpy()

#         # CPU path
#         else:
#             np_inputs = self._preprocess_numpy(image, prompt)
#             outputs = self.session.run([self.output_name], np_inputs)
#             logits = outputs[0]

#         return self._postprocess(logits, target_size=target_size)
    
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
    
        # cur_min = float(arr.min())
        # if self.running_min is None:
        #     # first frame: seed the low‐end
        #     self.running_min = cur_min
        # else:
        #     # subsequent frames: only ever reduce the low‐end
        #     self.running_min = min(self.running_min, cur_min)
        # # note: running_max stays at 1.0
        # span = self.running_max - self.running_min + self.eps
        # return (arr - self.running_min) / span

    def clipseg_hf_inference(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        resize_output_to_input: bool = True
    ) -> np.ndarray:
        """
        Run CLIPSeg on a PIL image or numpy array, return colorized mask as (H,W,3) uint8.
        """
        # 1) ensure PIL
        if isinstance(image, np.ndarray):
            # handle BGR->RGB if needed
            if image.ndim == 3 and image.shape[2] == 3:
                img = Image.fromarray(image)
            else:
                img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type {type(image)}")
        
        skip_norm = isinstance(prompt, str) and prompt.strip().lower() == "null"

        # 2) preprocess
        inputs = self.processor(images=img, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k,v in inputs.items()}

        # 3) inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape [1,1,H,W]

        # 4) postprocess logits to (H,W)
        arr = logits.cpu().squeeze().numpy().astype(np.float32)
        if skip_norm:
            # apply sigmoid directly
            prob = 1.0 / (1.0 + np.exp(-arr))
            mask_u8 = (prob * 255).astype(np.uint8)
        else:
            scaled = self._rescale_global(arr)
            mask_u8 = (scaled * 255).astype(np.uint8)
            # mn, mx = arr.min(), arr.max()
            # if mx - mn < 1e-9:
            #     scaled = np.zeros_like(arr)
            # else:
            #     scaled = (arr - mn) / (mx - mn)
            # mask_u8 = (scaled * 255).astype(np.uint8)
            # # prob = 1.0 / (1.0 + np.exp(-arr))
            # # scaled = prob
            # scaled = self._rescale_global(arr)
            # mask_u8 = (scaled * 255).astype(np.uint8)

        # arr = logits.cpu().squeeze()  # remove batch and channel dims
        # # arr may now be [H,W]
        # pm = arr.numpy().astype(np.float32)
        # # normalize
        # mn, mx = pm.min(), pm.max()
        # if mx - mn < 1e-9:
        #     norm = np.zeros_like(pm)
        # else:
        #     norm = (pm - mn) / (mx - mn)
        # mask_u8 = (norm * 255).astype(np.uint8)

        # 5) resize if needed
        if resize_output_to_input:
            target_size = img.size  # (W,H)
            mask_u8 = np.array(
                Image.fromarray(mask_u8).resize(target_size, resample=Image.BILINEAR)
            )

        # 6) colorize
        colorized = colorize_mask_fast(mask_u8, self.lut)
        overlayed = blend_overlay_gpu(image, colorized)
        return overlayed

# class CLIPSegModel:
#     def __init__(
#         self,
#         hf_model: str = "CIDAS/clipseg-rd64-refined",
#         device: Optional[str] = None,
#         cmap: str = "turbo",
#     ):
#         # Choose torch device
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         # HF processor
#         self.processor = CLIPSegProcessor.from_pretrained(hf_model, use_fast=True)

#     def clipseg_inference(
#         image: Image.Image,
#         prompt: str,
#         processor: CLIPSegProcessor,
#         model: CLIPSegForImageSegmentation,
#         device: str,
#         resize_output_to_input: bool = True
#     ) -> Image.Image:
#         """
#         Given a PIL image and a text prompt, return a single-channel mask 
#         (0-255) from CLIPSeg representing semantic probability.
#         """
#         # 1. Preprocess inputs for CLIPSeg
#         inputs = processor(text=prompt, images=image, return_tensors="pt")
#         for k, v in inputs.items():
#             inputs[k] = v.to(device)
        
#         # 2. Forward pass
#         with torch.no_grad():
#             outputs = model(**inputs)
#             # with torch.cuda.amp.autocast():
#             #     outputs = model(**inputs)
#             # Use outputs.logits instead of outputs.pred_masks (depends on HF version)
#             prob_map = outputs.logits #.sigmoid()  # shape: [batch_size, 1, height, width]

#         # 3. Extract the single-channel prob map and convert to PIL
#         prob_map_2d = prob_map[0].cpu().numpy()

#         # (B) Rescale the clipped logits to [0..1]
#         current_min = prob_map_2d.min()
#         current_max = prob_map_2d.max()
#         range_val = current_max - current_min

#         if range_val < 1e-9:
#             # If the entire map is the same value, everything becomes 0
#             scaled_prob_map = np.zeros_like(prob_map_2d, dtype=np.float32)
#         else:
#             scaled_prob_map = (prob_map_2d - current_min) / range_val  # => [0..1]

#         mask_pil = Image.fromarray((scaled_prob_map * 255).astype(np.uint8))
        
#         # 4. Optionally resize back to original resolution
#         if resize_output_to_input:
#             mask_pil = mask_pil.resize(image.size, resample=Image.BILINEAR)
        
#         return mask_pil


# def init_clipseg_model(
#     model_name: str = "CIDAS/clipseg-rd64-refined",
#     device: str = None
# ):
#     """
#     Load the CLIPSeg model and processor from HuggingFace.
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     processor = CLIPSegProcessor.from_pretrained(model_name)
#     model = CLIPSegForImageSegmentation.from_pretrained(model_name)
#     model.to(device)
    
#     return processor, model, device

# def clipseg_preprocess(frame_bgra):
#     # Convert BGRA → BGR (strip alpha), then → RGB
#     bgr = frame_bgra[..., :3]
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# def clipseg_inference_onnx_gpu(
#     image: Image.Image,
#     prompt: str,
#     processor,
#     session: ort.InferenceSession,
#     resize_output_to_input: bool = True,
#     device="cuda"
# ) -> Image.Image:
#     import torch
#     device = torch.device(device)  # Convert device string to torch.device

#     # 1. Preprocess input on GPU
#     inputs = processor(images=image, text=prompt, return_tensors="pt")
#     pixel_values = inputs["pixel_values"].to(device)
#     input_ids = inputs["input_ids"].to(device)
#     attention_mask = inputs["attention_mask"].to(device)

#     # 2. Prepare IO binding
#     io_binding = session.io_binding()

#     def bind_tensor(name, tensor):
#         io_binding.bind_input(
#             name=name,
#             device_type=device.type,  # e.g., 'cuda'
#             device_id=device.index if device.index is not None else 0,
#             element_type=np.float32 if tensor.dtype == torch.float32 else np.int64,
#             shape=tuple(tensor.shape),
#             buffer_ptr=tensor.data_ptr(),
#         )

#     # Bind inputs
#     bind_tensor("pixel_values", pixel_values)
#     bind_tensor("input_ids", input_ids)
#     bind_tensor("attention_mask", attention_mask)

#     # 3. Prepare GPU output buffer
#     output_name = session.get_outputs()[0].name
#     output_shape = session.get_outputs()[0].shape  # e.g. [1, 1, None, None]
#     # Convert dynamic dimensions to fixed integers.
#     fixed_output_shape = []
#     for i, dim in enumerate(output_shape):
#         if isinstance(dim, int):
#             fixed_output_shape.append(dim)
#         else:
#             # For dynamic dimensions, assume they match the corresponding dimension of the input.
#             # For CLIPSeg, outputs are typically of shape [1, 1, H, W] where H, W come from the input.
#             if i < len(pixel_values.shape):
#                 fixed_output_shape.append(pixel_values.shape[i])
#             else:
#                 fixed_output_shape.append(224)  # fallback fixed size if needed
#     fixed_output_shape = tuple(fixed_output_shape)

#     output_array = torch.empty(fixed_output_shape, dtype=torch.float32, device=device)
#     io_binding.bind_output(
#         name=output_name,
#         device_type=device.type,
#         device_id=device.index if device.index is not None else 0,
#         element_type=np.float32,
#         shape=fixed_output_shape,
#         buffer_ptr=output_array.data_ptr(),
#     )

#     # 4. Run inference with I/O binding
#     session.run_with_iobinding(io_binding)

#     # 5. Copy output to CPU for visualization
#     prob_map = output_array.squeeze().cpu().numpy()  # shape: (H, W)

#     # 6. Normalize and convert to mask
#     min_val = np.min(prob_map)
#     max_val = np.max(prob_map)
#     if max_val - min_val < 1e-9:
#         scaled = np.zeros_like(prob_map, dtype=np.float32)
#     else:
#         scaled = (prob_map - min_val) / (max_val - min_val)
#     mask_pil = Image.fromarray((scaled * 255).astype(np.uint8))

#     if resize_output_to_input:
#         mask_pil = mask_pil.resize(image.size, resample=Image.BILINEAR)

#     return mask_pil

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