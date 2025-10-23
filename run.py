# Optional config for better memory efficiency
import os
import sys
import cv2
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

sys.path.append(os.path.join(os.getcwd(), "sam2"))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# diffusers
from diffusers import StableDiffusionInpaintPipeline

import argparse 

# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

map_anything = MapAnything.from_pretrained("facebook/map-anything").to(device)
mogev2 = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)                             
sam2 = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2/checkpoints/sam2.1_hiera_large.pt"))
sd = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)

import pdb; pdb.set_trace()
