# Prediction interface for Cog ⚙️
# https://cog.run/python

import os

os.system("pip install --no-build-isolation -e .")

import time
import subprocess
import numpy as np
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2

DEVICE = "cuda"
MODEL_CACHE = "checkpoints"
BASE_URL = f"https://weights.replicate.delivery/default/sam-2/{MODEL_CACHE}/"


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # NOTE we download all weights no matter what
        # TODO should be optimised and lazy loaded tbh
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = [
            # "sam2_hiera_base_plus.pt",
            "sam2_hiera_large.pt",
            # "sam2_hiera_small.pt",
            # "sam2_hiera_tiny.pt",
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # TODO: Add Lazy loading for the other versions
        # self.model_configs = {
        #     "tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        #     "small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        #     "base": ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
        #     "large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
        # }

        sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        self.sam2 = build_sam2(
            model_cfg, sam2_checkpoint, device="cuda", apply_postprocessing=False
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2)

        # Enable bfloat16 and TF32 for better performance
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def predict(
        self,
        image: Path = Input(description="Input image"),
        points_per_side: int = Input(
            description="Points per side for mask generation", default=32
        ),
        pred_iou_thresh: float = Input(
            description="Predicted IOU threshold", default=0.88
        ),
        stability_score_thresh: float = Input(
            description="Stability score threshold", default=0.95
        ),
        use_m2m: bool = Input(description="Use M2M", default=True),
    ) -> Path:
        """Run a single prediction on the model"""
        # Load and preprocess the image
        input_image = Image.open(image)
        input_image = np.array(input_image.convert("RGB"))

        # Configure the mask generator
        self.mask_generator.points_per_side = points_per_side
        self.mask_generator.pred_iou_thresh = pred_iou_thresh
        self.mask_generator.stability_score_thresh = stability_score_thresh
        self.mask_generator.use_m2m = use_m2m

        # Generate masks
        masks = self.mask_generator.generate(input_image)

        # Visualize results
        plt.figure(figsize=(20, 20))
        plt.imshow(input_image)
        self.show_anns(masks)
        plt.axis("off")

        # Save the result
        output_path = Path("output.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return output_path

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

        ax.imshow(img)
