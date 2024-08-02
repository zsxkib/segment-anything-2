# Prediction interface for Cog ⚙️
# https://cog.run/python


import os
import cv2
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from typing import List
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path, BaseModel


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


class Output(BaseModel):
    combined_mask: Path
    individual_masks: List[Path]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        global build_sam2, SAM2AutomaticMaskGenerator
        os.system("pip install --no-build-isolation -e .")
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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
        self.model_configs = {
            "tiny": ("sam2_hiera_t.yaml", f"{MODEL_CACHE}/sam2_hiera_tiny.pt"),
            "small": ("sam2_hiera_s.yaml", f"{MODEL_CACHE}/sam2_hiera_small.pt"),
            "base": ("sam2_hiera_b+.yaml", f"{MODEL_CACHE}/sam2_hiera_base_plus.pt"),
            "large": ("sam2_hiera_l.yaml", f"{MODEL_CACHE}/sam2_hiera_large.pt"),
        }

        model_cfg, sam2_checkpoint = self.model_configs["large"]

        self.sam2 = build_sam2(
            model_cfg, sam2_checkpoint, device="cuda", apply_postprocessing=False
        )
        self.mask_generator = None
        self.last_params = None

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
        mask_limit: int = Input(
            default=-1,
            description="Maximum number of masks to return. If -1, all masks will be returned.",
        ),
        points_per_batch: int = Input(
            default=64,
            description="Number of points processed simultaneously by the model",
        ),
        crop_n_layers: int = Input(
            default=0,
            description="If >0, mask prediction will be run on crops of the image",
        ),
        box_nms_thresh: float = Input(
            default=0.7,
            description="Box IoU cutoff for non-maximal suppression to filter duplicate masks",
        ),
        crop_n_points_downscale_factor: int = Input(
            default=1,
            description="Scale factor for number of points sampled in crop layers",
        ),
        min_mask_region_area: int = Input(
            default=0, description="Minimum area for mask regions after postprocessing"
        ),
        multimask_output: bool = Input(
            default=False,
            description="Whether to output multiple masks at each grid point",
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        # Create a dictionary of the current parameters
        current_params = {
            "points_per_side": points_per_side,
            "points_per_batch": points_per_batch,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "use_m2m": use_m2m,
            "crop_n_layers": crop_n_layers,
            "box_nms_thresh": box_nms_thresh,
            "crop_n_points_downscale_factor": crop_n_points_downscale_factor,
            "min_mask_region_area": min_mask_region_area,
            "multimask_output": multimask_output,
        }

        # Check if we need to reinitialize the mask generator
        if self.mask_generator is None or self.last_params != current_params:
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2, **current_params
            )
            self.last_params = current_params

        # Load and preprocess the image
        input_image = Image.open(image)
        input_image = np.array(input_image.convert("RGB"))

        # Generate masks
        masks = self.mask_generator.generate(input_image)

        # Sort masks by area (largest first) and apply mask limit if specified
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        if mask_limit > 0:
            masks = masks[:mask_limit]

        # Create outputs folder if it doesn't exist
        outputs_folder = Path("outputs")
        outputs_folder.mkdir(exist_ok=True)

        # Generate and save combined colored mask
        combined_mask_path = outputs_folder / "combined_mask.png"
        self.save_combined_mask(input_image, masks, combined_mask_path)

        # Generate and save individual black and white masks
        individual_mask_paths = self.save_individual_masks(masks, outputs_folder)

        return Output(
            combined_mask=combined_mask_path, individual_masks=individual_mask_paths
        )

    def save_combined_mask(self, input_image, masks, output_path):
        plt.figure(figsize=(20, 20))
        plt.imshow(input_image)
        self.show_anns(masks)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def save_individual_masks(self, masks, output_folder):
        individual_mask_paths = []
        for i, mask in enumerate(masks):
            mask_image = mask["segmentation"].astype(np.uint8) * 255
            mask_path = output_folder / f"mask_{i}.png"
            Image.fromarray(mask_image).save(mask_path)
            individual_mask_paths.append(mask_path)
        return individual_mask_paths

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
