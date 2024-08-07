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


class Output(BaseModel):
    black_white_masks: List[Path]
    highlighted_frames: List[Path]


class Predictor(BasePredictor):
    def setup(self) -> None:
        global build_sam2_video_predictor

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            print("sam2 not found. Installing...")
            os.system("pip install --no-build-isolation -e .")
            from sam2.build_sam import build_sam2_video_predictor

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = ["sam2_hiera_large.pt"]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        model_cfg = "sam2_hiera_l.yaml"
        sam2_checkpoint = f"{MODEL_CACHE}/sam2_hiera_large.pt"

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

        # Enable bfloat16 and TF32 for better performance
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def predict(
        self,
        video: Path = Input(description="Input video file"),
        clicks: str = Input(
            description="List of click coordinates in format '[x,y],[x,y],...'"
        ),
        vis_frame_stride: int = Input(
            default=15, description="Stride for visualizing frames"
        ),
    ) -> Output:
        print("🚀 Starting prediction process...")
        start_time = time.time()

        # Create a temporary directory for video frames
        video_dir = "video_frames"
        os.makedirs(video_dir, exist_ok=True)
        print(f"📁 Created temporary directory: {video_dir}")

        # Use ffmpeg to extract frames from the video
        print("🎬 Extracting frames from video...")
        ffmpeg_start = time.time()
        ffmpeg_command = f"ffmpeg -i {video} -q:v 2 -start_number 0 {video_dir}/%05d.jpg"
        subprocess.run(ffmpeg_command, shell=True, check=True)
        print(
            f"✅ Frame extraction completed in {time.time() - ffmpeg_start:.2f} seconds"
        )

        # Get frame names
        frame_names = [
            p
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        print(f"🖼️ Total frames extracted: {len(frame_names)}")

        # Initialize the inference state
        print("🧠 Initializing inference state...")
        inference_start = time.time()
        inference_state = self.predictor.init_state(video_path=video_dir)
        print(
            f"✅ Inference state initialized in {time.time() - inference_start:.2f} seconds"
        )

        # Parse clicks
        print("👆 Parsing clicks...")
        click_list = clicks.strip("[]").split("],[")
        prompts = {}
        for i, click in enumerate(click_list):
            x, y = map(int, click.split(","))
            obj_id = i + 1
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1], np.int32)  # Assume all clicks are positive
            prompts[obj_id] = points, labels
            print(f"🔹 Click {i+1}: x={x}, y={y}")

            out_obj_ids, out_mask_logits = self.refine_mask(
                inference_state, 0, obj_id, points, labels
            )
        print(f"✅ Parsed {len(click_list)} clicks")

        # Propagate masks through the video
        print("🔄 Propagating masks through the video...")
        propagation_start = time.time()
        video_segments = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            if out_frame_idx % 50 == 0:
                print(f"🔹 Processed frame {out_frame_idx}")
        print(
            f"✅ Mask propagation completed in {time.time() - propagation_start:.2f} seconds"
        )

        # Create output directory
        output_dir = Path("predict_outputs")
        output_dir.mkdir(exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")

        # Render and save the segmentation results
        print("🎨 Rendering and saving segmentation results...")
        render_start = time.time()
        black_white_masks = []
        highlighted_frames = []
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            # Create black and white mask
            if video_segments[out_frame_idx]:  # Check if the dictionary is not empty
                first_mask = next(iter(video_segments[out_frame_idx].values()))
                combined_mask = np.zeros_like(first_mask.squeeze(), dtype=np.uint8)
                for out_mask in video_segments[out_frame_idx].values():
                    combined_mask |= out_mask.squeeze().astype(np.uint8)

                # Ensure the mask is 2D before saving
                if combined_mask.ndim == 3:
                    combined_mask = combined_mask.squeeze()

                bw_mask_path = output_dir / f"bw_mask_{out_frame_idx:05d}.png"
                Image.fromarray(combined_mask * 255).save(bw_mask_path)
                black_white_masks.append(bw_mask_path)

            # Create highlighted frame
            fig = plt.figure(figsize=(12, 8))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

            self.show_anns([mask for mask in video_segments[out_frame_idx].values()])
            for points, labels in prompts.values():
                self.show_points(points, labels, plt.gca())

            highlighted_frame_path = output_dir / f"highlighted_frame_{out_frame_idx:05d}.png"
            plt.savefig(highlighted_frame_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            highlighted_frames.append(highlighted_frame_path)

            if out_frame_idx % 50 == 0:
                print(f"🖼️ Processed frame {out_frame_idx}")
        print(f"✅ Rendering completed in {time.time() - render_start:.2f} seconds")

        # Clean up temporary directory
        print("🧹 Cleaning up temporary directory...")
        cleanup_start = time.time()
        for file in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, file))
        os.rmdir(video_dir)
        print(f"✅ Cleanup completed in {time.time() - cleanup_start:.2f} seconds")

        total_time = time.time() - start_time
        print(f"🏁 Prediction process completed in {total_time:.2f} seconds")

        return Output(
            black_white_masks=black_white_masks,
            highlighted_frames=highlighted_frames
        )

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: np.sum(x)), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.zeros((*sorted_anns[0].shape[-2:], 4))
        img[:, :, 3] = 0

        for i, ann in enumerate(sorted_anns):
            m = ann.squeeze().astype(bool)
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )

    def refine_mask(self, inference_state, frame_idx, obj_id, points, labels):
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        return out_obj_ids, out_mask_logits