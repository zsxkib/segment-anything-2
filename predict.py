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
        labels: str = Input(
            description="List of labels corresponding to clicks, e.g., '1,1,0,1'"
        ),
        affected_frames: str = Input(
            description="List of frame indices for each click, e.g., '0,0,150,0'"
        ),
        ann_obj_ids: str = Input(
            description="List of object IDs corresponding to each click, e.g., '1,1,1,2'"
        ),
        vis_frame_stride: int = Input(
            default=15, description="Stride for visualizing frames"
        ),
    ) -> Output:

        video_dir = "video_frames"
        os.makedirs(video_dir, exist_ok=True)

        ffmpeg_command = (
            f"ffmpeg -i {video} -q:v 2 -start_number 0 {video_dir}/%05d.jpg"
        )
        subprocess.run(ffmpeg_command, shell=True, check=True)

        frame_names = [
            p
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = self.predictor.init_state(video_path=video_dir)

        click_list = [
            list(map(int, click.split(",")))
            for click in clicks.strip("[]").split("],[")
        ]
        label_list = list(map(int, labels.split(",")))
        frame_list = list(map(int, affected_frames.split(",")))
        obj_id_list = list(map(int, ann_obj_ids.split(",")))

        if not (
            len(click_list) == len(label_list) == len(frame_list) == len(obj_id_list)
        ):
            raise ValueError(
                "The number of clicks, labels, affected frames, and object IDs must be the same."
            )

        output_dir = Path("predict_outputs")
        output_dir.mkdir(exist_ok=True)

        prompts = {}
        for i, (click, label, frame, obj_id) in enumerate(
            zip(click_list, label_list, frame_list, obj_id_list)
        ):
            x, y = click
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([label], np.int32)

            if obj_id not in prompts:
                prompts[obj_id] = []
            prompts[obj_id].append((frame, points, labels))

        video_segments = {}
        for obj_id, obj_prompts in prompts.items():
            for frame, points, labels in obj_prompts:
                out_obj_ids, out_mask_logits = self.refine_mask(
                    inference_state, frame, obj_id, points, labels
                )

                if frame not in video_segments:
                    video_segments[frame] = {}
                video_segments[frame][obj_id] = out_mask_logits[0].cpu().numpy()

                # Visualize each step
                fig = plt.figure(figsize=(12, 8))
                plt.title(f"frame {frame}, object {obj_id}")
                plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame])))
                self.show_points(points, labels, plt.gca())
                self.show_anns([(out_mask_logits[0] > 0.0).cpu().numpy()], [obj_id])

                step_output = output_dir / f"step_obj{obj_id}_frame_{frame}.png"
                plt.savefig(step_output, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

        # Propagate masks
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(inference_state):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id] = (
                    out_mask_logits[i].cpu().numpy()
                )

        # Visualize results
        black_white_masks = []
        highlighted_frames = []
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            fig = plt.figure(figsize=(12, 8))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

            combined_mask = np.zeros_like(
                next(iter(video_segments[out_frame_idx].values())).squeeze(),
                dtype=np.float32,
            )
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                mask = out_mask.squeeze()
                combined_mask = np.maximum(combined_mask, mask)

            # Apply threshold to get final binary mask
            final_mask = (combined_mask > 0.0).astype(np.uint8)

            for obj_id, obj_prompts in prompts.items():
                for frame, points, labels in obj_prompts:
                    if frame == out_frame_idx:
                        self.show_points(points, labels, plt.gca())

            bw_mask_path = output_dir / f"bw_mask_{out_frame_idx:05d}.png"
            Image.fromarray(final_mask * 255).save(bw_mask_path)
            black_white_masks.append(bw_mask_path)

            # Show the mask on the image
            self.show_anns([final_mask], [1])  # Use a single color for all objects

            highlighted_frame_path = (
                output_dir / f"highlighted_frame_{out_frame_idx:05d}.png"
            )
            plt.savefig(highlighted_frame_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            highlighted_frames.append(highlighted_frame_path)

        cleanup_start = time.time()
        for file in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, file))
        os.rmdir(video_dir)

        return Output(
            black_white_masks=black_white_masks, highlighted_frames=highlighted_frames
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

    def show_anns(self, anns, obj_ids):
        if len(anns) == 0:
            return
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.zeros((*anns[0].shape[-2:], 4))
        img[:, :, 3] = 0

        cmap = plt.get_cmap("tab10")
        for ann, obj_id in zip(anns, obj_ids):
            m = ann.squeeze().astype(bool)
            color = np.array([*cmap(obj_id % 10)[:3], 0.6])
            img[m] = color
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
