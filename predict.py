# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import cv2
import time
import torch
import logging
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


def extract_video_frames(video_path: Path, output_dir: Path):
    """
    Extract frames from a video file.

    Args:
    video_path (Path): Path to the input video file.
    output_dir (Path): Directory to save the extracted frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-q:v",
        "2",
        "-start_number",
        "0",
        f"{output_dir}/%05d.jpg",
    ]
    subprocess.run(ffmpeg_command, check=True)


def get_video_info(video_path: Path):
    """
    Get video information including FPS and presence of audio.

    Args:
    video_path (Path): Path to the input video file.

    Returns:
    dict: Contains 'fps' (float) and 'has_audio' (bool).
    """
    ffprobe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=r_frame_rate,nb_read_packets",
        "-of",
        "json",
        str(video_path),
    ]
    video_result = subprocess.run(ffprobe_command, capture_output=True, text=True)
    video_info = eval(video_result.stdout)

    ffprobe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "json",
        str(video_path),
    ]
    audio_result = subprocess.run(ffprobe_command, capture_output=True, text=True)
    audio_info = eval(audio_result.stdout)

    fps = eval(video_info["streams"][0]["r_frame_rate"])
    has_audio = (
        int(audio_info["streams"][0]["nb_read_packets"]) > 0
        if audio_info["streams"]
        else False
    )

    return {"fps": fps, "has_audio": has_audio}


def create_video_from_frames(
    input_pattern: str, output_path: Path, original_video_path: Path
):
    """
    Create a video from a series of frames, preserving original video properties.

    Args:
    input_pattern (str): Pattern for input frames (e.g., 'frame_%05d.png').
    output_path (Path): Path to save the output video.
    original_video_path (Path): Path to the original input video for reference.
    """
    video_info = get_video_info(original_video_path)

    ffmpeg_command = [
        "ffmpeg",
        "-framerate",
        str(video_info["fps"]),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",  # Force overwrite
    ]

    if video_info["has_audio"]:
        ffmpeg_command.extend(
            [
                "-i",
                str(original_video_path),
                "-c:a",
                "aac",
                "-map",
                "0:v",
                "-map",
                "1:a",
            ]
        )

    ffmpeg_command.append(str(output_path))

    logging.info(f"Running ffmpeg command: {' '.join(ffmpeg_command)}")

    try:
        result = subprocess.run(
            ffmpeg_command, check=True, capture_output=True, text=True
        )
        logging.info(f"ffmpeg stdout: {result.stdout}")
        logging.info(f"ffmpeg stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg command failed with return code {e.returncode}")
        logging.error(f"ffmpeg stdout: {e.stdout}")
        logging.error(f"ffmpeg stderr: {e.stderr}")
        raise

    if not output_path.exists():
        logging.error(f"Output file {output_path} was not created")
        raise FileNotFoundError(f"Output file {output_path} was not created")

    logging.info(f"Video created successfully: {output_path}")


class Output(BaseModel):
    black_white_masks: List[Path]
    highlighted_frames: List[Path]
    black_white_video: Path
    highlighted_video: Path


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
            default=1, description="Stride for visualizing frames"
        ),
    ) -> Output:
        logging.basicConfig(level=logging.INFO)

        video_dir = Path("video_frames")
        extract_video_frames(video, video_dir)
        frame_names = sorted(video_dir.glob("*.jpg"))
        logging.info(f"Extracted {len(frame_names)} frames from the video")

        inference_state = self.predictor.init_state(video_path=str(video_dir))
        prompts = self._parse_inputs(clicks, labels, affected_frames, ann_obj_ids)

        output_dir = Path("predict_outputs")
        output_dir.mkdir(exist_ok=True)

        video_segments = self._process_prompts(
            inference_state, prompts, frame_names, output_dir, video_dir
        )
        video_segments = self._propagate_masks(inference_state, video_segments)

        black_white_masks, highlighted_frames = self._visualize_results(
            video_segments,
            frame_names,
            prompts,
            vis_frame_stride,
            output_dir,
            video_dir,
        )

        self._cleanup(video_dir)

        bw_video_path = output_dir / "black_white_mask_video.mp4"
        highlighted_video_path = output_dir / "highlighted_frame_video.mp4"

        logging.info("Creating black and white mask video")
        create_video_from_frames(
            str(output_dir / "bw_mask_%05d.png"), bw_video_path, video
        )

        logging.info("Creating highlighted frame video")
        create_video_from_frames(
            str(output_dir / "highlighted_frame_%05d.png"),
            highlighted_video_path,
            video,
        )

        return Output(
            black_white_masks=black_white_masks,
            highlighted_frames=highlighted_frames,
            black_white_video=bw_video_path,
            highlighted_video=highlighted_video_path,
        )

    def _get_sorted_frame_names(self, video_dir):
        return sorted([p for p in video_dir.glob("*.jpg")], key=lambda p: int(p.stem))

    def _parse_inputs(self, clicks, labels, affected_frames, ann_obj_ids):
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

        prompts = {}
        for click, label, frame, obj_id in zip(
            click_list, label_list, frame_list, obj_id_list
        ):
            if obj_id not in prompts:
                prompts[obj_id] = []
            prompts[obj_id].append(
                (
                    frame,
                    np.array([click], dtype=np.float32),
                    np.array([label], np.int32),
                )
            )

        return prompts

    def _process_prompts(
        self, inference_state, prompts, frame_names, output_dir, video_dir
    ):
        video_segments = {}
        for obj_id, obj_prompts in prompts.items():
            for frame, points, labels in obj_prompts:
                out_obj_ids, out_mask_logits = self.refine_mask(
                    inference_state, frame, obj_id, points, labels
                )

                if frame not in video_segments:
                    video_segments[frame] = {}
                video_segments[frame][obj_id] = out_mask_logits[0].cpu().numpy()

                self._visualize_step(
                    frame,
                    obj_id,
                    points,
                    labels,
                    out_mask_logits,
                    frame_names,
                    output_dir,
                    video_dir,
                )

        return video_segments

    def _propagate_masks(self, inference_state, video_segments):
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
        return video_segments

    def _visualize_results(
        self,
        video_segments,
        frame_names,
        prompts,
        vis_frame_stride,
        output_dir,
        video_dir,
    ):
        black_white_masks = []
        highlighted_frames = []
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            bw_mask, highlighted_frame = self._create_frame_visualization(
                out_frame_idx,
                video_segments,
                prompts,
                frame_names,
                output_dir,
                video_dir,
            )
            black_white_masks.append(bw_mask)
            highlighted_frames.append(highlighted_frame)
        return black_white_masks, highlighted_frames

    def _cleanup(self, video_dir):
        for file in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, file))
        os.rmdir(video_dir)

    def _visualize_step(
        self,
        frame,
        obj_id,
        points,
        labels,
        out_mask_logits,
        frame_names,
        output_dir,
        video_dir,
    ):
        fig = plt.figure(figsize=(12, 8))
        plt.title(f"frame {frame}, object {obj_id}")
        plt.imshow(Image.open(frame_names[frame]))
        self.show_points(points, labels, plt.gca())
        self.show_anns([(out_mask_logits[0] > 0.0).cpu().numpy()], [obj_id])

        step_output = output_dir / f"step_obj{obj_id}_frame_{frame}.png"
        plt.savefig(step_output, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def _create_frame_visualization(
        self, out_frame_idx, video_segments, prompts, frame_names, output_dir, video_dir
    ):
        fig = plt.figure(figsize=(12, 8))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(frame_names[out_frame_idx]))

        combined_mask = np.zeros_like(
            next(iter(video_segments[out_frame_idx].values())).squeeze(),
            dtype=np.float32,
        )
        for mask in video_segments[out_frame_idx].values():
            combined_mask = np.maximum(combined_mask, mask.squeeze())

        final_mask = (combined_mask > 0.0).astype(np.uint8)

        for obj_id, obj_prompts in prompts.items():
            for frame, points, labels in obj_prompts:
                if frame == out_frame_idx:
                    self.show_points(points, labels, plt.gca())

        bw_mask_path = output_dir / f"bw_mask_{out_frame_idx:05d}.png"
        Image.fromarray(final_mask * 255).save(bw_mask_path)

        self.show_anns([final_mask], [1])

        # Save the figure with even dimensions
        highlighted_frame_path = (
            output_dir / f"highlighted_frame_{out_frame_idx:05d}.png"
        )
        plt.savefig(highlighted_frame_path, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)

        # Ensure even dimensions by padding if necessary
        img = Image.open(highlighted_frame_path)
        width, height = img.size
        new_width = width + (width % 2)
        new_height = height + (height % 2)
        if new_width != width or new_height != height:
            new_img = Image.new(img.mode, (new_width, new_height), (255, 255, 255, 0))
            new_img.paste(img, ((new_width - width) // 2, (new_height - height) // 2))
            new_img.save(highlighted_frame_path)

        return bw_mask_path, highlighted_frame_path

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
