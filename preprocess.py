"""
This script extracts 2D and 3D keypoints from 2D detections using the 4DHuman model. 
Please refer to the https://github.com/shubham-goel/4D-Humans/tree/main for installation instructions.
Plus à jour. Utilise SAM 3D body à la place. 

Author: Tianjian Jiang
Date: March 16, 2025
Modified by Simon Khan
"""

from pathlib import Path
import numpy as np
import torch
from tqdm import trange
from sam_3d_body.sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf
import argparse



def run_eval(model, image_dir, boxes, cam_int=None):
    num_frames, num_persons, _ = boxes.shape
    skels_2d = np.full((num_frames, num_persons, 25, 2), np.nan, dtype=np.float32)
    skels_3d = np.full((num_frames, num_persons, 25, 3), np.nan, dtype=np.float32)

    image_files = sorted(image_dir.glob("*.jpg"))
    assert len(image_files) >= num_frames, (
        f"Pas assez d'images dans {image_dir}: {len(image_files)} trouvées pour {num_frames} frames"
    )

    with torch.inference_mode():
        for frame in trange(num_frames, desc=image_dir.stem):
            img = image_files[frame]
            frame_boxes = boxes[frame]

            valid_mask = np.isfinite(frame_boxes).all(axis=1)

            if not np.any(valid_mask):
                continue

            valid_boxes = frame_boxes[valid_mask]
            frame_cam_int = None if cam_int is None else cam_int[frame]

            pred_2d, pred_3d = model(img, valid_boxes, cam_int=frame_cam_int)

            skels_2d[frame, valid_mask] = pred_2d
            skels_3d[frame, valid_mask] = pred_3d

    return skels_2d, skels_3d


def load_sequences(root):
    with open(root / "sequences_full.txt", "r") as f:
        sequences = f.read().splitlines()
    sequences = [s.strip() for s in sequences if s.strip() and not s.startswith("#")]
    return sequences


class SAM3D:
    """Wrapper autour de SAM 3D Body pour sortir des keypoints body25."""

    def __init__(self, device):
        ckpt_dir = "/home/BeeGFS/Laboratories/IBHGC/skhan/models/sam-3d-body-dinov3"
        model, model_cfg = load_sam_3d_body_hf(ckpt_dir)
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
        )
        self.device = device

    def sam3d_to_body25(self, kpt):
        indices_70_to_body25 = [
            0, 69, 6, 8, 41, 5, 7, 62, -1, 10, 12, 14, 9, 11, 13,
            2, 1, 4, 3, 15, 16, 17, 18, 19, 20,
        ]
        kp25 = kpt[..., indices_70_to_body25, :]
        kp25[..., 8, :] = (kpt[..., 9, :] + kpt[..., 10, :]) / 2
        return kp25

    def __call__(self, img, boxes=None, cam_int=None):
        if isinstance(img, Path):
            img = str(img)

        if cam_int is not None:
            if isinstance(cam_int, np.ndarray):
                cam_int = torch.from_numpy(cam_int).float().to(self.estimator.device)
            cam_int = cam_int.reshape(1, 3, 3)

        batch = self.estimator.process_one_image(
            img,
            bboxes=boxes,
            cam_int=cam_int,
            inference_type="body",
        )

        assert len(batch) == len(boxes), "Number of boxes and batch should be the same"

        kpt_2d = np.zeros((len(boxes), 70, 2), dtype=np.float32)
        kpt_3d = np.zeros((len(boxes), 70, 3), dtype=np.float32)

        for person_id, pitem in enumerate(batch):
            kpt_2d[person_id] = pitem["pred_keypoints_2d"]
            kpt_3d[person_id] = pitem["pred_keypoints_3d"]

        kpt_2d = self.sam3d_to_body25(kpt_2d)
        kpt_3d = self.sam3d_to_body25(kpt_3d)
        return kpt_2d, kpt_3d


def main(root, sequence=None):
    model = SAM3D("cuda")

    (root / "new_skel_2d").mkdir(parents=True, exist_ok=True)
    (root / "new_skel_3d").mkdir(parents=True, exist_ok=True)

    sequences = load_sequences(root)
    if sequence is not None:
        sequences = [s for s in sequences if s == sequence]

    for seq in sequences:
        camera = np.load(root / "cameras" / f"{seq}.npz")
        skel_2d_path = root / "new_skel_2d" / f"{seq}.npy"
        skel_3d_path = root / "new_skel_3d" / f"{seq}.npy"

        if skel_2d_path.exists() and skel_3d_path.exists():
            print(f"[skip] {seq}")
            continue

        cam_int = camera["K"]
        boxes = np.load(root / "boxes" / f"{seq}.npy")
        image_dir = root / "images" / seq

        print(f"[run] {seq}")
        skel_2d, skel_3d = run_eval(model, image_dir, boxes, cam_int)

        np.save(skel_2d_path, skel_2d)
        np.save(skel_3d_path, skel_3d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--sequence", type=str, default=None)
    args = parser.parse_args()

    main(Path(args.root), sequence=args.sequence)