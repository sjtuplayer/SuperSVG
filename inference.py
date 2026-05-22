import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
from skimage.segmentation import slic
from torchvision.utils import save_image

#as cpu fallback is brocken if diffvg is not compiled with gpu it put runtime error
#RuntimeError: diffvg not compiled with GPU
import torch
import pydiffvg
try:
    pydiffvg.set_use_gpu(torch.cuda.is_available())
except Exception:
    pydiffvg.set_use_gpu(False)

from models.attn_painter_superpixel import AttnPainterSVG
from models.refine_diff_module_multi_self_mlp import AttnPainterSVG as RefinePainterSVG
from models.refine_model_muti_self_dtw0 import CombinedModel as RefineModel
from util.utils import SignWithSigmoidGrad

MODEL_WIDTH = 128
RENDER_WIDTH = 512
PATCH_SIZE = 224
PATHS_PER_REGION = 128
PREDICT_BATCH_SIZE = 64
REFINE_WIDTH = 224
PATHS_PER_SEGMENT_FOR_SEGMENT_ESTIMATE = 64
HF_REPO_ID_DEFAULT = "JTUplayer/SuperSVG"
HF_WEIGHTS_PREFIX = "weights"

resize_224 = transforms.Resize([PATCH_SIZE, PATCH_SIZE])
resize_512 = transforms.Resize([RENDER_WIDTH, RENDER_WIDTH])
to_tensor_512 = transforms.Compose([transforms.ToTensor(), transforms.Resize([RENDER_WIDTH, RENDER_WIDTH])])


def get_args_parser():
    parser = argparse.ArgumentParser("Superpixel SVG rendering (coarse + global refine)")
    parser.add_argument("--output_dir", default="output", help="path where rendered images are saved")
    parser.add_argument("--device", default="cuda", help="device to use for rendering")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--input_path", type=str, default="test_images", help="path to image file or image folder")
    parser.add_argument("--path_num", type=int, default=1000, help="target SVG path count")
    parser.add_argument("--optimize_iter", type=int, default=0, help="fine-tuning iterations after rendering")
    parser.add_argument("--ckpt_dir", type=str, default="weights", help="checkpoint directory")
    parser.add_argument("--hf_repo_id", type=str, default=HF_REPO_ID_DEFAULT, help="Hugging Face repo id for auto-downloading weights")
    parser.add_argument("--refine_paths_per_segment", type=int, default=8, help="estimated refine paths per superpixel")
    parser.add_argument("--refine_batch_size", type=int, default=PREDICT_BATCH_SIZE, help="batch size for refine")
    return parser


def collect_input_files(input_path):
    path = Path(input_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in image_exts]
        if not files:
            raise FileNotFoundError(f"No image files found in folder: {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _download_weight_from_hf(filename: str, hf_repo_id: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for automatic checkpoint download. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    remote_path = f"{HF_WEIGHTS_PREFIX}/{filename}"
    downloaded = hf_hub_download(repo_id=hf_repo_id, filename=remote_path, repo_type="model")
    return Path(downloaded)


def get_ckpt_path(ckpt_dir, hf_repo_id):
    ckpt_dir_path = Path(ckpt_dir)
    if not ckpt_dir_path.is_absolute():
        ckpt_dir_path = (Path(__file__).resolve().parent / ckpt_dir_path).resolve()
    ckpt_path = ckpt_dir_path / "coarse.pt"
    if ckpt_path.exists():
        return ckpt_path
    print(f"[weights] Local checkpoint not found: {ckpt_path}. Downloading from {hf_repo_id} ...")
    return _download_weight_from_hf("coarse.pt", hf_repo_id)


def get_refine_ckpt_path(hf_repo_id):
    ckpt_path = Path(__file__).resolve().parent / "weights" / "refine.pt"
    if ckpt_path.exists():
        return ckpt_path
    print(f"[weights] Local refine checkpoint not found: {ckpt_path}. Downloading from {hf_repo_id} ...")
    return _download_weight_from_hf("refine.pt", hf_repo_id)


def build_refine_model(width: int, device: torch.device) -> RefineModel:
    refine_painter = RefinePainterSVG(
        stroke_num=8,
        path_num=4,
        width=width,
        control_num=False,
        num_loss=False,
        refine=True,
        self_attn_depth=4,
    )
    model = RefineModel(refine_painter, diff_model=None, path_num=4, width=width)
    model.to(device)
    model.eval()
    return model


def _extract_state_dict(raw_obj):
    if isinstance(raw_obj, dict) and "model_state_dict" in raw_obj:
        return raw_obj["model_state_dict"]
    return raw_obj


def _convert_refine_ckpt_keys(state_dict):
    return {key.replace("refine", "coarse"): value for key, value in state_dict.items() if "coarse" not in key}


def maybe_load_checkpoint(model: RefineModel, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Refine checkpoint not found: {ckpt_path}")
    raw = torch.load(str(ckpt_path), map_location=device)
    state_dict = _convert_refine_ckpt_keys(_extract_state_dict(raw))
    model.load_state_dict(state_dict, strict=False)


def get_segment_num_from_path_num(path_num):
    if path_num <= 0:
        raise ValueError(f"path_num must be > 0, got {path_num}")
    return max(1, int(path_num // PATHS_PER_SEGMENT_FOR_SEGMENT_ESTIMATE))


def ensure_stroke_dim_28(strokes):
    if strokes.size(-1) == 28:
        return strokes
    if strokes.size(-1) == 27:
        alpha = torch.ones(strokes.size(0), strokes.size(1), 1, device=strokes.device, dtype=strokes.dtype)
        return torch.cat([strokes, alpha], dim=-1)
    raise ValueError(f"Unsupported stroke dim {strokes.size(-1)}. Expected 27 or 28.")


def count_active_paths(strokes):
    if strokes.size(-1) >= 28:
        return int((strokes[0, :, -1] > 0.5).sum().item())
    return int(strokes.size(1))


@torch.no_grad()
def decode_by_id_map(image, model, device, num_of_segments=32):
    model.width = RENDER_WIDTH
    image_np = np.array(image.resize((RENDER_WIDTH, RENDER_WIDTH))) / 255.0
    segments = torch.from_numpy(slic(image_np, n_segments=num_of_segments, sigma=5, compactness=50))
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    _, h, w = image_tensor.size()

    crops = []
    masks = []
    coords = []
    kernel = np.ones((4, 4), np.uint8)
    for seg_id in range(1, int(segments.max()) + 1):
        seg_mask = (segments == seg_id).numpy().astype("uint8")
        seg_mask_dilate = cv2.dilate(seg_mask, kernel, iterations=1)

        seg_mask = torch.from_numpy(seg_mask)
        seg_mask_dilate = torch.from_numpy(seg_mask_dilate > 0.3).int()

        idxs = torch.nonzero(seg_mask_dilate)
        x1, x2 = idxs[:, 0].min(), idxs[:, 0].max()
        y1, y2 = idxs[:, 1].min(), idxs[:, 1].max()
        coords.append((x1 / h, (x2 + 1) / h, y1 / w, (y2 + 1) / w))

        crop = image_tensor * seg_mask_dilate.unsqueeze(0)
        crop = crop[:, x1 : x2 + 1, y1 : y2 + 1]
        mask = seg_mask[x1 : x2 + 1, y1 : y2 + 1].unsqueeze(0)
        crops.append(resize_224(crop))
        masks.append((resize_224(mask) > 0.5).float())

    crops = torch.stack(crops, dim=0).to(device).float()
    masks = torch.stack(masks, dim=0).to(device).float()

    if crops.size(0) <= PREDICT_BATCH_SIZE:
        strokes = model.predict_path(crops * masks - (1 - masks), num=PATHS_PER_REGION)
    else:
        batched_strokes = []
        for start in range(0, crops.size(0), PREDICT_BATCH_SIZE):
            end = min(start + PREDICT_BATCH_SIZE, crops.size(0))
            pred = model.predict_path(crops[start:end] * masks[start:end] - (1 - masks[start:end]), num=PATHS_PER_REGION)
            batched_strokes.append(pred)
        strokes = torch.cat(batched_strokes, dim=0)

    visible_strokes = []
    for crop_idx in range(len(crops)):
        for stroke in strokes[crop_idx]:
            if stroke[-1] == 1:
                x1, x2, y1, y2 = coords[crop_idx]
                global_stroke = stroke.clone()
                global_stroke[1:-4:2] = x1 + (x2 - x1) * global_stroke[1:-4:2]
                global_stroke[0:-4:2] = y1 + (y2 - y1) * global_stroke[0:-4:2]
                visible_strokes.append(global_stroke)
    if not visible_strokes:
        raise RuntimeError("No visible strokes predicted; try reducing path_num.")
    new_strokes = torch.stack(visible_strokes, dim=0).unsqueeze(0)
    output = model.rendering(new_strokes)[:, :3, :, :]
    return new_strokes, resize_512(output)


@torch.no_grad()
def global_slic_refine_once(image, coarse_strokes, coarse_model, refine_model, target_path_num, refine_paths_per_segment, refine_batch_size, device):
    coarse_strokes = ensure_stroke_dim_28(coarse_strokes)
    coarse_count = int(coarse_strokes.size(1))
    if coarse_count >= target_path_num:
        return coarse_strokes[:, :target_path_num]

    remaining_paths = target_path_num - coarse_count
    num_refine_segments = max(1, int(math.ceil(remaining_paths / max(1, refine_paths_per_segment))))

    image_np = np.array(image.resize((RENDER_WIDTH, RENDER_WIDTH))) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device).float()
    _, h, w = image_tensor.shape

    coarse_model.width = RENDER_WIDTH
    coarse_canvas = coarse_model.rendering(coarse_strokes)[:, :3, :, :]
    segment_map = slic(image_np, n_segments=num_refine_segments, sigma=5, compactness=20)
    max_seg = int(segment_map.max())
    if max_seg <= 0:
        return coarse_strokes

    refine_inputs = []
    refine_canvases = []
    region_infos = []
    for seg_id in range(1, max_seg + 1):
        seg_mask_np = (segment_map == seg_id).astype(np.float32)
        if int(seg_mask_np.sum()) == 0:
            continue
        idxs = np.argwhere(seg_mask_np > 0)
        if idxs.size == 0:
            continue
        x1 = int(idxs[:, 0].min())
        x2 = int(idxs[:, 0].max())
        y1 = int(idxs[:, 1].min())
        y2 = int(idxs[:, 1].max())
        if x2 <= x1 or y2 <= y1:
            continue

        local_img = image_tensor[:, x1 : x2 + 1, y1 : y2 + 1]
        local_canvas = coarse_canvas[0, :, x1 : x2 + 1, y1 : y2 + 1]
        local_mask = torch.from_numpy(seg_mask_np[x1 : x2 + 1, y1 : y2 + 1]).to(device).unsqueeze(0)

        local_img = resize_224(local_img)
        local_canvas = resize_224(local_canvas)
        local_mask = (resize_224(local_mask) > 0.5).float()

        refine_inputs.append(local_img * local_mask - (1 - local_mask))
        refine_canvases.append(local_canvas * local_mask - (1 - local_mask))
        region_infos.append(
            {
                "coords_norm": (x1 / h, (x2 + 1) / h, y1 / w, (y2 + 1) / w),
                "bbox": (x1, x2, y1, y2),
                "local_img": local_img,
                "local_mask": local_mask,
                "full_mask": torch.from_numpy(seg_mask_np).to(device),
            }
        )

    if not refine_inputs:
        return coarse_strokes

    refine_inputs = torch.stack(refine_inputs, dim=0).to(device).float()
    refine_canvases = torch.stack(refine_canvases, dim=0).to(device).float()
    predicted_strokes = []
    for start in range(0, refine_inputs.size(0), refine_batch_size):
        end = min(start + refine_batch_size, refine_inputs.size(0))
        pred = refine_model.coarse_model(x=refine_inputs[start:end], canvas=refine_canvases[start:end])
        predicted_strokes.append(ensure_stroke_dim_28(pred))
    predicted_strokes = torch.cat(predicted_strokes, dim=0)

    new_global_strokes = []
    for region_idx in range(predicted_strokes.size(0)):
        x1, x2, y1, y2 = region_infos[region_idx]["coords_norm"]
        for stroke in predicted_strokes[region_idx]:
            global_stroke = stroke.clone()
            global_stroke[1:-4:2] = x1 + (x2 - x1) * global_stroke[1:-4:2]
            global_stroke[0:-4:2] = y1 + (y2 - y1) * global_stroke[0:-4:2]
            new_global_strokes.append(global_stroke)
            if len(new_global_strokes) >= remaining_paths:
                break
        if len(new_global_strokes) >= remaining_paths:
            break

    if len(new_global_strokes) < remaining_paths and region_infos:
        if new_global_strokes:
            tmp_new_strokes = torch.stack(new_global_strokes, dim=0).unsqueeze(0).to(device)
            tmp_strokes = torch.cat([coarse_strokes, tmp_new_strokes], dim=1)
        else:
            tmp_strokes = coarse_strokes
        coarse_model.width = RENDER_WIDTH
        tmp_canvas = coarse_model.rendering(tmp_strokes)[:, :3, :, :]
        mse_map = ((tmp_canvas[0] - image_tensor) ** 2).mean(dim=0)

        # SLIC region count can be inaccurate. Route all remaining paths to the worst superpixel.
        region_mse_scores = []
        for info in region_infos:
            full_mask = info["full_mask"]
            mask_sum = full_mask.sum().clamp_min(1.0)
            region_mse = float((mse_map * full_mask).sum() / mask_sum)
            region_mse_scores.append(region_mse)
        region_rank = np.argsort(np.array(region_mse_scores))[::-1]
        paths_per_region_prediction = int(predicted_strokes.size(1))
        paths_per_region_prediction = max(1, paths_per_region_prediction)
        shortage = remaining_paths - len(new_global_strokes)
        extra_region_calls = int(math.ceil(shortage / paths_per_region_prediction))
        if extra_region_calls > 0:
            # Repeat top-MSE regions when shortage exceeds number of regions.
            selected_region_indices = [int(region_rank[i % len(region_rank)]) for i in range(extra_region_calls)]
            extra_refine_inputs = []
            extra_refine_canvases = []
            selected_infos = []
            for region_idx in selected_region_indices:
                info = region_infos[region_idx]
                cx1, cx2, cy1, cy2 = info["bbox"]
                local_canvas = tmp_canvas[0, :, cx1 : cx2 + 1, cy1 : cy2 + 1]
                local_canvas = resize_224(local_canvas)
                local_mask = info["local_mask"]
                refine_input = info["local_img"] * local_mask - (1 - local_mask)
                refine_canvas = local_canvas * local_mask - (1 - local_mask)
                extra_refine_inputs.append(refine_input)
                extra_refine_canvases.append(refine_canvas)
                selected_infos.append(info)

            extra_refine_inputs = torch.stack(extra_refine_inputs, dim=0).to(device).float()
            extra_refine_canvases = torch.stack(extra_refine_canvases, dim=0).to(device).float()
            extra_pred = refine_model.coarse_model(x=extra_refine_inputs, canvas=extra_refine_canvases)
            extra_pred = ensure_stroke_dim_28(extra_pred)

            for batch_idx in range(extra_pred.size(0)):
                nx1, nx2, ny1, ny2 = selected_infos[batch_idx]["coords_norm"]
                for stroke in extra_pred[batch_idx]:
                    global_stroke = stroke.clone()
                    global_stroke[1:-4:2] = nx1 + (nx2 - nx1) * global_stroke[1:-4:2]
                    global_stroke[0:-4:2] = ny1 + (ny2 - ny1) * global_stroke[0:-4:2]
                    new_global_strokes.append(global_stroke)
                    if len(new_global_strokes) >= remaining_paths:
                        break
                if len(new_global_strokes) >= remaining_paths:
                    break

    if not new_global_strokes:
        return coarse_strokes
    new_global_strokes = torch.stack(new_global_strokes, dim=0).unsqueeze(0).to(device)
    final_strokes = torch.cat([coarse_strokes, new_global_strokes], dim=1)
    return final_strokes[:, :target_path_num]


def fine_tune(image, strokes, model, device, iters=1000):
    target = to_tensor_512(image).to(device)
    model.width = RENDER_WIDTH
    strokes.requires_grad = True
    beta = torch.ones((1, int(strokes.size(1)), 1), device=device) * 0.01
    beta.requires_grad = True
    optimizer = torch.optim.AdamW([strokes, beta], lr=0.001, betas=(0.9, 0.95))
    for _ in range(iters + 1):
        new_beta = SignWithSigmoidGrad.apply(beta)
        if strokes.size(-1) == 27:
            pred_strokes = torch.cat([strokes, new_beta], dim=-1)
        elif strokes.size(-1) == 28:
            pred_strokes = torch.cat([strokes[:, :, :-1], new_beta], dim=-1)
        else:
            raise ValueError(f"Unsupported stroke dim {strokes.size(-1)} in fine_tune. Expected 27 or 28.")
        output = model.rendering(pred_strokes)[:, :3, :, :]
        loss_num = new_beta.sum()
        loss_pixel = ((output - target) ** 2).mean()
        loss = loss_pixel + loss_num * 1e-6
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return pred_strokes.detach(), output, loss_pixel


def main(args):
    print(f"job dir: {Path(__file__).resolve().parent}")
    print("{}".format(args).replace(", ", ",\n"))

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu or run on a CUDA machine.")
    device = requested_device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    ckpt_path = get_ckpt_path(args.ckpt_dir, args.hf_repo_id)
    refine_ckpt_path = get_refine_ckpt_path(args.hf_repo_id)
    files = collect_input_files(args.input_path)
    num_of_segments = get_segment_num_from_path_num(args.path_num)

    coarse_model = AttnPainterSVG(stroke_num=128, path_num=4, width=MODEL_WIDTH, control_num=False, num_loss=True)
    coarse_model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    coarse_model.to(device)
    coarse_model.eval()

    refine_model = build_refine_model(width=REFINE_WIDTH, device=device)
    maybe_load_checkpoint(refine_model, refine_ckpt_path, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Start rendering {len(files)} image(s)")
    print(f"path_num={args.path_num}, num_of_segments={num_of_segments}")
    start_time = time.time()
    average_mse = 0.0

    for idx, image_path in enumerate(files):
        image = Image.open(str(image_path)).convert("RGB")
        coarse_strokes, _ = decode_by_id_map(image, coarse_model, device, num_of_segments)
        final_strokes = global_slic_refine_once(
            image=image,
            coarse_strokes=coarse_strokes,
            coarse_model=coarse_model,
            refine_model=refine_model,
            target_path_num=args.path_num,
            refine_paths_per_segment=args.refine_paths_per_segment,
            refine_batch_size=args.refine_batch_size,
            device=device,
        )

        coarse_model.width = RENDER_WIDTH
        output = coarse_model.rendering(final_strokes)[:, :3, :, :]
        output = resize_512(output)
        if args.optimize_iter > 0:
            final_strokes, output, _ = fine_tune(image, final_strokes, coarse_model, device, args.optimize_iter)

        target = to_tensor_512(image).unsqueeze(0).to(device)
        mse_loss = ((output - target) ** 2).mean()
        average_mse += float(mse_loss)
        print(idx, mse_loss.item(), f"final_path_count={count_active_paths(final_strokes)}")

        output_png_path = output_dir / image_path.name
        output_svg_path = output_dir / f"{image_path.stem}.svg"
        save_image(output, str(output_png_path), normalize=False)
        coarse_model.width = RENDER_WIDTH
        coarse_model.rendering(final_strokes, save_svg_path=str(output_svg_path))

    print(average_mse / len(files))
    print(f"Rendering time {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
