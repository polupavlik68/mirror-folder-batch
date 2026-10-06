"""One photo at a time from a folder tree, then the same relative path on the way out.

The graph has to stay in two nodes. The picture must leave the first node, pass
through the edit, and come back to the second. One node on both sides would be
a cycle, and ComfyUI will not run that.
"""

from __future__ import annotations

import copy
import math
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

_ticks = 0
_active_input = ""
_starts: list[float] = []
_last_eta = "жду второе фото"


def _fmt_duration(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    if whole < 60:
        return f"{whole} с"
    minutes, sec = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes} мин {sec} с" if sec else f"{minutes} мин"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes} мин"


def eta_message(index: int, total: int) -> str:
    """Average gap between photos this session, then scale it to what is left."""
    global _last_eta
    _starts.append(time.time())
    number = index + 1
    left = max(0, total - index)
    if len(_starts) < 2:
        _last_eta = f"{number}/{total}, среднее появится со второго фото"
        return _last_eta
    deltas = [_starts[i] - _starts[i - 1] for i in range(1, len(_starts))]
    average = sum(deltas) / len(deltas)
    _last_eta = (
        f"{number}/{total}, среднее {_fmt_duration(average)}, "
        f"осталось ~{_fmt_duration(average * left)}"
    )
    return _last_eta


def _clean(path: str) -> Path:
    return Path(str(path).strip().strip('"'))


def _natural_key(text: str):
    parts = []
    buf = ""
    digit = None
    for ch in text:
        is_digit = ch.isdigit()
        if digit is None:
            digit = is_digit
            buf = ch
            continue
        if is_digit == digit:
            buf += ch
            continue
        parts.append(int(buf) if digit else buf.lower())
        buf = ch
        digit = is_digit
    if buf:
        parts.append(int(buf) if digit else buf.lower())
    return parts


def _jobs(input_folder: str, output_folder: str, skip_existing: bool) -> list[tuple[Path, Path]]:
    source = _clean(input_folder)
    target = _clean(output_folder)
    if not str(source):
        raise ValueError("Пакет: входная папка пустая.")
    if not source.is_dir():
        raise FileNotFoundError(f"Пакет: это не папка: {source}")
    if not str(target):
        raise ValueError("Пакет: папка результата пустая.")
    try:
        same = source.resolve() == target.resolve()
        inside = target.resolve().is_relative_to(source.resolve())
    except OSError:
        same, inside = False, False
    if same or inside:
        raise ValueError("Пакет: папка результата должна быть снаружи входной папки.")

    found = [
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    found.sort(key=lambda path: _natural_key(str(path.relative_to(source))))

    jobs = []
    for path in found:
        relative = path.relative_to(source)
        dest = target / relative
        if skip_existing and dest.is_file() and dest.stat().st_size > 0:
            continue
        jobs.append((path, relative))
    return jobs


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        rgb = image.convert("RGB")
        array = np.array(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _target_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Long side capped at max_side, never enlarged, both sides snapped to 32.

    Same rule as the resize node this used to sit next to: larger_side,
    dont_enlarge, lanczos, divisible_by 32.
    """
    if max_side <= 0 or width < 1 or height < 1:
        return width, height

    ideal_w, ideal_h = float(width), float(height)
    ratio = ideal_w / ideal_h if ideal_h else float("inf")
    if ideal_w > ideal_h:
        ideal_w = float(max_side)
        ideal_h = ideal_w / ratio if ratio else 0
    else:
        ideal_h = float(max_side)
        ideal_w = ideal_h * ratio
    if ideal_w * ideal_h > width * height:
        ideal_w, ideal_h = float(width), float(height)

    final_w = max(32, math.floor(ideal_w / 32) * 32)
    final_h = max(32, math.floor(ideal_h / 32) * 32)
    return final_w, final_h


def _resize_long_side(pixels: torch.Tensor, max_side: int) -> torch.Tensor:
    _batch, height, width, _channels = pixels.shape
    final_w, final_h = _target_size(width, height, int(max_side))
    if final_w == width and final_h == height:
        return pixels

    src_ratio = float(width) / float(height) if height else float("inf")
    target_ratio = float(final_w) / float(final_h) if final_h else float("inf")
    if src_ratio > target_ratio:
        scale_h = final_h
        scale_w = max(1, round(scale_h * src_ratio))
    else:
        scale_w = final_w
        scale_h = max(1, round(scale_w / src_ratio) if src_ratio else final_h)

    frame = pixels[0].detach().cpu().numpy()
    picture = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    picture = picture.resize((scale_w, scale_h), _LANCZOS)
    crop_x = max(0, (picture.width - final_w) // 2)
    crop_y = max(0, (picture.height - final_h) // 2)
    picture = picture.crop((crop_x, crop_y, crop_x + final_w, crop_y + final_h))
    array = np.array(picture).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class MirrorFolderBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": "", "multiline": True}),
                "output_folder": ("STRING", {"default": "", "multiline": True}),
                "skip_existing": ("BOOLEAN", {"default": True}),
                "max_side": ("INT", {
                    "default": 2048,
                    "min": 0,
                    "max": 8192,
                    "step": 32,
                    "tooltip": "Длинная сторона. Фото меньше не увеличиваются. 0 — не менять размер.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "relative_path", "output_folder")
    FUNCTION = "run"
    CATEGORY = "image/batch"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Следующее необработанное фото из папки и всех подпапок. "
        "Картинка идёт в обработку, путь — в ноду сохранения."
    )

    @classmethod
    def IS_CHANGED(cls, input_folder, output_folder, skip_existing, max_side):
        global _ticks
        _ticks += 1
        return f"batch:{_ticks}:{time.time_ns()}"

    def run(self, input_folder, output_folder, skip_existing, max_side):
        global _active_input
        _active_input = input_folder
        everything = _jobs(input_folder, output_folder, skip_existing=False)
        if not everything:
            raise ValueError("Пакет: во входной папке нет картинок.")
        pending = _jobs(input_folder, output_folder, True) if skip_existing else everything
        if not pending:
            raise ValueError("Пакет: всё уже сохранено, новых файлов нет.")

        total = len(everything)
        path, relative = pending[0]
        index = total - len(pending)
        image = _resize_long_side(_load_rgb(path), int(max_side))
        message = eta_message(index, total)
        print(f"[Пакет] {index + 1}/{total} {relative.as_posix()} | {message}")
        return {
            "ui": {"text": [message]},
            "result": (image, relative.as_posix(), output_folder),
        }


class MirrorFolderSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "relative_path": ("STRING", {"default": ""}),
                "output_folder": ("STRING", {"default": "", "multiline": True}),
                "jpeg_quality": ("INT", {"default": 95, "min": 80, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "image/batch"
    OUTPUT_NODE = True
    DESCRIPTION = "Кладёт картинку в папку результата, сохраняя подпапки и имя файла."

    def save(self, image, relative_path, output_folder, jpeg_quality):
        relative = Path(str(relative_path).strip().strip('"'))
        if relative.is_absolute() or ".." in relative.parts or not str(relative):
            raise ValueError(f"Сохранение: плохой относительный путь: {relative_path}")

        dest = _clean(output_folder) / relative
        dest.parent.mkdir(parents=True, exist_ok=True)

        frame = image[0] if isinstance(image, torch.Tensor) and image.ndim == 4 else image
        array = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] > 3:
            array = array[..., :3]
        picture = Image.fromarray(array, mode="RGB").convert("RGB")

        suffix = dest.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            picture.save(dest, format="JPEG", quality=int(jpeg_quality), subsampling=0)
        elif suffix == ".webp":
            picture.save(dest, format="WEBP", quality=int(jpeg_quality))
        else:
            if suffix != ".png":
                dest = dest.with_suffix(".png")
            picture.save(dest, format="PNG")

        print(f"[Сохранение] {dest}")
        _queue_next(output_folder)
        return (str(dest),)


def _queue_next(output_folder: str) -> None:
    """Queue one more run when this save was the last job still in flight."""
    try:
        pending = _jobs(_active_input, output_folder, skip_existing=True)
    except Exception as exc:
        print(f"[Сохранение] следующее не ставлю: {exc}")
        return
    if not pending:
        print("[Сохранение] папка готова")
        return

    from server import PromptServer

    queue = PromptServer.instance.prompt_queue
    running, queued = queue.get_current_queue_volatile()
    if queued or len(running) != 1:
        return

    item = copy.deepcopy(running[0])
    prompt_id = str(uuid.uuid4())
    number = float(item[0]) + 1
    tail = item[5] if len(item) > 5 else {}
    queue.put((number, prompt_id, item[2], item[3], item[4], tail))
    print(f"[Сохранение] следующее в очереди, осталось {len(pending)}")


def _ask_directory() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askdirectory(parent=root)
    root.destroy()
    return chosen or ""


def _register_browse_route() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:
        print(f"[Пакет] обзор папки недоступен: {exc}")
        return

    @PromptServer.instance.routes.post("/mirror_folder_batch/browse")
    async def browse_folder(_request):
        import asyncio

        path = await asyncio.to_thread(_ask_directory)
        return web.json_response({"path": path})


_register_browse_route()

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "MirrorFolderBatch": MirrorFolderBatch,
    "MirrorFolderSave": MirrorFolderSave,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MirrorFolderBatch": "Пакетная обработка",
    "MirrorFolderSave": "Сохранить в папки",
}
