#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inat_downloader.py - parallel, resumable iNaturalist photo downloader for a
deep-learning dataset.

Install:
    pip install requests pillow numpy
    pip install ultralytics        # optional: YOLO "is there a bird in the photo" check

Typical use:
    python inat_downloader.py --species-file birds_scientific.txt --out inat_birds --check-species
    python inat_downloader.py --species-file birds_scientific.txt --out inat_birds

Photo selection: only research-grade, wild, open-licence observations. Photos whose annotation says
"Evidence of presence = Organism" (i.e. the live bird, not a feather / track / scat) are requested first; the
remaining research-grade photos are used only when those run out. If PyTorch is installed, a bird detector
(ultralytics YOLO, or torchvision) additionally checks that a bird is really visible.

Output layout (everything lives under --out):
    dataset.csv              <- the ONLY file your DL project needs (one row per accepted image)
    rejected.csv             <- every rejected observation/photo + reason (never deleted)
    species_summary.csv      <- per-species counts / label ids (rewritten at each checkpoint)
    images/<Genus_species>/<photo_id>.jpg
    metadata/raw/<Genus_species>.jsonl   <- full raw iNaturalist JSON of each accepted observation
    state/state.json         <- cursors, label ids, resolved taxa (used to resume)
    logs/downloader.log

Live controls (type the letter, then press Enter):
    p = pause    r = resume    s = stop & save (continue later by re-running the same command)
    i = status   h = help      Ctrl+C = same as s (press twice to force-quit)
Same controls without a keyboard (nohup / IDE / ssh):
    echo pause  > <out>/control.txt      (pause | resume | stop | status)

Nothing is ever deleted or overwritten in dataset.csv / rejected.csv / images: they are
append-only, and a re-run rebuilds all de-duplication state from them.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import logging
import os
import queue
import re
import shutil
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from PIL import Image, ImageOps

try:  # Pillow >= 9.1
    RESAMPLE = Image.Resampling
except AttributeError:  # pragma: no cover
    RESAMPLE = Image  # type: ignore

SCHEMA_VERSION = 1
API_BASE_DEFAULT = "https://api.inaturalist.org/v1"
OPEN_LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nd,cc-by-nc-sa,cc-by-nc-nd"

DATASET_COLUMNS = [
    # --- what to feed the DL pipeline -------------------------------------------------
    "image_path", "image_abspath", "label_id", "species_scientific", "taxon_id",
    "common_name", "iconic_taxon", "selection_tier",
    # --- observation / taxon ----------------------------------------------------------
    "observation_id", "observation_uuid", "observation_url",
    "obs_taxon_id", "obs_taxon_name", "obs_taxon_rank", "species_guess",
    # --- photo ------------------------------------------------------------------------
    "photo_id", "photo_position", "photo_license", "photo_attribution", "photo_url", "size_variant",
    # --- user -------------------------------------------------------------------------
    "user_id", "user_login",
    # --- time -------------------------------------------------------------------------
    "observed_on", "time_observed_at", "observed_on_string", "observed_time_zone",
    "year", "month", "day", "hour", "week", "created_at", "updated_at",
    # --- location ---------------------------------------------------------------------
    "latitude", "longitude", "positional_accuracy", "public_positional_accuracy",
    "geoprivacy", "taxon_geoprivacy", "obscured", "place_guess", "place_ids",
    # --- community / quality of the observation ---------------------------------------
    "quality_grade", "captive", "identifications_count", "num_id_agreements",
    "num_id_disagreements", "faves_count", "comments_count",
    # --- annotations / free text ------------------------------------------------------
    "annotations", "life_stage", "sex", "alive_or_dead", "evidence_of_presence",
    "description", "tags", "observation_photos_count",
    # --- image file facts + quality metrics ------------------------------------------
    "width", "height", "file_size_bytes", "image_format", "exif_orientation", "sha256", "dhash",
    "sharpness", "brightness", "contrast", "aspect_ratio",
    # --- bird detector (empty if detector disabled) -----------------------------------
    "detector", "bird_detected", "bird_conf", "bird_num", "bird_x1", "bird_y1", "bird_x2",
    "bird_y2", "bird_area_frac", "bird_sharpness",
    "downloaded_at_utc",
]
NEW_COLUMNS = ("selection_tier", "bird_sharpness")  # columns added by newer script versions
REJECT_COLUMNS = ["timestamp_utc", "species_scientific", "observation_id", "photo_id",
                  "stage", "reason", "detail"]

# words in the description / tags that mean "this is not a photo of the live bird"
EVIDENCE_RE = re.compile(
    r"\b(feathers?|scats?|droppings?|poo|poop|feces|faeces|tracks?|footprints?|pellets?|"
    r"skulls?|skeletons?|bones?|carcass(?:es)?|roadkill|eggs?|eggshells?|empty nests?)\b",
    re.IGNORECASE)

POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)

log = logging.getLogger("inat")


# ------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------
def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_say_lock = threading.RLock()


def say(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _say_lock:
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            try:
                print(line.encode("ascii", "replace").decode("ascii"), flush=True)
            except Exception:
                pass
        except (OSError, ValueError):
            pass  # a broken console (e.g. Windows "invalid handle") must never kill a worker or the main loop
    log.info(msg)


class DiskFull(Exception):
    """The drive has (almost) no free space left."""


class TransientIO(Exception):
    """A file could not be written right now (locked by another program, ...). Retried on the next run."""


def is_disk_full(e: BaseException) -> bool:
    return getattr(e, "errno", None) == 28 or getattr(e, "winerror", None) in (39, 112)


def free_bytes(path: Path) -> Optional[int]:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return None


def safe_replace(src: Any, dst: Any, tries: int = 8) -> None:
    """os.replace with retries: on Windows another program (antivirus, Excel, an indexer, cloud sync)
    may hold the file for a moment, which raises PermissionError / WinError 5 / WinError 32."""
    delay = 0.2
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except FileNotFoundError:
            raise
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.7, 3.0)


def install_crash_logging(path: Path) -> None:
    """Every uncaught error (main thread or any worker thread) is written, in full, to logs/errors.log."""
    def write(kind: str, etype, evalue, tb) -> None:
        text = "".join(traceback.format_exception(etype, evalue, tb))
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"--- {utcnow()} {kind}\n{text}\n")
        except OSError:
            pass

    def main_hook(etype, evalue, tb) -> None:
        write("main thread", etype, evalue, tb)
        traceback.print_exception(etype, evalue, tb)

    def thread_hook(args) -> None:
        name = args.thread.name if args.thread else "?"
        write(f"thread {name}", args.exc_type, args.exc_value, args.exc_traceback)
        print(f"Exception in thread {name} (details in {path}):", file=sys.stderr)
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = main_hook
    threading.excepthook = thread_hook


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def clean_text(s: Any, limit: int = 1000) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()[:limit]


VARIANT_CAP = {"medium": 500, "large": 1024, "original": None}


def precheck_dims(photo: dict, variant: str, a: argparse.Namespace) -> Optional[Tuple[str, str]]:
    """Judge resolution from iNaturalist's metadata BEFORE downloading anything."""
    od = photo.get("original_dimensions") or {}
    w, h = od.get("width"), od.get("height")
    if not all(isinstance(v, (int, float)) and v > 0 for v in (w, h)):
        return None  # unknown -> checked after download
    cap = VARIANT_CAP.get(variant)
    if cap and max(w, h) > cap:  # the downloaded file is a scaled-down copy
        k = cap / max(w, h)
        w, h = w * k, h * k
    if min(w, h) < a.min_side:
        return "low_resolution_pre", f"would be {int(w)}x{int(h)}"
    if max(w, h) / min(w, h) > a.max_aspect:
        return "extreme_aspect_ratio_pre", f"{max(w, h) / min(w, h):.2f}"
    return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Stopped(Exception):
    """Raised inside a task when a stop was requested."""


class TooSlow(Exception):
    """A download kept exceeding the per-file time limit."""


class PermanentHTTPError(Exception):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url


# ------------------------------------------------------------------------------------
# HTTP, rate limiting, bandwidth budget
# ------------------------------------------------------------------------------------
class Http:
    def __init__(self, user_agent: str, stop: threading.Event):
        self.ua = user_agent
        self.stop = stop
        self.local = threading.local()
        self.net_errors = 0
        self.net_down = False
        self.last_net_error = 0.0
        self._last_msg = 0.0

    def session(self) -> requests.Session:
        s = getattr(self.local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = self.ua
            ad = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
            s.mount("http://", ad)
            s.mount("https://", ad)
            self.local.s = s
        return s

    def _request(self, url: str, params: Optional[dict], timeout: Tuple[int, int],
                 deadline: Optional[float]) -> requests.Response:
        """One request executed in a helper thread, so a stop request or the per-file time limit
        interrupts it even when the server only trickles a few bytes per second."""
        sess = self.session()
        cancel = threading.Event()
        box: Dict[str, Any] = {}

        def run() -> None:
            try:
                r = sess.get(url, params=params, timeout=timeout, stream=True)
                if r.status_code == 200:
                    chunks = []
                    for chunk in r.iter_content(16384):
                        if cancel.is_set():
                            r.close()
                            return
                        chunks.append(chunk)
                    r._content = b"".join(chunks)  # make r.content / r.json() work
                    r._content_consumed = True
                else:
                    r.close()
                box["r"] = r
            except BaseException as e:  # re-raised in the calling thread
                box["e"] = e

        th = threading.Thread(target=run, daemon=True, name="http-request")
        th.start()
        t0 = time.time()
        while th.is_alive():
            th.join(0.25)
            if self.stop.is_set():
                cancel.set()
                raise Stopped()
            if deadline and time.time() - t0 > deadline:
                cancel.set()
                raise TimeoutError("download exceeded the per-file time limit")
        if "e" in box:
            raise box["e"]
        return box["r"]

    def get(self, url: str, params: Optional[dict] = None, max_server_retries: int = 6,
            timeout: Tuple[int, int] = (10, 60), body_deadline: Optional[float] = None,
            max_slow_retries: int = 3) -> requests.Response:
        """GET with retries. Network problems are retried until they clear (or a stop is
        requested); 429/5xx are retried a limited number of times; other 4xx raise."""
        net_attempt = 0
        srv_attempt = 0
        slow_attempt = 0
        while True:
            if self.stop.is_set():
                raise Stopped()
            try:
                r = self._request(url, params, timeout, body_deadline)
            except TimeoutError:
                slow_attempt += 1
                log.warning("download too slow (>%ss): %s (attempt %s)", body_deadline, url, slow_attempt)
                if slow_attempt >= max_slow_retries:
                    raise TooSlow(url)
                continue
            except (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                self.net_errors += 1
                self.net_down = True
                self.last_net_error = time.time()
                wait = min(20, 2 ** min(net_attempt, 5))
                net_attempt += 1
                log.warning("network error (%s) on %s - retry in %ss", type(e).__name__, url, wait)
                if time.time() - self._last_msg > 20:
                    self._last_msg = time.time()
                    say(f"[NET] no connection ({type(e).__name__}) - waiting and retrying automatically; "
                        "everything resumes when the internet is back (type s + Enter to stop and save instead)")
                if self.stop.wait(wait):
                    raise Stopped()
                continue
            sc = r.status_code
            if self.net_down:
                self.net_down = False
                say("[NET] connection restored - continuing")
            if sc == 200:
                return r
            if sc == 429 or sc >= 500:
                srv_attempt += 1
                if srv_attempt > max_server_retries:
                    raise PermanentHTTPError(sc, url)
                ra = r.headers.get("Retry-After", "")
                wait = float(ra) if ra.isdigit() else min(120, 5 * 2 ** srv_attempt)
                log.warning("HTTP %s on %s - retry in %ss", sc, url, wait)
                if self.stop.wait(wait):
                    raise Stopped()
                continue
            raise PermanentHTTPError(sc, url)


class RateLimiter:
    def __init__(self, rps: float, stop: threading.Event):
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self.lock = threading.Lock()
        self.next = 0.0
        self.stop = stop

    def wait(self) -> None:
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.interval
        delay = t - now
        if delay > 0 and self.stop.wait(delay):
            raise Stopped()


class ByteBudget:
    """Keeps media downloads under N GB per rolling hour (iNaturalist asks for < 5 GB/h)."""

    def __init__(self, gb_per_hour: float):
        self.limit = gb_per_hour * 1e9
        self.lock = threading.Lock()
        self.events: collections.deque = collections.deque()
        self.sum = 0
        self.total = 0
        self.throttled = False

    def _prune(self, now: float) -> None:
        while self.events and self.events[0][0] < now - 3600:
            _, n = self.events.popleft()
            self.sum -= n

    def wait(self, stop: threading.Event) -> None:
        if not self.limit:
            return
        while not stop.is_set():
            with self.lock:
                now = time.time()
                self._prune(now)
                if self.sum < self.limit:
                    self.throttled = False
                    return
                self.throttled = True
                wait_s = self.events[0][0] + 3600 - now if self.events else 5
            if stop.wait(min(max(wait_s, 1.0), 30.0)):
                break
        raise Stopped()

    def add(self, n: int) -> None:
        with self.lock:
            self.events.append((time.time(), n))
            self.sum += n
            self.total += n


# ------------------------------------------------------------------------------------
# image analysis (quality tests + hashes)
# ------------------------------------------------------------------------------------
def analyze_image(data: bytes, a: argparse.Namespace):
    """Returns (info, rgb_image, None) or (None, None, (reason, detail))."""
    try:
        im = Image.open(io.BytesIO(data))
        fmt = im.format
        exif_orient = 1
        try:
            exif_orient = int(im.getexif().get(274, 1) or 1)
        except Exception:
            pass
        im.load()
    except Exception as e:  # corrupt / truncated / not an image
        return None, None, ("corrupt_image", f"{type(e).__name__}: {e}"[:150])
    if fmt not in ("JPEG", "PNG", "WEBP"):
        return None, None, ("unsupported_format", str(fmt))
    try:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    except Exception as e:
        return None, None, ("corrupt_image", f"{type(e).__name__}: {e}"[:150])

    w, h = im.size
    if min(w, h) < a.min_side:
        return None, None, ("low_resolution", f"{w}x{h}")
    ar = max(w, h) / max(1, min(w, h))
    if ar > a.max_aspect:
        return None, None, ("extreme_aspect_ratio", f"{ar:.2f}")

    g = im.convert("L")
    g.thumbnail((512, 512), RESAMPLE.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)
    brightness = float(arr.mean())
    contrast = float(arr.std())
    lap = (-4.0 * arr[1:-1, 1:-1] + arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:])
    hh, ww = lap.shape
    th, tw = max(1, hh // 4), max(1, ww // 4)
    tile_vars = [float(lap[i * th:(i + 1) * th, j * tw:(j + 1) * tw].var())
                 for i in range(4) for j in range(4)]
    # a photo with a sharp bird and a creamy blurred background must pass,
    # so use the sharpest tile instead of the global variance
    sharpness = max(tile_vars)

    if brightness < a.min_brightness or brightness > a.max_brightness:
        return None, None, ("bad_exposure", f"brightness={brightness:.0f}")
    if contrast < a.min_contrast:
        return None, None, ("low_contrast", f"contrast={contrast:.1f}")
    if sharpness < a.min_sharpness:
        return None, None, ("blurry", f"sharpness={sharpness:.1f}")

    small = np.asarray(g.resize((17, 16), RESAMPLE.BILINEAR), dtype=np.int16)
    bits = small[:, 1:] > small[:, :-1]
    packed = np.packbits(bits.flatten()).tobytes()  # 32 bytes = 256 bit dHash
    info = {
        "width": w, "height": h, "format": fmt, "exif_orientation": exif_orient,
        "brightness": round(brightness, 2), "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2), "aspect_ratio": round(ar, 3),
        "dhash_hex": packed.hex(), "dhash_arr": np.frombuffer(packed, dtype=np.uint64).copy(),
        "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
    }
    return info, im, None


def laplacian_var(gray: "np.ndarray") -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (-4.0 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def subject_sharpness(im: Image.Image, bbox: List[float]) -> Optional[float]:
    """Sharpness measured only inside the bird's box (the global test can be fooled by a sharp
    background branch while the bird itself is out of focus). None if the box is too small to judge."""
    w, h = im.size
    x1, y1, x2, y2 = bbox
    box = (max(0, int(x1 * w)), max(0, int(y1 * h)), min(w, int(x2 * w) + 1), min(h, int(y2 * h) + 1))
    crop = im.crop(box)
    if min(crop.size) < 48:
        return None
    g = crop.convert("L")
    g.thumbnail((256, 256), RESAMPLE.BILINEAR)  # only ever shrinks, so detail is never invented
    return laplacian_var(np.asarray(g, dtype=np.float32))


class YoloBirdDetector:
    """COCO 'bird' class detector (ultralytics). Used to make sure a bird is really in the photo."""

    BIRD_CLASS = 14

    def __init__(self, model_name: str, device: Optional[str], conf: float):
        from ultralytics import YOLO  # imported lazily: optional dependency
        self.model = YOLO(model_name)
        self.name = Path(model_name).name
        self.device = device
        self.conf = conf
        self.lock = threading.Lock()

    def detect(self, im: Image.Image) -> dict:
        with self.lock:
            res = self.model.predict(im, classes=[self.BIRD_CLASS], conf=self.conf,
                                     verbose=False, device=self.device)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return {"n": 0, "conf": 0.0, "bbox": None, "area": 0.0}
        xyxyn = boxes.xyxyn.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        areas = (xyxyn[:, 2] - xyxyn[:, 0]) * (xyxyn[:, 3] - xyxyn[:, 1])
        k = int(areas.argmax())  # the main subject = the biggest bird
        return {"n": int(len(confs)), "conf": float(confs[k]),
                "bbox": [float(v) for v in xyxyn[k]], "area": float(areas[k])}


class TorchvisionBirdDetector:
    """COCO 'bird' detector built on torchvision (needs only PyTorch + torchvision)."""

    def __init__(self, model_name: Optional[str], device: str, conf: float):
        import torch
        import torchvision
        from torchvision.models import detection as tvd
        self.torch = torch
        self.F = torchvision.transforms.functional
        name = model_name or ("resnet50_fpn_v2" if device.startswith("cuda") else "mobilenet_v3")
        if name in ("resnet50_fpn_v2", "fasterrcnn_resnet50_fpn_v2"):
            weights = tvd.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = tvd.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=conf)
        elif name in ("mobilenet_v3", "fasterrcnn_mobilenet_v3_large_fpn"):
            weights = tvd.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            model = tvd.fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=conf)
        else:
            raise ValueError(f"unknown torchvision model {name!r} (use resnet50_fpn_v2 or mobilenet_v3)")
        self.bird = weights.meta["categories"].index("bird")
        self.model = model.eval().to(device)
        self.device = device
        self.name = f"torchvision-{name}"
        self.lock = threading.Lock()

    def detect(self, im: Image.Image) -> dict:
        w, h = im.size
        with self.lock, self.torch.inference_mode():
            t = self.F.to_tensor(im).to(self.device)
            out = self.model([t])[0]
        boxes = out["boxes"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        keep = labels == self.bird
        if not keep.any():
            return {"n": 0, "conf": 0.0, "bbox": None, "area": 0.0}
        norm = np.clip(boxes[keep] / np.array([w, h, w, h], dtype=np.float32), 0.0, 1.0)
        sc = scores[keep]
        areas = (norm[:, 2] - norm[:, 0]) * (norm[:, 3] - norm[:, 1])
        k = int(areas.argmax())  # the main subject = the biggest bird
        return {"n": int(len(sc)), "conf": float(sc[k]), "bbox": [float(v) for v in norm[k]],
                "area": float(areas[k])}


def torch_device(requested: Optional[str]) -> Tuple[str, str]:
    """(device string, human description)"""
    try:
        import torch
    except Exception:
        return requested or "cpu", "CPU (PyTorch could not be imported)"
    if requested:
        return requested, requested
    if torch.cuda.is_available():
        return "cuda:0", f"GPU {torch.cuda.get_device_name(0)}"
    return "cpu", "CPU - this PyTorch build has no CUDA, so it is slow (install the CUDA build for the GPU)"


def build_detector(a: argparse.Namespace) -> Tuple[Optional[Any], str]:
    """Starts a bird detector and tests it on a blank image, so that a broken install or a failed
    weight download shows up now and not in the middle of a long run."""
    if a.detector == "auto":
        m = a.detector_model
        order = ["torchvision", "yolo"] if (m and not m.endswith(".pt")) else ["yolo", "torchvision"]
    else:
        order = [a.detector]
    device, devdesc = torch_device(a.detector_device)
    errors = []
    for kind in order:
        try:
            if kind == "yolo":
                det: Any = YoloBirdDetector(a.detector_model or "yolov8n.pt", device, a.detector_conf)
            else:
                det = TorchvisionBirdDetector(a.detector_model, device, a.detector_conf)
            blank = Image.new("RGB", (800, 600), (128, 128, 128))
            det.detect(blank)  # first call: warm-up
            t0 = time.time()
            for _ in range(3):
                det.detect(blank)
            ms = (time.time() - t0) / 3 * 1000
            return det, f"{det.name} on {devdesc}, ~{ms:.0f} ms per image"
        except Exception as e:
            errors.append(f"{kind}: {type(e).__name__}: {str(e)[:150]}")
    return None, "; ".join(errors)


# ------------------------------------------------------------------------------------
# observation parsing
# ------------------------------------------------------------------------------------
TERM_LABELS: Dict[int, str] = {}  # annotation attribute / value id -> lower-case label (filled at start-up)


def annotation_ids(an: dict) -> Tuple[Optional[int], Optional[int]]:
    aid = an.get("controlled_attribute_id")
    vid = an.get("controlled_value_id")
    if aid is None:
        aid = (an.get("controlled_attribute") or {}).get("id")
    if vid is None:
        vid = (an.get("controlled_value") or {}).get("id")
    cav = an.get("concatenated_attr_val")
    if (aid is None or vid is None) and isinstance(cav, str) and "|" in cav:
        x, y = cav.split("|", 1)
        try:
            aid, vid = int(x), int(y)
        except ValueError:
            pass
    return aid, vid


def parse_annotations(obs: dict) -> Dict[str, List[str]]:
    """Reads the observation's annotations. Works whether the API sends labels, ids only, or both."""
    out: Dict[str, List[str]] = {"life_stage": [], "sex": [], "alive_or_dead": [], "evidence": [], "all": []}
    fallback_attr = {1: "life stage", 9: "sex", 17: "alive or dead", 22: "evidence of presence"}
    for an in obs.get("annotations") or []:
        vs = an.get("vote_score")
        if isinstance(vs, (int, float)) and vs < 0:  # community disagrees with this annotation
            continue
        aid, vid = annotation_ids(an)
        attr = ((an.get("controlled_attribute") or {}).get("label") or TERM_LABELS.get(aid)
                or fallback_attr.get(aid) or "")
        val = ((an.get("controlled_value") or {}).get("label") or TERM_LABELS.get(vid) or "")
        al, vl = attr.lower(), val.lower()
        out["all"].append(f"{attr or aid}={val or vid}")
        if al.startswith("life stage"):
            out["life_stage"].append(vl)
        elif al == "sex":
            out["sex"].append(vl)
        elif al.startswith("alive"):
            out["alive_or_dead"].append(vl)
        elif al.startswith("evidence"):
            # an evidence value we cannot name is treated as "not the organism" (the safe side)
            out["evidence"].append(vl or f"value-{vid}")
    return out


def swap_variant(url: str, variant: str) -> str:
    return re.sub(r"/(square|small|medium|large|original)(\.[A-Za-z0-9]+)((?:\?.*)?)$",
                  lambda m: f"/{variant}{m.group(2)}{m.group(3)}", url)


def parse_latlon(obs: dict) -> Tuple[str, str]:
    loc = obs.get("location")
    if isinstance(loc, str) and "," in loc:
        lat, lon = loc.split(",", 1)
        return lat.strip(), lon.strip()
    coords = (obs.get("geojson") or {}).get("coordinates")
    if coords and len(coords) == 2:
        return str(coords[1]), str(coords[0])
    return "", ""


def build_row(sp: "Species", obs: dict, photo: dict, pos: int, variant: str, url: str,
              info: dict, rel_path: str, abs_path: str, det: Optional[dict],
              det_name: str, from_filter: bool = False, bird_sharp: Optional[float] = None) -> dict:
    ann = parse_annotations(obs)
    det_ = obs.get("observed_on_details") or {}
    taxon = obs.get("taxon") or {}
    user = obs.get("user") or {}
    lat, lon = parse_latlon(obs)
    bbox = (det or {}).get("bbox")
    row = {
        "image_path": rel_path, "image_abspath": abs_path, "label_id": sp.label_id,
        "species_scientific": sp.sci, "taxon_id": sp.taxon_id,
        "common_name": sp.common or "", "iconic_taxon": taxon.get("iconic_taxon_name") or "",
        "selection_tier": ("organism_annotated" if (from_filter or "organism" in ann["evidence"])
                           else "no_annotation"),
        "observation_id": obs.get("id"), "observation_uuid": obs.get("uuid") or "",
        "observation_url": obs.get("uri") or f"https://www.inaturalist.org/observations/{obs.get('id')}",
        "obs_taxon_id": taxon.get("id"), "obs_taxon_name": taxon.get("name") or "",
        "obs_taxon_rank": taxon.get("rank") or "", "species_guess": clean_text(obs.get("species_guess"), 200),
        "photo_id": photo.get("id"), "photo_position": pos,
        "photo_license": photo.get("license_code") or "",
        "photo_attribution": clean_text(photo.get("attribution"), 300),
        "photo_url": url, "size_variant": variant,
        "user_id": user.get("id"), "user_login": user.get("login") or "",
        "observed_on": obs.get("observed_on") or det_.get("date") or "",
        "time_observed_at": obs.get("time_observed_at") or "",
        "observed_on_string": clean_text(obs.get("observed_on_string"), 100),
        "observed_time_zone": obs.get("observed_time_zone") or "",
        "year": det_.get("year", ""), "month": det_.get("month", ""), "day": det_.get("day", ""),
        "hour": det_.get("hour", ""), "week": det_.get("week", ""),
        "created_at": obs.get("created_at") or "", "updated_at": obs.get("updated_at") or "",
        "latitude": lat, "longitude": lon,
        "positional_accuracy": obs.get("positional_accuracy") if obs.get("positional_accuracy") is not None else "",
        "public_positional_accuracy": obs.get("public_positional_accuracy") if obs.get("public_positional_accuracy") is not None else "",
        "geoprivacy": obs.get("geoprivacy") or "", "taxon_geoprivacy": obs.get("taxon_geoprivacy") or "",
        "obscured": bool(obs.get("obscured")), "place_guess": clean_text(obs.get("place_guess"), 300),
        "place_ids": "|".join(str(x) for x in (obs.get("place_ids") or [])),
        "quality_grade": obs.get("quality_grade") or "", "captive": bool(obs.get("captive")),
        "identifications_count": len(obs.get("identifications") or []),
        "num_id_agreements": obs.get("num_identification_agreements", ""),
        "num_id_disagreements": obs.get("num_identification_disagreements", ""),
        "faves_count": obs.get("faves_count", ""), "comments_count": obs.get("comments_count", ""),
        "annotations": "; ".join(ann["all"]), "life_stage": "|".join(ann["life_stage"]),
        "sex": "|".join(ann["sex"]), "alive_or_dead": "|".join(ann["alive_or_dead"]),
        "evidence_of_presence": "|".join(ann["evidence"]),
        "description": clean_text(obs.get("description")),
        "tags": "|".join(str(t) for t in (obs.get("tags") or [])),
        "observation_photos_count": len(obs.get("photos") or []),
        "width": info["width"], "height": info["height"], "file_size_bytes": info["size"],
        "image_format": info["format"], "exif_orientation": info["exif_orientation"],
        "sha256": info["sha256"], "dhash": info["dhash_hex"], "sharpness": info["sharpness"],
        "brightness": info["brightness"], "contrast": info["contrast"],
        "aspect_ratio": info["aspect_ratio"],
        "detector": det_name if det is not None else "",
        "bird_detected": (det["n"] > 0) if det is not None else "",
        "bird_conf": round(det["conf"], 4) if det is not None else "",
        "bird_num": det["n"] if det is not None else "",
        "bird_x1": round(bbox[0], 5) if bbox else "", "bird_y1": round(bbox[1], 5) if bbox else "",
        "bird_x2": round(bbox[2], 5) if bbox else "", "bird_y2": round(bbox[3], 5) if bbox else "",
        "bird_area_frac": round(det["area"], 5) if det is not None else "",
        "bird_sharpness": round(bird_sharp, 2) if bird_sharp is not None else "",
        "downloaded_at_utc": utcnow(),
    }
    return row


# ------------------------------------------------------------------------------------
# data structures
# ------------------------------------------------------------------------------------
@dataclass
class Species:
    sci: str
    slug: str
    label_id: int
    taxon_id: int
    name: str
    common: str
    obs_count: int = 0


@dataclass
class Stream:
    sp: Species
    month: Optional[int]
    key: str
    tier: int = 1  # 0 = observations annotated "Evidence of presence = Organism", 1 = all research grade
    cursor: Optional[int] = None
    exhausted: bool = False
    inflight: Set[int] = field(default_factory=set)


def repair_csv_tail(path: Path) -> None:
    """If a hard crash left a half-written last line, move it to a .tail.bak file."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        data = f.read()
        cut = data.rfind(b"\n") + 1
        with open(str(path) + ".tail.bak", "ab") as b:
            b.write(data[cut:] + b"\n")
        f.truncate(cut)
    say(f"[REPAIR] {path.name}: incomplete last line moved to {path.name}.tail.bak")


class Store:
    """Append-only CSV storage + all de-duplication state (rebuilt from the CSVs on start)."""

    def __init__(self, out: Path, a: argparse.Namespace):
        self.out = out
        self.a = a
        self.lock = threading.RLock()
        self.counts: collections.Counter = collections.Counter()
        self.photo_ids: Set[str] = set()
        self.processed: Set[int] = set()
        self.sha: Set[str] = set()
        self.userday: Set[tuple] = set()
        self.userday_inflight: Set[tuple] = set()
        self.rej_counts: collections.Counter = collections.Counter()
        self.hash_arr = np.zeros((1024, 4), dtype=np.uint64)
        self.hash_n = 0
        self.total_accepted = 0
        self.excluded_existing = 0
        self.sum_bytes = 0
        self.n_bytes = 0
        self.session_accepted = 0
        self.session_rejected = 0
        self.raw_files: Dict[str, Any] = {}
        self._unsynced = 0
        self.dataset_path = out / "dataset.csv"
        self.rejected_path = out / "rejected.csv"
        self._load()
        self._open()

    # -- keys ---------------------------------------------------------------------
    def ud_key(self, sci: str, user_id: Any, date: str) -> tuple:
        if self.a.user_day_scope == "species":
            return (sci, str(user_id), date)
        return (str(user_id), date)

    # -- load ---------------------------------------------------------------------
    def _load(self) -> None:
        repair_csv_tail(self.dataset_path)
        repair_csv_tail(self.rejected_path)
        self._migrate_dataset_layout()
        if self.dataset_path.exists() and self.dataset_path.stat().st_size > 0:
            with open(self.dataset_path, newline="", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                if rd.fieldnames != DATASET_COLUMNS:
                    raise SystemExit(
                        "ERROR: existing dataset.csv has a different column layout than this script "
                        "version. Use a new --out folder (nothing was modified).")
                for row in rd:
                    self._register(row)
        if self.rejected_path.exists() and self.rejected_path.stat().st_size > 0:
            with open(self.rejected_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        self.processed.add(int(row["observation_id"]))
                    except (ValueError, KeyError):
                        pass
                    self.rej_counts[row.get("reason", "")] += 1

    def _migrate_dataset_layout(self) -> None:
        """Older script versions wrote fewer columns. Add the missing ones (keeping a backup copy)."""
        p = self.dataset_path
        if not p.exists() or p.stat().st_size == 0:
            return
        with open(p, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header == DATASET_COLUMNS:
            return
        missing = [c for c in DATASET_COLUMNS if c not in header]
        extra = [c for c in header if c not in DATASET_COLUMNS]
        if (extra or not missing or not all(c in NEW_COLUMNS for c in missing)
                or header != [c for c in DATASET_COLUMNS if c in header]):
            return  # unknown layout: reported later with a clear error
        bak = p.with_name(f"dataset.csv.bak_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(p, bak)
        tmp = p.with_name("dataset.csv.migrating")
        with open(p, newline="", encoding="utf-8") as fin, \
                open(tmp, "w", newline="", encoding="utf-8") as fout:
            wr = csv.DictWriter(fout, fieldnames=DATASET_COLUMNS, lineterminator="\n", restval="")
            wr.writeheader()
            for row in csv.DictReader(fin):
                if "selection_tier" in missing:
                    ev = (row.get("evidence_of_presence") or "").lower()
                    row["selection_tier"] = "organism_annotated" if "organism" in ev else "no_annotation"
                wr.writerow(row)
        safe_replace(tmp, p)
        say(f"[MIGRATE] dataset.csv upgraded with the new column(s) {', '.join(missing)}; backup kept as {bak.name}")

    def _counts_toward_quota(self, row: dict) -> bool:
        if not self.a.topup_small:
            return True
        try:
            return float(row["bird_area_frac"]) >= self.a.min_bird_area
        except (ValueError, KeyError, TypeError):
            return True  # no detector value stored: cannot judge, so it counts

    def _register(self, row: dict) -> None:
        sci = row["species_scientific"]
        if self._counts_toward_quota(row):
            self.counts[sci] += 1
        else:
            self.excluded_existing += 1
        self.total_accepted += 1
        self.photo_ids.add(str(row["photo_id"]))
        try:
            self.sum_bytes += int(row["file_size_bytes"])
            self.n_bytes += 1
        except (ValueError, KeyError, TypeError):
            pass
        self.sha.add(row["sha256"])
        self.userday.add(self.ud_key(sci, row["user_id"], row["observed_on"]))
        try:
            self.processed.add(int(row["observation_id"]))
        except ValueError:
            pass
        try:
            h = np.frombuffer(bytes.fromhex(row["dhash"]), dtype=np.uint64)
            if h.shape == (4,):
                self._add_hash(h)
        except ValueError:
            pass

    def _add_hash(self, h: np.ndarray) -> None:
        if self.hash_n >= len(self.hash_arr):
            bigger = np.zeros((len(self.hash_arr) * 2, 4), dtype=np.uint64)
            bigger[:self.hash_n] = self.hash_arr[:self.hash_n]
            self.hash_arr = bigger
        self.hash_arr[self.hash_n] = h
        self.hash_n += 1

    # -- open files ---------------------------------------------------------------
    def _open(self) -> None:
        new_ds = not self.dataset_path.exists() or self.dataset_path.stat().st_size == 0
        self._ds_f = open(self.dataset_path, "a", newline="", encoding="utf-8")
        self._ds_w = csv.DictWriter(self._ds_f, fieldnames=DATASET_COLUMNS, lineterminator="\n",
                                    extrasaction="ignore", restval="")
        if new_ds:
            self._ds_w.writeheader()
            self._ds_f.flush()
        new_rj = not self.rejected_path.exists() or self.rejected_path.stat().st_size == 0
        self._rj_f = open(self.rejected_path, "a", newline="", encoding="utf-8")
        self._rj_w = csv.DictWriter(self._rj_f, fieldnames=REJECT_COLUMNS, lineterminator="\n",
                                    extrasaction="ignore", restval="")
        if new_rj:
            self._rj_w.writeheader()
            self._rj_f.flush()

    # -- user/day claims ----------------------------------------------------------
    def avg_bytes(self) -> int:
        return int(self.sum_bytes / self.n_bytes) if self.n_bytes else 450_000

    def claim_userday(self, key: tuple) -> bool:
        with self.lock:
            if key in self.userday or key in self.userday_inflight:
                return False
            self.userday_inflight.add(key)
            return True

    def release_userday(self, key: tuple) -> None:
        with self.lock:
            self.userday_inflight.discard(key)

    # -- writing ------------------------------------------------------------------
    def _near_dup(self, h: np.ndarray) -> bool:
        if self.hash_n == 0:
            return False
        x = np.bitwise_xor(self.hash_arr[:self.hash_n], h)
        d = POP16[x.view(np.uint16)].sum(axis=1, dtype=np.int32)
        return bool(d.min() <= self.a.dup_hamming)

    def commit(self, sp: Species, obs: dict, row: dict, data: bytes, h: np.ndarray) -> str:
        with self.lock:
            if self.counts[sp.sci] >= self.a.per_species:
                return "full"
            if str(row["photo_id"]) in self.photo_ids:
                return "dup_photo_id"
            if row["sha256"] in self.sha:
                return "dup_exact_image"
            if self._near_dup(h):
                return "dup_near_image"
            free = free_bytes(self.out)
            if free is not None and free < len(data) + 50_000_000:
                raise DiskFull()
            dest = self.out / row["image_path"]
            tmp = dest.with_name(dest.name + ".part")
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(data)
                safe_replace(tmp, dest)
            except OSError as e:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                if is_disk_full(e):
                    raise DiskFull() from e
                raise TransientIO(f"{type(e).__name__}: {e}") from e
            try:
                if not self.a.no_raw_json:
                    fh = self.raw_files.get(sp.slug)
                    if fh is None:
                        p = self.out / "metadata" / "raw"
                        p.mkdir(parents=True, exist_ok=True)
                        fh = open(p / f"{sp.slug}.jsonl", "a", encoding="utf-8")
                        self.raw_files[sp.slug] = fh
                    fh.write(json.dumps({"photo_id": row["photo_id"], "observation": obs},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                self._ds_w.writerow(row)
                self._ds_f.flush()
            except OSError as e:
                if is_disk_full(e):
                    raise DiskFull() from e
                raise TransientIO(f"{type(e).__name__}: {e}") from e
            self._register(row)
            self.userday_inflight.discard(self.ud_key(sp.sci, row["user_id"], row["observed_on"]))
            self.session_accepted += 1
            self._unsynced += 1
            if self._unsynced >= 100:
                self.sync()
            return "ok"

    def reject(self, sci: str, obs_id: Any, photo_id: Any, stage: str, reason: str,
               detail: str = "") -> None:
        with self.lock:
            try:
                self.processed.add(int(obs_id))
            except (TypeError, ValueError):
                pass
            self.rej_counts[reason] += 1
            self.session_rejected += 1
            try:
                self._rj_w.writerow({"timestamp_utc": utcnow(), "species_scientific": sci,
                                     "observation_id": obs_id, "photo_id": photo_id, "stage": stage,
                                     "reason": reason, "detail": clean_text(detail, 200)})
                self._rj_f.flush()
            except OSError as e:  # recording a rejection must never stop a worker
                log.warning("could not write rejected.csv: %s", e)
            if not self.a.quiet_rejects:
                say(f"[SKIP] {sci} | obs {obs_id}: {reason}" + (f" ({clean_text(detail, 80)})" if detail else ""))

    def sync(self) -> None:
        with self.lock:
            for f in (self._ds_f, self._rj_f, *self.raw_files.values()):
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except OSError:
                    pass
            self._unsynced = 0

    def close(self) -> None:
        self.sync()
        with self.lock:
            for f in (self._ds_f, self._rj_f, *self.raw_files.values()):
                try:
                    f.close()
                except OSError:
                    pass


# ------------------------------------------------------------------------------------
# keyboard listener
# ------------------------------------------------------------------------------------
COMMANDS = {"p": "p", "pause": "p", "r": "r", "resume": "r", "s": "s", "stop": "s", "q": "s",
            "quit": "s", "i": "i", "status": "i", "h": "h", "help": "h", "?": "h"}


class CommandReader(threading.Thread):
    """Reads typed commands (letter + Enter) from the terminal and applies them."""

    def __init__(self, on_cmd):
        super().__init__(daemon=True)
        self.on_cmd = on_cmd

    def run(self) -> None:
        try:
            while True:
                line = sys.stdin.readline()
                if line == "":  # EOF: terminal closed
                    return
                word = line.strip().lower()
                if not word:
                    continue
                if word in COMMANDS:
                    self.on_cmd(COMMANDS[word])
                else:
                    say(f"[INPUT] unknown command {word!r}. Type h + Enter for help.")
        except Exception as e:  # keyboard control is a convenience, never fatal
            log.warning("command reader stopped: %s", e)


# ------------------------------------------------------------------------------------
# the application
# ------------------------------------------------------------------------------------
class App:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = Path(a.out).resolve()
        self.stop_event = threading.Event()
        self.run_event = threading.Event()
        self.run_event.set()
        self.keys_done = threading.Event()
        self.producer_done = threading.Event()
        self.paused = False
        self.reader_started = False
        self.pause_announced = False
        self.producer_gated = False
        self.api_lat = 0.0
        self.dl_lat = 0.0
        self.producer_state = "starting"
        self.stop_announced = False
        self.stop_time: Optional[float] = None
        self.state_lock = threading.RLock()
        self.active = 0
        self.active_lock = threading.Lock()
        self.pending: collections.Counter = collections.Counter()
        self.rr_species = 0
        self.rr_stream: Dict[str, int] = collections.defaultdict(int)
        self.species: List[Species] = []
        self.unresolved: List[str] = []
        self.streams: Dict[str, List[Stream]] = {}
        self.q: "queue.Queue" = queue.Queue(maxsize=max(4, a.workers * 3))
        self.t0 = time.time()
        self.last_status = 0.0
        self.last_checkpoint = 0.0
        self.detector: Optional[Any] = None
        self.organism_term: Optional[Tuple[int, int]] = None
        self.tier0_disabled = False
        self.tier0_checked = False
        self.organism_check_failed = False
        self.evidence_attr: Optional[int] = None
        self.organism_value: Optional[int] = None
        self.bad_values: List[int] = []
        self.evidence_exclusion = False
        self.ann_samples = 0
        self.disk_paused = False
        self.last_io_warn = 0.0
        self.last_disk_check = 0.0
        self.api_fail: Dict[str, int] = {}
        self.seen_obs = 0
        self.seen_ann = 0
        self.ann_note_shown = False
        self.det_name = ""
        self.lock_path = self.out / "run.lock"
        self.state_path = self.out / "state" / "state.json"

        ua = "inat-dl-dataset-downloader/1.0 (+research use)"
        if a.contact:
            ua += f" contact:{a.contact}"
        self.http = Http(ua, self.stop_event)
        self.api_limiter = RateLimiter(a.api_rps, self.stop_event)
        self.budget = ByteBudget(a.max_gb_per_hour)

    # -- setup --------------------------------------------------------------------
    def setup_dirs(self) -> None:
        for d in ("", "images", "metadata/raw", "state", "logs"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        install_crash_logging(self.out / "logs" / "errors.log")
        logging.basicConfig(filename=str(self.out / "logs" / "downloader.log"),
                            level=getattr(logging, self.a.log_level.upper(), logging.INFO),
                            format="%(asctime)s %(levelname)s %(threadName)s %(message)s")

    def acquire_lock(self) -> None:
        if self.lock_path.exists() and not self.a.force_unlock:
            m = re.search(r"pid=(\d+)", self.lock_path.read_text(errors="ignore"))
            old_pid = int(m.group(1)) if m else 0
            if old_pid and old_pid != os.getpid() and pid_alive(old_pid):
                raise SystemExit(
                    f"ERROR: another instance (process {old_pid}) seems to be running on this folder.\n"
                    "Close it first. If you are sure it is not running, re-run with --force-unlock.")
            say("[INFO] found a leftover run.lock from a previous run that is no longer running - ignoring it")
        self.lock_path.write_text(f"pid={os.getpid()} started={utcnow()}\n")

    def release_lock(self) -> None:
        try:
            self.lock_path.unlink()
        except OSError:
            pass

    def load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                say("[WARN] state.json unreadable - cursors restart from the top "
                    "(dataset.csv / rejected.csv still prevent any duplicates)")
        return {}

    def filter_signature(self) -> str:
        a = self.a
        return json.dumps({"q": a.quality_grade, "lic": a.licenses, "cap": a.allow_captive,
                           "months": not a.no_month_spread, "term": list(self.organism_term or []), "excl": self.evidence_exclusion,
                           "v": SCHEMA_VERSION}, sort_keys=True)

    def read_species_file(self) -> List[str]:
        names, seen = [], set()
        for line in Path(self.a.species_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                line = line.split("|")[-1].strip()
            name = " ".join(line.split())
            if name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
        if self.a.limit_species:
            names = names[: self.a.limit_species]
        return names

    def resolve_taxon(self, sci: str) -> Optional[dict]:
        self.api_limiter.wait()
        r = self.http.get(f"{self.a.api_base}/taxa",
                          params={"q": sci, "rank": "species", "per_page": 10, "is_active": "true"})
        results = r.json().get("results", [])
        low = sci.lower()
        exact = [t for t in results if (t.get("name") or "").lower() == low]
        cand = exact or [t for t in results if (t.get("matched_term") or "").lower() == low]
        if not cand:
            return None
        t = cand[0]
        return {"id": t["id"], "name": t["name"], "common": t.get("preferred_common_name") or "",
                "obs_count": t.get("observations_count") or 0,
                "matched_as": "exact" if exact else "synonym"}

    def resolve_all(self, names: List[str], state: dict) -> None:
        taxa = state.get("taxa", {})
        labels = state.get("labels", {})
        next_label = (max(labels.values()) + 1) if labels else 0
        for sci in names:
            info = taxa.get(sci)
            if not info:
                info = self.resolve_taxon(sci)
                if info:
                    taxa[sci] = info
            if not info:
                self.unresolved.append(sci)
                continue
            if sci not in labels:
                labels[sci] = next_label
                next_label += 1
            self.species.append(Species(sci=sci, slug=slugify(sci), label_id=labels[sci],
                                        taxon_id=int(info["id"]), name=info["name"],
                                        common=info.get("common", ""), obs_count=info.get("obs_count", 0)))
        state["taxa"], state["labels"] = taxa, labels
        self.state = state

    def load_terms(self) -> None:
        """Looks up the ids of the annotation 'Evidence of Presence' and its values (not hard-coded)."""
        info = self.state.get("terms")
        if not info:
            try:
                self.api_limiter.wait()
                r = self.http.get(f"{self.a.api_base}/controlled_terms")
                results = r.json().get("results", [])
                labels: Dict[str, str] = {}
                ev_attr = organism = None
                bad: List[int] = []
                for term in results:
                    tl = (term.get("label") or "").strip()
                    if term.get("id") is not None:
                        labels[str(int(term["id"]))] = tl.lower()
                    vals = term.get("values") or term.get("controlled_values") or []
                    for v in vals:
                        if v.get("id") is not None:
                            labels[str(int(v["id"]))] = (v.get("label") or "").strip().lower()
                    if "evidence" in tl.lower():
                        ev_attr = int(term["id"])
                        for v in vals:
                            if (v.get("label") or "").strip().lower() == "organism":
                                organism = int(v["id"])
                            elif v.get("id") is not None:
                                bad.append(int(v["id"]))
                if ev_attr is None or organism is None:
                    snippet = json.dumps(results[:1], ensure_ascii=False)[:300] if results else "(empty answer)"
                    say("[WARN] could not read the 'Evidence of Presence' annotation from iNaturalist; "
                        f"unexpected answer format: {snippet}")
                else:
                    info = {"labels": labels, "evidence_attr": ev_attr, "organism": organism, "bad": bad}
                    self.state["terms"] = info
            except (PermanentHTTPError, ValueError, KeyError, TypeError) as e:
                say(f"[WARN] could not load the annotation terms from iNaturalist ({e})")
        if info:
            TERM_LABELS.update({int(k): v for k, v in info["labels"].items()})
            self.evidence_attr = int(info["evidence_attr"])
            self.organism_value = int(info["organism"])
            self.bad_values = [int(x) for x in info["bad"]]

    def find_organism_term(self) -> Optional[Tuple[int, int]]:
        """ids of the annotation 'Evidence of presence = Organism'."""
        a = self.a
        if a.no_organism_first:
            return None
        if a.organism_term != "auto":
            try:
                x, y = a.organism_term.split(":")
                return int(x), int(y)
            except ValueError:
                raise SystemExit("ERROR: --organism-term must look like 22:24 (attribute:value ids)")
        if self.evidence_attr is not None and self.organism_value is not None:
            return self.evidence_attr, self.organism_value
        return None

    def verify_evidence_exclusion(self) -> bool:
        """Can the server itself leave out photos annotated as feather / scat / track / ...?
        Checked by comparing result counts (the parameter is recent, so it is not taken on trust)."""
        if self.a.allow_evidence or self.evidence_attr is None or not self.bad_values:
            return False
        bad_csv = ",".join(str(v) for v in self.bad_values)
        try:
            for sp in self.species[:5]:
                base = self.base_filter_params(sp)
                n_all = self.api_total(base)
                n_bad = self.api_total(dict(base, term_id=self.evidence_attr, term_value_id=bad_csv))
                n_ex = self.api_total(dict(base, term_id_or_unknown=self.evidence_attr,
                                          without_term_value_id=bad_csv))
                say(f"[CHECK] {sp.sci}: {n_all} research-grade observations, {n_bad} annotated as feather/"
                    f"scat/track/..., {n_ex} left after excluding them (expected about {n_all - n_bad})")
                if n_bad == 0:
                    continue
                if abs((n_all - n_bad) - n_ex) <= max(5, 0.05 * n_all):
                    return True
                say("[WARN] the server does not apply the exclusion as expected - the program applies the "
                    "same rule itself to every observation instead")
                return False
        except (PermanentHTTPError, ValueError, KeyError, TypeError) as e:
            say(f"[WARN] could not verify the exclusion filter ({e}) - the program applies the rule itself")
            return False
        say("[INFO] could not verify the exclusion filter (no such annotated photos among the first species) - "
            "the program applies the rule itself to every observation")
        return False

    def api_total(self, params: dict) -> int:
        self.api_limiter.wait()
        r = self.http.get(f"{self.a.api_base}/observations", params=dict(params, per_page=1))
        return int(r.json().get("total_results") or 0)

    def base_filter_params(self, sp: Species) -> Dict[str, Any]:
        a = self.a
        p: Dict[str, Any] = {"taxon_id": sp.taxon_id, "quality_grade": a.quality_grade, "photos": "true"}
        if not a.allow_captive:
            p["captive"] = "false"
        if a.licenses != "any":
            p["photo_license"] = OPEN_LICENSES if a.licenses == "open" else a.licenses
        return p

    def verify_organism_filter(self) -> bool:
        """Compares result counts with and without the annotation filter, so it works even if
        the search results carry no annotation data."""
        assert self.organism_term
        tid, vid = self.organism_term
        try:
            for sp in self.species[:3]:
                base = self.base_filter_params(sp)
                total_all = self.api_total(base)
                total_org = self.api_total(dict(base, term_id=tid, term_value_id=vid))
                say(f"[CHECK] {sp.sci}: {total_all} research-grade observations, "
                    f"{total_org} of them annotated as the live organism")
                if total_all == 0:
                    continue
                if total_org >= total_all:
                    say("[WARN] the server ignores the annotation filter (same count with and without it) - "
                        "using all research-grade photos instead")
                    self.organism_check_failed = True
                    return False
                if total_org > 0:
                    return True
        except (PermanentHTTPError, ValueError, KeyError, TypeError) as e:
            say(f"[WARN] could not verify the annotation filter ({e}) - using all research-grade photos")
            self.organism_check_failed = True
            return False
        say("[WARN] no annotated photos found for the first species - using all research-grade photos")
        self.organism_check_failed = True
        return False

    def print_coverage(self) -> None:
        term = self.find_organism_term()
        if not term:
            print("could not find the 'Organism' annotation on iNaturalist")
            return
        print(f"\n{'species':32} {'research-grade':>15} {'annotated organism':>19}")
        for sp in self.species:
            base = self.base_filter_params(sp)
            n_all = self.api_total(base)
            n_org = self.api_total(dict(base, term_id=term[0], term_value_id=term[1]))
            print(f"{sp.sci:32} {n_all:>15} {n_org:>19}", flush=True)
        print()

    def usable(self, s: Stream) -> bool:
        return not s.exhausted and not (s.tier == 0 and self.tier0_disabled)

    def build_streams(self, saved: dict) -> None:
        use_saved = saved.get("signature") == self.filter_signature()
        if saved.get("signature") and not use_saved:
            say("[INFO] filter settings changed since the last run: paging restarts from the top "
                "(already-collected images are still skipped)")
        sv = saved.get("streams", {}) if use_saved else {}
        months: List[Optional[int]] = [None] if self.a.no_month_spread else list(range(1, 13))
        tiers = [0, 1] if self.organism_term else [1]
        for sp in self.species:
            lst = []
            for tier in tiers:
                for m in months:
                    key = f"{sp.sci}|t{tier}|{m if m else 'all'}"
                    s = Stream(sp=sp, month=m, key=key, tier=tier)
                    if key in sv:
                        s.cursor = sv[key].get("cursor")
                        s.exhausted = bool(sv[key].get("exhausted"))
                    lst.append(s)
            self.streams[sp.sci] = lst
            self.rr_stream[sp.sci] = sp.label_id % len(months)  # species start on different months

    def setup_detector(self) -> None:
        mode = self.a.detector
        if mode == "none":
            say("[INFO] bird detector: OFF (only metadata checks are used to make sure photos show the bird)")
            return
        say("[INFO] starting the bird detector (first start may download model weights) ...")
        det, info = build_detector(self.a)
        if det is None:
            if mode != "auto":
                raise SystemExit(f"ERROR: bird detector '{mode}' could not start: {info}")
            say(f"[WARN] no bird detector available ({info}). Only metadata checks are used to make "
                "sure photos show the bird. Install ultralytics, or torch + torchvision, to enable it.")
            return
        self.detector = det
        self.det_name = det.name
        say(f"[INFO] bird detector ON: {info}; a photo needs a bird with confidence >= "
            f"{self.a.detector_conf} covering >= {self.a.min_bird_area:.0%} of the image")

    # -- control ------------------------------------------------------------------
    def apply_command(self, key: str) -> None:
        if key == "p":
            if self.paused:
                say("[PAUSE] already paused - press r to resume")
                return
            self.paused = True
            self.pause_announced = False
            self.run_event.clear()
            say(f"[PAUSE] pause requested - finishing {self.active} in-flight item(s), "
                "no new work will start. Type r + Enter to resume.")
        elif key == "r":
            if not self.paused:
                say("[RESUME] not paused")
                return
            self.paused = False
            self.disk_paused = False
            self.run_event.set()
            say("[RESUME] resumed")
        elif key in ("s", "q"):
            if self.stop_event.is_set():
                return
            say("[STOP] stop requested - saving now; downloads that are still running are abandoned "
                "and simply redone when you run the same command again")
            self.stop_time = time.time()
            self.stop_event.set()
            self.run_event.set()
        elif key == "i":
            self.print_status(force=True)
        elif key in ("h", "?"):
            say("commands (type + Enter): p=pause  r=resume  s=stop&save  i=status  h=help   (Ctrl+C = stop&save)")

    def poll_control_file(self) -> None:
        p = self.out / "control.txt"
        if not p.exists():
            return
        try:
            word = (p.read_text().strip().lower().split() or [""])[0]
        except OSError:
            return
        try:
            p.unlink()
        except OSError:
            pass
        mapping = {"pause": "p", "resume": "r", "stop": "s", "status": "i"}
        if word in mapping:
            say(f"[CONTROL] control.txt -> {word}")
            self.apply_command(mapping[word])
        else:
            say(f"[CONTROL] unknown command in control.txt: {word!r} (use pause/resume/stop/status)")

    def gate(self, producer: bool = False) -> None:
        while not self.run_event.is_set():
            if self.stop_event.is_set():
                raise Stopped()
            if producer:
                self.producer_gated = True
            self.run_event.wait(0.3)
        if producer:
            self.producer_gated = False
        if self.stop_event.is_set():
            raise Stopped()

    # -- API paging ---------------------------------------------------------------
    def fetch_page(self, st: Stream) -> List[dict]:
        a = self.a
        params: Dict[str, Any] = {
            "taxon_id": st.sp.taxon_id, "quality_grade": a.quality_grade, "photos": "true",
            "per_page": a.page_size, "order_by": "id", "order": "desc",
        }
        if not a.allow_captive:
            params["captive"] = "false"
        if a.licenses != "any":
            params["photo_license"] = OPEN_LICENSES if a.licenses == "open" else a.licenses
        if st.tier == 0 and self.organism_term:  # ask only for photos annotated as the live organism
            params["term_id"], params["term_value_id"] = self.organism_term
        if st.tier != 0 and self.evidence_exclusion:  # leave out feather / scat / track / ... photos
            params["term_id_or_unknown"] = self.evidence_attr
            params["without_term_value_id"] = ",".join(str(v) for v in self.bad_values)
        if st.cursor:
            params["id_below"] = st.cursor
        if st.month:
            params["month"] = st.month
        self.api_limiter.wait()
        t0 = time.time()
        r = self.http.get(f"{a.api_base}/observations", params=params)
        dt = time.time() - t0
        self.api_lat = dt if not self.api_lat else 0.8 * self.api_lat + 0.2 * dt
        return r.json().get("results", [])

    # -- producer -----------------------------------------------------------------
    def species_needs_work(self, sp: Species) -> bool:
        return (self.store.counts[sp.sci] < self.a.per_species
                and any(self.usable(s) for s in self.streams[sp.sci]))

    def producer(self) -> None:
        try:
            self.producer_loop()
        except Stopped:
            pass
        except Exception as e:  # pragma: no cover
            log.exception("producer crashed")
            say(f"[ERROR] producer crashed: {type(e).__name__}: {e}")
        finally:
            self.producer_done.set()

    def producer_loop(self) -> None:
        n = len(self.species)
        target = self.a.per_species
        while not self.stop_event.is_set():
            self.gate(producer=True)
            picked = None
            for k in range(n):
                sp = self.species[(self.rr_species + k) % n]
                if not self.species_needs_work(sp):
                    continue
                need = target - self.store.counts[sp.sci] - self.pending[sp.sci]
                if need <= 0:
                    continue  # enough tasks already in flight for this species
                picked = (sp, need)
                self.rr_species = (self.rr_species + k + 1) % n
                break
            if picked is None:
                if not any(self.species_needs_work(sp) for sp in self.species):
                    return  # everything is full or exhausted
                self.stop_event.wait(0.3)  # waiting for in-flight tasks to settle
                continue
            sp, need = picked
            live_all = [s for s in self.streams[sp.sci] if self.usable(s)]
            tmin = min(s.tier for s in live_all)  # annotated-organism streams first, the rest only when those ran out
            live = [s for s in live_all if s.tier == tmin]
            st = live[self.rr_stream[sp.sci] % len(live)]
            self.rr_stream[sp.sci] += 1
            try:
                self.producer_state = "asking iNaturalist for a page"
                page = self.fetch_page(st)
            except PermanentHTTPError as e:
                say(f"[ERROR] API refused request for {sp.sci} ({st.key}): HTTP {e.status} - stream skipped")
                st.exhausted = True
                if st.tier == 0 and e.status in (400, 422) and not self.tier0_disabled:
                    self.tier0_disabled = True
                    say("[WARN] the API rejected the annotation filter - using all research-grade photos instead")
                continue
            except (ValueError, KeyError, TypeError, requests.RequestException) as e:
                # e.g. the site answered "200 OK" with a maintenance page instead of JSON
                fails = self.api_fail[st.key] = self.api_fail.get(st.key, 0) + 1
                say(f"[WARN] unexpected answer from iNaturalist ({type(e).__name__}); retrying ({fails}/6)")
                if fails >= 6:
                    st.exhausted = True
                    say(f"[ERROR] giving up on {st.key} for this run after 6 failed attempts")
                self.stop_event.wait(min(60, 5 * fails))
                continue
            self.api_fail.pop(st.key, None)
            self.dispatch(st, page, need)

    def dispatch(self, st: Stream, page: List[dict], need: int) -> None:
        for o in page:
            self.seen_obs += 1
            if o.get("annotations"):
                self.seen_ann += 1
                if self.ann_samples < 3:  # helps to debug the API's annotation format (see logs/downloader.log)
                    self.ann_samples += 1
                    log.info("sample annotation from the API: %s", json.dumps(o["annotations"][0])[:500])
        if self.seen_obs >= 300 and self.seen_ann == 0 and not self.ann_note_shown:
            self.ann_note_shown = True
            say("[NOTE] none of the first 300 observations came with annotation data, so the annotation-based "
                "checks (dead / egg / feather / track) cannot run; the server-side filter and the bird "
                "detector are what keep non-bird photos out")
        if st.tier == 0 and not self.tier0_checked and len(page) >= 5 and any(o.get("annotations") for o in page):
            self.tier0_checked = True  # annotations are visible: double-check that the filter really works
            good = sum(1 for o in page if "organism" in parse_annotations(o)["evidence"])
            if good < 0.6 * len(page):  # a working server-side filter returns (almost) only annotated photos
                self.tier0_disabled = True
                say(f"[WARN] the annotation filter does not seem to work (only {good} of {len(page)} returned "
                    "photos carry the 'Organism' annotation) - using all research-grade photos instead")
                return
        dispatch_n = min(len(page), max(5, int(need * 1.6) + 1))
        sent = considered = 0
        last_id = None
        for obs in page:
            if sent >= dispatch_n:
                break
            considered += 1
            oid = obs["id"]
            last_id = oid
            if oid in self.store.processed:
                continue
            with self.state_lock:
                st.inflight.add(oid)
                self.pending[st.sp.sci] += 1
            while True:
                try:
                    self.q.put((st, obs), timeout=0.5)
                    self.producer_gated = False
                    self.producer_state = "feeding workers"
                    break
                except queue.Full:
                    self.producer_state = "waiting: all workers busy"
                    if self.stop_event.is_set():
                        raise Stopped()
                    if not self.run_event.is_set():  # paused while waiting for queue space
                        self.producer_gated = True
            sent += 1
        with self.state_lock:
            if last_id is not None:
                st.cursor = last_id
            if len(page) == 0 or (considered == len(page) and len(page) < self.a.page_size):
                st.exhausted = True
        if st.exhausted and st.tier == 0 and all(
                x.exhausted for x in self.streams[st.sp.sci] if x.tier == 0):
            say(f"[TIER] {st.sp.sci}: annotated-organism photos used up - now also taking the other "
                f"research-grade photos (checked by the filters{' and the bird detector' if self.detector else ''})")

    # -- workers ------------------------------------------------------------------
    def worker(self) -> None:
        while True:
            try:
                self.gate()
            except Stopped:
                return
            try:
                st, obs = self.q.get(timeout=0.5)
            except queue.Empty:
                if self.producer_done.is_set() and self.q.empty():
                    return
                if self.stop_event.is_set():
                    return
                continue
            with self.active_lock:
                self.active += 1
            finished = False
            try:
                finished = self.handle(st, obs)
            except Stopped:
                finished = False  # stays "in flight": the saved cursor will cover it on resume
            except TransientIO as e:
                log.warning("could not write files for observation %s: %s", obs.get("id"), e)
                self.warn_io(f"an image file ({e})")
                finished = False  # not recorded as processed: it is simply tried again on the next run
            except Exception as e:
                log.exception("unexpected error on observation %s", obs.get("id"))
                try:
                    self.store.reject(st.sp.sci, obs.get("id"), "", "exception", "exception",
                                      f"{type(e).__name__}: {e}")
                except Exception:
                    log.exception("could not record the failure either")
                finished = True
            finally:
                with self.active_lock:
                    self.active -= 1
                with self.state_lock:
                    self.pending[st.sp.sci] -= 1
                    if finished:
                        st.inflight.discard(obs["id"])

    def meta_check(self, obs: dict) -> Optional[Tuple[str, str]]:
        a = self.a
        if obs.get("quality_grade") not in a.quality_set:
            return "quality_grade", str(obs.get("quality_grade"))
        if not a.allow_captive and obs.get("captive"):
            return "captive", ""
        if not obs.get("taxon"):
            return "no_taxon", ""
        if (obs.get("user") or {}).get("id") is None or not (
                obs.get("observed_on") or (obs.get("observed_on_details") or {}).get("date")):
            return "missing_user_or_date", ""
        ann = parse_annotations(obs)
        bad = [v for v in ann["evidence"] if v != "organism"]
        if bad and not a.allow_evidence:
            return "evidence_not_organism", ",".join(bad)
        if "egg" in ann["life_stage"]:
            return "life_stage_egg", ""
        if not a.allow_dead and "dead" in ann["alive_or_dead"]:
            return "dead_organism", ""
        if not a.no_keyword_filter:
            text = " ".join([clean_text(obs.get("description")), " ".join(map(str, obs.get("tags") or [])),
                             clean_text(obs.get("species_guess"))])
            m = EVIDENCE_RE.search(text)
            if m:
                return "keyword_not_the_bird", m.group(0)
        return None

    def handle(self, st: Stream, obs: dict) -> bool:
        sp = st.sp
        sci, oid = sp.sci, obs["id"]
        if self.store.counts[sci] >= self.a.per_species:
            return True  # species already complete: just drop it (not recorded, can be reused later)
        why = self.meta_check(obs)
        if why:
            self.store.reject(sci, oid, "", "metadata", why[0], why[1])
            return True
        # cheap resolution/aspect check from metadata BEFORE claiming the user/day or downloading
        photos = []
        for ph in [p for p in (obs.get("photos") or []) if p.get("url")][: self.a.max_photos_tried]:
            bad = precheck_dims(ph, self.a.size, self.a)
            if bad:
                self.store.reject(sci, oid, ph.get("id"), "precheck", bad[0], bad[1])
            else:
                photos.append(ph)
        if not photos:
            if not obs.get("photos"):
                self.store.reject(sci, oid, "", "metadata", "no_photo")
            return True
        date = obs.get("observed_on") or (obs.get("observed_on_details") or {}).get("date")
        key = self.store.ud_key(sci, obs["user"]["id"], date)
        if not self.store.claim_userday(key):
            self.store.reject(sci, oid, "", "dedup", "duplicate_user_day",
                              f"user {obs['user']['id']} on {date}")
            return True
        accepted = False
        try:
            for pos, ph in enumerate(photos):
                res = self.try_photo(sp, obs, ph, pos,
                                     from_filter=(st.tier == 0 and not self.tier0_disabled))
                if res == "ok":
                    accepted = True
                    break
                if res == "full":
                    break
        finally:
            if not accepted:
                self.store.release_userday(key)
        return True

    def download_photo(self, photo: dict) -> Tuple[bytes, str, str]:
        url = photo["url"]
        variants: List[str] = []
        for v in (self.a.size, "large", "original", "medium"):
            if v not in variants:
                variants.append(v)
        last: Optional[PermanentHTTPError] = None
        for v in variants:
            u = swap_variant(url, v)
            self.budget.wait(self.stop_event)
            t0 = time.time()
            try:
                r = self.http.get(u, max_server_retries=5, timeout=(10, 30),
                                  body_deadline=self.a.download_timeout)
                dt = time.time() - t0
                self.dl_lat = dt if not self.dl_lat else 0.9 * self.dl_lat + 0.1 * dt
            except PermanentHTTPError as e:
                if e.status in (403, 404):
                    last = e
                    continue
                raise
            self.budget.add(len(r.content))
            return r.content, v, u
        raise last or PermanentHTTPError(404, url)

    def try_photo(self, sp: Species, obs: dict, ph: dict, pos: int, from_filter: bool = False) -> str:
        a, st = self.a, self.store
        pid, oid = ph.get("id"), obs["id"]
        if str(pid) in st.photo_ids:
            st.reject(sp.sci, oid, pid, "dedup", "dup_photo_id")
            return "fail"
        try:
            data, variant, url = self.download_photo(ph)
        except TooSlow:
            st.reject(sp.sci, oid, pid, "download", "download_too_slow",
                      f"> {a.download_timeout:.0f}s x3")
            return "fail"
        except PermanentHTTPError as e:
            st.reject(sp.sci, oid, pid, "download", "photo_unavailable", f"HTTP {e.status}")
            return "fail"
        info, im, bad = analyze_image(data, a)
        if bad:
            st.reject(sp.sci, oid, pid, "quality", bad[0], bad[1])
            return "fail"
        det = None
        bird_sharp: Optional[float] = None
        if self.detector is not None:
            det = self.detector.detect(im)
            if det["n"] == 0:
                st.reject(sp.sci, oid, pid, "detector", "no_bird_detected")
                return "fail"
            if det["area"] < a.min_bird_area:
                st.reject(sp.sci, oid, pid, "detector", "bird_too_small", f"area={det['area']:.4f}")
                return "fail"
            if det["bbox"]:
                bird_sharp = subject_sharpness(im, det["bbox"])
                if bird_sharp is not None and bird_sharp < a.min_bird_sharpness:
                    st.reject(sp.sci, oid, pid, "quality", "bird_blurry", f"bird sharpness={bird_sharp:.1f}")
                    return "fail"
        ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[info["format"]]
        rel = f"images/{sp.slug}/{pid}.{ext}"
        row = build_row(sp, obs, ph, pos, variant, url, info, rel,
                        str((self.out / rel).resolve()), det, self.det_name, from_filter, bird_sharp)
        res = ""
        for _ in range(100000):
            try:
                res = st.commit(sp, obs, row, data, info["dhash_arr"])
                break
            except DiskFull:
                self.wait_for_space()  # blocks (paused) until there is room again, then tries once more
        if res == "ok":
            tgt = len(self.species) * a.per_species
            say(f"[OK  ] {sp.sci}  image {st.counts[sp.sci]}/{a.per_species} | total {st.total_accepted}/{tgt}"
                f" | photo {pid} {info['width']}x{info['height']} {info['size'] // 1024} KB | obs {oid}")
            return res
        if res == "full":
            return res
        st.reject(sp.sci, oid, pid, "dedup", res)
        return "fail"

    # -- state / reporting --------------------------------------------------------
    def warn_io(self, what: str) -> None:
        if time.time() - self.last_io_warn > 60:
            self.last_io_warn = time.time()
            say(f"[IO] could not update {what} - another program may have it open (Excel, an editor, antivirus, "
                "cloud sync). Downloading continues and this is retried automatically. "
                "Tip: never open files of the output folder directly while downloading - copy them first.")

    def enter_disk_pause(self, free: Optional[int]) -> None:
        with self.state_lock:
            if self.disk_paused:
                return
            self.disk_paused = True
            self.paused = True
            self.pause_announced = False
            self.run_event.clear()
        gb = f"{free / 1e9:.2f} GB" if free is not None else "very little"
        say(f"[DISK] only {gb} free on the drive of {self.out} - paused automatically. Free some space: the run "
            f"resumes by itself when more than {self.a.min_free_gb + 0.5:.1f} GB are free (or type r + Enter).")

    def wait_for_space(self) -> None:
        self.enter_disk_pause(free_bytes(self.out))
        while self.disk_paused and not self.stop_event.is_set():
            self.stop_event.wait(2.0)
        if self.stop_event.is_set():
            raise Stopped()

    def check_disk(self) -> None:
        free = free_bytes(self.out)
        if free is None or self.a.min_free_gb <= 0:
            return
        if free < self.a.min_free_gb * 1e9:
            if not self.disk_paused and not self.stop_event.is_set():
                self.enter_disk_pause(free)
        elif self.disk_paused and free >= (self.a.min_free_gb + 0.5) * 1e9:
            self.disk_paused = False
            self.paused = False
            self.run_event.set()
            say("[DISK] enough free space again - resumed")

    def save_state(self) -> None:
        with self.state_lock:
            streams = {}
            for lst in self.streams.values():
                for s in lst:
                    cur = (max(s.inflight) + 1) if s.inflight else s.cursor
                    streams[s.key] = {"cursor": cur, "exhausted": bool(s.exhausted and not s.inflight)}
            state = dict(self.state)
            state.update({"schema": SCHEMA_VERSION, "signature": self.filter_signature(),
                          "streams": streams, "updated_at": utcnow()})
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        safe_replace(tmp, self.state_path, tries=3)

    def write_summary(self) -> None:
        p = self.out / "species_summary.csv"
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["label_id", "species_scientific", "taxon_id", "common_name",
                        "accepted", "target", "inat_total_observations", "status"])
            for sp in self.species:
                c = self.store.counts[sp.sci]
                if c >= self.a.per_species:
                    status = "complete"
                elif not any(self.usable(s) for s in self.streams[sp.sci]):
                    status = "exhausted (not enough usable photos on iNaturalist)"
                else:
                    status = "in progress"
                w.writerow([sp.label_id, sp.sci, sp.taxon_id, sp.common, c, self.a.per_species,
                            sp.obs_count, status])
        safe_replace(tmp, p, tries=3)

    def checkpoint(self) -> None:
        for name, fn in (("the dataset files", self.store.sync), ("state/state.json", self.save_state),
                         ("species_summary.csv", self.write_summary)):
            try:
                fn()
            except Exception as e:  # a locked or full drive must not stop the download
                log.warning("checkpoint step '%s' failed: %s: %s", name, type(e).__name__, e)
                self.warn_io(f"{name} ({type(e).__name__})")

    def print_status(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_status < self.a.status_interval:
            return
        self.last_status = now
        tgt = len(self.species) * self.a.per_species
        done = sum(min(self.store.counts[s.sci], self.a.per_species) for s in self.species)
        full = sum(1 for s in self.species if self.store.counts[s.sci] >= self.a.per_species)
        el = max(1.0, now - self.t0)
        rate = self.store.session_accepted / el
        if self.stop_event.is_set():
            state = f"STOPPING ({self.active} still finishing)"
        elif self.paused:
            state = "PAUSED"
        elif time.time() - self.http.last_net_error < 45 and self.http.net_down:
            state = "NO INTERNET - retrying"
        elif self.budget.throttled:
            state = "THROTTLED (bandwidth cap)"
        else:
            state = "running"
        say(f"[{state}] images {done}/{tgt} ({100.0 * done / max(1, tgt):.1f}%) | species full {full}/{len(self.species)}"
            f" | rejected(this run) {self.store.session_rejected} | active {self.active} | queue {self.q.qsize()}"
            f" | {rate:.2f} img/s | downloaded {self.budget.total / 1e9:.2f} GB | net-errors {self.http.net_errors}"
            f" | API {self.api_lat:.1f}s/call | download {self.dl_lat:.1f}s/img | feeder: {self.producer_state}")

    # -- main ---------------------------------------------------------------------
    def run(self) -> int:
        a = self.a
        self.setup_dirs()
        names = self.read_species_file()
        saved = self.load_state()
        say(f"resolving {len(names)} species on iNaturalist ...")
        try:
            self.resolve_all(names, saved)
        except (Stopped, PermanentHTTPError) as e:
            raise SystemExit(f"ERROR while resolving taxa: {e}")
        if a.check_species:
            self.print_species_check()
            if a.coverage:
                self.print_coverage()
            self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
            return 0
        if not self.species:
            raise SystemExit("ERROR: no species could be resolved.")
        if self.unresolved:
            say(f"[WARN] {len(self.unresolved)} name(s) not found on iNaturalist and skipped: "
                + ", ".join(self.unresolved) + "   (use --check-species to inspect)")

        self.load_terms()
        self.organism_term = self.find_organism_term()
        if self.organism_term and not self.verify_organism_filter():
            self.organism_term = None
        self.evidence_exclusion = self.verify_evidence_exclusion()
        if self.evidence_exclusion:
            say("[INFO] evidence filter ON (asked from the server): observations annotated as feather, scat, "
                "track, bone, egg, ... are never requested")
        elif self.a.allow_evidence:
            say("[INFO] evidence filter OFF (--allow-evidence)")
        else:
            say("[INFO] evidence filter: applied by the program to every observation before downloading")
        if self.organism_term:
            say("[INFO] annotation-first mode ON: photos annotated 'Evidence of presence = Organism' are "
                "requested first; other research-grade photos are used only when those run out")
        elif a.no_organism_first:
            say("[INFO] annotation-first mode OFF")
        elif not self.organism_check_failed:
            say("[WARN] could not look up the 'Organism' annotation on iNaturalist - using all "
                "research-grade photos (still filtered by annotations, keywords and the bird detector)")

        self.acquire_lock()
        try:
            self.store = Store(self.out, a)
            self.build_streams(saved)
            self.setup_detector()
            self.install_signals()
            return self.run_loop()
        finally:
            self.release_lock()

    def print_species_check(self) -> None:
        print(f"\n{'name in file':32} {'iNat taxon':32} {'id':>9} {'obs':>9}  note")
        for sci in self.read_species_file():
            t = self.state["taxa"].get(sci)
            if not t:
                print(f"{sci:32} {'-- NOT FOUND --':32}")
                continue
            note = "" if t["matched_as"] == "exact" else "(synonym: iNat uses a different current name)"
            if t.get("obs_count", 0) < self.a.per_species:
                note += " FEWER observations than --per-species"
            print(f"{sci:32} {t['name']:32} {t['id']:>9} {t.get('obs_count', 0):>9}  {note}")
        print()

    def install_signals(self) -> None:
        def handler(signum, frame):
            if self.stop_event.is_set():
                say("[FORCE] second interrupt - exiting now (CSV/images are intact; "
                    "cursors from the last checkpoint will be used)")
                os._exit(130)
            self.apply_command("s")
        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handler)

    def run_loop(self) -> int:
        a = self.a
        start_total = self.store.total_accepted
        say(f"output folder : {self.out}")
        say(f"already in dataset.csv: {start_total} images, {len(self.store.processed)} observations processed before")
        if self.store.excluded_existing:
            say(f"[TOP-UP] {self.store.excluded_existing} of the existing images show a bird smaller than "
                f"--min-bird-area {a.min_bird_area}: they stay in dataset.csv but do not count toward the quota, "
                "so those species are topped up with closer photos")
        free = free_bytes(self.out)
        if free is not None:
            done = sum(min(self.store.counts[s.sci], a.per_species) for s in self.species)
            remaining = max(0, len(self.species) * a.per_species - done)
            need = remaining * self.store.avg_bytes()
            say(f"[DISK] {free / 1e9:.1f} GB free on the drive of the output folder; about {need / 1e9:.1f} GB "
                f"are still needed for the remaining {remaining} images (estimate)")
            if free < need * 1.1 + a.min_free_gb * 1e9:
                say("[WARN] this may not be enough space - free some, or the run pauses by itself when it gets low")
        say(f"workers={a.workers}  size={a.size}  per-species={a.per_species}  "
            f"bandwidth cap={'off' if not a.max_gb_per_hour else str(a.max_gb_per_hour) + ' GB/h'}  "
            f"api={a.api_rps}/s")
        if sys.stdin.isatty():
            CommandReader(self.apply_command).start()
            self.reader_started = True
            say("commands (type + Enter): p=pause  r=resume  s=stop&save  i=status  h=help   (Ctrl+C = stop&save)")
        else:
            say(f"no keyboard attached: control with  echo pause|resume|stop > {self.out / 'control.txt'}")
        workers = [threading.Thread(target=self.worker, name=f"worker-{i}", daemon=True)
                   for i in range(a.workers)]
        prod = threading.Thread(target=self.producer, name="producer", daemon=True)
        for t in workers:
            t.start()
        prod.start()
        self.last_checkpoint = time.time()
        try:
            while True:
                time.sleep(0.4)
                self.poll_control_file()
                if self.paused and not self.pause_announced and self.active == 0 \
                        and (self.producer_gated or self.producer_done.is_set()):
                    self.pause_announced = True
                    self.checkpoint()
                    say("[PAUSED] all workers idle, state saved. Type r + Enter to resume or s + Enter to stop.")
                self.print_status()
                if time.time() - self.last_checkpoint > 15:
                    self.last_checkpoint = time.time()
                    self.checkpoint()
                if time.time() - self.last_disk_check > 10:
                    self.last_disk_check = time.time()
                    self.check_disk()
                if not any(t.is_alive() for t in workers):
                    if self.producer_done.is_set() or not prod.is_alive():
                        break
                    say("[ERROR] all download workers stopped unexpectedly (details: logs/errors.log) - "
                        "stopping safely; run the same command again to continue")
                    self.stop_time = time.time()
                    self.stop_event.set()
                    break
                if self.stop_time and time.time() - self.stop_time > 20:
                    say("[STOP] some downloads are still stuck - abandoning them (they will be redone on resume)")
                    break
        finally:
            self.stop_event.set()
            self.run_event.set()
            deadline = time.time() + 3
            for t in workers:
                t.join(timeout=max(0.05, deadline - time.time()))
            prod.join(timeout=max(0.05, deadline - time.time()))
            self.keys_done.set()
            self.checkpoint()
            try:
                self.store.close()
            except Exception as e:
                log.warning("closing files failed: %s", e)
        return self.finish(start_total)

    def finish(self, start_total: int) -> int:
        a = self.a
        complete = all(self.store.counts[s.sci] >= a.per_species for s in self.species)
        gained = self.store.total_accepted - start_total
        short = [(s.sci, self.store.counts[s.sci]) for s in self.species
                 if self.store.counts[s.sci] < a.per_species]
        if complete:
            say(f"[DONE] dataset complete: {self.store.total_accepted} images in dataset.csv "
                f"(+{gained} this run).")
        elif self.stop_event.is_set() and any(self.species_needs_work(s) for s in self.species):
            say(f"[STOPPED] state saved. {self.store.total_accepted} images so far (+{gained} this run). "
                "Run the same command again to continue exactly where it stopped.")
        else:
            say(f"[FINISHED] {self.store.total_accepted} images (+{gained} this run). "
                f"{len(short)} species have fewer than {a.per_species} usable images "
                "(see species_summary.csv).")
        if short and len(short) <= 25:
            say("short species: " + ", ".join(f"{n} ({c})" for n, c in short))
        if self.store.rej_counts:
            top = ", ".join(f"{k}:{v}" for k, v in self.store.rej_counts.most_common(8))
            say(f"rejection reasons so far: {top}")
        return 0


# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel, resumable iNaturalist downloader (images + metadata -> one CSV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Live commands (type + Enter): p=pause r=resume s=stop&save i=status h=help. "
               "Without a keyboard: write pause|resume|stop|status into <out>/control.txt")
    g = p.add_argument_group("main")
    g.add_argument("--species-file", required=True, help="text file, one scientific name per line")
    g.add_argument("--out", default="inat_birds", help="output folder (re-use it to resume)")
    g.add_argument("--per-species", type=int, default=500)
    g.add_argument("--workers", type=int, default=12, help="parallel image workers")
    g.add_argument("--size", default="large", choices=["medium", "large", "original"],
                   help="photo size (large = max 1024px, original = up to 2048px)")
    g.add_argument("--check-species", action="store_true",
                   help="only resolve the names on iNaturalist, print a table and exit")
    g.add_argument("--coverage", action="store_true",
                   help="with --check-species: also show, per species, how many observations are annotated as the live organism")
    g.add_argument("--limit-species", type=int, default=0, help="debug: only the first N species")
    g.add_argument("--force-unlock", action="store_true", help="ignore a stale run.lock")
    g.add_argument("--demo", action="store_true",
                   help="quick demo: first 3 species, 5 images each, no YOLO needed, output in demo_run/")
    f = p.add_argument_group("observation filters")
    f.add_argument("--quality-grade", default="research", help="comma list: research,needs_id,casual")
    f.add_argument("--licenses", default="open",
                   help="'open' = all CC licences + CC0 (recommended: images come from the open-data bucket), "
                        "'any', or a comma list such as cc0,cc-by")
    f.add_argument("--allow-evidence", action="store_true",
                   help="do NOT skip observations whose annotation 'Evidence of Presence' is feather / scat / "
                        "track / ... (default: skipped)")
    f.add_argument("--no-organism-first", action="store_true",
                   help="do NOT request photos annotated 'Evidence of presence = Organism' first")
    f.add_argument("--organism-term", default="auto",
                   help="annotation filter as attribute:value ids (e.g. 22:24); auto = look it up on iNaturalist")
    f.add_argument("--allow-captive", action="store_true", help="also accept captive/cultivated birds")
    f.add_argument("--allow-dead", action="store_true", help="also accept photos annotated as dead")
    f.add_argument("--no-keyword-filter", action="store_true",
                   help="do not reject descriptions/tags that mention feather, scat, tracks, eggs ...")
    f.add_argument("--no-month-spread", action="store_true",
                   help="do not balance photos across the 12 months (default: balanced)")
    f.add_argument("--user-day-scope", choices=["global", "species"], default="species",
                   help="one photo per user per day PER SPECIES (default), or across the whole dataset")
    f.add_argument("--max-photos-tried", type=int, default=2,
                   help="if an observation's 1st photo fails the checks, try up to N photos (still max 1 kept)")
    q = p.add_argument_group("image quality tests (all values are stored in dataset.csv)")
    q.add_argument("--min-side", type=int, default=384, help="min length of the shorter side (px)")
    q.add_argument("--max-aspect", type=float, default=3.0)
    q.add_argument("--min-sharpness", type=float, default=25.0,
                   help="Laplacian variance of the sharpest of 16 tiles (512px thumbnail)")
    q.add_argument("--min-bird-sharpness", type=float, default=8.0,
                   help="blur test on the bird itself (needs the bird detector): Laplacian variance inside the bird "
                        "box. Deliberately low: it only removes clearly out-of-focus birds; 0 = off")
    q.add_argument("--min-brightness", type=float, default=20.0)
    q.add_argument("--max-brightness", type=float, default=240.0)
    q.add_argument("--min-contrast", type=float, default=8.0)
    q.add_argument("--dup-hamming", type=int, default=10,
                   help="256-bit dHash distance <= this counts as a near-duplicate image")
    d = p.add_argument_group("bird detector (is a bird really in the photo?)")
    d.add_argument("--detector", choices=["auto", "yolo", "torchvision", "none"], default="auto",
                   help="auto = ultralytics YOLO if installed, else torchvision (needs only PyTorch)")
    d.add_argument("--detector-model", default=None,
                   help="yolo: weights file (default yolov8n.pt); torchvision: resnet50_fpn_v2 (default on GPU) "
                        "or mobilenet_v3 (default on CPU)")
    d.add_argument("--detector-conf", type=float, default=0.30)
    d.add_argument("--detector-device", default=None, help="e.g. cpu, cuda:0")
    d.add_argument("--min-bird-area", type=float, default=0.05,
                   help="min fraction of the image covered by the biggest bird box, i.e. how close the bird "
                        "must be (needs the bird detector; 0.05 = the box covers at least 5%% of the photo)")
    d.add_argument("--topup-small", action="store_true",
                   help="photos ALREADY in dataset.csv whose bird covers less than --min-bird-area stay in the "
                        "file but no longer count toward --per-species, so a re-run replaces them with closer ones")
    n = p.add_argument_group("network / politeness")
    n.add_argument("--api-base", default=API_BASE_DEFAULT)
    n.add_argument("--api-rps", type=float, default=1.0, help="API requests per second (iNat asks <= 1)")
    n.add_argument("--max-gb-per-hour", type=float, default=4.5,
                   help="media bandwidth cap (iNat asks < 5 GB/h); 0 = unlimited")
    n.add_argument("--page-size", type=int, default=50, help="observations per API page (<= 200)")
    n.add_argument("--download-timeout", type=float, default=90.0,
                   help="max seconds for ONE image download attempt (3 attempts, then the photo is skipped)")
    n.add_argument("--contact", default="", help="your e-mail, added to the User-Agent (polite)")
    o = p.add_argument_group("misc")
    o.add_argument("--no-raw-json", action="store_true", help="do not save full raw observation JSON")
    o.add_argument("--quiet-rejects", action="store_true", help="do not print a line for every skipped photo")
    o.add_argument("--min-free-gb", type=float, default=1.0,
                   help="pause automatically when the drive has less free space than this (0 = off)")
    o.add_argument("--status-interval", type=float, default=15.0)
    o.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    if a.demo:
        a.per_species = 5
        a.limit_species = a.limit_species or 3
        a.detector = "none"
        a.workers = min(a.workers, 4)
        if a.out == "inat_birds":
            a.out = "demo_run"
    a.page_size = max(1, min(200, a.page_size))
    a.quality_set = {s.strip() for s in a.quality_grade.split(",") if s.strip()}
    a.api_base = a.api_base.rstrip("/")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    app = App(a)
    try:
        rc = app.run()
    except SystemExit:
        raise
    except BaseException:
        sys.excepthook(*sys.exc_info())  # prints it and writes it to logs/errors.log
        sys.stdout.flush()
        sys.stderr.flush()
        logging.shutdown()
        os._exit(1)  # do not wait for the terminal-command thread
    if app.reader_started:
        # The terminal-command thread may still be blocked waiting for Enter; on some systems
        # (notably Windows) that keeps the process alive after everything is finished.
        # All files are already closed and synced at this point, so exit right away.
        sys.stdout.flush()
        sys.stderr.flush()
        logging.shutdown()
        os._exit(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
