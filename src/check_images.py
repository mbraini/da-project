#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_images.py  (v3)  --  automated image checking for the bird dataset
=========================================================================

Every image gets one verdict:  KEEP | REVIEW | REMOVE
  * REMOVE only with clear evidence. Every uncertain case is REVIEW.
  * Nothing is deleted. Verdicts are written as new columns in dataset.csv.

What counts as a violation
  1. A human or any human body part (any size).
  2. Another SPECIES in the picture: a non-bird animal, or a second bird species.
     Several birds are fine as long as they are the same species: the birds are compared with the
     label AND with each other, and 2+ birds of one species are never flagged for being two birds.
     The detector's class name is never proof: a detection that sits on the labeled bird is
     ignored, and an animal detection must be verified as "not a bird" before it can REMOVE.
  3. A re-captured photo = the bird is shown INSIDE a screen / phone / monitor / printed photo.
     This needs PHYSICAL evidence: a screen/print region that contains the bird, and a second
     model that confirms the region really is a display/print (looking at the region with its
     surroundings, so the bezel/frame is visible). The old whole-image "recapture score" is
     NOT used for decisions any more (it fired on ordinary photos).

Stages (run them in this order; each one is resumable)
  run      detectors + species check on all images           (GPU, keyboard controls)
  devices  screens/prints: find regions, check the bird is inside, verify them (GPU+CPU, keyboard controls)
  hires    optional high-resolution pass on suspicious images (GPU, keyboard controls)
  apply    write verdict columns into dataset.csv
  stats    numbers that show how selective each signal is
  review   picture grids to check decisions by eye

Keyboard controls while a stage is working (type the letter, press Enter):
  p = pause     r = resume     s = stop: save everything and exit safely
Run the same command again later: finished images are skipped automatically.

Install
  pip install ultralytics open_clip_torch tqdm opencv-python pandas matplotlib pillow
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:  # PIL fallback is used for reading; moire / quads are skipped
    cv2 = None

SCHEMA = 2

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
COCO_NAMES = {0: "person", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
              20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
              62: "tv", 63: "laptop", 67: "cell phone", 73: "book"}
COCO_ANIMALS = {"cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
SCREEN_NAMES = {"tv", "laptop", "cell phone", "book"}

HUMAN_PARTS = ["human hand", "human finger", "human foot", "human arm", "human leg", "human face"]
HUMAN_PERSON = ["person"]
BIRD_COMPETITORS = ["bird", "bird leg", "bird claw", "bird wing", "bird beak", "bird feather"]
ANIMALS_OW = ["deer", "rabbit", "squirrel", "rodent", "monkey", "snake", "lizard", "frog", "fish",
              "insect", "butterfly", "spider", "bat", "fox", "wolf", "dog", "cat", "cow", "horse",
              "sheep", "goat", "pig", "turtle", "crocodile", "kangaroo", "elephant", "giraffe",
              "zebra", "lion", "bear", "seal"]
PREY = {"snake", "lizard", "frog", "fish", "insect", "butterfly", "spider", "rodent",
        "squirrel", "rabbit", "bat"}
SCREEN_B = ["computer monitor", "television", "smartphone", "tablet", "printed photograph",
            "photo frame", "magazine", "poster", "book page"]
OW_VOCAB = HUMAN_PARTS + HUMAN_PERSON + BIRD_COMPETITORS + ANIMALS_OW + SCREEN_B
OW_KEEP = set(HUMAN_PARTS) | set(HUMAN_PERSON) | set(ANIMALS_OW) | set(SCREEN_B)

SIGLIP_PROMPTS = {  # whole image (hand_held decisions + a weak global margin); 4 prompts per group
    "screen": ["a photo of a computer screen displaying a photo of a bird",
               "a photograph taken of a smartphone screen showing a bird",
               "a photo of a television screen showing a bird",
               "a photo of a tablet or laptop display with a picture of a bird"],
    "print": ["a photograph of a printed photo of a bird",
              "a photo of a magazine or book page with a bird picture",
              "a photo of a framed picture or poster of a bird",
              "a photo of a postcard or paper print of a bird"],
    "hand_held": ["a photo of a bird held in a person's hand",
                  "a person's hand holding a bird",
                  "a bird being held by a person",
                  "a bird in the hand of a bird ringer"],
    "normal": ["a nature photograph of a bird in the wild",
               "a close-up photo of a bird perched on a branch",
               "a photo of a bird flying in the sky",
               "a photo of a bird on the ground or on water"],
}
VERIFY_PROMPTS = {  # crop of an animal detection: is it actually a bird?
    "bird": ["a photo of a bird", "a photo of a large flightless bird such as an ostrich or emu",
             "a photo of a waterbird", "a photo of a bird of prey", "a photo of a bird standing on the ground",
             "a photo of a bird flock"],
    "other": ["a photo of a dog", "a photo of a cow", "a photo of a horse", "a photo of a sheep",
              "a photo of a cat", "a photo of a deer", "a photo of a rabbit", "a photo of a monkey",
              "a photo of a fish", "a photo of an insect", "a photo of a reptile", "a photo of a mammal",
              "a photo of a person"],
}
DEVICE_PROMPTS = {  # region (with its surroundings): is it a display / print showing a picture?
    "display": ["a photo of a smartphone showing a picture",
                "a photo of a computer monitor or television displaying an image",
                "a photo of a laptop or tablet screen displaying a picture",
                "a photo of a printed photograph or a magazine page",
                "a photo of a framed picture or a poster",
                "a photo of a book page with a photograph"],
    "real": ["a photo of a real bird outdoors in nature",
             "a photo of a bird in the wild",
             "a landscape photo with trees and sky",
             "a photo of a building, a window or a door",
             "a photo of a rock, a wall or the ground",
             "a photo of a person or an animal outdoors"],
}

CHECK_COLS = ["img_check", "img_check_reasons", "chk_human_score", "chk_hand_held",
              "chk_animal_score", "chk_recapture", "chk_device_contain", "chk_p_display",
              "chk_moire_z", "chk_margin_z", "chk_n_birds", "chk_bird_species_check", "chk_label_suspect"]


# --------------------------------------------------------------------------
# keyboard controls (p / r / s + Enter)
# --------------------------------------------------------------------------
class Controller:
    HELP = "Controls (type the letter, press Enter):  p = pause   r = resume   s = stop & save"

    def __init__(self):
        self.running = threading.Event()
        self.running.set()
        self.stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        print(self.HELP, flush=True)
        self._thread.start()

    def _reader(self):
        while not self.stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":  # stdin closed (non-interactive run)
                break
            cmd = line.strip().lower()
            if cmd == "p":
                if self.running.is_set():
                    self.running.clear()
                    print("\n[PAUSED] after the current batch.  r = resume, s = stop & save", flush=True)
            elif cmd == "r":
                if not self.running.is_set():
                    self.running.set()
                    print("\n[RESUMED]", flush=True)
            elif cmd == "s":
                self.stop.set()
                self.running.set()
                print("\n[STOPPING] finishing the current batch, saving progress...", flush=True)
            elif cmd:
                print(f"\nUnknown command '{cmd}'. {self.HELP}", flush=True)

    def checkpoint(self):
        """Call between batches. Blocks while paused. Returns True if we must stop."""
        announced = False
        while not self.running.is_set() and not self.stop.is_set():
            if not announced:
                print("[PAUSED] waiting...", flush=True)
                announced = True
            time.sleep(0.2)
        return self.stop.is_set()


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a, b):
    i = inter_area(a, b)
    u = box_area(a) + box_area(b) - i
    return i / u if u > 0 else 0.0


def contained(inner, outer):
    """fraction of `inner` that lies inside `outer`"""
    ai = box_area(inner)
    return inter_area(inner, outer) / ai if ai > 0 else 0.0


def expand(b, frac, w, h):
    bw, bh = b[2] - b[0], b[3] - b[1]
    return [max(0, b[0] - frac * bw), max(0, b[1] - frac * bh),
            min(w, b[2] + frac * bw), min(h, b[3] + frac * bh)]


def same_object(det_box, bird_boxes):
    """True if a non-bird detection sits on one of the birds (the detector just named the bird wrongly)."""
    for bb in bird_boxes:
        if iou(det_box, bb) >= 0.30:
            return True
        if contained(det_box, bb) >= 0.60:  # detection lies inside the bird
            return True
        if contained(bb, det_box) >= 0.60 and box_area(det_box) <= 2.5 * box_area(bb):
            return True  # bird fills the detection box
    return False


def norm_main_box(b, w, h):
    """Box of the labeled bird from dataset.csv (pixels or 0-1 fractions) -> pixel box or None."""
    if b is None:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    if any(np.isnan(v) for v in (x1, y1, x2, y2)):
        return None
    if max(x1, y1, x2, y2) <= 1.5:
        x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [int(x1), int(y1), int(x2), int(y2)]


# --------------------------------------------------------------------------
# cache + image reading
# --------------------------------------------------------------------------
def load_cache(path):
    recs = {}
    p = Path(path)
    if not p.exists():
        return recs
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # last line may be half-written after a crash
                continue
            if r.get("v") == SCHEMA and "photo_id" in r:
                recs[r["photo_id"]] = r
    return recs


class CacheWriter:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def write_many(self, recs):
        for r in recs:
            r["v"] = SCHEMA
            self.f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.f.flush()
        try:
            os.fsync(self.f.fileno())
        except OSError:
            pass

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def moire_score(arr, n=384):
    """Strength of isolated periodic peaks in the spectrum (screens / halftone prints produce them).
    Only used as SUPPORTING evidence (fences and nets also produce peaks)."""
    if cv2 is None:
        return None
    try:
        g = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        n = min(n, h, w)
        if n < 128:
            return None
        y0, x0 = (h - n) // 2, (w - n) // 2
        c = g[y0:y0 + n, x0:x0 + n].astype(np.float32)
        c -= c.mean()
        win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
        m = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(c * win)))).astype(np.float32)
        m8 = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        bg = cv2.medianBlur(m8, 15)
        resid = m8.astype(np.float32) - bg.astype(np.float32)
        yy, xx = np.ogrid[:n, :n]
        r = np.hypot(yy - n / 2, xx - n / 2) / (n / 2)
        mask = (r > 0.10) & (r < 0.95) & (np.abs(yy - n / 2) > 3) & (np.abs(xx - n / 2) > 3)
        return round(float(np.percentile(resid[mask], 99.9)) / 255.0, 5)
    except Exception:
        return None


def find_quads(arr, max_side=640):
    """Rectangle-like regions (screens, phones, photo prints, frames). Candidates only: a region counts
    only if it contains the bird AND a model confirms it is a display/print."""
    if cv2 is None:
        return []
    try:
        h, w = arr.shape[:2]
        s = max_side / max(h, w) if max(h, w) > max_side else 1.0
        im = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else arr
        g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.dilate(cv2.Canny(g, 40, 120), None, iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        H, W = g.shape
        img_area, out = float(H * W), []
        for c in cnts:
            if cv2.contourArea(c) < 0.04 * img_area:
                continue
            ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(ap) != 4 or not cv2.isContourConvex(ap):
                continue
            area = cv2.contourArea(ap)
            if area < 0.05 * img_area or area > 0.90 * img_area:
                continue
            rw, rh = cv2.minAreaRect(ap)[1]
            if rw <= 0 or rh <= 0 or area / (rw * rh) < 0.88:
                continue
            x, y, bw, bh = cv2.boundingRect(ap)
            out.append({"box": [int(x / s), int(y / s), int((x + bw) / s), int((y + bh) / s)],
                        "frac": round(area / img_area, 3), "fill": round(area / (rw * rh), 3)})
        out.sort(key=lambda d: -d["frac"] * d["fill"])
        return out[:3]
    except Exception:
        return []


def load_one(row, size, want_moire):
    """Read one image (BGR) + a small RGB copy for SigLIP + moire score. Runs in worker threads."""
    key, path, label = row
    arr = None
    try:
        if cv2 is not None:
            arr = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            from PIL import Image, ImageOps
            im = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
            arr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
    except Exception:
        arr = None
    if arr is None:
        return key, label, None, None, None
    small = None
    if size:
        if cv2 is not None:
            small = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        else:
            from PIL import Image
            small = np.asarray(Image.fromarray(arr[:, :, ::-1]).resize((size, size)))
        small = np.ascontiguousarray(small)
    return key, label, arr, small, (moire_score(arr) if want_moire else None)


def iter_batches(rows, batch, workers, fn):
    """Decode images in worker threads (JPEG decoding releases the GIL) and yield lists of fn(row)."""
    ex = ThreadPoolExecutor(max_workers=workers)
    pending, it = deque(), iter(rows)

    def fill():
        while len(pending) < batch * 3:
            try:
                row = next(it)
            except StopIteration:
                return
            pending.append(ex.submit(fn, row))

    try:
        fill()
        while pending:
            out = []
            while pending and len(out) < batch:
                out.append(pending.popleft().result())
                fill()
            yield out
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def make_crop(arr, box, pad=0.10, size=224):
    from PIL import Image, ImageOps
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * (1 + pad)
    a1, b1 = int(max(0, round(cx - s / 2))), int(max(0, round(cy - s / 2)))
    a2, b2 = int(min(w, round(cx + s / 2))), int(min(h, round(cy + s / 2)))
    rgb = np.ascontiguousarray(arr[b1:b2, a1:a2, ::-1])
    return ImageOps.pad(Image.fromarray(rgb), (size, size), color=(128, 128, 128))


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class Models:
    """mode 'full': YOLO + YOLO-World + SigLIP + BioCLIP | 'dets': only the two detectors | 'siglip': only SigLIP."""

    def __init__(self, a, species, mode="full"):
        import torch
        import open_clip

        self.torch = torch
        self.cuda = torch.cuda.is_available()
        self.dev = "cuda" if self.cuda else "cpu"
        if not self.cuda:
            print("WARNING: CUDA is not available. This will be very slow.")
        self.species, self.a = species, a
        self.sig_size = 0

        if mode in ("full", "dets"):
            from ultralytics import YOLO, YOLOWorld
            try:
                self.A = YOLO(a.det_weights)
            except Exception as e:
                print(f"could not load {a.det_weights} ({e}); falling back to yolo11m.pt")
                self.A = YOLO("yolo11m.pt")
            self.a_classes = list(COCO_NAMES)
            try:
                self.B = YOLOWorld(a.ow_weights)
            except Exception as e:
                print(f"could not load {a.ow_weights} ({e}); falling back to yolov8s-world.pt")
                self.B = YOLOWorld("yolov8s-world.pt")
            self.B.set_classes(OW_VOCAB)
        if mode == "dets":
            return

        self.sig, _, _ = open_clip.create_model_and_transforms(a.siglip_model, pretrained=a.siglip_pretrained)
        self.sig = self.sig.to(self.dev).eval()
        self.tok = open_clip.get_tokenizer(a.siglip_model)
        cfg = getattr(self.sig.visual, "preprocess_cfg", {}) or {}
        size = cfg.get("size", 224)
        self.sig_size = int(size[0] if isinstance(size, (tuple, list)) else size)
        self.sig_mean = torch.tensor(cfg.get("mean", (0.5, 0.5, 0.5)), device=self.dev).view(1, 3, 1, 1)
        self.sig_std = torch.tensor(cfg.get("std", (0.5, 0.5, 0.5)), device=self.dev).view(1, 3, 1, 1)
        self.sig_scale = self.sig.logit_scale.exp().item()
        bias = getattr(self.sig, "logit_bias", None)
        self.sig_bias = float(bias.item()) if bias is not None else 0.0
        self.g_main = self._text_groups(SIGLIP_PROMPTS)
        self.g_verify = self._text_groups(VERIFY_PROMPTS)
        self.g_dev = self._text_groups(DEVICE_PROMPTS)
        if mode == "siglip":
            return

        self.bio, _, self.bio_pre = open_clip.create_model_and_transforms(a.bio_model)
        self.bio = self.bio.to(self.dev).eval()
        btok = open_clip.get_tokenizer(a.bio_model)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cuda):
            bt = self.bio.encode_text(btok([f"a photo of {s}." for s in species]).to(self.dev))
        bt = bt.float()
        self.bio_text = bt / bt.norm(dim=-1, keepdim=True)
        self.bio_scale = self.bio.logit_scale.exp().item()

    # ---- detectors -------------------------------------------------------
    def _yolo(self, model, arrs, imgsz, conf, classes=None):
        kw = dict(imgsz=imgsz, conf=conf, half=self.cuda, device=0 if self.cuda else "cpu", verbose=False)
        if classes is not None:
            kw["classes"] = classes
        return model.predict(arrs, **kw)

    def det_A(self, arrs, imgsz, conf):
        out = []
        for r in self._yolo(self.A, arrs, imgsz, conf, self.a_classes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            out.append([{"n": COCO_NAMES[c], "conf": round(float(p), 3), "box": [int(v) for v in b]}
                        for b, p, c in zip(xyxy, cf, cl) if c in COCO_NAMES])
        return out

    def det_B(self, arrs, imgsz, conf):
        out = []
        for r in self._yolo(self.B, arrs, imgsz, conf):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names
            out.append([{"n": names[int(c)], "conf": round(float(p), 3), "box": [int(v) for v in b]}
                        for b, p, c in zip(xyxy, cf, cl) if names[int(c)] in OW_KEEP])
        return out

    # ---- SigLIP ----------------------------------------------------------
    def _text_groups(self, prompts_by_group):
        torch = self.torch
        names, prompts, gidx = list(prompts_by_group), [], []
        for gi, g in enumerate(names):
            for p in prompts_by_group[g]:
                prompts.append(p)
                gidx.append(gi)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cuda):
            tf = self.sig.encode_text(self.tok(prompts).to(self.dev))
        tf = tf.float()
        return names, tf / tf.norm(dim=-1, keepdim=True), torch.tensor(gidx, device=self.dev)

    def _group_probs(self, smalls, groups):
        """-> names, softmax over groups (N, G), max cosine per group (N, G)"""
        torch = self.torch
        names, tf, gidx = groups
        x = torch.from_numpy(np.stack(smalls)).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.sig_mean) / self.sig_std
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cuda):
            f = self.sig.encode_image(x)
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        cos = f @ tf.T
        logits = self.sig_scale * cos + self.sig_bias
        gl = torch.stack([logits[:, gidx == gi].max(dim=1).values for gi in range(len(names))], dim=1)
        gc = torch.stack([cos[:, gidx == gi].max(dim=1).values for gi in range(len(names))], dim=1)
        return names, gl.softmax(dim=1).cpu().numpy(), gc.cpu().numpy()

    def siglip(self, smalls):
        names, p, _ = self._group_probs(smalls, self.g_main)
        res = []
        for row in p:
            d = {n: round(float(v), 4) for n, v in zip(names, row)}
            d["recapture"] = round(d["screen"] + d["print"], 4)  # kept for reference only
            res.append(d)
        return res

    def siglip_margin(self, smalls):
        """max cosine(screen/print prompts) - cosine(normal prompts); weak global hint (used only as a z-score)."""
        names, _, gc = self._group_probs(smalls, self.g_main)
        i, j, k = names.index("screen"), names.index("print"), names.index("normal")
        return [float(max(r[i], r[j]) - r[k]) for r in gc]

    def verify_bird(self, smalls):
        names, p, _ = self._group_probs(smalls, self.g_verify)
        i = names.index("bird")
        return [float(row[i]) for row in p]

    def verify_display(self, smalls):
        names, p, _ = self._group_probs(smalls, self.g_dev)
        i = names.index("display")
        return [float(row[i]) for row in p]

    # ---- BioCLIP ---------------------------------------------------------
    def bio_probs(self, crops):
        """-> (probabilities over the target species (n, S), normalized image features (n, d))"""
        torch = self.torch
        out, feats = [], []
        with torch.no_grad():
            for i in range(0, len(crops), self.a.bio_batch):
                x = torch.stack([self.bio_pre(c) for c in crops[i:i + self.a.bio_batch]]).to(self.dev)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cuda):
                    f = self.bio.encode_image(x)
                f = f.float()
                f = f / f.norm(dim=-1, keepdim=True)
                out.append((self.bio_scale * f @ self.bio_text.T).softmax(-1).cpu().numpy())
                feats.append(f.cpu().numpy())
        if not out:
            return np.zeros((0, len(self.species))), np.zeros((0, 1))
        return np.concatenate(out), np.concatenate(feats)


def load_models(a, species, mode="full"):  # separate function so tests can replace it
    return Models(a, species, mode)


# --------------------------------------------------------------------------
# table helpers
# --------------------------------------------------------------------------
BOX_COLS = ["bird_x1", "bird_y1", "bird_x2", "bird_y2"]


def read_table(a):
    want = {"photo_id", "image_path", "species_scientific"} | set(BOX_COLS)
    df = pd.read_csv(a.csv, usecols=lambda c: c in want)
    df["photo_id"] = df["photo_id"].astype(str)
    return df


def main_boxes_of(df):
    if not all(c in df.columns for c in BOX_COLS):
        print("note: bird_x1..bird_y2 not found in the csv; the labeled-bird box cannot be used")
        return {}
    return {k: row for k, row in zip(df["photo_id"], df[BOX_COLS].itertuples(index=False, name=None))}


def select_birds(A, w, h, a):
    cand = []
    for d in A:
        if d["n"] != "bird" or d["conf"] < a.bird_conf:
            continue
        bw, bh = d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]
        if min(bw, bh) >= a.min_side and (bw * bh) / (w * h) >= a.min_area_frac:
            cand.append(d["box"])
    return cand


def finish(writer, code=0):
    writer.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # avoids a hang/crash at exit caused by the stdin-reading thread


# --------------------------------------------------------------------------
# stage 1: run
# --------------------------------------------------------------------------
def process_batch(models, ok, main_boxes, sp_idx, a):
    """ok: list of (key, label, arr, small, moire). Returns list of records."""
    arrs = [x[2] for x in ok]
    A = models.det_A(arrs, a.imgsz, a.a_conf)
    B = models.det_B(arrs, a.imgsz, a.b_conf)
    S = models.siglip([x[3] for x in ok])

    recs, jobs = [], []
    for i, (k, lab, arr, small, mz) in enumerate(ok):
        h, w = arr.shape[:2]
        boxes = select_birds(A[i], w, h, a)
        mb = norm_main_box(main_boxes.get(k), w, h)  # the labeled bird (from the earlier detector)
        if mb is not None and not any(iou(mb, bb) >= 0.5 for bb in boxes):
            boxes.append(mb)
        rec = {"photo_id": k, "w": w, "h": h, "A": A[i], "B": B[i], "S": S[i], "moire": mz,
               "main": mb, "bird_boxes": boxes, "birds": []}
        recs.append(rec)
        # every non-bird animal detection: skip it if it is just the bird, otherwise verify "not a bird"
        for det in [d for d in A[i] if d["n"] in COCO_ANIMALS] + [d for d in B[i] if d["n"] in ANIMALS_OW]:
            if same_object(det["box"], boxes):
                det["self"] = True
                continue
            bw, bh = det["box"][2] - det["box"][0], det["box"][3] - det["box"][1]
            if min(bw, bh) >= a.verify_min_side:
                jobs.append((i, det, np.asarray(make_crop(arr, det["box"], 0.15, models.sig_size))))
    if jobs:
        for (i, det, _), p in zip(jobs, models.verify_bird([j[2] for j in jobs])):
            det["p_bird"] = round(p, 3)
            if p >= a.verify_bird:  # it is a bird that the detector named wrongly
                det["is_bird"] = True
                bb = recs[i]["bird_boxes"]
                if not any(iou(det["box"], x) >= 0.5 for x in bb):
                    bb.append(det["box"])

    crops, owner = [], []
    for ri, rec in enumerate(recs):
        rec["bird_boxes"].sort(key=lambda b: -box_area(b))
        rec["n_birds"] = len(rec["bird_boxes"])
        if rec["n_birds"] >= 2:  # species check only when several birds are present
            for bx in rec["bird_boxes"][:a.max_birds]:
                rec["birds"].append({"box": [int(v) for v in bx]})
                crops.append(make_crop(ok[ri][2], bx))
                owner.append((ri, sp_idx.get(ok[ri][1], -1)))
    if crops:
        P, F = models.bio_probs(crops)
        cursor = [0] * len(recs)
        members = {}  # image index -> crop rows
        for j, (ri, li) in enumerate(owner):
            b = recs[ri]["birds"][cursor[ri]]
            cursor[ri] += 1
            members.setdefault(ri, []).append(j)
            top = int(P[j].argmax())
            b["p_label"] = round(float(P[j, li]), 4) if li >= 0 else 0.0
            b["top1"] = models.species[top]
            b["p_top1"] = round(float(P[j, top]), 4)
        for ri, rows_ in members.items():  # how similar are the birds of this image to EACH OTHER?
            recs[ri]["pair_cos"] = [[x, y, round(float(F[rows_[x]] @ F[rows_[y]]), 3)]
                                    for x in range(len(rows_)) for y in range(x + 1, len(rows_))]
    return recs


def run(a):
    from tqdm import tqdm

    df = read_table(a)
    species = sorted(df["species_scientific"].dropna().unique())
    sp_idx = {s: i for i, s in enumerate(species)}
    main_boxes = main_boxes_of(df)
    done = set(load_cache(a.cache))
    rows = [(k, str(Path(a.root) / ip), lab)
            for k, ip, lab in zip(df["photo_id"], df["image_path"], df["species_scientific"]) if k not in done]
    if a.limit:
        rows = rows[:a.limit]
    print(f"images in table: {len(df)} | already done: {len(done)} | to do now: {len(rows)}")
    if not rows:
        print("Nothing to do. Next: `devices`, optional `hires`, then `apply`.")
        return

    models = load_models(a, species, "full")
    writer = CacheWriter(a.cache)
    ctl = Controller()
    ctl.start()
    n_done, t0 = 0, time.time()
    bar = tqdm(total=len(rows), desc="run", unit="img")
    try:
        for batch in iter_batches(rows, a.batch, a.workers, lambda r: load_one(r, models.sig_size, True)):
            if ctl.checkpoint():
                break
            out = [{"photo_id": x[0], "error": "read_failed"} for x in batch if x[2] is None]
            ok = [x for x in batch if x[2] is not None]
            if ok:
                out.extend(process_batch(models, ok, main_boxes, sp_idx, a))
            writer.write_many(out)  # a batch is saved only when it is complete
            n_done += len(batch)
            bar.update(len(batch))
    except KeyboardInterrupt:
        print("\nCtrl+C received: progress is saved.")
    bar.close()
    dt = max(time.time() - t0, 1e-9)
    total_done = len(done) + n_done
    print(f"{'stopped' if ctl.stop.is_set() else 'finished'}: {n_done} images this session "
          f"({n_done / dt:.1f} img/s) | total done: {total_done}/{len(df)}")
    if total_done < len(df):
        print("Run the same command again to continue.")
    finish(writer)


# --------------------------------------------------------------------------
# stage 2: devices (screens / prints that contain the bird)
# --------------------------------------------------------------------------
def dev_jobs(rec, quads, a):
    """Device candidates that contain a bird and therefore need verification: [(kind, index, box)]."""
    area = float(rec["w"] * rec["h"])
    bb = rec.get("bird_boxes", [])
    cands = []
    for i, d in enumerate(rec["A"]):
        if d["n"] in SCREEN_NAMES and d["conf"] >= a.dev_conf_a:
            cands.append(("A", i, d["box"], d["conf"]))
    for i, d in enumerate(rec["B"]):
        if d["n"] in SCREEN_B and d["conf"] >= a.dev_conf_b:
            cands.append(("B", i, d["box"], d["conf"]))
    for j, q in enumerate(quads):
        cands.append(("quad", j, q["box"], 0.0))
    jobs = []
    for kind, idx, box, conf in cands:
        frac = box_area(box) / area
        cont = max([contained(b, box) for b in bb], default=0.0)
        if cont >= a.contain_verify and 0 < frac <= 0.98:
            jobs.append((conf, kind, idx, box))
    jobs.sort(key=lambda t: -t[0])
    return [(k, i, b) for _, k, i, b in jobs[:a.max_dev_jobs]]


def load_dev(row, size):
    key, label, arr, small, _ = load_one(row, size, False)
    return key, label, arr, small, (find_quads(arr) if arr is not None else [])


def devices(a):
    from tqdm import tqdm

    df = read_table(a)
    paths = dict(zip(df["photo_id"], df["image_path"]))
    labels = dict(zip(df["photo_id"], df["species_scientific"]))
    recs = load_cache(a.cache)
    if not recs:
        sys.exit("main cache is empty: run `run` first")
    done = set(load_cache(a.dev_cache))
    todo = [(k, str(Path(a.root) / paths[k]), labels[k]) for k, r in recs.items()
            if "error" not in r and k not in done and k in paths]
    if a.limit:
        todo = todo[:a.limit]
    print(f"images to check for screens/prints: {len(todo)} (already done: {len(done)})")
    if not todo:
        print("Nothing to do. Next: `apply`.")
        return

    models = load_models(a, [], "siglip")
    writer = CacheWriter(a.dev_cache)
    ctl = Controller()
    ctl.start()
    bar = tqdm(total=len(todo), desc="devices", unit="img")
    n_done = 0
    try:
        for batch in iter_batches(todo, a.batch, a.workers, lambda r: load_dev(r, models.sig_size)):
            if ctl.checkpoint():
                break
            out = [{"photo_id": x[0], "margin": None, "quads": [], "pd_A": {}, "pd_B": {}} for x in batch if x[2] is None]
            ok = [x for x in batch if x[2] is not None]
            if ok:
                margins = models.siglip_margin([x[3] for x in ok])
                jobs, outs = [], []
                for (k, lab, arr, small, quads), m in zip(ok, margins):
                    ro = {"photo_id": k, "margin": round(m, 4), "quads": quads, "pd_A": {}, "pd_B": {}}
                    for kind, idx, box in dev_jobs(recs[k], quads, a):
                        # the region WITH its surroundings (25% padding) so the bezel / frame is visible
                        jobs.append((len(outs), kind, idx, np.asarray(make_crop(arr, box, 0.25, models.sig_size))))
                    outs.append(ro)
                if jobs:
                    for (ri, kind, idx, _), p in zip(jobs, models.verify_display([j[3] for j in jobs])):
                        p = round(p, 3)
                        if kind == "quad":
                            outs[ri]["quads"][idx]["p_display"] = p
                        elif kind == "A":
                            outs[ri]["pd_A"][str(idx)] = p
                        else:
                            outs[ri]["pd_B"][str(idx)] = p
                out.extend(outs)
            writer.write_many(out)
            n_done += len(batch)
            bar.update(len(batch))
    except KeyboardInterrupt:
        print("\nCtrl+C received: progress is saved.")
    bar.close()
    print(f"{'stopped' if ctl.stop.is_set() else 'finished'}: {n_done} images this session.")
    finish(writer)


# --------------------------------------------------------------------------
# decision rules (pure python; thresholds can be changed without re-running anything)
# --------------------------------------------------------------------------
def robust_stats(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) < 20:
        return None
    med = float(np.median(v))
    return med, max(float(1.4826 * np.median(np.abs(v - med))), 1e-6)


def ds_stats(recs, devs):
    return {"moire": robust_stats([r.get("moire") for r in recs.values()]),
            "margin": robust_stats([d.get("margin") for d in devs.values()])}


def zscore(value, stat):
    return 0.0 if (value is None or stat is None) else (value - stat[0]) / stat[1]


def device_list(rec, hr, dv, a):
    """[(box, source, conf, p_display)] -- p_display is None when the region was not verified."""
    pdA, pdB = (dv or {}).get("pd_A", {}), (dv or {}).get("pd_B", {})
    out = []
    for i, d in enumerate(rec["A"]):
        if d["n"] in SCREEN_NAMES and d["conf"] >= a.dev_conf_a:
            out.append((d["box"], "A", d["conf"], pdA.get(str(i))))
    for i, d in enumerate(rec["B"]):
        if d["n"] in SCREEN_B and d["conf"] >= a.dev_conf_b:
            out.append((d["box"], "B", d["conf"], pdB.get(str(i))))
    if hr:
        out += [(d["box"], "A", d["conf"], None) for d in hr["A"] if d["n"] in SCREEN_NAMES and d["conf"] >= a.dev_conf_a]
        out += [(d["box"], "B", d["conf"], None) for d in hr["B"] if d["n"] in SCREEN_B and d["conf"] >= a.dev_conf_b]
    out += [(q["box"], "quad", 0.0, q.get("p_display")) for q in (dv or {}).get("quads", [])]
    return out


def decide(rec, hr, dv, label, a, st=None):
    w, h = rec["w"], rec["h"]
    img_area = float(w * h)
    A = list(rec["A"]) + (list(hr["A"]) if hr else [])
    B = list(rec["B"]) + (list(hr["B"]) if hr else [])
    S = rec.get("S", {})
    st = st or {}
    bird_boxes = list(rec.get("bird_boxes", [])) + \
                 [d["box"] for d in A if d["n"] == "bird" and d["conf"] >= a.bird_conf]
    remove, review = [], []

    # ---- 1. humans (any size) -----------------------------------------------------------
    a_person = max([d["conf"] for d in A if d["n"] == "person"], default=0.0)
    b_part = max([d["conf"] for d in B if d["n"] in HUMAN_PARTS], default=0.0)
    b_person = max([d["conf"] for d in B if d["n"] in HUMAN_PERSON], default=0.0)
    hand_held = float(S.get("hand_held", 0.0))
    if a_person >= a.person_remove:
        remove.append("human:person")
    elif b_part >= a.part_remove:
        remove.append("human:body_part")
    elif a_person >= a.person_agree and max(b_part, b_person) >= a.agree_conf:
        remove.append("human:two_detectors")
    elif a_person >= a.review_conf or b_part >= a.review_conf or b_person >= a.review_conf \
            or hand_held >= a.hand_held_review:
        review.append("human:weak_evidence")

    # ---- 2a. another species: non-bird animal --------------------------------------------
    animal_score = 0.0
    for d in A + B:
        if not (d["n"] in COCO_ANIMALS or d["n"] in ANIMALS_OW):
            continue
        if d.get("is_bird") or d.get("self") or same_object(d["box"], bird_boxes):
            continue
        pb = d.get("p_bird")  # None: tiny crop or high-res pass -> unverified
        if pb is not None and pb >= a.verify_bird:
            continue
        src_a = d in A
        rv = a.animal_review_a if src_a else a.animal_review_b
        rm = a.animal_remove_a if src_a else a.animal_remove_b
        frac = box_area(d["box"]) / img_area
        near_bird = any(inter_area(d["box"], expand(bb, 0.25, w, h)) > 0 for bb in bird_boxes)
        animal_score = max(animal_score, d["conf"])
        if d["n"] in PREY and near_bird and d["conf"] >= rv:
            review.append("other_species:possible_prey")  # prey near the beak -> always REVIEW
        elif pb is not None and pb <= a.nonbird_max and d["conf"] >= rm and frac >= a.animal_min_frac:
            remove.append("other_species:non_bird_animal")
        elif d["conf"] >= rv:
            review.append("other_species:possible_non_bird_animal")

    # ---- 2b. another species: a second bird species ---------------------------------------
    # Several birds are NOT a problem by themselves. They are a problem only when they are not the
    # same species: the birds are compared with the label AND with each other.
    birds = rec.get("birds", [])
    pair_cos = {(int(i), int(j)): c for i, j, c in rec.get("pair_cos", [])}

    def cos_of(i, j):
        return pair_cos.get((min(i, j), max(i, j)))

    def side(b):
        return min(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1])

    is_match = [b["top1"] == label or b["p_label"] >= a.match_thr for b in birds]
    n_match = sum(is_match)
    big = [i for i, b in enumerate(birds) if side(b) >= a.bio_min_side]  # crops large enough to judge
    species_check = "single_bird" if len(birds) < 2 else "same_species"
    if len(birds) >= 2:
        sure = [i for i in big if birds[i]["p_top1"] >= a.strong_top_p]
        diff_pairs = [(i, j) for x, i in enumerate(sure) for j in sure[x + 1:]
                      if birds[i]["top1"] != birds[j]["top1"]]
        if diff_pairs:  # two birds that are confidently different species
            clearly_diff = [p for p in diff_pairs if cos_of(*p) is None or cos_of(*p) < a.same_cos]
            if clearly_diff:
                remove.append("other_species:other_bird")
                species_check = "different_species"
            else:  # the model says "different" but the crops look alike -> let a human decide
                review.append("other_species:possible_other_bird")
                species_check = "uncertain"
        else:
            nonmatch = [i for i in big if not is_match[i]]
            if n_match >= 1 and nonmatch:
                matches = [i for i, m in enumerate(is_match) if m]
                strong_mis = [i for i in nonmatch if birds[i]["p_label"] < a.strong_mis_p
                              and birds[i]["p_top1"] >= a.strong_top_p and birds[i]["top1"] != label
                              and all(cos_of(i, m) is None or cos_of(i, m) < a.same_cos for m in matches)]
                if strong_mis:
                    remove.append("other_species:other_bird")
                    species_check = "different_species"
                else:
                    review.append("other_species:possible_other_bird")
                    species_check = "uncertain"
            elif n_match == 0 and len(big) >= 2 and len({birds[i]["top1"] for i in big}) >= 2:
                review.append("other_species:possible_other_bird")  # no bird matches, and they differ
                species_check = "uncertain"
            # all birds match the label, or all birds are the same species -> same species: no flag
    label_suspect = int(len(birds) >= 1 and n_match == 0 and any(not m for m in is_match))  # information only

    # ---- 3. re-captured photo: the bird must be INSIDE a verified screen / print region -----------
    mz = zscore(rec.get("moire"), st.get("moire"))
    gz = zscore((dv or {}).get("margin"), st.get("margin"))
    devs = device_list(rec, hr, dv, a)
    best_cont, best_pd = 0.0, None
    for box, src, conf, pd_ in devs:
        frac = box_area(box) / img_area
        cont = max([contained(bb, box) for bb in bird_boxes], default=0.0)
        agree = any(s2 != src and iou(box, b2) >= 0.4 for b2, s2, _, _ in devs)
        strong_src = (src != "quad" and conf >= a.dev_strong_conf) or agree
        if cont >= 0.85 and frac < a.dev_full_frac:
            best_cont = max(best_cont, cont)
            if pd_ is not None:
                best_pd = pd_ if best_pd is None else max(best_pd, pd_)
        if frac >= a.dev_full_frac:  # the "screen" is the whole picture: nothing to verify against
            if src != "quad" and conf >= a.dev_strong_conf and cont >= 0.85:
                review.append("recapture:full_frame_device")
            continue
        if cont >= a.contain_remove:
            if pd_ is not None and pd_ >= a.display_remove and (strong_src or mz >= a.moire_support):
                remove.append("recapture:bird_inside_screen_or_print")
            elif strong_src or (pd_ is not None and pd_ >= a.display_review):
                review.append("recapture:bird_in_device_unconfirmed")
        elif cont >= a.contain_review and strong_src and (pd_ is None or pd_ >= a.display_review):
            review.append("recapture:bird_partly_in_device")
    if gz >= a.margin_z and mz >= a.moire_support:  # both global signals are extreme at the same time
        review.append("recapture:global_and_pattern_outlier")

    verdict = "REMOVE" if remove else ("REVIEW" if review else "KEEP")
    return {
        "img_check": verdict,
        "img_check_reasons": ";".join(sorted(set(remove)) + sorted(set(review))),
        "chk_human_score": round(max(a_person, b_part, b_person), 3),
        "chk_hand_held": round(hand_held, 3),
        "chk_animal_score": round(animal_score, 3),
        "chk_recapture": round(float(S.get("recapture", 0.0)), 3),  # reference only, not used for decisions
        "chk_device_contain": round(best_cont, 3),
        "chk_p_display": None if best_pd is None else round(best_pd, 3),
        "chk_moire_z": round(mz, 2),
        "chk_margin_z": round(gz, 2),
        "chk_n_birds": int(rec.get("n_birds", len(bird_boxes))),
        "chk_bird_species_check": species_check,
        "chk_label_suspect": label_suspect,
    }


def weak_signal(rec, a, st):
    """Images that deserve the slow high-resolution pass."""
    S = rec.get("S", {})
    if S.get("hand_held", 0) >= a.hi_hand:
        return True
    for d in list(rec["A"]) + list(rec["B"]):
        if d.get("self") or d.get("is_bird"):
            continue
        if d["n"] == "person" or d["n"] in HUMAN_PARTS:
            if d["conf"] >= a.hi_human:
                return True
        elif d["n"] in ANIMALS_OW or d["n"] in COCO_ANIMALS:
            if d["conf"] >= a.hi_animal:
                return True
    return False


# --------------------------------------------------------------------------
# stage 3: high-resolution + tiled pass on suspicious images
# --------------------------------------------------------------------------
def tile_views(arr, overlap=0.15):
    h, w = arr.shape[:2]
    tw, th = int(w * (0.5 + overlap)), int(h * (0.5 + overlap))
    return [(x, y, arr[y:y + th, x:x + tw]) for y in (0, h - th) for x in (0, w - tw)]


def hires(a):
    from tqdm import tqdm

    df = read_table(a)
    labels = dict(zip(df["photo_id"], df["species_scientific"]))
    paths = dict(zip(df["photo_id"], df["image_path"]))
    recs = load_cache(a.cache)
    if not recs:
        sys.exit("main cache is empty: run `run` first")
    devs = load_cache(a.dev_cache)
    st = ds_stats(recs, devs)
    done = set(load_cache(a.hires_cache))
    todo = []
    for k, rec in recs.items():
        if "error" in rec or k in done or k not in labels:
            continue
        verdict = decide(rec, None, devs.get(k), labels[k], a, st)["img_check"]
        if verdict == "REMOVE":
            continue  # already clear: no need to look harder
        if verdict == "REVIEW" or weak_signal(rec, a, st):
            todo.append((k, str(Path(a.root) / paths[k]), labels[k]))
    if a.limit:
        todo = todo[:a.limit]
    print(f"suspicious images for the high-res pass: {len(todo)} (already done: {len(done)})")
    if not todo:
        print("Nothing to do. Next: `apply`.")
        return

    models = load_models(a, [], "dets")
    writer = CacheWriter(a.hires_cache)
    ctl = Controller()
    ctl.start()
    bar = tqdm(total=len(todo), desc="hires", unit="img")
    n_done = 0
    try:
        for batch in iter_batches(todo, a.hires_batch, a.workers, lambda r: load_one(r, 0, False)):
            if ctl.checkpoint():
                break
            out = [{"photo_id": x[0], "A": [], "B": []} for x in batch if x[2] is None]
            ok = [x for x in batch if x[2] is not None]
            if ok:
                full = [x[2] for x in ok]
                views, owner = [], []
                for i, x in enumerate(ok):
                    for (ox, oy, t) in tile_views(x[2]):
                        views.append(np.ascontiguousarray(t))
                        owner.append((i, ox, oy))
                res = [{"photo_id": x[0], "A": [], "B": []} for x in ok]
                for kind, fn in (("A", models.det_A), ("B", models.det_B)):
                    conf = a.a_conf if kind == "A" else a.b_conf
                    for i, dets in enumerate(fn(full, a.hires_imgsz, conf)):
                        res[i][kind].extend(dets)
                    for (i, ox, oy), dets in zip(owner, fn(views, a.tile_imgsz, conf)):
                        for d in dets:
                            b = d["box"]
                            d["box"] = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
                            res[i][kind].append(d)
                out.extend(res)
            writer.write_many(out)
            n_done += len(batch)
            bar.update(len(batch))
    except KeyboardInterrupt:
        print("\nCtrl+C received: progress is saved.")
    bar.close()
    print(f"{'stopped' if ctl.stop.is_set() else 'finished'}: {n_done} images this session.")
    finish(writer)


# --------------------------------------------------------------------------
# apply / stats / review
# --------------------------------------------------------------------------
def apply(a):
    recs = load_cache(a.cache)
    if not recs:
        sys.exit(f"cache is empty or missing (schema v{SCHEMA}): {a.cache}")
    hrs, devs = load_cache(a.hires_cache), load_cache(a.dev_cache)
    if not devs:
        print("WARNING: no `devices` results found: screens/prints are NOT verified. Run `devices` first.")
    st = ds_stats(recs, devs)
    csv = Path(a.csv)
    df = pd.read_csv(csv, low_memory=False)
    df = df.drop(columns=[c for c in CHECK_COLS if c in df.columns])
    out = []
    for pid, lab in zip(df["photo_id"].astype(str), df["species_scientific"]):
        r = recs.get(pid)
        out.append({} if (r is None or "error" in r) else decide(r, hrs.get(pid), devs.get(pid), lab, a, st))
    res = pd.DataFrame(out, index=df.index).reindex(columns=CHECK_COLS)
    for c in ["chk_n_birds", "chk_label_suspect"]:
        res[c] = res[c].astype("Int64")
    df = pd.concat([df, res], axis=1)

    bak = csv.with_name(csv.stem + "_backup_before_imagecheck.csv")
    if not bak.exists():
        shutil.copy2(csv, bak)
        print(f"backup saved: {bak}")
    tmp = csv.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv)

    print(f"verdicts written to {csv}: {int(df['img_check'].notna().sum())} of {len(df)} rows")
    print(df["img_check"].value_counts(dropna=False).to_string())
    reasons = df["img_check_reasons"].dropna().str.split(";").explode()
    reasons = reasons[reasons != ""]
    if len(reasons):
        print("\nreasons:")
        print(reasons.value_counts().to_string())
    print(f"\nhigh-res pass: {len(hrs)} images | device check: {len(devs)} images")


def stats(a):
    recs, devs, hrs = load_cache(a.cache), load_cache(a.dev_cache), load_cache(a.hires_cache)
    if not recs:
        sys.exit("cache is empty")
    st = ds_stats(recs, devs)
    df = read_table(a)
    labels = dict(zip(df["photo_id"], df["species_scientific"]))
    n = len(recs)
    rc = np.array([r["S"].get("recapture", 0) for r in recs.values() if "S" in r])
    print(f"images in cache: {n} | device check done for: {len(devs)}")
    print("\nglobal SigLIP 'recapture' softmax score (NOT used for decisions any more):")
    for t in (0.35, 0.5, 0.8, 0.95):
        print(f"  score >= {t}: {(rc >= t).sum()} images ({100 * (rc >= t).mean():.1f}%)")
    print("  -> a signal that fires on that many images cannot be evidence of a rare event.")
    cnt = {"any_device": 0, "bird_inside": 0, "verified_display>=0.7": 0, "quad_only": 0}
    for k, r in recs.items():
        if "error" in r:
            continue
        dl = device_list(r, hrs.get(k), devs.get(k), a)
        if not dl:
            continue
        cnt["any_device"] += 1
        inside = [(b, s, c, p) for b, s, c, p in dl
                  if max([contained(bb, b) for bb in r.get("bird_boxes", [])], default=0) >= 0.85]
        if inside:
            cnt["bird_inside"] += 1
            if any(p is not None and p >= 0.7 for *_, p in inside):
                cnt["verified_display>=0.7"] += 1
            if all(s == "quad" for _, s, _, _ in inside):
                cnt["quad_only"] += 1
    print("\ndevice evidence (images):")
    for k, v in cnt.items():
        print(f"  {k}: {v}")
    ver = {"REMOVE": 0, "REVIEW": 0, "KEEP": 0}
    rr = {}
    for k, r in recs.items():
        if "error" in r or k not in labels:
            continue
        d = decide(r, hrs.get(k), devs.get(k), labels[k], a, st)
        ver[d["img_check"]] += 1
        for x in d["img_check_reasons"].split(";"):
            if x.startswith("recapture"):
                rr[x] = rr.get(x, 0) + 1
    multi = {}
    for k, r in recs.items():
        if "error" in r or k not in labels:
            continue
        if len(r.get("birds", [])) >= 2:
            c = decide(r, hrs.get(k), devs.get(k), labels[k], a, st)["chk_bird_species_check"]
            multi[c] = multi.get(c, 0) + 1
    print("\nimages with 2+ birds, by species check:", multi if multi else "none")
    print("\nverdicts:", ver)
    print("recapture reasons:", rr if rr else "none")


def review(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps

    recs, hrs, devs = load_cache(a.cache), load_cache(a.hires_cache), load_cache(a.dev_cache)
    st = ds_stats(recs, devs)
    df = pd.read_csv(a.csv, usecols=["photo_id", "image_path", "species_scientific"])
    df["photo_id"] = df["photo_id"].astype(str)
    rows = []
    for k, ip, lab in zip(df["photo_id"], df["image_path"], df["species_scientific"]):
        r = recs.get(k)
        if r is None or "error" in r:
            continue
        d = decide(r, hrs.get(k), devs.get(k), lab, a, st)
        if d["img_check"] != a.category.upper():
            continue
        if a.reason and a.reason not in d["img_check_reasons"]:
            continue
        rows.append((k, ip, lab, d))
    print(f"{len(rows)} images with category={a.category}" + (f", reason contains '{a.reason}'" if a.reason else ""))
    if not rows:
        return
    if a.shuffle:
        np.random.default_rng(a.seed).shuffle(rows)
    page = rows[a.offset:a.offset + a.n]
    cols = 4
    nrows = int(np.ceil(len(page) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (k, ip, lab, d) in zip(axes, page):
        im = ImageOps.exif_transpose(Image.open(Path(a.root) / ip).convert("RGB"))
        ax.imshow(im)
        r, hr, dv = recs[k], hrs.get(k), devs.get(k)
        for bb in r.get("bird_boxes", []):
            ax.add_patch(patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, lw=1.4, ec="lime"))
        for det in list(r["A"]) + list(r["B"]) + (list(hr["A"]) + list(hr["B"]) if hr else []):
            n = det["n"]
            if n == "bird" or det.get("self") or det.get("is_bird"):
                continue
            if n == "person" or n in HUMAN_PARTS:
                c, t = "red", "human"
            elif n in SCREEN_NAMES or n in SCREEN_B:
                c, t = "magenta", "screen/print"
            else:
                c, t = "deepskyblue", "animal" + (f" p_bird={det['p_bird']:.2f}" if "p_bird" in det else "")
            x1, y1, x2, y2 = det["box"]
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.6, ec=c))
            ax.text(x1, y1, f"{t} {det['conf']:.2f}", color=c, fontsize=7, backgroundcolor="black")
        for q in (dv or {}).get("quads", []):
            x1, y1, x2, y2 = q["box"]
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.2, ec="orange", ls="--"))
            if "p_display" in q:
                ax.text(x2, y2, f"quad p_display={q['p_display']:.2f}", color="orange", fontsize=7,
                        backgroundcolor="black", ha="right", va="bottom")
        ax.set_title(f"{k} | {lab}\n{d['img_check_reasons'] or '-'}", fontsize=7)
        ax.axis("off")
    for ax in axes[len(page):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(a.out, dpi=105)
    print(f"saved {a.out} ({len(page)} images; offset={a.offset}).  "
          "red=human, blue=animal, magenta=screen/print, orange dashed=rectangle candidate, green=bird")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--csv", default="../inat_birds/dataset.csv")
        p.add_argument("--root", default="../inat_birds", help="folder that image_path is relative to")
        p.add_argument("--cache", default="../inat_birds/imagecheck_v2_cache.jsonl")
        p.add_argument("--hires-cache", default="../inat_birds/imagecheck_v2_hires_cache.jsonl")
        p.add_argument("--dev-cache", default="../inat_birds/imagecheck_v3_devices_cache.jsonl")
        # humans
        p.add_argument("--person-remove", type=float, default=0.60)
        p.add_argument("--part-remove", type=float, default=0.50)
        p.add_argument("--person-agree", type=float, default=0.35)
        p.add_argument("--agree-conf", type=float, default=0.30)
        p.add_argument("--review-conf", type=float, default=0.15)
        p.add_argument("--hand-held-review", type=float, default=0.50)
        # other animals
        p.add_argument("--verify-bird", type=float, default=0.50)
        p.add_argument("--nonbird-max", type=float, default=0.20)
        p.add_argument("--verify-min-side", type=int, default=48)
        p.add_argument("--animal-remove-a", type=float, default=0.60)
        p.add_argument("--animal-remove-b", type=float, default=0.50)
        p.add_argument("--animal-review-a", type=float, default=0.25)
        p.add_argument("--animal-review-b", type=float, default=0.20)
        p.add_argument("--animal-min-frac", type=float, default=0.01)
        # second bird species
        p.add_argument("--bird-conf", type=float, default=0.25)
        p.add_argument("--match-thr", type=float, default=0.20)
        p.add_argument("--mismatch-thr", type=float, default=0.05)
        p.add_argument("--strong-mis-p", type=float, default=0.02)
        p.add_argument("--strong-top-p", type=float, default=0.70)
        p.add_argument("--bio-min-side", type=int, default=64)
        p.add_argument("--same-cos", type=float, default=0.85,
                       help="birds of one image whose crop similarity is >= this are treated as the same species "
                            "(a REMOVE becomes REVIEW)")
        # re-captured photo (screen / print regions that contain the bird)
        p.add_argument("--dev-conf-a", type=float, default=0.30, help="COCO tv/laptop/phone/book confidence")
        p.add_argument("--dev-conf-b", type=float, default=0.25, help="open-vocab screen/print confidence")
        p.add_argument("--dev-strong-conf", type=float, default=0.50, help="detector confidence counted as strong")
        p.add_argument("--contain-verify", type=float, default=0.50, help="bird fraction inside region to bother verifying")
        p.add_argument("--contain-remove", type=float, default=0.85, help="bird fraction inside region for REMOVE/REVIEW")
        p.add_argument("--contain-review", type=float, default=0.60)
        p.add_argument("--dev-full-frac", type=float, default=0.90, help="region covering this much of the image = whole picture")
        p.add_argument("--display-remove", type=float, default=0.70, help="verifier P(display/print) needed for REMOVE")
        p.add_argument("--display-review", type=float, default=0.40)
        p.add_argument("--moire-support", type=float, default=4.0, help="robust z of the periodic-pattern score (supporting)")
        p.add_argument("--margin-z", type=float, default=4.0, help="robust z of the global SigLIP margin (supporting)")
        p.add_argument("--max-dev-jobs", type=int, default=6)
        # second-pass selection
        p.add_argument("--hi-human", type=float, default=0.10)
        p.add_argument("--hi-animal", type=float, default=0.12)
        p.add_argument("--hi-hand", type=float, default=0.20)

    def models_args(p):
        p.add_argument("--det-weights", default="yolo26m.pt")
        p.add_argument("--ow-weights", default="yolov8s-worldv2.pt")
        p.add_argument("--siglip-model", default="ViT-B-16-SigLIP")
        p.add_argument("--siglip-pretrained", default="webli")
        p.add_argument("--bio-model", default="hf-hub:imageomics/bioclip-2",
                       help="use hf-hub:imageomics/bioclip for the lighter first version")
        p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
        p.add_argument("--a-conf", type=float, default=0.15)
        p.add_argument("--b-conf", type=float, default=0.10)
        p.add_argument("--bio-batch", type=int, default=64)
        p.add_argument("--min-side", type=int, default=40)
        p.add_argument("--min-area-frac", type=float, default=0.003)
        p.add_argument("--max-birds", type=int, default=8)
        p.add_argument("--limit", type=int, default=0, help="process only N images (quick test)")

    r = sub.add_parser("run", help="detectors + species check (resumable; p/r/s controls)")
    common(r)
    models_args(r)
    r.add_argument("--imgsz", type=int, default=640)
    r.add_argument("--batch", type=int, default=32)
    r.set_defaults(fn=run)

    d = sub.add_parser("devices", help="screens/prints: regions, bird-inside check, verification (resumable; p/r/s controls)")
    common(d)
    models_args(d)
    d.add_argument("--batch", type=int, default=32)
    d.set_defaults(fn=devices)

    h = sub.add_parser("hires", help="high-res + tiled pass on suspicious images (resumable; p/r/s controls)")
    common(h)
    models_args(h)
    h.add_argument("--hires-imgsz", type=int, default=1280)
    h.add_argument("--tile-imgsz", type=int, default=960)
    h.add_argument("--hires-batch", type=int, default=8)
    h.set_defaults(fn=hires)

    p = sub.add_parser("apply", help="write verdict columns into dataset.csv")
    common(p)
    p.set_defaults(fn=apply)

    s = sub.add_parser("stats", help="how selective is each signal?")
    common(s)
    s.set_defaults(fn=stats)

    v = sub.add_parser("review", help="save a picture grid to inspect decisions")
    common(v)
    v.add_argument("--category", default="REVIEW", choices=["REVIEW", "REMOVE", "KEEP", "review", "remove", "keep"])
    v.add_argument("--reason", default="", help="only reasons containing this text, e.g. human, recapture, other_species")
    v.add_argument("--n", type=int, default=24)
    v.add_argument("--offset", type=int, default=0)
    v.add_argument("--shuffle", action="store_true")
    v.add_argument("--seed", type=int, default=42)
    v.add_argument("--out", default="imagecheck_review.png")
    v.set_defaults(fn=review)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
