"""
Brownian Motion Camera – Particle Tracker
==========================================
Berkeley Advanced Lab – BMC Experiment
Python/Tkinter implementation using the IDS peak SDK.

Requirements
------------
    pip install ids_peak ids_peak_ipl numpy opencv-python Pillow

The IDS peak SDK (transport-layer .cti files) must be installed separately:
    https://en.ids-imaging.com/ids-peak.html

If no IDS camera is detected the app starts in Demo mode automatically.
Demo mode can also be toggled at any time from the Camera panel in the GUI.

Windows / Linux both supported.
"""

import csv
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ── IDS peak optional import ──────────────────────────────────────────────────
try:
    from ids_peak import ids_peak
    from ids_peak import ids_peak_ipl_extension
    _IDS_AVAILABLE = True
except ImportError:
    _IDS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
BUFFER_COUNT   = 5
DISPLAY_W      = 640
DISPLAY_H      = 512
MAX_TRAIL      = 60      # frames of trail to keep per particle
DEMO_N         = 10      # synthetic particles in demo mode
EXP_ALPHA      = 0.05    # exponential background smoothing (5 % new frame)
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# Colours (BGR)
C_TRAIL        = (0,   200, 255)
C_TRACKED      = (0,   80,  220)   # red-ish in BGR→RGB display
C_UNTRACKED    = (200, 120,  0)    # blue-ish
C_TEXT         = (255, 255, 255)
C_ROI          = (0,   255,   0)


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Particle:
    pid:       int
    x:         float
    y:         float
    area:      float
    tracked:   bool = False
    trail:     deque = field(default_factory=lambda: deque(maxlen=MAX_TRAIL))
    # statistics
    t0:        float = 0.0          # time of first sighting
    positions: list  = field(default_factory=list)   # [(t, x, y), ...]
    dr2:       list  = field(default_factory=list)   # displacement² from origin


# ─────────────────────────────────────────────────────────────────────────────
#  IDS Camera back-end
# ─────────────────────────────────────────────────────────────────────────────
class IDSCamera:
    """Thin wrapper around the IDS peak generic SDK for one camera."""

    def __init__(self, force_demo: bool = False):
        self._device        = None
        self._data_stream   = None
        self._remote_nm     = None
        self._running       = False
        self._model         = "DEMO"

        if force_demo or not _IDS_AVAILABLE:
            return

        ids_peak.Library.Initialize()
        dm = ids_peak.DeviceManager.Instance()
        dm.Update()
        if dm.Devices().empty():
            print("[INFO] No IDS camera found – falling back to demo mode.")
            return

        self._device    = dm.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        self._remote_nm = self._device.RemoteDevice().NodeMaps()[0]
        self._data_stream = self._device.DataStreams()[0].OpenDataStream()
        payload = self._remote_nm.FindNode("PayloadSize").Value()
        for _ in range(BUFFER_COUNT):
            buf = self._data_stream.AllocAndAnnounceBuffer(payload)
            self._data_stream.QueueBuffer(buf)
        try:
            self._model = self._remote_nm.FindNode("DeviceModelName").Value()
        except Exception:
            self._model = "IDS Camera"

    @property
    def model(self):
        return self._model

    @property
    def is_real(self):
        return self._device is not None

    def start(self):
        self._running = True
        if not self.is_real:
            return
        self._remote_nm.FindNode("TLParamsLocked").SetValue(1)
        self._data_stream.StartAcquisition(
            ids_peak.AcquisitionStartMode_Default,
            ids_peak.DataStream.INFINITE_NUMBER)
        self._remote_nm.FindNode("AcquisitionStart").Execute()
        self._remote_nm.FindNode("AcquisitionStart").WaitUntilDone()

    def stop(self):
        self._running = False
        if not self.is_real:
            return
        try:
            self._remote_nm.FindNode("AcquisitionStop").Execute()
            self._remote_nm.FindNode("AcquisitionStop").WaitUntilDone()
            self._data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            self._data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for buf in self._data_stream.AnnouncedBuffers():
                self._data_stream.RevokeBuffer(buf)
            self._remote_nm.FindNode("TLParamsLocked").SetValue(0)
        except Exception as e:
            print(f"[WARN] stop: {e}")

    def grab_frame(self) -> Optional[np.ndarray]:
        if not self._running:
            return None
        if not self.is_real:
            return _demo_frame()
        try:
            buf = self._data_stream.WaitForFinishedBuffer(1000)
            ipl  = ids_peak_ipl_extension.BufferToImage(buf)
            conv = ipl.ConvertTo(ids_peak_ipl_extension.PixelFormatName_BGR8)
            w, h = conv.Width(), conv.Height()
            frame = np.frombuffer(conv.Data(), dtype=np.uint8).reshape(h, w, 3).copy()
            self._data_stream.QueueBuffer(buf)
            return frame
        except Exception as e:
            print(f"[WARN] grab_frame: {e}")
            return None

    def set_exposure(self, us: float):
        if not self.is_real:
            return
        try:
            node = self._remote_nm.FindNode("ExposureTime")
            mn   = node.Minimum()
            mx   = node.Maximum()
            node.SetValue(float(np.clip(us, mn, mx)))
        except Exception as e:
            print(f"[WARN] set_exposure: {e}")

    def set_gain(self, gain: float):
        if not self.is_real:
            return
        try:
            node = self._remote_nm.FindNode("Gain")
            mn   = node.Minimum()
            mx   = node.Maximum()
            node.SetValue(float(np.clip(gain, mn, mx)))
        except Exception as e:
            print(f"[WARN] set_gain: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Demo frame generator (synthetic Brownian particles)
# ─────────────────────────────────────────────────────────────────────────────
_demo = None

def _demo_frame() -> np.ndarray:
    global _demo
    if _demo is None:
        rng = np.random.default_rng(42)
        _demo = {
            "pos": rng.uniform(40, [DISPLAY_W-40, DISPLAY_H-40], (DEMO_N, 2)),
            "rng": rng,
            "t":   0,
        }
    d = _demo
    d["t"] += 1
    # Brownian step
    d["pos"] += d["rng"].normal(0, 1.5, (DEMO_N, 2))
    d["pos"][:, 0] = np.clip(d["pos"][:, 0], 20, DISPLAY_W - 20)
    d["pos"][:, 1] = np.clip(d["pos"][:, 1], 20, DISPLAY_H - 20)

    frame = np.full((DISPLAY_H, DISPLAY_W, 3), 18, dtype=np.uint8)
    noise = d["rng"].integers(0, 20, (DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)

    for i, (x, y) in enumerate(d["pos"]):
        r = int(5 + 2 * np.sin(d["t"] / 20 + i))
        cv2.circle(frame, (int(x), int(y)), r, (210, 230, 255), -1)
        cv2.GaussianBlur(frame[
            max(0, int(y)-r-4):int(y)+r+4,
            max(0, int(x)-r-4):int(x)+r+4],
            (5, 5), 0,
            frame[
                max(0, int(y)-r-4):int(y)+r+4,
                max(0, int(x)-r-4):int(x)+r+4])
    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Blob finder  (Berkeley BMC algorithm)
# ─────────────────────────────────────────────────────────────────────────────
class BlobFinder:
    """
    Implements the Berkeley BMC blob-finding algorithm:
      1. Exponential background subtraction
      2. Z-score thresholding
      3. Connected component analysis
      4. Centroiding (intensity-weighted centre-of-mass)
    """

    def __init__(self):
        self._bg: Optional[np.ndarray] = None   # running average background

    def reset(self):
        self._bg = None

    def process(self,
                gray: np.ndarray,
                zscore_thresh: float = 2.5,
                min_area: int = 20,
                max_area: int = 2000,
                blur: int = 3,
                subtract_bg: bool = True
                ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """
        Returns
        -------
        post_img  : uint8 grayscale showing what the blob-finder sees
        vis_img   : BGR image with circles drawn on subtracted view
        blobs     : list of dicts  {x, y, area, bbox}
        """
        # ── 1. Background subtraction (exponential smoothing) ─────────────
        f = gray.astype(np.float32)
        if self._bg is None:
            self._bg = f.copy()
        else:
            self._bg = EXP_ALPHA * f + (1.0 - EXP_ALPHA) * self._bg

        if subtract_bg:
            subtracted = np.clip(f - self._bg, 0, 255).astype(np.uint8)
        else:
            subtracted = gray.copy()

        # ── 2. Blur ──────────────────────────────────────────────────────
        k = blur if blur % 2 == 1 else blur + 1
        blurred = cv2.GaussianBlur(subtracted, (k, k), 0)

        # ── 3. Z-score threshold ─────────────────────────────────────────
        mu  = float(blurred.mean())
        std = float(blurred.std()) + 1e-6
        thresh_val = mu + zscore_thresh * std
        thresh_val = np.clip(thresh_val, 1, 254)
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # ── 4. Connected components + centroiding ────────────────────────
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8)

        blobs = []
        for lbl in range(1, n_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue
            x0 = stats[lbl, cv2.CC_STAT_LEFT]
            y0 = stats[lbl, cv2.CC_STAT_TOP]
            w  = stats[lbl, cv2.CC_STAT_WIDTH]
            h  = stats[lbl, cv2.CC_STAT_HEIGHT]

            # Intensity-weighted centroid (sub-pixel accuracy)
            mask    = (labels == lbl)
            ys, xs  = np.where(mask)
            weights = blurred[ys, xs].astype(np.float64)
            W       = weights.sum()
            if W == 0:
                cx, cy = float(xs.mean()), float(ys.mean())
            else:
                cx = float((xs * weights).sum() / W)
                cy = float((ys * weights).sum() / W)

            blobs.append({"x": cx, "y": cy, "area": area,
                          "bbox": (x0, y0, w, h)})

        # post-processing visualisation (grayscale → BGR for consistency)
        post_vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        return cleaned, post_vis, blobs


# ─────────────────────────────────────────────────────────────────────────────
#  Particle tracker  (nearest-neighbour with cluster-level global optimisation)
# ─────────────────────────────────────────────────────────────────────────────
class ParticleTracker:
    """
    Implements the Berkeley BMC nearest-neighbour tracker with per-cluster
    global optimisation (factorial search within small groups).
    """

    def __init__(self):
        self._tracks: dict[int, Particle] = {}
        self._next_id   = 0
        self._t_origin  = None   # wall-clock time of first frame
        self._frame_t   = 0.0

    def reset(self):
        self._tracks   = {}
        self._next_id  = 0
        self._t_origin = None

    def update(self, blobs: list[dict], t: float, reach: float = 30.0
               ) -> list[Particle]:
        """
        Match blobs to existing tracks and return the full list of
        current Particle objects.
        """
        if self._t_origin is None:
            self._t_origin = t
        self._frame_t = t - self._t_origin

        prev = list(self._tracks.values())
        prev_xy = np.array([[p.x, p.y] for p in prev]) if prev else np.empty((0, 2))
        new_xy  = np.array([[b["x"], b["y"]] for b in blobs]) if blobs else np.empty((0, 2))

        matched_prev = set()
        matched_new  = set()

        # ── Build distance matrix & cluster ──────────────────────────────
        if len(prev_xy) and len(new_xy):
            dist = np.sqrt(
                ((prev_xy[:, None, :] - new_xy[None, :, :]) ** 2).sum(axis=2))
            # Candidate pairs within reach
            pi_cands, ni_cands = np.where(dist < reach)

            # Group into independent clusters
            from collections import defaultdict
            adj_p = defaultdict(set)
            adj_n = defaultdict(set)
            for pi, ni in zip(pi_cands, ni_cands):
                adj_p[pi].add(ni)
                adj_n[ni].add(pi)

            visited_p, visited_n = set(), set()
            clusters = []

            def collect(pi):
                stack_p = [pi]
                cp, cn = [], []
                while stack_p:
                    p = stack_p.pop()
                    if p in visited_p:
                        continue
                    visited_p.add(p)
                    cp.append(p)
                    for n in adj_p[p]:
                        if n not in visited_n:
                            visited_n.add(n)
                            cn.append(n)
                            for pp in adj_n[n]:
                                stack_p.append(pp)
                return cp, cn

            for pi in range(len(prev_xy)):
                if pi not in visited_p and pi in adj_p:
                    cp, cn = collect(pi)
                    if cp and cn:
                        clusters.append((cp, cn))

            # ── Per-cluster optimisation ─────────────────────────────────
            import itertools
            for cp, cn in clusters:
                if len(cp) <= 1 and len(cn) <= 1:
                    # trivial
                    pi, ni = cp[0], cn[0]
                    matched_prev.add(pi)
                    matched_new.add(ni)
                    p = prev[pi]
                    p.x, p.y  = blobs[ni]["x"], blobs[ni]["y"]
                    p.area     = blobs[ni]["area"]
                    p.tracked  = True
                    p.trail.append((p.x, p.y))
                    _record(p, self._frame_t)
                    continue

                # factorial search if cluster is small
                if len(cp) <= 6 and len(cn) <= 6:
                    best_cost, best_perm = np.inf, None
                    for perm in itertools.permutations(cn):
                        cost = sum(dist[pi, ni] ** 2
                                   for pi, ni in zip(cp, perm[:len(cp)]))
                        if cost < best_cost:
                            best_cost, best_perm = cost, perm
                    for pi, ni in zip(cp, best_perm[:len(cp)]):
                        matched_prev.add(pi)
                        matched_new.add(ni)
                        p = prev[pi]
                        p.x, p.y  = blobs[ni]["x"], blobs[ni]["y"]
                        p.area     = blobs[ni]["area"]
                        p.tracked  = True
                        p.trail.append((p.x, p.y))
                        _record(p, self._frame_t)
                else:
                    # fall back to greedy for large clusters
                    sub = dist[np.ix_(cp, cn)]
                    order = np.argsort(sub.min(axis=1))
                    used_n = set()
                    for pi_local in order:
                        pi = cp[pi_local]
                        row = sub[pi_local].copy()
                        row[list(used_n)] = np.inf
                        ni_local = int(row.argmin())
                        if row[ni_local] < reach:
                            ni = cn[ni_local]
                            used_n.add(ni_local)
                            matched_prev.add(pi)
                            matched_new.add(ni)
                            p = prev[pi]
                            p.x, p.y  = blobs[ni]["x"], blobs[ni]["y"]
                            p.area     = blobs[ni]["area"]
                            p.tracked  = True
                            p.trail.append((p.x, p.y))
                            _record(p, self._frame_t)

        # Mark unmatched old tracks as lost (keep for trail display one frame)
        for i, p in enumerate(prev):
            if i not in matched_prev:
                p.tracked = False

        # Remove tracks that have been lost for > 5 frames
        stale = [pid for pid, p in self._tracks.items()
                 if not p.tracked and len(p.trail) > 0 and
                 p.trail[-1] == p.trail[-1]]  # placeholder; real staleness below

        # Proper stale removal via a frame counter
        for pid in list(self._tracks.keys()):
            p = self._tracks[pid]
            if not p.tracked:
                if not hasattr(p, "_lost_frames"):
                    p._lost_frames = 0
                p._lost_frames += 1
                if p._lost_frames > 5:
                    del self._tracks[pid]
            else:
                p._lost_frames = 0

        # New particles
        for ni, blob in enumerate(blobs):
            if ni not in matched_new:
                pid = self._next_id
                self._next_id += 1
                p = Particle(pid=pid, x=blob["x"], y=blob["y"],
                             area=blob["area"], tracked=False,
                             t0=self._frame_t)
                p.trail.append((p.x, p.y))
                p.positions.append((self._frame_t, p.x, p.y))
                self._tracks[pid] = p

        return list(self._tracks.values())

    @property
    def tracks(self) -> dict:
        return self._tracks

    def diffusion_coeff(self, pid: int) -> Optional[float]:
        """
        D from slope of ⟨r²⟩ vs t (origin-fixed least squares).
        Returns D in px²/s, or None if insufficient data.
        """
        p = self._tracks.get(pid)
        if p is None or len(p.positions) < 5:
            return None
        arr = np.array(p.positions)
        t    = arr[:, 0]
        x0, y0 = arr[0, 1], arr[0, 2]
        dr2  = (arr[:, 1] - x0) ** 2 + (arr[:, 2] - y0) ** 2
        # Force zero intercept: slope = (t · r²) / (t · t)
        slope = float(np.dot(t, dr2) / np.dot(t, t))
        return slope / 4.0  # 2D: ⟨r²⟩ = 4Dt


def _record(p: Particle, t: float):
    p.positions.append((t, p.x, p.y))
    if len(p.positions) > 1:
        x0, y0 = p.positions[0][1], p.positions[0][2]
        dr2 = (p.x - x0) ** 2 + (p.y - y0) ** 2
        p.dr2.append((t, dr2))


# ─────────────────────────────────────────────────────────────────────────────
#  Main GUI Application
# ─────────────────────────────────────────────────────────────────────────────
class BrownianApp(tk.Tk):
    # ── init ──────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.title("Brownian Motion Camera – Particle Tracker  |  Berkeley Advanced Lab")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)

        # Demo mode: start in demo if no real camera is available
        no_real_cam = (not _IDS_AVAILABLE) or self._probe_no_camera()
        self._demo_mode     = tk.BooleanVar(value=no_real_cam)

        self._camera       = IDSCamera(force_demo=no_real_cam)
        self._blob_finder  = BlobFinder()
        self._tracker      = ParticleTracker()

        self._running       = False
        self._find_blobs    = tk.BooleanVar(value=False)
        self._track_parts   = tk.BooleanVar(value=False)
        self._subtract_bg   = tk.BooleanVar(value=True)
        self._recording     = tk.BooleanVar(value=False)

        # ROI in the pass-through image (x, y, w, h) normalised to canvas
        self._roi = [0.15, 0.10, 0.70, 0.80]   # fractions of image
        self._drag_state = None

        # Frame queue (producer → consumer)
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._recorded_frames: list[np.ndarray] = []

        # FPS tracking
        self._fps_buf = deque(maxlen=30)
        self._last_t  = time.perf_counter()

        # Save directory
        self._save_dir = tk.StringVar(value=str(Path.home() / "Documents"))

        self._build_ui()
        self._update_status(f"Camera: {self._camera.model}  |  "
                            f"{'DEMO' if self._demo_mode.get() else 'Real camera'} mode")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    @staticmethod
    def _probe_no_camera() -> bool:
        """Return True if no IDS camera is connected (quick non-opening check)."""
        if not _IDS_AVAILABLE:
            return True
        try:
            ids_peak.Library.Initialize()
            dm = ids_peak.DeviceManager.Instance()
            dm.Update()
            result = dm.Devices().empty()
            # Do NOT close the library here – IDSCamera.__init__ will reuse it
            return result
        except Exception:
            return True

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Colour palette ────────────────────────────────────────────────
        BG  = "#1e1e2e"
        PNL = "#2a2a3e"
        ACC = "#7c6af7"
        FG  = "#cdd6f4"
        DIM = "#585b70"

        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".",         background=BG,  foreground=FG, font=("Segoe UI", 9))
        s.configure("TFrame",    background=BG)
        s.configure("TLabel",    background=BG,  foreground=FG)
        s.configure("TLabelframe",       background=PNL, foreground=ACC,
                    bordercolor=ACC, relief="groove")
        s.configure("TLabelframe.Label", background=PNL, foreground=ACC,
                    font=("Segoe UI", 9, "bold"))
        s.configure("TCheckbutton", background=BG, foreground=FG,
                    indicatorcolor=ACC)
        s.configure("TScale",       background=BG, troughcolor=PNL,
                    slidercolor=ACC)
        s.configure("TButton", background=PNL, foreground=FG,
                    bordercolor=ACC, focusthickness=0)
        s.map("TButton", background=[("active", ACC)])

        # ── Top title bar ─────────────────────────────────────────────────
        title_bar = tk.Frame(self, bg=PNL, height=36)
        title_bar.pack(fill="x", side="top")
        tk.Label(title_bar, text="  Brownian Motion Camera  |  Particle Tracker",
                 bg=PNL, fg=ACC,
                 font=("Segoe UI", 11, "bold")).pack(side="left", pady=4)
        self._fps_lbl = tk.Label(title_bar, text="0.0 fps",
                                 bg=PNL, fg=DIM,
                                 font=("Segoe UI", 9))
        self._fps_lbl.pack(side="right", padx=12, pady=4)

        # ── Main area  (video panels left + control panel right) ──────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=6, pady=4)

        # ── Left: two video panels stacked ────────────────────────────────
        video_frame = tk.Frame(main, bg=BG)
        video_frame.pack(side="left", fill="both", expand=True)

        # Pass-through
        pt_lbl_frame = ttk.LabelFrame(video_frame, text="Pass-Through  (click → move ROI  |  Shift+click → resize)")
        pt_lbl_frame.pack(fill="both", expand=True, padx=2, pady=2)
        self._pt_canvas = tk.Canvas(pt_lbl_frame, width=DISPLAY_W, height=DISPLAY_H,
                                    bg="black", highlightthickness=0, cursor="crosshair")
        self._pt_canvas.pack()
        self._pt_canvas.bind("<Button-1>",        self._roi_move)
        self._pt_canvas.bind("<Shift-Button-1>",  self._roi_resize)

        # Images window (processed)
        img_lbl_frame = ttk.LabelFrame(video_frame, text="Images  (top: raw ROI + tracks  |  bottom: post-processed blobs)")
        img_lbl_frame.pack(fill="both", expand=True, padx=2, pady=2)
        self._img_canvas = tk.Canvas(img_lbl_frame, width=DISPLAY_W, height=DISPLAY_H,
                                     bg="black", highlightthickness=0)
        self._img_canvas.pack()

        # ── Right: control panel ───────────────────────────────────────────
        ctrl = tk.Frame(main, bg=BG, width=300)
        ctrl.pack(side="right", fill="y", padx=4, pady=0)
        ctrl.pack_propagate(False)

        # ── Camera controls ────────────────────────────────────────────────
        cam_frame = ttk.LabelFrame(ctrl, text="Camera")
        cam_frame.pack(fill="x", padx=4, pady=4)

        self._cam_btn = ttk.Button(cam_frame, text="▶  Camera Go",
                                   command=self._toggle_camera)
        self._cam_btn.pack(fill="x", padx=6, pady=6)

        ttk.Checkbutton(cam_frame, text="Demo mode  (synthetic Brownian particles)",
                        variable=self._demo_mode,
                        command=self._on_demo_toggle).pack(anchor="w", padx=8, pady=(0, 6))

        ttk.Label(cam_frame, text="Exposure (µs)").pack(anchor="w", padx=8)
        self._exposure_var = tk.DoubleVar(value=10000)
        exp_scale = ttk.Scale(cam_frame, variable=self._exposure_var,
                              from_=100, to=100000, orient="horizontal",
                              command=self._on_exposure)
        exp_scale.pack(fill="x", padx=8, pady=2)
        self._exp_val_lbl = ttk.Label(cam_frame, text="10000 µs")
        self._exp_val_lbl.pack(anchor="e", padx=8)

        ttk.Label(cam_frame, text="Gain").pack(anchor="w", padx=8)
        self._gain_var = tk.DoubleVar(value=1.0)
        gain_scale = ttk.Scale(cam_frame, variable=self._gain_var,
                               from_=1.0, to=16.0, orient="horizontal",
                               command=self._on_gain)
        gain_scale.pack(fill="x", padx=8, pady=2)
        self._gain_val_lbl = ttk.Label(cam_frame, text="1.0×")
        self._gain_val_lbl.pack(anchor="e", padx=8)

        # ── Blob finder ────────────────────────────────────────────────────
        blob_frame = ttk.LabelFrame(ctrl, text="Blob Finder")
        blob_frame.pack(fill="x", padx=4, pady=4)

        ttk.Checkbutton(blob_frame, text="Find Particles",
                        variable=self._find_blobs,
                        command=self._on_toggle_blobs).pack(anchor="w", padx=8, pady=4)
        ttk.Checkbutton(blob_frame, text="Subtract Background (exp. smoothing)",
                        variable=self._subtract_bg).pack(anchor="w", padx=8)

        self._zscore_var = tk.DoubleVar(value=2.5)
        self._add_slider(blob_frame, "Z-Score Threshold",
                         self._zscore_var, 0.5, 8.0, "%.1f")

        self._blur_var = tk.IntVar(value=3)
        self._add_slider(blob_frame, "Gaussian Blur (px)",
                         self._blur_var, 1, 15, "%d", integer=True)

        self._min_area_var = tk.IntVar(value=20)
        self._add_slider(blob_frame, "Min Particle Area (px²)",
                         self._min_area_var, 5, 500, "%d", integer=True)

        self._max_area_var = tk.IntVar(value=2000)
        self._add_slider(blob_frame, "Max Particle Area (px²)",
                         self._max_area_var, 100, 5000, "%d", integer=True)

        # ── Particle tracker ───────────────────────────────────────────────
        track_frame = ttk.LabelFrame(ctrl, text="Particle Tracker")
        track_frame.pack(fill="x", padx=4, pady=4)

        ttk.Checkbutton(track_frame, text="Track Particles",
                        variable=self._track_parts).pack(anchor="w", padx=8, pady=4)

        self._reach_var = tk.DoubleVar(value=30.0)
        self._add_slider(track_frame, "Tracking Reach (px)",
                         self._reach_var, 5, 120, "%.0f")

        self._trail_var = tk.IntVar(value=MAX_TRAIL)
        self._add_slider(track_frame, "Trail Length (frames)",
                         self._trail_var, 5, MAX_TRAIL, "%d", integer=True)

        # ── Statistics readout ─────────────────────────────────────────────
        stats_frame = ttk.LabelFrame(ctrl, text="Statistics")
        stats_frame.pack(fill="x", padx=4, pady=4)
        self._stats_text = tk.Text(stats_frame, height=6, width=34,
                                   bg="#1a1a28", fg="#a6e3a1",
                                   font=("Courier New", 8),
                                   relief="flat", state="disabled",
                                   insertbackground="white")
        self._stats_text.pack(padx=6, pady=4, fill="x")

        # ── Data capture ───────────────────────────────────────────────────
        data_frame = ttk.LabelFrame(ctrl, text="Data Capture")
        data_frame.pack(fill="x", padx=4, pady=4)

        ttk.Button(data_frame, text="Set Start Time  (zero history)",
                   command=self._set_start_time).pack(fill="x", padx=6, pady=3)

        ttk.Checkbutton(data_frame, text="Record Bitmaps",
                        variable=self._recording).pack(anchor="w", padx=8)

        dir_row = tk.Frame(data_frame, bg=BG)
        dir_row.pack(fill="x", padx=6, pady=2)
        ttk.Label(dir_row, text="Save dir:").pack(side="left")
        ttk.Entry(dir_row, textvariable=self._save_dir, width=18).pack(side="left", padx=2)
        ttk.Button(dir_row, text="…", width=2,
                   command=self._browse_dir).pack(side="left")

        ttk.Button(data_frame, text="💾  Save Particle Data (CSV)",
                   command=self._save_data).pack(fill="x", padx=6, pady=3)
        ttk.Button(data_frame, text="💾  Save Recorded Bitmaps",
                   command=self._save_bitmaps).pack(fill="x", padx=6, pady=2)

        # ── Status bar ─────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready.")
        status_bar = tk.Label(self, textvariable=self._status_var,
                              bg="#181825", fg=DIM,
                              font=("Segoe UI", 8), anchor="w")
        status_bar.pack(fill="x", side="bottom", padx=6, pady=2)

    def _add_slider(self, parent, label, var, mn, mx, fmt, integer=False):
        ttk.Label(parent, text=label).pack(anchor="w", padx=8, pady=(4, 0))
        row = tk.Frame(parent, bg="#1e1e2e")
        row.pack(fill="x", padx=8, pady=2)
        val_lbl = ttk.Label(row, text=fmt % var.get(), width=7)
        val_lbl.pack(side="right")
        def _update(v):
            val_lbl.config(text=fmt % (int(float(v)) if integer else float(v)))
        scale = ttk.Scale(row, variable=var, from_=mn, to=mx,
                          orient="horizontal", command=_update)
        scale.pack(side="left", fill="x", expand=True)

    # ── Camera callbacks ───────────────────────────────────────────────────
    def _on_demo_toggle(self):
        """Switch between demo and real-camera mode. Restarts acquisition if running."""
        was_running = self._running
        if was_running:
            self._running = False
            self._camera.stop()

        # Rebuild camera backend with new demo flag
        self._camera = IDSCamera(force_demo=self._demo_mode.get())
        self._blob_finder.reset()
        self._tracker.reset()
        # Drain any stale frames
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break

        mode_str = "DEMO" if self._demo_mode.get() else f"Real ({self._camera.model})"
        self._update_status(f"Switched to {mode_str} mode.")

        if was_running:
            self._camera.start()
            self._running = True
            self._acquisition_thread = threading.Thread(
                target=self._acquire_loop, daemon=True)
            self._acquisition_thread.start()
            self._display_loop()

    def _toggle_camera(self):
        if not self._running:
            self._camera.start()
            self._running = True
            self._cam_btn.config(text="⏹  Camera Stop")
            self._update_status(f"Acquiring from {self._camera.model}")
            self._acquisition_thread = threading.Thread(
                target=self._acquire_loop, daemon=True)
            self._acquisition_thread.start()
            self._display_loop()
        else:
            self._running = False
            self._camera.stop()
            self._cam_btn.config(text="▶  Camera Go")
            self._update_status("Stopped.")

    def _acquire_loop(self):
        """Background thread: grab frames and push to queue."""
        while self._running:
            frame = self._camera.grab_frame()
            if frame is not None:
                if self._frame_q.full():
                    try:
                        self._frame_q.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_q.put(frame)
            else:
                time.sleep(0.005)

    def _display_loop(self):
        """Tkinter-thread: consume latest frame, process, display."""
        try:
            frame = self._frame_q.get_nowait()
        except queue.Empty:
            self.after(20, self._display_loop)
            return

        # FPS
        now = time.perf_counter()
        self._fps_buf.append(1.0 / max(now - self._last_t, 1e-6))
        self._last_t = now
        fps = float(np.mean(self._fps_buf))
        self._fps_lbl.config(text=f"{fps:.1f} fps")

        # Record
        if self._recording.get():
            self._recorded_frames.append(frame.copy())

        # Resize frame to display size
        h, w = frame.shape[:2]
        if (w, h) != (DISPLAY_W, DISPLAY_H):
            frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H),
                               interpolation=cv2.INTER_LINEAR)

        # ── Pass-through with ROI box ──────────────────────────────────
        pt_display = frame.copy()
        rx, ry, rw, rh = self._roi_pixels()
        cv2.rectangle(pt_display,
                      (rx, ry), (rx + rw, ry + rh),
                      C_ROI, 2)
        self._show_on_canvas(self._pt_canvas, pt_display)

        # ── ROI crop ──────────────────────────────────────────────────
        roi_img = frame[ry:ry + rh, rx:rx + rw].copy()
        if roi_img.size == 0:
            self.after(20, self._display_loop)
            return

        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        # ── Blob finding ──────────────────────────────────────────────
        particles = []
        post_display = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

        if self._find_blobs.get():
            _, post_vis, blobs = self._blob_finder.process(
                roi_gray,
                zscore_thresh=self._zscore_var.get(),
                min_area=int(self._min_area_var.get()),
                max_area=int(self._max_area_var.get()),
                blur=int(self._blur_var.get()),
                subtract_bg=self._subtract_bg.get())

            # Scale post image back to ROI size then embed in full display
            post_scaled = cv2.resize(post_vis,
                                     (DISPLAY_W, DISPLAY_H),
                                     interpolation=cv2.INTER_NEAREST)

            # ── Tracking ──────────────────────────────────────────────
            if self._track_parts.get():
                particles = self._tracker.update(
                    blobs, time.perf_counter(),
                    reach=self._reach_var.get())
            else:
                # Untracked blobs – just wrap in Particle-like objects
                for b in blobs:
                    particles.append(Particle(pid=-1, x=b["x"], y=b["y"],
                                              area=b["area"]))

            # Draw on post image
            scale_x = DISPLAY_W / max(roi_img.shape[1], 1)
            scale_y = DISPLAY_H / max(roi_img.shape[0], 1)
            for p in particles:
                sx, sy = int(p.x * scale_x), int(p.y * scale_y)
                colour = C_TRACKED if p.tracked else C_UNTRACKED
                cv2.circle(post_scaled, (sx, sy), 8, colour, 2)
                if p.pid >= 0:
                    cv2.putText(post_scaled, str(p.pid),
                                (sx + 9, sy - 6), FONT, 0.35, colour, 1)

            post_display = post_scaled

        # ── Images-window top half: raw ROI + tracks ──────────────────
        top_half = cv2.resize(roi_img, (DISPLAY_W, DISPLAY_H // 2),
                              interpolation=cv2.INTER_LINEAR)
        bot_half = cv2.resize(post_display, (DISPLAY_W, DISPLAY_H // 2),
                              interpolation=cv2.INTER_NEAREST)

        # Draw trails on top half
        if self._track_parts.get():
            trail_len = int(self._trail_var.get())
            scale_x = DISPLAY_W / max(roi_img.shape[1], 1)
            scale_y = (DISPLAY_H // 2) / max(roi_img.shape[0], 1)
            for p in particles:
                pts = list(p.trail)[-trail_len:]
                if len(pts) < 2:
                    continue
                for i in range(1, len(pts)):
                    pt1 = (int(pts[i-1][0] * scale_x),
                           int(pts[i-1][1] * scale_y))
                    pt2 = (int(pts[i][0]   * scale_x),
                           int(pts[i][1]   * scale_y))
                    alpha = i / len(pts)
                    colour = tuple(int(c * alpha) for c in C_TRAIL)
                    cv2.line(top_half, pt1, pt2, colour, 1)
                # Circle at current position
                cx = int(p.x * scale_x)
                cy = int(p.y * scale_y)
                c = C_TRACKED if p.tracked else C_UNTRACKED
                cv2.circle(top_half, (cx, cy), 6, c, 2)

        combined = np.vstack([top_half, bot_half])
        # Divider line
        cv2.line(combined, (0, DISPLAY_H // 2),
                 (DISPLAY_W, DISPLAY_H // 2), (80, 80, 80), 1)
        self._show_on_canvas(self._img_canvas, combined)

        # ── Stats ─────────────────────────────────────────────────────
        self._update_stats(particles, fps)

        if self._running:
            self.after(20, self._display_loop)

    # ── Display helpers ────────────────────────────────────────────────────
    def _show_on_canvas(self, canvas, bgr: np.ndarray):
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor="nw", image=imgtk)
        canvas._img = imgtk   # prevent GC

    def _roi_pixels(self):
        rx = int(self._roi[0] * DISPLAY_W)
        ry = int(self._roi[1] * DISPLAY_H)
        rw = int(self._roi[2] * DISPLAY_W)
        rh = int(self._roi[3] * DISPLAY_H)
        # clamp
        rx = max(0, min(rx, DISPLAY_W - rw))
        ry = max(0, min(ry, DISPLAY_H - rh))
        return rx, ry, rw, rh

    def _roi_move(self, e):
        cx, cy = e.x / DISPLAY_W, e.y / DISPLAY_H
        hw, hh = self._roi[2] / 2, self._roi[3] / 2
        self._roi[0] = max(0, min(cx - hw, 1 - self._roi[2]))
        self._roi[1] = max(0, min(cy - hh, 1 - self._roi[3]))

    def _roi_resize(self, e):
        cx, cy = e.x / DISPLAY_W, e.y / DISPLAY_H
        x0, y0 = self._roi[0], self._roi[1]
        self._roi[2] = max(0.05, min(cx - x0, 1.0 - x0))
        self._roi[3] = max(0.05, min(cy - y0, 1.0 - y0))

    # ── Blob / track toggle ────────────────────────────────────────────────
    def _on_toggle_blobs(self):
        if self._find_blobs.get():
            self._blob_finder.reset()
        else:
            self._find_blobs.set(False)
            self._track_parts.set(False)

    # ── Camera settings ────────────────────────────────────────────────────
    def _on_exposure(self, v):
        val = float(v)
        self._exp_val_lbl.config(text=f"{int(val)} µs")
        self._camera.set_exposure(val)

    def _on_gain(self, v):
        val = float(v)
        self._gain_val_lbl.config(text=f"{val:.2f}×")
        self._camera.set_gain(val)

    # ── Data capture ───────────────────────────────────────────────────────
    def _set_start_time(self):
        self._tracker.reset()
        self._blob_finder.reset()
        self._update_status("Start time zeroed – history cleared.")

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self._save_dir.get())
        if d:
            self._save_dir.set(d)

    def _save_data(self):
        """
        Save per-particle trajectory data in Berkeley BMC format:
        header + per-particle block: count, then x y t dx dy dt dr² Δr²
        """
        tracks = self._tracker.tracks
        if not tracks:
            messagebox.showwarning("No data", "No particle tracks to save.")
            return
        path = Path(self._save_dir.get()) / f"particle_data_{_ts()}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["# x", "y", "time", "dx", "dy", "dt",
                        "dr^2", "DisplacementSquared"])
            for pid, p in tracks.items():
                pos = p.positions
                if len(pos) < 2:
                    continue
                w.writerow([f"# Particle {pid}"])
                w.writerow([len(pos)])
                prev_t, prev_x, prev_y = pos[0]
                x0, y0 = prev_x, prev_y
                for i, (t, x, y) in enumerate(pos):
                    dx   = x - prev_x if i > 0 else 0
                    dy   = y - prev_y if i > 0 else 0
                    dt   = t - prev_t if i > 0 else 0
                    dr2  = dx**2 + dy**2
                    Δr2  = (x - x0)**2 + (y - y0)**2
                    w.writerow([f"{x:.3f}", f"{y:.3f}", f"{t:.4f}",
                                f"{dx:.3f}", f"{dy:.3f}", f"{dt:.4f}",
                                f"{dr2:.3f}", f"{Δr2:.3f}"])
                    prev_t, prev_x, prev_y = t, x, y
        self._update_status(f"Saved → {path}")
        messagebox.showinfo("Saved", str(path))

    def _save_bitmaps(self):
        if not self._recorded_frames:
            messagebox.showwarning("No frames", "No frames recorded yet.")
            return
        folder = Path(self._save_dir.get()) / f"movie_{_ts()}"
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / f"{folder.name} frames.txt"
        with open(log_path, "w") as fh:
            fh.write("filename\ttime_s\n")
            t0 = time.time()
            for i, frm in enumerate(self._recorded_frames):
                fname = f"frame_{i:05d}.png"
                cv2.imwrite(str(folder / fname), frm)
                fh.write(f"{fname}\t{i/max(len(self._recorded_frames),1)*10:.4f}\n")
        self._recorded_frames.clear()
        self._update_status(f"Saved {i+1} frames → {folder}")
        messagebox.showinfo("Saved", str(folder))

    # ── Statistics panel ───────────────────────────────────────────────────
    def _update_stats(self, particles, fps):
        tracked   = [p for p in particles if p.tracked]
        untracked = [p for p in particles if not p.tracked]

        lines = [
            f"Particles visible : {len(particles)}",
            f"  Tracked (red)   : {len(tracked)}",
            f"  Untracked (blue): {len(untracked)}",
            "",
        ]
        # Diffusion coefficients for top-3 longest tracks
        ds = []
        for p in sorted(tracked, key=lambda x: -len(x.positions))[:5]:
            d = self._tracker.diffusion_coeff(p.pid)
            if d is not None:
                ds.append(d)
                lines.append(f"  P{p.pid:03d}  D={d:.2f} px²/s")
        if ds:
            weighted = float(np.mean(ds))
            lines += ["", f"  ⟨D⟩ = {weighted:.2f} px²/s"]

        text = "\n".join(lines)
        self._stats_text.config(state="normal")
        self._stats_text.delete("1.0", "end")
        self._stats_text.insert("end", text)
        self._stats_text.config(state="disabled")

    # ── Status bar ─────────────────────────────────────────────────────────
    def _update_status(self, msg):
        self._status_var.set(msg)

    # ── Close ─────────────────────────────────────────────────────────────
    def _on_close(self):
        self._running = False
        self._camera.stop()
        if _IDS_AVAILABLE:
            try:
                ids_peak.Library.Close()
            except Exception:
                pass
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ts():
    return time.strftime("%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = BrownianApp()
    app.mainloop()