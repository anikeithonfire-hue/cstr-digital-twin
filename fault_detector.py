import numpy as np
from collections import deque


class FaultDetector:
    """
    Real-time fault detection using a rolling Z-score method.

    How it works
    ────────────
    1. WARMUP phase  — first WARMUP samples are used to learn what "normal"
       looks like (baseline mean and standard deviation).
    2. DETECTION phase — for every new reading, compute:
           z = |reading − baseline_mean| / baseline_std
       If z exceeds Z_THRESH, a fault is declared.

    This mirrors industrial alarm management standards (ISA-18.2).
    """

    WARMUP   = 100    # samples before detection starts  (~7 seconds at 200 ms ticks)
    Z_THRESH = 3.5    # standard deviations before fault alarm

    def __init__(self):
        # Circular buffers that fill during warmup
        self._T_buf  = deque(maxlen=self.WARMUP)
        self._Ca_buf = deque(maxlen=self.WARMUP)

        self._count       = 0
        self.ready        = False
        self.baseline     = {}     # {"T_mean", "T_std", "Ca_mean", "Ca_std"}
        self.score        = 0.0    # latest z-score (shown on dashboard)
        self.status       = f"Warming up (0 / {self.WARMUP})"
        self.fault_active = False

    # ─────────────────────────────────────────────────────────
    def update(self, T, Ca, P, F):
        """
        Feed one new set of filtered sensor readings.
        Returns True if a fault is detected, False otherwise.
        """
        self._count += 1
        self._T_buf.append(T)
        self._Ca_buf.append(Ca)

        # ── Warmup: learn baseline ────────────────────────────
        if self._count < self.WARMUP:
            self.score  = 0.0
            self.status = f"Warming up ({self._count} / {self.WARMUP})"
            return False

        if self._count == self.WARMUP:
            self.baseline = {
                "T_mean" : np.mean(self._T_buf),
                "T_std"  : max(np.std(self._T_buf),  0.1),
                "Ca_mean": np.mean(self._Ca_buf),
                "Ca_std" : max(np.std(self._Ca_buf), 0.001),
            }
            self.ready = True

        # ── Detection: rolling Z-score ────────────────────────
        z_T  = abs(T  - self.baseline["T_mean"])  / self.baseline["T_std"]
        z_Ca = abs(Ca - self.baseline["Ca_mean"]) / self.baseline["Ca_std"]

        self.score        = round(max(z_T, z_Ca), 2)
        self.fault_active = self.score > self.Z_THRESH

        if self.fault_active:
            self.status = f"FAULT DETECTED  (z = {self.score})"
        else:
            self.status = f"Normal  (z = {self.score})"

        return self.fault_active