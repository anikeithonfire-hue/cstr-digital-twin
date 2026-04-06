class KalmanFilter1D:
    """
    Single-variable Kalman filter.

    Reduces high-frequency sensor noise while tracking the true signal.
    Two noise parameters control the trade-off:
        Q  — process noise variance   (how much the true value changes per step)
        R  — measurement noise variance (how noisy the sensor is)
    Large Q → filter trusts new measurements more (faster response, less smooth).
    Large R → filter trusts its own prediction more  (smoother, slower response).
    """

    def __init__(self, Q=1e-4, R=0.64, x0=350.0):
        self.Q = Q       # Process noise variance
        self.R = R       # Measurement noise variance
        self.x = x0      # Initial state estimate
        self.P = 1.0     # Initial estimate covariance (uncertainty)

    # ─────────────────────────────────────────────────────────
    def update(self, z):
        """
        Feed one new sensor reading z.
        Returns the filtered (smoothed) estimate.

        Prediction step
        ───────────────
          x_pred = x          (no control input → state doesn't change on its own)
          P_pred = P + Q      (uncertainty grows a little each step)

        Update step
        ───────────
          K = P_pred / (P_pred + R)       Kalman gain (0 → ignore measurement, 1 → trust it fully)
          x = x_pred + K × (z − x_pred)  Fuse prediction with measurement
          P = (1 − K) × P_pred            Reduce uncertainty after seeing data
        """
        # Prediction
        x_pred = self.x
        P_pred = self.P + self.Q

        # Kalman gain
        K = P_pred / (P_pred + self.R)

        # Update
        self.x = x_pred + K * (z - x_pred)
        self.P = (1.0 - K) * P_pred

        return self.x