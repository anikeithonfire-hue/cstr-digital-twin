class PIDController:
    """
    Proportional–Integral–Derivative controller with anti-windup.

    Controls the CSTR temperature by computing the required coolant
    temperature (Tc) that the jacket should maintain.

    Output logic
    ────────────
    error > 0  →  T < setpoint  →  reactor is too cold  →  raise Tc (less cooling)
    error < 0  →  T > setpoint  →  reactor is too hot   →  lower Tc (more cooling)

    Anti-windup clamps the integral term to prevent it from accumulating
    to extreme values when the output is saturated.
    """

    def __init__(
        self,
        Kp=5.0,
        Ki=0.05,
        Kd=1.0,
        setpoint=350.0,
        out_min=280.0,
        out_max=380.0,
    ):
        self.Kp       = Kp
        self.Ki       = Ki
        self.Kd       = Kd
        self.setpoint = setpoint
        self.out_min  = out_min
        self.out_max  = out_max

        self._integral = 0.0
        self._prev_err = 0.0

    # ─────────────────────────────────────────────────────────
    def compute(self, measurement, dt=0.05):
        """
        Compute the new coolant temperature command.

        measurement : latest (filtered) reactor temperature reading (K)
        dt          : time elapsed since last call (min)
        Returns (Tc, error) tuple.
        """
        error = self.setpoint - measurement

        # Integral with anti-windup clamping
        self._integral += error * dt
        self._integral  = max(-500.0, min(500.0, self._integral))

        # Derivative
        derivative     = (error - self._prev_err) / max(dt, 1e-6)
        self._prev_err  = error

        # Coolant temperature command
        # Base of 300 K = nominal coolant supply temperature
        Tc = 300.0 + self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        Tc = max(self.out_min, min(self.out_max, Tc))

        return Tc, error

    # ─────────────────────────────────────────────────────────
    def set_setpoint(self, sp):
        """Change the target temperature and reset integrator state."""
        if sp != self.setpoint:
            self.setpoint  = sp
            self._integral = 0.0
            
            self._prev_err = 0.0