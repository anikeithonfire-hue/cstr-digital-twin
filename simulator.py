import numpy as np


class CSTRSimulator:
    """
    Simulates a Continuous Stirred Tank Reactor (CSTR).
    Uses Euler integration of the energy and mass balance ODEs.
    Generates realistic noisy sensor readings for:
        - Temperature (K)
        - Reactant concentration Ca (mol/L)
        - Pressure (kPa)
        - Feed flow rate (L/min)
    """

    def __init__(self):
        # ── Physical parameters ──────────────────────────────
        self.V   = 100.0     # Reactor volume (L)
        self.rho = 1000.0    # Liquid density (g/L)
        self.Cp  = 0.239     # Heat capacity (J/g/K)
        self.dH  = -5e4      # Heat of reaction (J/mol)  [exothermic → negative]
        self.Ea  = 72750.0   # Activation energy (J/mol)
        self.R   = 8.314     # Gas constant (J/mol/K)
        self.k0  = 7.2e10    # Pre-exponential factor (1/min)
        self.UA  = 5e4       # Overall heat transfer coefficient × area (J/min/K)

        # ── Feed conditions ──────────────────────────────────
        self.F   = 100.0     # Feed flow rate (L/min)
        self.Caf = 1.0       # Feed concentration of A (mol/L)
        self.Tf  = 350.0     # Feed temperature (K)

        # ── Initial state variables ──────────────────────────
        self.T  = 350.0      # Reactor temperature (K)
        self.Ca = 0.5        # Concentration of A (mol/L)
        self.t  = 0.0        # Simulation time (min)
        self.dt = 0.05       # Time step (min) ≈ 3 seconds real-time

        # ── Sensor noise standard deviations ─────────────────
        self.T_noise  = 0.8   # K
        self.Ca_noise = 0.008 # mol/L
        self.P_noise  = 0.3   # kPa
        self.F_noise  = 1.5   # L/min

        # ── Fault injection ──────────────────────────────────
        self.fault_active = False
        self.fault_type   = None

    # ─────────────────────────────────────────────────────────
    def _reaction_rate(self, T, Ca):
        """Arrhenius equation: k(T) × Ca"""
        k = self.k0 * np.exp(-self.Ea / (self.R * T))
        return k * Ca

    # ─────────────────────────────────────────────────────────
    def step(self, Tc):
        """
        Advance simulation by one time step.
        Tc  : coolant temperature (K) — set by the PID controller
        Returns a dict of sensor measurements.
        """
        T, Ca, F = self.T, self.Ca, self.F

        # Apply fault disturbances
        if self.fault_active:
            if self.fault_type == "flow_drop":
                F = 50.0                        # feed flow drops to 50 %
            elif self.fault_type == "heat_loss":
                Tc = min(Tc + 35.0, 400.0)     # cooling system partially fails

        r = self._reaction_rate(T, Ca)

        # Mass balance:   dCa/dt = (F/V)(Caf - Ca) - r
        dCa_dt = (F / self.V) * (self.Caf - Ca) - r

        # Energy balance: dT/dt  = (F/V)(Tf - T) + (−ΔH/ρCp)·r + (UA/VρCp)(Tc − T)
        dT_dt = (
            (F / self.V) * (self.Tf - T)
            + (-self.dH / (self.rho * self.Cp)) * r
            + (self.UA  / (self.V * self.rho * self.Cp)) * (Tc - T)
        )

        # Euler integration
        self.Ca = float(np.clip(self.Ca + dCa_dt * self.dt, 0.0, self.Caf))
        self.T  = float(np.clip(self.T  + dT_dt  * self.dt, 300.0, 500.0))
        self.t += self.dt

        # ── Sensor readings (true value + Gaussian noise) ────
        T_meas  = self.T  + np.random.normal(0, self.T_noise)
        Ca_meas = self.Ca + np.random.normal(0, self.Ca_noise)
        P_meas  = 101.3 + 0.08 * (self.T - 350.0) + np.random.normal(0, self.P_noise)
        F_meas  = F + np.random.normal(0, self.F_noise)

        return {
            "time"   : round(self.t, 3),
            "T_true" : self.T,
            "Ca_true": self.Ca,
            "T_meas" : T_meas,
            "Ca_meas": Ca_meas,
            "P_meas" : P_meas,
            "F_meas" : F_meas,
        }

    # ─────────────────────────────────────────────────────────
    def inject_fault(self, fault_type):
        """Activate a fault: 'flow_drop' or 'heat_loss'"""
        self.fault_active = True
        self.fault_type   = fault_type

    def clear_fault(self):
        """Remove the active fault and reset flow rate"""
        self.fault_active = False
        self.fault_type   = None
        self.F = 100.0