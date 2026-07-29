import math

class CyclingPhysics:
    def __init__(self, mass, cda, crr):
        self.mass = mass      # Rider + Bike (kg)
        self.cda = cda        # Aero coefficient
        self.crr = crr        # Rolling resistance
        self.g = 9.81         # Gravity m/s^2
        self.rho_sea = 1.225  # Standard air density

    def calculate_power(self, v_m_s, v_prev_m_s, ele_m, ele_prev_m, dt, temp_c=20, wind_speed=0.0, drivetrain_eff=0.97):
        """
        Energy-based power estimate over a sampling interval using altitude for potential energy.

        Parameters:
        - v_m_s: current speed (m/s)
        - v_prev_m_s: previous speed (m/s)
        - ele_m: current elevation (m)
        - ele_prev_m: previous elevation (m)
        - dt: time delta between samples (s)
        - temp_c: ambient temperature in Celsius (optional, used to adjust air density)
        - wind_speed: longitudinal wind speed (m/s), positive means tailwind (optional)
        - drivetrain_eff: drivetrain efficiency (0 < eff <= 1)

        Returns:
        - Estimated rider power (W) required to produce the observed change in energy
          plus modeled losses. Can be negative if the system lost energy (regeneration/braking).
        """
        if dt <= 0:
            return 0.0

        # Mass and gravity
        m = self.mass
        g = self.g

        # 1) Change in mechanical energy (potential + kinetic) over the interval
        dE_potential = m * g * (ele_m - ele_prev_m)
        dE_kinetic = 0.5 * m * (v_m_s**2 - v_prev_m_s**2)
        dE = dE_potential + dE_kinetic
        dE_dt = dE / dt

        # 2) Aerodynamic power (use simple cubic relationship). Adjust air density slightly with temperature.
        #    If wind_speed is positive (tailwind), relative velocity reduces; if negative (headwind), increases.
        rho = self.rho_sea * (273.15 / (temp_c + 273.15))
        v_rel = max(0.0, v_m_s - wind_speed)
        p_aero = 0.5 * rho * self.cda * (v_rel ** 3)

        # 3) Rolling resistance power (Crr * m * g * v)
        p_roll = self.crr * m * g * v_m_s

        # Sum requirements and account for drivetrain losses (divide by efficiency)
        mechanical_required = dE_dt + p_aero + p_roll
        if drivetrain_eff <= 0:
            drivetrain_eff = 0.97

        rider_power = mechanical_required / drivetrain_eff

        return rider_power
