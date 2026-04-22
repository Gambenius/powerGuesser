import math

class CyclingPhysics:
    def __init__(self, mass, cda, crr):
        self.mass = mass      # Rider + Bike (kg)
        self.cda = cda        # Aero coefficient
        self.crr = crr        # Rolling resistance
        self.g = 9.81         # Gravity m/s^2
        self.rho_sea = 1.225  # Standard air density

    def calculate_power(self, v_m_s, v_prev, grade, dt, temp_c=20):
        if v_m_s <= 0 or dt <= 0:
            return 0
        
        # 1. Gravity Force
        f_gravity = self.mass * self.g * math.sin(math.atan(grade))
        
        # 2. Rolling Resistance
        f_rolling = self.mass * self.g * math.cos(math.atan(grade)) * self.crr
        
        # 3. Aerodynamic Drag
        # (Simplified: assumes no wind for now)
        f_drag = 0.5 * self.cda * self.rho_sea * (v_m_s**2)
        
        # 4. Acceleration Force (Inertia)
        accel = (v_m_s - v_prev) / dt
        f_accel = self.mass * accel
        
        # Total Power = Total Force * Velocity
        # Add 3% for drivetrain loss (0.97 efficiency)
        total_power = (f_gravity + f_rolling + f_drag + f_accel) * v_m_s
        return max(0, total_power / 0.97)