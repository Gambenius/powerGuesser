import math
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt

class CyclingPhysics:
    def __init__(self, mass, cda, crr):
        self.mass = mass      # Rider + Bike (kg)
        self.cda = cda        # Aero coefficient
        self.crr = crr        # Rolling resistance
        self.g = 9.81         # Gravity m/s^2
        self.rho_sea = 1.225  # Standard air density

    def calculate_power(self, v_m_s, v_prev, delta_elevation_m, distance_m, dt, temp_c=20):
        if dt <= 0 or distance_m <= 0:
            return 0

        average_speed = max(0, (v_prev + v_m_s) / 2)
        kinetic_energy = 0.5 * self.mass * (v_m_s**2 - v_prev**2)
        potential_energy = self.mass * self.g * delta_elevation_m
        rolling_energy = self.mass * self.g * self.crr * distance_m
        aerodynamic_energy = 0.5 * self.cda * self.rho_sea * average_speed**2 * distance_m

        total_energy = (
            kinetic_energy
            + potential_energy
            + rolling_energy
            + aerodynamic_energy
        )
        return max(0, total_energy / dt / 0.97)

    def estimate_series(self, speed, elevation_delta, distance, dt, cadence):
        speed = np.asarray(speed, dtype=float)
        dt = np.asarray(dt, dtype=float)
        distance = np.asarray(distance, dtype=float)
        previous_speed = np.roll(speed, 1)
        average_speed = np.maximum(0, (previous_speed + speed) / 2)
        energy = (
            0.5 * self.mass * (speed**2 - previous_speed**2)
            + self.mass * self.g * np.asarray(elevation_delta)
            + self.mass * self.g * self.crr * distance
            + 0.5 * self.cda * self.rho_sea * average_speed**2 * distance
        )
        powers = np.zeros(len(speed))
        valid = (dt > 0) & (distance > 0) & (np.asarray(cadence) > 0)
        np.divide(energy, dt * 0.97, out=powers, where=valid)
        powers = np.maximum(0, powers)
        if len(powers):
            powers[0] = 0
        return powers


def lowpass_power(values, dt, cutoff_seconds=15):
    """Low-pass filter power while preserving the signal length."""
    values = np.asarray(values, dtype=float)
    if len(values) < 15:
        return values
    sample_rate = 1 / max(float(np.median(dt)), 0.1)
    cutoff = 1 / cutoff_seconds
    if cutoff >= sample_rate / 2:
        return values
    b, a = butter(2, cutoff, btype='low', fs=sample_rate)
    return filtfilt(b, a, values)


def optimize_parameters(initial, measured_power, speed, elevation_delta, distance, dt, cadence):
    """Fit CdA and Crr with fixed mass using robust low-pass residuals."""
    measured_filtered = lowpass_power(measured_power, dt)
    valid = np.isfinite(measured_filtered) & (measured_filtered > 0)
    mass = float(initial[0])

    def estimate(parameters):
        physics = CyclingPhysics(mass, parameters[0], parameters[1])
        estimated = physics.estimate_series(speed, elevation_delta, distance, dt, cadence)
        return lowpass_power(estimated, dt)

    def residuals(parameters):
        return (estimate(parameters)[valid] - measured_filtered[valid]) / 25

    result = least_squares(
        residuals,
        x0=np.asarray(initial[1:], dtype=float),
        bounds=([0.15, 0.001], [0.60, 0.015]),
        loss='soft_l1',
        f_scale=1.0,
        max_nfev=100,
    )
    optimized = np.array([mass, result.x[0], result.x[1]])
    base_power = estimate(result.x)[valid].mean()
    uncertainty = []
    for parameter_index, step in enumerate((0.01, 0.0005)):
        plus = result.x.copy()
        minus = result.x.copy()
        plus[parameter_index] += step
        minus[parameter_index] -= step
        sensitivity = (estimate(plus)[valid].mean() - estimate(minus)[valid].mean()) / (2 * step)
        delta = 5 / abs(sensitivity) if abs(sensitivity) > 1e-9 else float('inf')
        uncertainty.append(delta)
    return optimized, np.array(uncertainty), result.cost, result.optimality
