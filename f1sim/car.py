"""Car structure: mass/geometry, tire (Pacejka), aero, powertrain, brakes."""
import json
from dataclasses import dataclass, field


@dataclass
class TireParams:
    # Pacejka Magic Formula coefficients (simplified, combined-slip ignored;
    # lateral and longitudinal treated independently and capped by a friction
    # ellipse in vehicle_dynamics).
    B_lat: float = 10.0
    C_lat: float = 1.9
    D_lat: float = 1.0     # peak mu at nominal load (scaled by load sensitivity)
    E_lat: float = 0.97
    B_long: float = 11.5
    C_long: float = 1.65
    D_long: float = 1.1
    E_long: float = 0.97
    nominal_load_N: float = 3000.0   # Fz0 used for load-sensitivity scaling
    load_sensitivity: float = 0.15   # mu falls off as load rises above Fz0


@dataclass
class Car:
    name: str = "generic_race_car"
    mass_kg: float = 750.0
    wheelbase_m: float = 3.0
    track_width_front_m: float = 1.6
    track_width_rear_m: float = 1.55
    cg_height_m: float = 0.30
    weight_dist_front: float = 0.46   # fraction of static weight on front axle
    roll_stiffness_front_frac: float = 0.5  # fraction of total roll stiffness at front

    # Aero: downforce/drag = 0.5 * rho * v^2 * (Cl*A or Cd*A)
    ClA_front: float = 1.5
    ClA_rear: float = 1.8
    CdA: float = 1.0
    air_density: float = 1.225

    # Powertrain
    max_power_kw: float = 550.0
    drivetrain: str = "RWD"           # RWD, FWD, AWD
    max_engine_accel_mps2: float = 12.0  # traction-independent cap (e.g. torque/rpm limit at low speed)

    # Brakes
    max_brake_decel_mps2: float = 45.0  # ceiling before tire-limited

    tires: TireParams = field(default_factory=TireParams)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        tire_data = data.pop("tires", {})
        return cls(tires=TireParams(**tire_data), **data)

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @property
    def weight_dist_rear(self):
        return 1.0 - self.weight_dist_front

    @property
    def static_load_front_N(self):
        return self.mass_kg * 9.81 * self.weight_dist_front

    @property
    def static_load_rear_N(self):
        return self.mass_kg * 9.81 * self.weight_dist_rear
