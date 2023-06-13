from dataclasses import dataclass

@dataclass
class Power_Budget:
    """
    Contains constants for the power budget.
        """
    
    mass_battery_kg: float
    # Constant required power
    avionics = 233.8  # [W]
    airco = 2783.1  # [W]
    battery_cooling = 30 * self.mass_battery_kg /72  # [W]
    autopilot = 140.1  # [W]
    trim = 50.1  # [W]
    passenger_power = 377.1 # [W]
    external_lights = 108.5  # [W]
    deice = 2783.1  # [W]

    # Non continuous power
    landing_gear = 46.4  # [W]
    wing_rot_mech = 19500  # [W]'

    @property
    def P_continuous(self):
        return self.avionics + self.airco + self.battery_cooling + self.autopilot + self.trim + self.passenger_power \
               + self.external_lights + self.deice
    @property
    def E_landing_gear(self, t_lg_rot):
        return self.landing_gear * t_lg_rot
    @property
    def E_wing_rot(self, t_transition):
        return self.wing_rot_mech * t_transition
    @property
    def Total_power_budget(self):
        return self.avionics + self.airco + self.battery_cooling + self.autopilot + self.trim + self.passenger_power \
               + self.external_lights + self.deice + self.landing_gear + self.wing_rot_mech

PB = Power_Budget(886.2)
print(PB.P_continuous())
print(PB.Total_power_budget())  