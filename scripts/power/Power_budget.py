"""
Contains constants for the power budget.
"""

class Power_Budget:
    def __init__(self, m):
        # Constant required power
        self.avionics = 233.8  # [W]
        self.airco = 2783.1  # [W]
        self.battery_cooling = 30 * m/72  # [W]
        self.autopilot = 140.1  # [W]
        self.trim = 50.1  # [W]
        self.passenger_power = 377.1 # [W]
        self.external_lights = 108.5  # [W]
        self.deice = 2783.1  # [W]

        # Non continuous power
        self.landing_gear = 46.4  # [W]
        self.wing_rot_mech = 19500  # [W]

    def P_continuous(self):
        return self.avionics + self.airco + self.battery_cooling + self.autopilot + self.trim + self.passenger_power \
               + self.external_lights + self.deice

    def E_landing_gear(self, t_lg_rot):
        return self.landing_gear * t_lg_rot

    def E_wing_rot(self, t_transition):
        return self.wing_rot_mech * t_transition
    
    def Total_power_budget(self):
        return self.avionics + self.airco + self.battery_cooling + self.autopilot + self.trim + self.passenger_power \
               + self.external_lights + self.deice + self.landing_gear + self.wing_rot_mech

PB = Power_Budget(886.2)
print(PB.P_continuous())
print(PB.Total_power_budget())  