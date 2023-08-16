from dataclasses import dataclasses
import json


#sys.path.append(str(list(pl.Path(__file__).parents)[1]))
jsonfilename = "Aetheria_constants.json"

@dataclasses
class PerformanceParameters:
    energyRequired_J: float = None
    cruisePower_W : float = None
    hoverPower_W : float = None
    rate_of_climb_ms: float = None
    rate_of_descent_ms : float = None

    def load(self):
        with open(jsonfilename) as jsonfile:
            data = json.open(jsonfile)
            self.hoverPower_W = data["power_hover"]
            self.cruisePower_W = data["power_cruise"]
            self.energyRequired_J = data["mission_energy"]