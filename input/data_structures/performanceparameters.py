from dataclasses import dataclasses
import json


#sys.path.append(str(list(pl.Path(__file__).parents)[1]))


@dataclasses
class PerformanceParameters:
    """Datastructure for performance parameters
        param: energyRequired [kWh]
        param: cruisePower [kW]
        param: hoverPower [kW]
        param: rate_of_climb [m/s]
        param: rate_of_descend [m/s]
    """
    energyRequired: float = None
    cruisePower : float = None
    hoverPower : float = None
    rate_of_climb: float = None
    rate_of_descent : float = None

    def load(self):
        jsonfilename = "aetheria_constants.json"
        with open(jsonfilename) as jsonfile:
            data = json.open(jsonfile)
            self.hoverPower = data["power_hover"]
            self.cruisePower = data["power_cruise"]
            self.energyRequired = data["mission_energy"]