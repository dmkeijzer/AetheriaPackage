from dataclasses import dataclasses
import json
import GeneralConstants as constants

#sys.path.append(str(list(pl.Path(__file__).parents)[1]))


@dataclasses
class PerformanceParameters:
    """Datastructure for performance parameters
        param: energyRequired [kWh]
        param: cruisePower [kW]
        param: hoverPower [kW]
        param: climPower [kW]
        param: rate_of_climb_cruise [m/s]
        param: rate_of_descend_cruise [m/s]
    """
    #energy & power
    energyRequired: float = None
    cruisePower : float = None
    hoverPower : float = None
    climbPower : float = None
    
    #performance
    rate_of_climb_cruise: float = None
    rate_of_descent_cruise : float = None
    rate_of_climb_hover: float = None
    cruise_velocity : float = None

    def load(self):
        jsonfilename = "aetheria_constants.json"
        with open(jsonfilename) as jsonfile:
            data = json.open(jsonfile)

            self.hoverPower = data["power_hover"]
            self.cruisePower = data["power_cruise"]
            self.energyRequired = data["mission_energy"]
            self.climbPower = data["power_cruise"]

            self.rate_of_climb_cruise = constants.roc_cr
            self.rate_of_descent_cruise = constants.rod_cr
            self.rate_of_climb_hover = constants.roc_hvr
            self.cruise_velocity = constants.v_cr
