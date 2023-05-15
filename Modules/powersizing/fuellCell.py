
class FuellCellSizing:
    """This class is to estimate the parameters of a Fuell Cell"""
    def __init__(self, sp_power_den:float ,vol_power_den:float, efficiency: float,  cost:float):
        """
            :param: sp_power_den: Energy Density of the battery [kWh/kg]
            :param: vol_en_den: Volumetric Density [kWh/l]
            :param: sp_power_den: Power Density of the battery [kW/kg]
            :param: CostDensity: Cost per Wh of the battery [US$/kWh]
            """
        self.PowerDensity= sp_power_den
        #self.power = power_required
        self.VolumeDensity = vol_power_den
        self.Cost = cost
        self.Efficiency = efficiency

    def mass(self,power):
        """
        :param power: Power requirement for the fuell cell[kW]
        :param sp_P_den: Power density of the fuell cell[kW/kg]
        :return: Mass of the battery
        """
        return power / self.PowerDensity

    def volume(self,power):
        """
        :param Powery: Power required from the fuell cell [kW]
        :param vol_en_den: Volumetric energy density of the fuell cell [kW/l]
        :return: Volume of the fuell cell [m^3]
        """
        return power /self.VolumeDensity  * 0.001

    def price(self,power):
        """
        :param power: Required power for the fuell cell [Wh]
        :param cost: Cost per Wh of the battery [US$/kW]
        :return: Approx cost of the battery [US$]
        """
        return power * self.Cost

