import numpy as plt


class Battery:
    """This class is to estimate the parameters of a battery"""
    def __init__(self, sp_en_den, vol_en_den, tot_energy, cost, DoD, P_den, P_max, safety, EOL_C):
        """
        :param sp_en_den: Specific energy density of the battery [Wh/kg]
        :param vol_en_den: Volumetric energy density of the battery [Wh/l]
        :param tot_energy: Required total energy for the battery [Wh]
        :param cost: Cost per kWh of the battery [US$/kWh]
        :param DoD: Max depth of discharge for the battery [-], af fraction (so range 0 - 1)
        :param P_den: Power density of the cells [W/kg]
        :param P_max: Maximum required power of the cells [W]
        :param safety: Safety factor for all additional components [-]
        :param EOL_C: Required fraction of initial capacity that is required at EOL [-]
        """
        self.sp_en_den = sp_en_den
        self.energy = tot_energy
        self.vol_en_den = vol_en_den
        self.cost = cost
        self.DoD = DoD
        self.P_den = P_den
        self.P_max = P_max
        self.safety = safety
        self.EOL_C = EOL_C

    def mass(self):
        m_en = self.energy / self.sp_en_den
        m_p = self.P_max / self.P_den
        print('m_en, m_p', m_en, m_p)
        bat_mass = plt.maximum(m_en, m_p)
        return bat_mass / (self.DoD * self.EOL_C) * self.safety

    def volume(self):
        return self.mass() * self.sp_en_den / self.vol_en_den * 0.001

    def price(self):
        return self.mass() * self.sp_en_den * self.cost / 1000  # divide by 1000 since cost is in kWh

    def mass_both(self):
        m_en = self.energy / self.sp_en_den / (self.DoD * self.EOL_C) * self.safety
        m_p = self.P_max / self.P_den / (self.DoD * self.EOL_C) * self.safety
        return m_en, m_p