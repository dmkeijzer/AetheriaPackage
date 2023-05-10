import numpy as np


def closestNumber(n, m):
    # Find the quotient
    q = int(n / m)

    # 1st possible closest number
    n1 = m * q

    # 2nd possible closest number
    if ((n * m) > 0):
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))

    # if true, then n1 is the required closest number
    if n1 > n:
        return n1
    else:
        return n2


class redundancy_power:
    def __init__(self, V_motor, E_tot, V_cell, C_cell, n_mot, n_bat_mot, per_mot):
        """
        :param V_motor: Voltage required for the electric motors [V]
        :param E_tot: total energy required [kWh]
        :param V_cell: voltage of a single cell [V]
        :param C_cell: capacitance of a single cell [Ah]
        :param n_mot: number of motors [-]
        :param n_bat_mot: number of batteries per motor [-]
        :param per_mot: ratio of power going to the motors, of total energy [-]
        """
        self.V_motor = V_motor
        self.E_tot = E_tot
        self.V_cell = V_cell
        self.C_cell = C_cell
        self.E_cell = V_cell * C_cell
        self.n_mot = n_mot
        self.n_bat_mot = n_bat_mot
        self.per_mot = per_mot

    def N_cells_mot(self):
        return int(np.ceil(self.E_tot * self.per_mot / self.E_cell * 1000))

    def N_cells_misc(self):
        return int(np.ceil(self.E_tot * (1 - self.per_mot) / self.E_cell * 1000))

    def N_cells_tot(self):
        return self.N_cells_mot() + self.N_cells_misc()

    def N_ser(self):
        return int(np.ceil(self.V_motor / self.V_cell))

    def N_par(self):
        return int(np.ceil(self.N_cells_mot() / self.N_ser()))

    def N_par_new(self):
        return closestNumber(self.N_par(), self.n_mot * self.n_bat_mot)

    def N_cells_mot_new(self):
        return self.N_ser() * self.N_par_new()

    def N_cells_new(self):
        return self.N_cells_mot_new() + self.N_cells_misc()

    def increase_mot(self):
        increase = self.N_cells_mot_new() - self.N_cells_mot()
        perc = np.round(((increase) / self.N_cells_mot() * 100), 5)
        return perc

    def increase(self):
        increase = self.N_cells_new() - self.N_cells_tot()
        perc = np.round(((increase) / self.N_cells_tot() * 100), 3)
        return increase, perc
