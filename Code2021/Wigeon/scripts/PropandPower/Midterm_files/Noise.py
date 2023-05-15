import numpy as np


class Noise:
    def __init__(self, P_br_cr, P_br_h, D, B, N_p_cr, N_p_h, rpm_hover, rpm_cruise, sound_speed, M_t_h=None, M_t_cr=None):
        """
        :param P_br_cr: Engine power in kW at cruise [kW]
        :param P_br_h: Engine power in kW at hover [kW]
        :param D: Propeller diameter [m]
        :param B:  Number of blades [-]
        :param N_p_cr: Number of propellers at cruise [-]
        :param N_p_h: Number of propellers at hover [-]
        :param sound_speed: Speed of sound [m/s]
        :param M_t_h: Mach number at propeller tip during hover [-]
        :param M_t_cr: Mach number at propeller tip during cruise [-]
        """
        self.P_cr = P_br_cr
        self.P_h = P_br_h
        self.D = D
        self.B = B
        self.N_p_cr = N_p_cr
        self.N_p_h = N_p_h
        self.rpm_h = rpm_hover
        self.rpm_cr = rpm_cruise
        if M_t_h is None:
            self.M_t_h = np.pi*D*rpm_hover / (sound_speed*60)
        else:
            self.M_t_h = M_t_h

        if M_t_cr is None:
            self.M_t_cr = np.pi*D*rpm_cruise / (sound_speed*60)
        else:
            self.M_t_cr = M_t_cr

    def SPL_cr(self):
        return 83.4 + 15.3*np.log10(self.P_cr) - 20*np.log10(self.D) + 38.5*self.M_t_cr - 3*(self.B-2) + 10*np.log10(self.N_p_cr)

    def SPL_hover(self):
        return 83.4 + 15.3*np.log10(self.P_h) - 20*np.log10(self.D) + 38.5*self.M_t_h - 3*(self.B-2) + 10*np.log10(self.N_p_h)


def sum_noise(noises):
    tens = np.ones((np.shape(noises))) * 10
    summed_noise = 10 * np.log10(np.sum(np.power(tens, np.array(noises)/10)))
    return summed_noise


def noise_at_distance(noise, distance):
    return 1
