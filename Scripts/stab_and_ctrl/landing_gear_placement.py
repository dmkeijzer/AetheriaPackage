import numpy as np
from matplotlib import pyplot as plt
import Final_optimization.constants_final as const

class LandingGearCalc:
    """
    Class to calculate optimal landing gear position based on constraints.
    Based on Lecture 9 of AE3211-I Systems Engineering and Aerospace Design
    The coordinate system has its origin at the nose the height of the bottom
    of the fuselage. The x-axis points backwards, the y-axis points towards
    starboard and the z-axis points upwards.

    @author Jakob Schoser
    """
    def __init__(self, x_ng: float, x_tg: float, tw_ng: float,
                 tilt_max_rad: float, y_tilt: float, tilt_tip_rad: float,
                 b: float, taper: float, z_wing: float, gamma: float,
                 y_max_rotor: float, z_offset_rotor: float, rotor_rad: float,
                 l_fus: float, h_fus: float):
        """
        :param x_ng: x-location of the nose landing gears
        :param x_tg: x-location of the rear landing gears
        :param tw_ng: maximum track width of the front landing gears
        (limited by the rotation of the wings in the front)
        :param tilt_max_rad: length of the root chord of the rotating part
        of the front wing behind the tilting point (i.e., the radius of the
        circle segment traced out by the trailing edge)
        :param y_tilt: y-location where the rotating part of the front wing
        starts
        :param tilt_tip_rad: length of the tip chord behind the rotation axis
        :param b: span of the front wing
        :param taper: taper ratio of the front wing
        :param z_wing: z-position of the front wing in the symmetry plane
        :param gamma: dihedral of the front wing
        :param y_max_rotor: y-position of the most outboard rotor
        :param z_offset_rotor: offset of the rotor line in z-direction
        (compared to wing)
        :param rotor_rad: radius of rotors on the front wing
        :param l_fus: length of the fuselage
        :param h_fus: height of the fuselage
        """
        self.x_ng = x_ng
        self.x_tg = x_tg
        self.tw_ng = tw_ng
        self.tilt_max_rad = tilt_max_rad
        self.y_tilt = y_tilt
        self.tilt_tip_rad = tilt_tip_rad
        self.b = b
        self.taper = taper
        self.z_wing = z_wing
        self.gamma = gamma
        self.y_max_rotor = y_max_rotor
        self.z_offset_rotor = z_offset_rotor
        self.rotor_rad = rotor_rad
        self.l_fus = l_fus
        self.h_fus = h_fus

    def calc_min_h_root(self, tw_tg, phi):
        z_root = (self.z_wing + self.y_tilt * np.tan(self.gamma)
                  - self.tilt_max_rad)
        y_dist = self.y_tilt - tw_tg / 2
        return y_dist * np.tan(phi) - z_root

    def calc_min_h_tip(self, tw_tg, phi):
        z_tip = (self.z_wing + self.b / 2 * np.tan(self.gamma)
                 - self.tilt_tip_rad)
        y_dist = self.b / 2 - tw_tg / 2
        return y_dist * np.tan(phi) - z_tip

    def calc_min_h_rotor(self, tw_tg, phi):
        crit_rad = max(self.rotor_rad, const.h_wt_1)
        z_rot = (self.z_wing + self.b / 2 * np.tan(self.gamma) - crit_rad)
        y_dist = self.y_max_rotor - tw_tg / 2
        return y_dist * np.tan(phi) - z_rot

    def calc_min_h_tipback(self, theta):
        return np.tan(theta) * (self.l_fus - self.x_tg) - self.h_fus

    def calc_psi(self, tw_tg, z_cg, h_lg, x_cg):
        alpha = np.arctan((tw_tg - self.tw_ng) / 2 / (self.x_tg - self.x_ng))
        bn = x_cg - self.x_ng + self.tw_ng / 2 / np.tan(alpha)
        c = bn * np.sin(alpha)
        return np.arctan((z_cg + h_lg) / c)

    def calc_ng_lf(self, x_cg):
        return (self.x_tg - x_cg) / (self.x_tg - self.x_ng)

    def calc_tg_lf(self, x_cg):
        return 1 - self.calc_ng_lf(x_cg)

    def calc_cg_tipback(self, x_cg, z_cg, h_lg):
        return np.arctan((self.x_tg - x_cg) / (h_lg + z_cg))

    def optimum_placement(self, x_cg_range: list,
                          z_cg_max: float, theta: float, phi: float,
                          psi: float, min_lf: float, tw_tg_max=4.,
                          tw_tg_res=20) -> tuple:
        # TODO: consider compression of the landing gear
        # TODO: consider y-offset of CG in turnover angle
        """
        Place the landing gear with respect to geometric constraints
        :param x_cg_range: x-range of CG locations
        :param z_cg_max: z-location of the highest CG
        :param theta: Pitch angle limit
        :param phi: Lateral ground clearance angle
        :param psi: Turnover angle
        :param min_lf: Minimum fraction of the weight to be carried
        by the least loaded pair of gears
        :param tw_tg_max: Maximum track with of the tail gear that is tested
        for
        :param tw_tg_res: Resolution of the array containing all track widths
        to be tried out
        :return: (track width of the main landing gear, height of the
        main landing gear). (None, None) if no feasible configuration
        was found with the given parameters
        """

        if (self.calc_ng_lf(x_cg_range[1]) < min_lf
                or self.calc_tg_lf(x_cg_range[0]) < min_lf):
            print(self.x_ng, x_cg_range, self.x_tg)
            print('fractions', self.calc_ng_lf(x_cg_range[0]),
                  self.calc_ng_lf(x_cg_range[1]))

            return None, None, "load on one landing gear too small"
        reasons = ["tailcone tipback", "rotated wing root clearance",
                   "rotated wing tip clearance",
                   "non-rotated wing rotor clearance"]

        tw_tg_list = np.linspace(self.tw_ng, tw_tg_max, tw_tg_res)
        min_h_tipback = (np.ones(tw_tg_list.shape)
                         * self.calc_min_h_tipback(theta))
        min_h_root = self.calc_min_h_root(tw_tg_list, phi)
        min_h_tip = self.calc_min_h_tip(tw_tg_list, phi)
        min_h_rotor = self.calc_min_h_rotor(tw_tg_list, phi)
        h_mat = np.array([min_h_tipback, min_h_root, min_h_tip, min_h_rotor])
        h_list = h_mat.max(axis=0)
        psi_list = self.calc_psi(tw_tg_list, z_cg_max, h_list, x_cg_range[1])

        if np.nanmin(psi_list) > psi:
            return None, None, "could not satisfy turn-over requirement"

        selected_idx = np.argmin(np.abs(psi_list[np.logical_not(np.isnan(psi_list))] - psi))
        tw_tg = tw_tg_list[selected_idx]
        h = h_list[selected_idx]

        plt.subplot(121)
        plt.title("height")
        plt.plot(tw_tg_list, min_h_tipback, label="tipback")
        plt.plot(tw_tg_list, min_h_root, label="root")
        plt.plot(tw_tg_list, min_h_tip, label="tip")
        plt.plot(tw_tg_list, min_h_rotor, label="rotor")
        plt.legend()

        plt.subplot(122)
        plt.title("psi")
        plt.plot(tw_tg_list, np.rad2deg(psi_list))
        plt.axhline(55)
        plt.show()

        if self.calc_cg_tipback(x_cg_range[1], z_cg_max, h) < theta:
            return None, None, "could not satisfy tipback requirement for CG"

        reason = reasons[np.argmax(h_mat[:, selected_idx])]

        return tw_tg, h, reason

    def plot_lg(self, x_cg_range: list, z_cg_max: float,
                x_ng: float, x_tg: float, tw_ng: float, tw_tg: float,
                h: float):
        """
        Create a side and front view plot of the given landing gear
        configuration.
        :param x_cg_range: x-range of CG locations
        :param z_cg_max: z-location of the highest CG
        :param x_ng: x-location of the nose gear
        :param x_tg: x-location of the tail gear
        :param tw_ng: track width of the nose  gear
        :param tw_tg: track width of the tail gear
        :param h: height of the landing gear
        """
        # plot side view
        plt.subplot(211)
        # plot CG range
        plt.plot(x_cg_range, [z_cg_max, z_cg_max], color="k", marker="o",
                 label="CG range")
        # plot nose gear
        plt.scatter([x_ng], [-h], color="tab:blue", label="Nose gear")
        # plot main landing gear
        plt.scatter([x_tg], [-h], color="tab:orange",
                    label="Tail gear")
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')

        # plot front view
        plt.subplot(212)
        # plot CG
        plt.scatter([0], [z_cg_max], color="k", label="CG range")
        # plot nose gear
        plt.scatter([-tw_ng/2, tw_ng/2], [-h, -h], color="tab:blue",
                    label="Nose gear")
        # plot tail gear
        plt.scatter([-tw_tg/2, tw_tg/2], [-h, -h], color="tab:orange",
                    label="Tail gear")
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')


if __name__ == "__main__":
    x_ng = 1.75
    x_tg = 6.5
    tw_ng = 1.85
    tilt_max_rad = 1
    y_tilt = 0.45
    tilt_tip_rad = 0.6
    b = 8.2
    taper = 0.45
    z_wing = 0.3
    gamma = np.deg2rad(-0.5)
    y_max_rotor = b/2
    z_offset_rotor = 0
    rotor_rad = 0.6
    l_fus = 7.5
    h_fus = 1.7
    x_cg_range = [2.54, 2.61]
    z_cg = 0.4*1.7
    lgc = LandingGearCalc(x_ng, x_tg, tw_ng, tilt_max_rad, y_tilt, tilt_tip_rad, b, taper, z_wing, gamma, y_max_rotor, z_offset_rotor, rotor_rad, l_fus, h_fus)
    tw, h, reason = lgc.optimum_placement(x_cg_range, z_cg, np.deg2rad(15), np.deg2rad(5), np.deg2rad(55), 0.08)
    print("tw_tg:", tw, "h:", h, "reason:", reason)
    lgc.plot_lg(x_cg_range, z_cg, x_ng, x_tg, tw_ng, tw, h)
    plt.show()

