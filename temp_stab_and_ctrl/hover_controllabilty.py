"""
ACAI hover controllability criterion calculation. Based on:

Controllability Test Procedures for a class of hexacopters
Matlab files for "Controllability Analysis for a Class of Multirotors Subject to Rotor Failure/Wear"

By Guang-Xun Du, Quan Quan, Binxian Yang and Kai-Yuan Cai
Beihang Universty, Beijing, China
05/05/2012
"""

import numpy as np
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from matplotlib import colors as mc
import colorsys
from dataclasses import dataclass
from itertools import combinations
import constants as consts

# numbers of states and control variables
# Hard-coded because they would not make sense as arguments
n_states = 8  # x = [h, phi, theta, psi, vh, p, q, r]
n_ctrl = 4  # u = F - G = [T, L, M, N] - [ma*g, 0, 0, 0]


@dataclass
class Rotor:
    """
    Container for data related to one rotor.
    @author: Jakob Schoser
    """
    x: float  # x (longitudinal) position w.r.t the centre of mass [m]
    y: float  # y (lateral) position w.r.t the centre of mass [m]
    K: float  # maximum lift [N]
    ku: float  # ratio between reactive torque and lift [m]
    eta: float  # efficiency parameter [-]
    ccw: bool  # rotation direction [-]


class HoverControlCalcBase:
    """
    A hover controllability calculator that can handle any aircraft geometry.
    The coordinate system in this class has its origin at the nose, with the
    x-axis pointing backwards and the y-axis pointing towards starboard.
    @author: Jakob Schoser
    """
    def __init__(self, m: float, rotors: list):
        """
        Construct an ACAI calculator object for a given aircraft.
        :param m: Mass of the aircraft.
        :param rotors: List containing all rotors.
        """
        self.m = m
        self.rotors = rotors

    def plot_rotors(self):
        """
        Create a plot of the rotors indicating their position,
        spinning direction and effectiveness. Orange rotors turn clockwise,
        blue rotors turn counter-clockwise. The darker the colour, the lower
        the effectiveness (black corresponds to eta = 0).
        """
        for rotor in self.rotors:
            color = "tab:blue" if rotor.ccw else "tab:orange"
            color = colorsys.rgb_to_hls(*mc.to_rgb(color))
            color = colorsys.hls_to_rgb(color[0], rotor.eta*color[1], color[2])
            plt.scatter(rotor.x, rotor.y, color=color)

    def acai(self, cg: list) -> float:
        """
        Calculate the ACAI
        :param cg: [x, y]-location of the CG of the aircraft
        :return: ACAI
        """
        # control effectiveness matrix (equation 5)
        Bf = np.zeros((n_ctrl, len(self.rotors)))

        fc = np.zeros(len(self.rotors))

        for i, r in enumerate(self.rotors):
            Bf[0, i] = r.eta  # lift
            Bf[1, i] = -r.eta * (r.y - cg[1])  # roll torque
            Bf[2, i] = r.eta * (r.x - cg[0])  # pitch torque
            Bf[3, i] = (-1 if not r.ccw else 1) * r.eta * r.ku  # yaw torque

            fc[i] = 0.5 * r.K

        if np.linalg.matrix_rank(Bf) < n_ctrl:
            return -1E6

        G = np.zeros(n_ctrl)
        G[0] = self.m * consts.g

        m = Bf.shape[1]
        M = np.arange(m)
        S1 = np.asarray([s for s in combinations(M, 3)])
        S2 = np.asarray([np.setdiff1d(M, s) for s in S1])
        d = np.zeros(S1.shape[0])

        for j in range(S1.shape[0]):
            # equation 11
            B1 = Bf[:, S1[j, :]]
            B2 = Bf[:, S2[j, :]]

            if np.linalg.matrix_rank(B1) == 3:
                # equation 15
                L = np.diag(2 * fc[S2[j, :]])

                # equation 16
                xi = null_space(B1.transpose())
                xi /= np.linalg.norm(xi)

                # equation 14
                xiB2 = xi.transpose() @ B2
                d[j] = (0.5 * np.sign(xiB2) @ L @ xiB2.transpose()
                        - np.abs(xi.transpose() @ (Bf @ fc - G)))

            else:
                d[j] = 1E6

        # equation 13
        return np.sign(np.min(d)) * np.min(np.abs(d))

    def rank_cab(self, Jx=1., Jy=1., Jz=1.) -> int:
        """
        Calculate the rank of the controllability matrix, which must be 8 for
        the aircraft to be controllable. The moments of inertia are accessible
        as optional parameters, but they do not affect the output.
        :param Jx: Moment of inertia about the x-axis
        :param Jy: Moment of inertia about the y-axis
        :param Jz: Moment of inertia about the z-axis
        :return: Rank of the controllability matrix.
        """
        # state matrix
        A = np.zeros((n_states, n_states))
        A[:n_states // 2, n_states // 2:] = np.eye(n_states // 2)

        # control matrix
        B = np.zeros((n_states, n_ctrl))
        Jf_inv = np.diag([-1 / self.m, 1 / Jx, 1 / Jy, 1 / Jz])
        B[n_states // 2:, :] = Jf_inv

        # controllability system
        Cab = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(8)])
        return np.linalg.matrix_rank(Cab)

    def controllable(self, cg: list, margin=1e-10) -> bool:
        """
        Check both criteria for controllability: rank(C(AB)) = 8 and ACAI > 0.
        :param cg: [x, y]-location of the CG of the aircraft
        :param margin: smallest value of the ACAI that counts as controllable
        :return: Boolean indicating whether the aircraft is controllable or not
        """
        if self.rank_cab() < n_states or self.acai(cg) <= margin:
            return False

        return True

    def calc_x_cg_range(self, x_min: float, x_max: float,
                        dx: float, y: float) -> tuple:
        """
        Calculate the allowable CG range in x-direction for a given y-location
        :param x_min: Minimum x-location in the considered CG range
        :param x_max: Maximum x-location in the considered CG range
        :param dx: Resolution of CG range evaluation
        :param y: y-location of the CG of the aircraft
        :return: (most forward CG, most aft CG). (None, None) if the aircraft
        is never controllable
        """
        x_front, x_aft = None, None
        was_controllable = False

        for x in np.arange(x_min, x_max, dx):
            controllable = self.controllable([x, y])
            if not was_controllable and controllable:
                x_front = x
            elif was_controllable and not controllable:
                x_aft = x
                break
            was_controllable = controllable

        return x_front, x_aft

    def reset_rotors(self):
        """
        Reset the efficiency of all rotors to 1
        """
        for r in self.rotors:
            r.eta = 1

    def fail_rotors(self, failed: list):
        """
        Set the efficiency of the given motors to 0 (simulating a failure)
        :param failed: List of indices for the rotors that failed
        """
        self.reset_rotors()
        for f in failed:
            self.rotors[f].eta = 0

    def find_max_allowable_rotor_failures(self, cg: list,
                                          max_n_failures: int) -> list:
        """
        Find the combinations of rotor failures that allow for the highest
        number of rotor failures without becoming uncontrollable.
        :param cg: [x, y]-location of the CG of the aircraft
        :param max_n_failures: Maximum number of failures considered
        :return: Lists of the different controllable failure patterns,
        sorted into sub-lists by number of engine failures. Each failure
        pattern is represented by a list of rotor indices
        """
        self.reset_rotors()
        idcs = list(range(len(self.rotors)))  # rotor indices

        # return an empty list if the aircraft is not controllable with all
        # rotors active
        if not self.controllable(cg):
            return []

        controllable_combos = []

        for n_failures in range(1, min(len(self.rotors), max_n_failures) + 1):
            if not controllable_combos:
                test_combos = [[i] for i in idcs]

            # based on the controllable combinations in the last step,
            # determine the combinations to be tested in this step
            else:
                test_combos = []
                for combo in controllable_combos[-1]:
                    for i in range(combo[-1] + 1, len(self.rotors)):
                        test_combos.append(combo + [i])

            controllable_combos.append([])

            for combo in test_combos:
                self.fail_rotors(combo)
                if self.controllable(cg):
                    controllable_combos[-1].append(combo)

            # return an empty list if no controllable failure combinations
            # were found in the last iteration
            if not controllable_combos[-1]:
                break

        self.reset_rotors()
        return controllable_combos

    def calc_crit_x_cg_range(self, x_min: float, x_max: float, dx: float,
                             failure_eval_cg: list, n_failures: list) -> list:
        """
        Calculate the allowable CG range in x-direction for a given y-location,
        such that the CG location is not critical for controllability
        :param x_min: Minimum x-location in the considered CG range
        :param x_max: Maximum x-location in the considered CG range
        :param dx: Resolution of CG range evaluation
        :param failure_eval_cg: [x, y]-location of the CG that is used to
        find the maximum allowable rotor failures.
        :param n_failures: List of the different numbers of engine failures
        for which the critical CG limits will be calculated
        :return: [(most forward CG, most aft CG, rotor failures leading to
        front limit, rotor failures leading to aft limit) for each number
        of engine failures]. The return values are replaced by
        (None, None, None, None) if the aircraft is not controllable with
        any CG location
        """
        # assume central CG to find allowable rotor failures. This is the
        # most lenient of CGs, so it is guaranteed that all relevant failure
        # combinations will be considered in the next step
        failure_combos = self.find_max_allowable_rotor_failures(
            failure_eval_cg, max(n_failures)
        )

        return_list = []
        for failed in n_failures:
            failures = failure_combos[failed - 1]

            x_front, x_aft = None, None
            limiting_failures_front, limiting_failures_aft = None, None
            for combo in failures:
                self.fail_rotors(combo)
                _x_front, _x_aft = self.calc_x_cg_range(x_min, x_max, dx,
                                                        failure_eval_cg[1])

                # go to the next combination if this one cannot be controllable
                # with the given CG
                if _x_front is None:
                    continue

                if x_front is None:
                    x_front = _x_front
                    x_aft = _x_aft

                    limiting_failures_front = combo
                    limiting_failures_aft = combo
                else:
                    if _x_front > x_front:
                        x_front = _x_front
                        limiting_failures_front = combo
                    if _x_aft < x_aft:
                        x_aft = _x_aft
                        limiting_failures_aft = combo

            return_list.append((
                x_front, x_aft, limiting_failures_front, limiting_failures_aft
            ))

        self.reset_rotors()
        return return_list


class HoverControlCalcTandem(HoverControlCalcBase):
    """
    An extension of the HoverControlCalcBase which makes it easier to
    set up the rotors for a tandem wing aircraft.
    @author Jakob Schoser
    """
    def __init__(self, m: float, n_rot_f: int, n_rot_r: int, x_wf: float,
                 x_wr: float, rot_y_range_f: list, rot_y_range_r: list,
                 K: float, ku: float):
        """
        Construct an ACAI calculator object for a given tandem wing aircraft.
        The default rotation direction is set to be inboard rotating.
        :param m: Mass of the aircraft
        :param n_rot_f: Number of rotors on the front wing (must be even)
        :param n_rot_r: Number of rotors on the rear wing (must be even)
        :param x_wf: x-location of the rotors on the front wing
        :param x_wr: x-location of the rotors on the rear wing
        :param rot_y_range_f: y-range inside of which the rotors on the front
        starboard half-wing are distributed
        :param rot_y_range_r: y-range inside of which the rotors on the rear
        starboard half-wing are distributed
        :param K: Maximum thrust per rotor
        :param ku: Ratio between reactive torque and thrust for each rotor
        """
        rotors = []

        xs = n_rot_f * [x_wf] + n_rot_r * [x_wr]
        ys = np.concatenate((
            np.linspace(-rot_y_range_f[1], -rot_y_range_f[0], n_rot_f // 2),
            np.linspace(rot_y_range_f[0], rot_y_range_f[1], n_rot_f // 2),
            np.linspace(-rot_y_range_r[1], -rot_y_range_r[0], n_rot_r // 2),
            np.linspace(rot_y_range_r[0], rot_y_range_r[1], n_rot_r // 2)
        ))

        ccws = (n_rot_f // 2 * [True] + n_rot_f // 2 * [False]
                + n_rot_r // 2 * [True] + n_rot_r // 2 * [False])

        for i in range(n_rot_f + n_rot_r):
            rotors.append(Rotor(xs[i], ys[i], K, ku, 1, ccws[i]))

        super().__init__(m, rotors)

if __name__ == "__main__":
    import json

    f = open('')
    data = json.load(f)
    istandem = False
## From JSON
    Sproj = data["StotS"] * data["S"]
    m = data["mtom"]
    n_rot_f =     ##must be even
    n_rot_r =     ##must be even

    if istandem:
        rot_y_range_f = [0, data["b1"]/2]
        rot_y_range_r = [0, data["b2"]/2]
    else:
        rot_y_range_f = [0, data["b"] / 2]
        rot_y_range_r = [data["bh"]/2, data["bh"]/2]
    K = 9.80665*m*1.2*(1+1/(n_rot_r+n_rot_f-1))*(1+1.225*4*Sproj/(m*9.80665))/(n_rot_r+n_rot_f)

    ## NOT from JSON
    ku = 0.1
    x_wf =
    x_wr =

    hovercalc = HoverControlCalcTandem(m, n_rot_f, n_rot_r, x_wf, x_wr, rot_y_range_f, rot_y_range_r, K, ku)

    hovercalc_same_instance= HoverControlCalcBase(m, rotors)


    x_min = 0
    x_max = data["lfus"]
    dx = 0.01
    failure_eval_cg = [??,0]
    n_failures = [0,1,2]
    cg_range = hovercalc_same_instance.calc_crit_x_cg_range(x_min, x_max, dx,failure_eval_cg, n_failures)

