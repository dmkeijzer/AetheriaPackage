class CgCalculator:
    """
    Class to calculate the CG and generate loading diagrams for the aircraft.
    The coordinate system has its origin at the nose the height of the bottom
    of the fuselage. The x-axis points backwards, the y-axis points towards
    starboard and the z-axis points upwards.

    @author: Jakob Schoser
    """
    def __init__(self, m_wf: float, m_wr: float, m_fus: float, m_bat: float,
                 m_cargo: float, m_pax: float, m_pil: float, cg_fus: list,
                 cg_bat: list, cg_cargo: list, cg_pax: list, cg_pil: list,
                 m_vt: float, cg_vt: float):
        """
        Constructs a CG calculator object for a given aircraft. It takes the
        masses and CG of all components, except for the wings where the CG
        is not fixed since it will be optimised during the design for stability
        and control.
        :param m_wf: Mass of the front wing (incl. engines)
        :param m_wr: Mass of the rear wing (incl. engines)
        :param m_fus: Empty mass of the aircraft without the wings and battery
        :param m_bat: Mass of the battery
        :param m_cargo: Mass of the cargo
        :param m_pax: Mass of one passenger (incl. personal luggage)
        :param m_pil: Mass of the pilot
        :param cg_fus: [x, y, z]-location of the CG of m_fus
        :param cg_bat: [x, y, z]-location of the CG of the battery
        :param cg_cargo: [x, y, z]-location of the CG of the cargo
        :param cg_pax: [x, y, z]-locations of the CGs of all passengers in a list
        :param cg_pil: [x, y, z]-location of the CG of the pilot
        """
        self.m_wf = m_wf
        self.m_wr = m_wr
        self.m_fus = m_fus
        self.m_bat = m_bat
        self.m_cargo = m_cargo
        self.m_pax = m_pax
        self.m_pil = m_pil
        self.m_vt  = m_vt

        self.cg_fus = cg_fus
        self.cg_bat = cg_bat
        self.cg_cargo = cg_cargo
        self.cg_pax = cg_pax
        self.cg_pil = cg_pil
        self.cg_vt  = cg_vt

    def calc_cg(self, cg_wf: list, cg_wr: list, loaded_cargo: bool,
                seated_pax: list, seated_pil: bool) -> tuple:
        """
        Calculates the CG of the aircraft for given wing positions and
        occupied seats.
        :param cg_wf: [x, z]-location of the CG of the front wing
        :param cg_wr: [x, z]-location of the CG of the rear wing
        :param loaded_cargo: boolean indicating whether cargo has been loaded
        :param seated_pax: list of passenger indices that are seated
        :param seated_pil: boolean indicating whether pilot is seated
        :return: [x, y, z]-location of the CG of the aircraft
        """
        x = (self.m_wf * cg_wf[0] + self.m_wr * cg_wr[0]
             + self.m_fus * self.cg_fus[0] + self.m_bat * self.cg_bat[0] + self.m_vt*self.cg_vt)
        # assume that the CGs of the wings are on the symmetry plane
        y = self.m_fus * self.cg_fus[1] + self.m_bat * self.cg_bat[1]
        z = (self.m_wf * cg_wf[1] + self.m_wr * cg_wr[1]
             + self.m_fus * self.cg_fus[2] + self.m_bat * self.cg_bat[2])
        m = self.m_wf + self.m_wr + self.m_fus + self.m_bat + self.m_vt

        if loaded_cargo:
            x += self.m_cargo * self.cg_cargo[0]
            y += self.m_cargo * self.cg_cargo[1]
            z += self.m_cargo * self.cg_cargo[2]
            m += self.m_cargo

        for seated in seated_pax:
            x += self.m_pax * self.cg_pax[seated][0]
            y += self.m_pax * self.cg_pax[seated][1]
            z += self.m_pax * self.cg_pax[seated][2]
            m += self.m_pax

        if seated_pil:
            x += self.m_pil * self.cg_pil[0]
            y += self.m_pil * self.cg_pil[1]
            z += self.m_pil * self.cg_pil[2]
            m += self.m_pil

        x /= m
        y /= m
        z /= m

        return x, y, z

    def calc_cg_range(self, cg_wf: list, cg_wr: list,
                      order=("cargo", "pil", 0, 1, 2, 3)) -> tuple:
        """
        Calculate the CG range during loading of the aircraft.
        :param cg_wf: [x, z]-location of the CG of the front wing
        :param cg_wr: [x, z]-location of the CG of the rear wing
        :param order: Order of loading different parts. May contain "cargo",
        "pil", and numbers indicating passenger IDs starting from 0.
        :return: [most forward CG, most aft CG],
        [most port CG, most starboard CG], [lowest CG, highest CG]
        """

        x_front, x_aft = None, None
        y_port, y_star = None, None
        z_bottom, z_top = None, None

        loaded_cargo = False
        seated_pax = []
        seated_pil = False

        for item in order:
            if item == "cargo":
                loaded_cargo = True
            elif item == "pil":
                seated_pil = True
            else:
                seated_pax.append(item)

            x, y, z = self.calc_cg(cg_wf, cg_wr, loaded_cargo,
                                   seated_pax, seated_pil)

            if x_front is None:
                x_front, x_aft = x, x
                y_port, y_star = y, y
                z_bottom, z_top = z, z
            else:
                x_front = min(x_front, x)
                x_aft = max(x_aft, x)

                y_port = min(y_port, y)
                y_star = max(y_star, y)

                z_bottom = min(z_bottom, z)
                z_top = max(z_top, z)

        return [x_front, x_aft], [y_star, y_port], [z_bottom, z_top]
