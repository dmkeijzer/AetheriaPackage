from stab_and_ctrl.hover_controllabilty import *


def test_hexacopter_PNPNPN_fully_functional():
    ma = 1.535  # [kg]

    r = 0.275  # [m]
    psi = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])  # [rad]
    x = r * np.cos(psi)  # [m]
    y = r * np.sin(psi)  # [m]
    ccw = np.array([False, True, False, True, False, True])
    ku = 0.1  # [m]
    eta = 1  # [-]
    K = 6.125  # [N]

    rotors = [Rotor(x[i], y[i], K, ku, eta, ccw[i]) for i in range(6)]
    calc = HoverControlCalcBase(ma, rotors)

    assert calc.controllable([0, 0]) and abs(calc.acai([0, 0]) - 1.4861) < 1E-4


def test_hexacopter_PNPNPN_failure1():
    ma = 1.535  # [kg]

    r = 0.275  # [m]
    psi = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])  # [rad]
    x = r * np.cos(psi)  # [m]
    y = r * np.sin(psi)  # [m]
    ccw = np.array([False, True, False, True, False, True])
    ku = 0.1  # [m]
    etas = [0, 1, 1, 1, 1, 1]  # [-]
    K = 6.125  # [N]

    rotors = [Rotor(x[i], y[i], K, ku, etas[i], ccw[i]) for i in range(6)]
    calc = HoverControlCalcBase(ma, rotors)

    assert not calc.controllable([0, 0]) and abs(calc.acai([0, 0])) < 1E-4


def test_hexacopter_PPNNPN_fully_functional():
    ma = 1.535  # [kg]

    r = 0.275  # [m]
    psi = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])  # [rad]
    x = r * np.cos(psi)  # [m]
    y = r * np.sin(psi)  # [m]
    ccw = np.array([False, False, True, True, False, True])
    ku = 0.1  # [m]
    etas = [1, 1, 1, 1, 1, 1]  # [-]
    K = 6.125  # [N]

    rotors = [Rotor(x[i], y[i], K, ku, etas[i], ccw[i]) for i in range(6)]
    calc = HoverControlCalcBase(ma, rotors)

    assert calc.controllable([0, 0]) and abs(calc.acai([0, 0]) - 1.1295) < 1E-4


def test_hexacopter_PPNNPN_failure1():
    ma = 1.535  # [kg]

    r = 0.275  # [m]
    psi = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])  # [rad]
    x = r * np.cos(psi)  # [m]
    y = r * np.sin(psi)  # [m]
    ccw = np.array([False, False, True, True, False, True])
    ku = 0.1  # [m]
    etas = [0, 1, 1, 1, 1, 1]  # [-]
    K = 6.125  # [N]

    rotors = [Rotor(x[i], y[i], K, ku, etas[i], ccw[i]) for i in range(6)]
    calc = HoverControlCalcBase(ma, rotors)

    assert calc.controllable([0, 0]) and abs(calc.acai([0, 0]) - 0.7213) < 1E-4


def test_hexacopter_PPNNPN_failure5():
    ma = 1.535  # [kg]

    r = 0.275  # [m]
    psi = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])  # [rad]
    x = r * np.cos(psi)  # [m]
    y = r * np.sin(psi)  # [m]
    ccw = np.array([False, False, True, True, False, True])
    ku = 0.1  # [m]
    etas = [1, 1, 1, 1, 0, 1]  # [-]
    K = 6.125  # [N]

    rotors = [Rotor(x[i], y[i], K, ku, etas[i], ccw[i]) for i in range(6)]
    calc = HoverControlCalcBase(ma, rotors)

    assert not calc.controllable([0, 0]) and abs(calc.acai([0, 0])) < 1E-4


def test_hexacopter_PPNNPN_heavier():
    ma = 3.535  # [kg]

    r = 0.275  # [m]
    psi = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])  # [rad]
    x = r * np.cos(psi)  # [m]
    y = r * np.sin(psi)  # [m]
    ccw = np.array([False, False, True, True, False, True])
    ku = 0.1  # [m]
    etas = [1, 1, 1, 1, 1, 1]  # [-]
    K = 6.125  # [N]

    rotors = [Rotor(x[i], y[i], K, ku, etas[i], ccw[i]) for i in range(6)]
    calc = HoverControlCalcBase(ma, rotors)

    assert calc.controllable([0, 0]) and abs(calc.acai([0, 0]) - 0.1591) < 1E-4


def test_hexacopter_PPNNPN_wider():
    ma = 1.535  # [kg]

    r = 0.475  # [m]
    psi = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])  # [rad]
    x = r * np.cos(psi)  # [m]
    y = r * np.sin(psi)  # [m]
    ccw = np.array([False, False, True, True, False, True])
    ku = 0.1  # [m]
    etas = [1, 1, 1, 1, 1, 1]  # [-]
    K = 6.125  # [N]

    rotors = [Rotor(x[i], y[i], K, ku, etas[i], ccw[i]) for i in range(6)]
    calc = HoverControlCalcBase(ma, rotors)

    assert calc.controllable([0, 0]) and abs(calc.acai([0, 0]) - 1.1903) < 1E-4
