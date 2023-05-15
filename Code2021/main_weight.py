import structures.Weight as w

mtom = 1972  # maximum take-off mass from statistical data - Class I estimation
S1, S2 = 5.5, 5.5  # surface areas of wing one and two
A = 14  # aspect ratio of a wing, not aircraft
n_ult = 3.2 * 1.5  # 3.2 is the max we found, 1.5 is the safety factor
Pmax = 15.25  # this is defined as maximum perimeter in Roskam, so i took top down view of the fuselage perimeter
lf = 7.2  # length of fuselage

# From project guide: 95 kg per pax (including luggage)
n_pax = 5  # number of passengers (pilot included)
m_pax = 88  # assume average mass of a passenger according to Google
cargo_m = (95-m_pax)*n_pax  # Use difference of pax+luggage - pax to get cargo mass

n_prop = 16  # number of engines

pos_fus = 3.6  # fuselage centre of mass away from the nose
pos_lgear = 4  # landing gear position away from the nose
pos_frontwing, pos_backwing = 0.5, 7  # positions of the wings away from the nose
m_prop = [30] * 16  # list of mass of engines (so 30 kg per engine with nacelle and propeller)
pos_prop = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
            6.7]  # 8 on front wing and 8 on back wing
wing = w.Wing(mtom, S1, S2, n_ult, A, [pos_frontwing, pos_backwing])
fuselage = w.Fuselage(mtom, Pmax, lf, n_pax, pos_fus)
lgear = w.LandingGear(mtom, pos_lgear)
props = w.Propulsion(n_prop, m_prop, pos_prop)
weight = w.Weight(m_pax, wing, fuselage, lgear, props, cargo_m=cargo_m, cargo_pos=6, battery_m=400, battery_pos=3.6,
                p_pax=[1.5, 3, 3, 4.2, 4.2])
print(weight.print_weight_fractions())




