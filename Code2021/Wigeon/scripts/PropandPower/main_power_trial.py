import Final_optimization.constants_final as const
import battery as bat

sp_en_den = const.sp_en_den
vol_en_den = const.vol_en_den
bat_cost = const.bat_cost
DoD = const.DoD
P_den = const.P_den
EOL_C = const.EOL_C

energy = 301111.1
P_max = 1809362.3556091622
safety = 1.0

bat = bat.Battery(sp_en_den, vol_en_den, energy, bat_cost, DoD, P_den, P_max, safety, EOL_C)


print("mass", bat.mass())
print()
print("volume", bat.volume())
print()
print("cost", bat.price())

print("mass_energy", bat.mass_both()[0])
print("mass_power", bat.mass_both()[1])

if bat.mass_both()[0] > bat.mass_both()[1]:
    print("Energy limiting")
else:
    print("Power limiting")

