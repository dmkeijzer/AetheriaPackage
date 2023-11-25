import sys
sys.path.append(r".")

from AetheriaPackage.integration import run_integration



mission, wing, engine, aero, fuselage, stability, power = run_integration(r"input/initial_estimate.json" )


print(mission)
