"""
Trade-off of the power system
"""

# Weights of the criteria
weights = {"Specific energy density": 1,
           "Volumetric energy density": 0.8,
           "Power density": 0.8,
           "Safety score": 0,
           "Operational life": 0.5,
           "Cost": 0.3}

# List of power system options
li_ion = {"Specific energy density": 250,
          "Volumetric energy density": 650,
          "Power density": 1000,
          "Safety score": 3,
          "Operational life": 1000,
          "Cost": 80}  # we expect the price of an average battery pack to be around $94/kWh by 2024 and $62/kWh by 2030
# https://about.bnef.com/blog/behind-scenes-take-lithium-ion-battery-prices/
# 125 -> https://www.osti.gov/pages/servlets/purl/1400215

li_S = {"Specific energy density": 550,
        "Volumetric energy density": 650,
        "Power density": 10000,
        "Safety score": 3,
        "Operational life": 800,
        "Cost": 87}  # with Li-S potentially available at about €72 per kWh – 30% less than comparable Li-ion technology
# https://horizon-magazine.eu/article/cheaper-lighter-and-more-energy-dense-promise-lithium-sulphur-batteries.html

li_metal = {"Specific energy density": 500,
            "Volumetric energy density": 1000,
            "Power density": 1000,
            "Safety score": 4,
            "Operational life": 800,
            "Cost": 100}

solid_st = {"Specific energy density": 500,
            "Volumetric energy density": 1000,
            "Power density": 10000,
            "Safety score": 5,
            "Operational life": 10000,
            "Cost": 100}

GH2_fuelcell = {"Specific energy density": 1,
                "Volumetric energy density": 1,
                "Power density": 750,
                "Safety score": 1,
                "Operational life": 1,
                "Cost": 10}

LH2_fuelcell = {"Specific energy density": 1500,
                "Volumetric energy density": 1000,
                "Power density": 850,
                "Safety score": 1,
                "Operational life": 1,
                "Cost": 10}  # $/kWh


def score(system):
    return (weights["Specific energy density"]*system["Specific energy density"] / li_ion["Specific energy density"] +
            weights["Volumetric energy density"]*system["Volumetric energy density"] / li_ion["Volumetric energy density"] +
            weights["Power density"]*system["Power density"] / li_ion["Power density"] +
            weights["Safety score"]*system["Safety score"] / li_ion["Safety score"] +
            weights["Operational life"]*system["Operational life"] / li_ion["Operational life"] +
            weights["Cost"]*(1/system["Cost"]) * li_ion["Cost"]) / sum(weights.values() )


print("Trade-off scores:")
print("Lithium-ion:", score(li_ion))
print("Lithium-sulfur:", score(li_S))
print("Lithium-metal by SES:", score(li_metal))
print("Solid state:", score(solid_st))
print("High pressure gas hydrogen fuel cell:", score(GH2_fuelcell))
print("Liquid hydrogen fuel cell:", score(LH2_fuelcell))
