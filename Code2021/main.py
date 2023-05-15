from structures.Material import Material
from structures.SolveLoads import SolveACLoads

aluminum = Material.load(file='data/materials.csv', material='Al 6061', Condition='T6')

print(aluminum)


print(SolveACLoads(1, 0.5, 1.5))
