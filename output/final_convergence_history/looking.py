import pandas as pd
import os
import matplotlib.pyplot as plt

files = os.listdir()

print(files[-2])
file = files[-2]

df = pd.read_csv(file)

y = df.iloc[:,1]
print(y)

x = [i for i in range(1,len(y)+1)]
    