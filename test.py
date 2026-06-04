from numcompute.io import read_csv
import numpy as np  

file_path = r"C:\Users\aarib\OneDrive\Documents\AdelUni\Semester 1\ProgrammingAI\programming_task_1\data\Iris.csv"
data = read_csv(file_path)
# print(data)
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.preprocessing import OneHotEncoder

X = np.array([
    [25, "Male"],
    [30, "Female"],
    [22, "Female"]
])

#pipeline applying encoding step
pipe = Pipeline([
    ("encode", OneHotEncoder())
])

X_out = pipe.fit_transform(X)

print("Original:\n", X)
print("\nPipeline Output:\n", X_out)

data_out = pipe.fit_transform(data)
print("\nData Original:\n", data)
print("\nData Pipeline Output:\n", data_out)

from numcompute.stats import mean, median, std, minimum, maximum, quantiles, describe

# For stats.py

print("\n===== STAT TESTS =====")

# Normal Case 

X = np.array([1, 2, 3, 4, 5])
print("Mean:", mean(X))                  # 3
print("Median:", median(X))              # 3
print("Std:", std(X))                   # ~1.414
print("Min:", minimum(X))               # 1
print("Max:", maximum(X))               # 5
print("Quantiles:", quantiles(X, [25, 50, 75]))

datax = data[1:,1].astype(float)
print("\nData Mean:", mean(datax)) 
print("Data Median:", median(datax))
print("Data Std:", std(datax))     
print("Data Min:", minimum(datax))
print("Data Max:", maximum(datax))
print("Data Quantiles:", quantiles(datax, [25, 50, 75]))
print("\nData Describe:\n", describe(datax))