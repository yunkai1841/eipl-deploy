import numpy as np
from sklearn.metrics import mean_squared_error

file1 = input("file1: ")
file2 = input("file2: ")

a = np.load(file1)
b = np.load(file2)

mses = []
for i in range(len(a)):
    print(a[i])
    print(b[i])
    mse = mean_squared_error(a[i], b[i])
    print(mse)
    mses.append(mse)

print("mean mse: {}".format(np.mean(mses)))
print("max mse: {}".format(max(mses)))
print("min mse: {}".format(min(mses)))
