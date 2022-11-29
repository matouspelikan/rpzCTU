import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([1, 0, 1])

print(a)
print(b)

print(a.T[b==1].T)
print(a[:, b==1])


print(b[np.where(b>0)])
