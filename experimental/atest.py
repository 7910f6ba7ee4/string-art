import numpy as np

a = np.array([[1,2,3,4], [1,2,3,5]])
print(np.unravel_index(np.argmin(a, axis=None), a.shape))
a[0, 0] = 100
print(np.unravel_index(np.argmin(a, axis=None), a.shape))