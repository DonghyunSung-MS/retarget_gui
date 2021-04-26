from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt


#2d case
class NDSphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.dim = center.shape[0]

    def generate_sample(self, N):

        vec = np.random.rand(N, self.dim) - 0.5
        vec = vec/np.linalg.norm(vec, axis=1, keepdims=True)
        return self.radius * vec + self.center.reshape(1, -1) + 0.1*np.random.randn(N, self.dim)

sphere = NDSphere(np.array([3., 1.]),1.0)

data = sphere.generate_sample(100)

# plt.plot(data[:,0], data[:,1], 'o')
# plt.show()

g = np.mean(data, axis=0, keepdims=False)
g = np.hstack([g, np.zeros(1)]).reshape(-1, 1)
H = - np.identity(sphere.dim + 1)
H[-1, -1] = 1.0
print(g)
print(H)

print(inv(H) @ g)