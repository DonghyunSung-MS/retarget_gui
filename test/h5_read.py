import h5py
import numpy as np

h5f = h5py.File("test1.h5", "r")

for key in h5f.keys():
    print(h5f[key].shape)