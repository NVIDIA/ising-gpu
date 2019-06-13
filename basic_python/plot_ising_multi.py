import glob
import matplotlib.pyplot as plt
import numpy as np

files = sorted(glob.glob("final_rank*.txt"))

if (len(files) == 0):
    raise Exception("Could not find any lattice files. Expecting files named 'final_rank*.txt' for processing")

lattice = np.loadtxt(files[0], dtype=np.int32)
for i,f in enumerate(files):
    if i == 0: continue
    lattice = np.concatenate((lattice, np.loadtxt(f, dtype=np.int32)))

plt.imshow(lattice)
plt.title('Final Lattice Configuration')
plt.colorbar()
plt.show()

