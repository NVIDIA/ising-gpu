import numpy as np
import matplotlib.pyplot as plt

lattice = np.loadtxt("final.txt", dtype=np.int32)
plt.imshow(lattice)
plt.title('Final Lattice Configuration')
plt.colorbar()
plt.show()

