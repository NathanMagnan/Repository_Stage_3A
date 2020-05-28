## imports
import numpy as np
import matplotlib.pyplot as plt
import os

print("All imports successful")

## Drawing
print("starting to plot the results")
n_kernels = 8
Performances = [(0.248043259118587, 0.005362297514592487), (0.28031679770028595, 0.055898993975172596), (0.18340983108760123, 0.029205087580071478), (0.1836357742395285, 0.016012992592860896), (0.1622188950611747, 0.02839762103846572), (0.1687407823984622, 0.03461049551622683), (0.16162315968701507, 0.024040478889329774), (0.160260570051421, 0.005299883541867323)]
Kernel_names = ['RBF isotropic', 'Exponential', 'Matern32', 'Matern52', 'anisotropic', 'bounded', ' prior', 'sgc']

figure = plt.figure()
ax = figure.gca()

ax.set_title("Performances of the different kernels")
ax.set_xlabel("Kernels")
ax.set_ylabel("Performance (arbitrary unit)")

for i in range(n_kernels):
    ax.errorbar(x = [Kernel_names[i]], y = Performances[i][0], yerr = Performances[i][1], fmt = "o")

my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_file = 'Comparison_between_kernels'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)

print("results plotted and saved")