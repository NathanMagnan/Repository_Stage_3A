## imports
import numpy as np
import matplotlib.pyplot as plt

print("All imports successful")

## Drawing
print("starting to plot the results")
n_kernels = 5
Performances = [(1.7431731128219348, 0.03424299249080456), (1.1104506855727245, 0.12047927606391293), (1.4527061862685788, 0.21480454208667255), (1.1294659311180384, 0.1205993185368657), (1.226891725125497, 0.0651931666909746)]
Kernel_names = ['RBF isotropic', 'RBF anisotropic', 'Exponential', 'Matern32', 'Matern52']

figure = plt.figure()
ax = figure.gca()

ax.set_title("Performances of the different kernels")
ax.set_xlabel("Kernels")
ax.set_ylabel("Performance (arbitrary unit)")

for i in range(n_kernels):
    ax.errorbar(x = [Kernel_names[i]], y = Performances[i][0], yerr = Performances[i][1], fmt = "o")

my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_file = 'Comparison between_kernels'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file, format = 'pdf')

print("results plotted and saved")