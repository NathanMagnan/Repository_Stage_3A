## imports
import numpy as np
import matplotlib.pyplot as plt
import os

print("All imports successful")

## Drawing
print("starting to plot the results")
n_kernels = 5
Performances = [(1.1391121990209963, 0.02625892867056413), (1.743217622405576, 0.03425304067544455), (1.4526183087437272, 0.21462319865916168), (1.1294199012053834, 0.12058342522599383), (1.2268530535357431, 0.06521446989864907)]
Kernel_names = ['RBF anisotropic', 'RBF isotropic', 'Exponential', 'Matern32', 'Matern52']

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
plt.savefig(my_file, format = 'pdf')

print("results plotted and saved")