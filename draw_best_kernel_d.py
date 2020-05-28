## imports
import numpy as np
import matplotlib.pyplot as plt
import os

print("All imports successful")

## Drawing
print("starting to plot the results")
n_kernels = 8
Performances = [(1.5839707633350024, 0.28221748772229766), (1.1658135630347024, 0.27904929863404476), (1.4214899503229779, 0.207887393767021), (1.5647038147870622, 0.22500253101212803), (1.4218844429272284, 0.2993943034053358), (1.4732638325191334, 0.28289909676520386), (1.4685413460244996, 0.2880984936896126), (1.5642631954170114, 0.33143822780299864)]
Kernel_names = ['RBF isotropic', 'Exponential', 'Matern32', 'Matern52', 'RBF anisotropic', 'bounded', 'prior', 'sgc']

figure = plt.figure()
ax = figure.gca()

ax.set_title("Performances of the different kernels")
ax.set_xlabel("Kernels")
ax.set_ylabel("Performance (arbitrary unit)")

for i in range(n_kernels):
    ax.errorbar(x = [Kernel_names[i]], y = Performances[i][0], yerr = Performances[i][1], fmt = "o")

my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_file = 'Comparison_between_kernels_d'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)

print("results plotted and saved")