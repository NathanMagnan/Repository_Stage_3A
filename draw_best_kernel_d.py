## imports
import numpy as np
import matplotlib.pyplot as plt
import os

print("All imports successful")

## Drawing
print("starting to plot the results")
n_kernels = 8
Performances = [(0.24350464841398053, 0.043515257941273325), (0.4572760334266503, 0.12035178619521882), (0.28233076385302236, 0.03785708973648219), (0.24885080895492231, 0.0326197704483331), (0.20685441533249155, 0.043335109478190005), (0.21738036659538545, 0.04168739738025657), (0.21536767999660644, 0.04212083437061767), (0.26802912985880367, 0.04303800122469888)]
Kernel_names = ['RBF isotropic', 'Exponential', 'Matern32', 'Matern52', 'RBF anisotropic', 'bounded', 'prior', 'sgc']

figure = plt.figure()
ax = figure.gca()

ax.set_title("Performances of the different kernels")
ax.set_xlabel("Kernels")
ax.set_ylabel("Performance (arbitrary unit)")

for i in range(n_kernels):
    ax.errorbar(x = [Kernel_names[i]], y = Performances[i][0], yerr = Performances[i][1], fmt = "o")

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Comparison_between_kernels_d_2'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()

print("results plotted and saved")