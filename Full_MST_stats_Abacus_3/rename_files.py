import numpy as np
import os

path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Full_MST_stats_Abacus_3'
list_files = os.listdir(path)

Numbers = []
for i in range(len(list_files)):
    old_file_name = list_files[i]
    
    old_file_name_split = old_file_name.split("_")
    
    old_number = int(old_file_name_split[3])
    if not (old_number in Numbers):
        Numbers.append(old_number)
    new_number = Numbers.index(old_number)
    
    new_file_name_split = old_file_name_split.copy()
    new_file_name_split[3] = str(new_number)
    
    new_file_name = '_'.join(new_file_name_split)
    
    old_file_path = os.path.join(path, old_file_name)
    new_file_path = os.path.join(path, new_file_name)
    os.rename(old_file_path, new_file_path)