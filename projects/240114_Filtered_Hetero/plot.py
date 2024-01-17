#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

oc_acc = [45.00,70.35,63.33]
os_acc = [32.14,62.71,43.05]
fc_acc = [37.98,40.15,43.08]
fs_acc = [35.68,38.84,38.57]
re_acc = [80.74,57.82,0.0]
ms_acc = [91.04,79.99,0.0]

oc_std = [1.57,3.32,1.65]
os_std = [1.53,1.46,1.40]
fc_std = [1.09,3.38,2.95]
fs_std = [2.33,1.43,1.83]
re_std = [0.47,2.45,0.0]
ms_std = [0.69,0.04,0.0]

acc_dict = {'original_chameleon_accuracy': oc_acc, 'original_squirrel_accuracy': os_acc,
            'filtered_chameleon_accuracy': fc_acc, 'filtered_squirrel_accuracy': fs_acc,
            'roman_empire_accuracy': re_acc, 'minesweeper_accuracy': ms_acc, 
            
            'original_chameleon_std': oc_std, 'original_squirrel_std': os_std,
            'filtered_chameleon_std': fc_std, 'filtered_squirrel_std': fs_std,
            'roman_empire_std': re_std, 'minesweeper_std': ms_std, 
            }

df = pd.DataFrame(acc_dict, index=['GAT', 'ACM-GCN', 'GREET'])
# df.head()


#%%

chameleon_results = df.plot.bar(rot=0, 
                                y=['original_chameleon_accuracy', 'filtered_chameleon_accuracy'],
                                yerr=[df.original_chameleon_std, df.filtered_chameleon_std]
                                )

squirrel_results = df.plot.bar(rot=0, 
                                y=['original_squirrel_accuracy', 'filtered_squirrel_accuracy'],
                                yerr=[df.original_squirrel_std, df.filtered_squirrel_std]
                                )

roman_empire_results = df.plot.bar(rot=0, y='roman_empire_accuracy', yerr=df.roman_empire_std)

minesweeper_results = df.plot.bar(rot=0, y='minesweeper_accuracy', yerr=df.minesweeper_std)


# %%
#! Chameleon
chameleon_results = df.plot.bar(rot=0, 
                                y=['original_chameleon_accuracy', 'filtered_chameleon_accuracy'],
                                yerr=[df.original_chameleon_std, df.filtered_chameleon_std]
                                )
for container in chameleon_results.containers[1::2]:
    chameleon_results.bar_label(container, label_type='center')
    chameleon_results.legend(bbox_to_anchor=(1, 1.02), loc='upper left')


# %%
#! Squirrel
squirrel_results = df.plot.bar(rot=0, 
                                y=['original_squirrel_accuracy', 'filtered_squirrel_accuracy'],
                                yerr=[df.original_squirrel_std, df.filtered_squirrel_std]
                                )
for container in squirrel_results.containers[1::2]:
    squirrel_results.bar_label(container, label_type='center')
    squirrel_results.legend(bbox_to_anchor=(1, 1.02), loc='upper left')

# %%
#! Roman-empire
roman_empire_results = df.plot.bar(rot=0, 
                                y='roman_empire_accuracy',
                                yerr=df.roman_empire_std
                                )
for container in roman_empire_results.containers[1::2]:
    roman_empire_results.bar_label(container, label_type='center')
    roman_empire_results.legend(bbox_to_anchor=(1, 1.02), loc='upper left')


# %%
#! Minesweeper
minesweeper_results = df.plot.bar(rot=0, 
                                y='minesweeper_accuracy',
                                yerr=df.minesweeper_std
                                )
for container in minesweeper_results.containers[1::2]:
    minesweeper_results.bar_label(container, label_type='center')
    minesweeper_results.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
# %%
