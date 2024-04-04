# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 00:47:38 2024

@author: Nathan
"""

# check_dmg_index = dmg_index[:,16,8:-1]
import matplotlib.pyplot as plt


# count = 1
# plt.figure(figsize=(10,10))
# for i in range(16,24):
#     check_dmg_index = dmg_index[:,i,8:-1]
#     plt.subplot(4,2,count)
#     plt.contourf(check_dmg_index)
#     count += 1
#     plt.colorbar()
# plt.show()

count = 1
monotonicity = np.zeros((np.shape(dmg_index)[0], np.shape(dmg_index)[1]), dtype= bool)
for i in range(np.shape(dmg_index)[1]):
    check_dmg_index = dmg_index[:,i,:]
    monotonicity[:,i] = np.all(check_dmg_index[:, 1:] >= check_dmg_index[:, :-1], axis=1)