import numpy as np


xi_xy = np.load('inputs/xi_xy_test_Nxi6_NR99.npy')
xi_z = np.load('inputs/xi_z_test_Nxi6_NR99.npy')
xi_por = np.load('inputs/xi_por_test_Nxi6_NR99.npy')

xi_site = np.concatenate((xi_xy, xi_z, xi_por), axis = 1)

print(xi_site.shape)

np.save('inputs/xi_site_test.npy', xi_site)