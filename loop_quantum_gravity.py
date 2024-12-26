import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paramètres de la simulation
grid_size = 10  # Taille de la grille 3D
density = np.random.rand(grid_size, grid_size, grid_size)  # Densité aléatoire pour chaque point de la grille

# Création des coordonnées de la grille
x, y, z = np.indices((grid_size, grid_size, grid_size))

# Visualisation de la grille 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Boucles sur chaque point de la grille
for i in range(grid_size):
    for j in range(grid_size):
        for k in range(grid_size):
            # Dessiner un point si la densité est au-dessus d'un seuil (pour rendre la visualisation plus claire)
            if density[i, j, k] > 0.5:
                ax.scatter(i, j, k, c='b', marker='o', s=density[i, j, k] * 100)

# Configuration des axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Simulation de la Gravité Quantique en Boucles')

plt.show()