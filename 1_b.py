import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator

# Definir la función tridimensional f_b
def f_b(x):
    x1, x2, x3 = x
    return 0.75 * np.exp(-((10*x1 - 2)**2) / 4 - ((9*x2 - 2)**2) / 4) + \
           0.65 * np.exp(-((9*x1 + 1)**2) / 9 - ((10*x2 + 1)**2) / 2) + \
           0.55 * np.exp(-((9*x1 - 6)**2) / 4 - ((9*x2 - 3)**2) / 4) - \
           0.01 * np.exp(-((9*x1 - 7)**2) / 4 - ((9*x2 - 3)**2) / 4)

# Generar datos para la interpolación
x1_values = np.linspace(-1, 1, 10)
x2_values = np.linspace(-1, 1, 10)
x3_values = np.linspace(-1, 1, 10)
X1, X2, X3 = np.meshgrid(x1_values, x2_values, x3_values)
data = f_b((X1, X2, X3))

# Crear el interpolador tridimensional
interpolator = RegularGridInterpolator((x1_values, x2_values, x3_values), data)

# Generar una malla de puntos dentro del rango [-1, 1] en cada dimensión
x = np.linspace(-1, 1, 50)
X, Y, Z = np.meshgrid(x, x, x)
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

# Evaluar la función interpolada en los puntos de la malla
interpolated_values = interpolator(points).reshape(X.shape)

# Aplanar las matrices 3D generadas por np.meshgrid
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

# Evaluar la función interpolada en los puntos de la malla
interpolated_values = interpolator(np.column_stack((X, Y, Z))).reshape(X.shape)

# Graficar la función interpolada
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X, Y, Z, cmap=plt.cm.viridis, linewidth=0.2, antialiased=True)  # Superficie interpolada
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Interpolación tridimensional')
plt.show()

