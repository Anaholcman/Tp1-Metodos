import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline

def original_function(x1, x2):
    return 0.75 * np.exp(-((10*x1 - 2)**2) / 4 - ((9*x2 - 2)**2) / 4) + \
        0.65 * np.exp(-((9*x1 + 1)**2) / 9 - ((10*x2 + 1)**2) / 2) + \
        0.55 * np.exp(-((9*x1 - 6)**2) / 4 - ((9*x2 - 3)**2) / 4) - \
        0.01 * np.exp(-((9*x1 - 7)**2) / 4 - ((9*x2 - 3)**2) / 4)

x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)

# Evaluar la función original en los puntos
X1, X2 = np.meshgrid(x1, x2)
Z = original_function(X1, X2)

f = RectBivariateSpline(x1, x2, Z)

# Definir puntos de malla más fina para la superficie interpolada
x1_new = np.linspace(-1, 1, 20)
x2_new = np.linspace(-1, 1, 20)
X1_new, X2_new = np.meshgrid(x1_new, x2_new)
Z_new = f(x1_new, x2_new)

error = np.abs(original_function(X1_new, X2_new) - f(x1_new, x2_new))

fig = plt.figure(figsize=(12, 6))# Graficar la función original


ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(X1, X2, Z, cmap='viridis')
ax1.set_title('Función Original')

# Graficar la función interpolada
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(X1_new, X2_new, Z_new, cmap='plasma')
ax2.set_title('Función Interpolada')

# Graficar el error
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(X1_new.flatten(), X2_new.flatten(), error.flatten(), 'b.')
ax3.set_title('Error')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')


# Mostrar los gráficos
plt.tight_layout()
plt.show()

print("Error máximo:", np.max(error))
