import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline

def original_function(x1, x2):
    return 0.75 * np.exp(-((10*x1 - 2)**2) / 4 - ((9*x2 - 2)**2) / 4) + \
        0.65 * np.exp(-((9*x1 + 1)**2) / 9 - ((10*x2 + 1)**2) / 2) + \
        0.55 * np.exp(-((9*x1 - 6)**2) / 4 - ((9*x2 - 3)**2) / 4) - \
        0.01 * np.exp(-((9*x1 - 7)**2) / 4 - ((9*x2 - 3)**2) / 4)



#Puntos para graficar la funcion original
x1_orig = np.linspace(-1, 1, 100)
x2_orig = np.linspace(-1, 1, 100)
x1_mesh_orig, x2_mesh_orig = np.meshgrid(x1_orig, x2_orig)
z_mesh_orig = original_function(x1_mesh_orig, x2_mesh_orig)


#Puntos para interpolar
x1 = np.linspace(-1, 1, 20)
x2 = np.linspace(-1, 1, 20)
X1, X2 = np.meshgrid(x1, x2)
Z = original_function(X1, X2)
f = RectBivariateSpline(x1, x2, Z) #interpolacion

x1_interp = np.linspace(-1, 1, 20)
x2_interp = np.linspace(-1, 1, 20)
z_interp = f(x1_interp, x2_interp)
x1_interp_mesh, x2_interp_mesh = np.meshgrid(x1_interp, x2_interp)

#puntos chevbyshev
def chebyshev_points(a, b, n):
    k = np.arange(1, n+1)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) / (2 * n) * np.pi)

x1_cheb_interp = np.sort(chebyshev_points(-1, 1, 20))
x2_cheb_interp = np.sort(chebyshev_points(-1, 1, 20))
z_cheb = f(x1_cheb_interp, x2_cheb_interp)
X1_cheb_mesh, X2_cheb_mesh = np.meshgrid(x1_cheb_interp, x2_cheb_interp)





fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(x1_mesh_orig, x2_mesh_orig, z_mesh_orig, cmap='viridis')
ax1.set_title('Función Original')

# función interpolada
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(x1_interp_mesh, x2_interp_mesh, z_interp, cmap='plasma')
ax2.set_title('Función Interpolada')

#  función chebvyshev
ax3 = fig.add_subplot(1, 3, 3, projection='3d') 
ax3.plot_surface(X1_cheb_mesh, X2_cheb_mesh, z_cheb, cmap='Blues')#dame otrp cmap
ax3.set_title('Función chebvyshev')

# Calcular el error relativo de ambas
error_interp = np.abs(original_function(x1_interp_mesh, x2_interp_mesh) - f(x1_interp, x2_interp))/np.abs(original_function(x1_interp_mesh, x2_interp_mesh))
error_interp_abs = np.abs(original_function(x1_interp_mesh, x2_interp_mesh) - f(x1_interp, x2_interp))
error_cheb = np.abs(original_function(x1_cheb_interp, x2_cheb_interp) - f(x1_cheb_interp, x2_cheb_interp))/np.abs(original_function(x1_cheb_interp, x2_cheb_interp))
error_cheb_abs = np.abs(original_function(x1_cheb_interp, x2_cheb_interp) - f(x1_cheb_interp, x2_cheb_interp))

error_abs_max = np.max(error_interp_abs)
error_rel_max = np.max(error_interp)
error_cheb_rel_max = np.max(error_cheb)
error_cheb_abs_max = np.max(error_cheb_abs)

equi_text = f"Error relativo:\n{error_rel_max}\nError absoluto:\n{error_abs_max}"
cheb_text = f"Error relativo:\n{error_cheb_rel_max}\nError absoluto:\n{error_cheb_abs_max}"

## quiero que abajo de los graficos de interpolacion y de chevbyshev aparezca el error
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.48, 0.05, equi_text, ha="center", fontsize=10)
plt.figtext(0.76, 0.05, cheb_text, ha="center", fontsize=10)

##hacer un grafico que muestre mientras mas puntos el error de cada una de las interpolaciones


plt.tight_layout()
plt.show()

