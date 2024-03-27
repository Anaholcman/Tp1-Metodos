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
x1 = np.linspace(-1, 1, 15)
x2 = np.linspace(-1, 1, 15)
X1, X2 = np.meshgrid(x1, x2)
Z = original_function(X1, X2)
f = RectBivariateSpline(x1, x2, Z) #interpolacion

x1_interp = np.linspace(-1, 1, 100) ##para graf
x2_interp = np.linspace(-1, 1, 100)
z_interp = f(x1_interp, x2_interp)
x1_interp_mesh, x2_interp_mesh = np.meshgrid(x1_interp, x2_interp)

#puntos chevbyshev
def chebyshev_points(a, b, n):
    k = np.arange(1, n+1)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) / (2 * n) * np.pi)

x1_cheb = np.sort(chebyshev_points(-1, 1, 15))
x2_cheb = np.sort(chebyshev_points(-1, 1, 15))
X1_cheb_mesh, X2_cheb_mesh = np.meshgrid(x1_cheb, x2_cheb)
z_cheb = original_function(X1_cheb_mesh, X2_cheb_mesh)

f_b = RectBivariateSpline(x1_cheb, x2_cheb, z_cheb)
z_b_interp = f_b(x1_interp, x2_interp)


def absolute_error(func, x1_evaluar, x2_evaluar):
    return np.abs(original_function(x1_evaluar, x2_evaluar)-func(x1_evaluar, x2_evaluar))

def relative_error(func, x1_evaluar, x2_evaluar):
    return np.abs(original_function(x1_evaluar, x2_evaluar)-func(x1_evaluar, x2_evaluar))/np.abs(original_function(x1_evaluar, x2_evaluar))


error_abs = absolute_error(f, x1_interp, x2_interp)
error_rel = relative_error(f, x1_interp, x2_interp)
error_cheb_abs = absolute_error(f_b, x1_interp, x2_interp)
error_cheb_rel = relative_error(f_b, x1_interp, x2_interp)


'''
error_interp = np.abs(original_function(x1_interp_mesh, x2_interp_mesh) - f(x1_interp, x2_interp))/np.abs(original_function(x1_interp_mesh, x2_interp_mesh))
error_interp_abs = np.abs(original_function(x1_interp_mesh, x2_interp_mesh) - f(x1_interp, x2_interp))
error_cheb = np.abs(original_function(x1_cheb, x2_cheb) - f_b(x1_cheb, x2_cheb))/np.abs(original_function(x1_cheb, x2_cheb))
error_cheb_abs = np.abs(original_function(x1_cheb, x2_cheb) - f_b(x1_cheb, x2_cheb))'''

error_abs_max = np.median(error_abs)
error_rel_max = np.median(error_rel)
error_cheb_rel_max = np.median(error_cheb_rel)
error_cheb_abs_max = np.median(error_cheb_abs)

equi_text = f"Error relativo:\n{error_rel_max}\nError absoluto:\n{error_abs_max}"
cheb_text = f"Error relativo:\n{error_cheb_rel_max}\nError absoluto:\n{error_cheb_abs_max}"




fig = plt.figure(figsize=(12, 9))

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(x1_mesh_orig, x2_mesh_orig, z_mesh_orig, cmap='viridis')
ax1.set_title('Función Original')

# función interpolada
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(x1_interp_mesh, x2_interp_mesh, z_interp, cmap='plasma')
ax2.text2D(0.05, -0.2, equi_text, transform=ax2.transAxes, fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax2.set_title('Función Interpolada')

#  función chebvyshev
ax3 = fig.add_subplot(1, 3, 3, projection='3d') 
ax3.plot_surface(x1_interp_mesh, x2_interp_mesh, z_b_interp, cmap='Blues')
ax3.text2D(0.05,-0.2, cheb_text, transform=ax3.transAxes, fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax3.set_title('Función chebvyshev')

plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.show()

# hacer una funcion que muestre como cambia el error en funcion de como aumenta la cantidad de nodos
def plot_error_vs_nodes():
    num_nodes = np.arange(5, 50, 5)
    error_abs_max = []
    error_rel_max = []
    error_cheb_rel_max = []
    error_cheb_abs_max = []

    for n in num_nodes:
        x1 = np.linspace(-1, 1, n)
        x2 = np.linspace(-1, 1, n)
        X1, X2 = np.meshgrid(x1, x2)
        Z = original_function(X1, X2)
        f = RectBivariateSpline(x1, x2, Z)

        x1_interp = np.linspace(-1, 1, 100)
        x2_interp = np.linspace(-1, 1, 100)
        z_interp = f(x1_interp, x2_interp)

        error_abs = absolute_error(f, x1_interp, x2_interp)
        error_rel = relative_error(f, x1_interp, x2_interp)

        error_abs_max.append(np.median(error_abs))
        error_rel_max.append(np.median(error_rel))

        x1_cheb = np.sort(chebyshev_points(-1, 1, n))
        x2_cheb = np.sort(chebyshev_points(-1, 1, n))
        X1_cheb_mesh, X2_cheb_mesh = np.meshgrid(x1_cheb, x2_cheb)
        z_cheb = original_function(X1_cheb_mesh, X2_cheb_mesh)

        f_b = RectBivariateSpline(x1_cheb, x2_cheb, z_cheb)
        z_b_interp = f_b(x1_interp, x2_interp)

        error_cheb_abs = absolute_error(f_b, x1_interp, x2_interp)
        error_cheb_rel = relative_error(f_b, x1_interp, x2_interp)

        error_cheb_abs_max.append(np.median(error_cheb_abs))
        error_cheb_rel_max.append(np.median(error_cheb_rel))

    plt.plot(num_nodes, error_abs_max, label='Absolute Error')
    plt.plot(num_nodes, error_rel_max, label='Relative Error')
    plt.plot(num_nodes, error_cheb_abs_max, label='Chebyshev Absolute Error')
    plt.plot(num_nodes, error_cheb_rel_max, label='Chebyshev Relative Error')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Error')
    plt.title('Error vs Number of Nodes')
    plt.legend()
    plt.show()

plot_error_vs_nodes()
