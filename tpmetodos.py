import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import interp2d

def f_a(x):
    return (0.3**(abs(x))) * np.sin(4*x) - np.tanh(2*x) + 2

def f_b(x1, x2):
    return 0.75 * np.exp(-((10*x1 - 2)**2) / 4 - ((9*x2 - 2)**2) / 4) + \
           0.65 * np.exp(-((9*x1 + 1)**2) / 9 - ((10*x2 + 1)**2) / 2) + \
           0.55 * np.exp(-((9*x1 - 6)**2) / 4 - ((9*x2 - 3)**2) / 4) - \
           0.01 * np.exp(-((9*x1 - 7)**2) / 4 - ((9*x2 - 3)**2) / 4)

def equispaced_dataset_build(start, stop, points_number, f): # points_number>=2
    data_set = []
    for x_i in np.linspace(start, stop, points_number):     # genera puntos equiespaciados
        data_set.append([x_i, f(x_i)])
    return data_set

def lagrange_interpolation(data_set):
    x_values = [point[0] for point in data_set]
    y_values = [point[1] for point in data_set]
    interpolator = lagrange(x_values, y_values)     
    return interpolator                             

def lagrange_iterator(start, stop, f, degree):
    degree_evolution = {}
    for i in range(2,degree+2):                                 # quiero generar hasta n+1 puntos, que me dará el mayor grado que quiero (n)
        data_set = equispaced_dataset_build(start, stop, i, f)
        degree_evolution[i-1]=lagrange_interpolation(data_set)
    return degree_evolution

def error_median(interpolator, f, x_values):
    y_values_interp = interpolator(x_values)
    y_values_exact = f(x_values)
    error = np.abs(y_values_exact - y_values_interp)
    return np.median(error)

def main():

    start = -4
    stop = 4
    degree = 30
    x_values = np.linspace(start, stop, 1000)
    errors = []
    interpolators_dict = lagrange_iterator(start, stop, f_a, degree)
    
    for i in range(2, degree+2):
        interpolator = interpolators_dict[i-1]
        error = error_median(interpolator, f_a, x_values)
        errors.append(error)
    
    plt.plot(range(1, degree + 1), errors, marker='o')
    plt.xlabel('Grado del polinomio interpolante')
    plt.ylabel('Error promedio de interpolación')
    plt.title('Error de interpolación vs. Grado del polinomio')
    plt.grid(True)
    plt.show()

    # podemos evaluar cual es el grado del polinomio que se corresponde con el menor error, y comparar ambos gráficos


    
    # Graficar la función original
    y_values = f_a(x_values)
    plt.plot(x_values, y_values, label='f_a(x)')
    
    # Obtener y graficar el polinomio interpolante con 20 puntos
    data_set_20 = equispaced_dataset_build(start, stop, 10, f_a)
    interpolator_20 = lagrange_interpolation(data_set_20)
    yp_values_20 = interpolator_20(x_values)
    plt.plot(x_values, yp_values_20, label='p_a(x)', linestyle='--')
    
    # Graficar los puntos utilizados por el polinomio interpolante
    x_interp_points = [point[0] for point in data_set_20]
    y_interp_points = [point[1] for point in data_set_20]
    plt.scatter(x_interp_points, y_interp_points, color='red', label='Puntos de interpolación')
    
    plt.xlabel('x')
    plt.ylabel('f_a(x)')
    plt.title('Gráfico de la función f_a(x) y su interpolación')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Generar valores para x1 y x2
    x1_values = np.linspace(-2, 2, 100)
    x2_values = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1_values, x2_values)

    # Calcular los valores de f_b para cada punto en la malla
    Z = f_b(X1, X2)

    # Configuración de la figura
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la función original
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

    # Etiquetas de los ejes y título
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f_b(x1, x2)')
    ax.set_title('Gráfico de la función f_b(x1, x2)')

    # Mostrar la gráfica
    plt.show()

if __name__ == "__main__":
    main()



