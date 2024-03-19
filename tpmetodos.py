from sympy import *
import numpy as np

x = Symbol('x', real=True)

def f_a(x):
    return (0.3) ** abs(x) * sin(4 * x) - tanh(2 * x) + 2

def f_a_prima(x):
    return diff(f_a(x), x)

def f_a_prima2(x):
    return diff(f_a_prima(x), x)

def newton_raphson(f, f_prima, p0, maxIter, e):
    i = 1
    while (i < maxIter):
        try:
            f_primaval = f_prima(p0)
        except ValueError:
            print(f"La función no es diferenciable en x = {p0}")
            return None

        p = p0 - f(p0)/f_primaval
        if abs(p - p0).evalf() < e:  # Evaluar la expresión y comparar con la tolerancia
            break
        else:
            i += 1
            p0 = p
    return p

# Definir el rango y el tamaño del intervalo
rango_inicio = -4
rango_fin = 4
intervalo = 1

print("Raíces encontradas en cada intervalo:")

# Iterar sobre cada intervalo
for i in range(rango_inicio, rango_fin, intervalo):
    inicio_intervalo = i
    fin_intervalo = i + intervalo
    
    # Encontrar la raíz en el intervalo usando Newton-Raphson
    p0 = (inicio_intervalo + fin_intervalo) / 2
    raiz = newton_raphson(f_a, f_a_prima, p0, 1000, 0.001)
    
    if raiz is not None:
        print(f"Entre {inicio_intervalo} y {fin_intervalo} hay una raíz en {raiz.evalf()}")


''''
x_values = np.linspace(-4, 4, 400)
y_values = f_a(x_values)

plt.plot(x_values, y_values)
plt.title('Gráfico de la función f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
"""
"""
'''

def newton_raphson(f, f_prima, p0, maxIter, e):
    i=1
    while (i<maxIter):
        p = p0 - f(p0)/f_prima(p0)
        if (abs(p-p0) < e):
            break
        else:
            i+=1
            p0 = p
    return abs(p-p0)

def secante(f, p0, p1, maxIter, e):
    i=1
    while (i<maxIter):
        p = p1 - f(p1)*(p1-p0)/(f(p1)-f(p0))
        if (abs(p-p1) < e):
            break
        else:
            i+=1
            p0 = p1
            p1 = p
    return abs(p-p1)


def biseccion(f, x0, x1, maxIter, e):
    i=1
    while (i<maxIter):
        p = (x0+x1)/2
        if (f(x0)*f(p) < 0):
            x1 = p
        else:
            x0 = p
        if (abs(x1-x0) < e):
            break
        else:
            i+=1
    return abs(x1-x0)
""""""