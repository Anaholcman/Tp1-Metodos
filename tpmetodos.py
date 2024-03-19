from sympy import symbols, diff
from sympy import sin, tanh


def biseccion(func, a, b, tol=1e-6, max_iter=1000):

    if func(a) * func(b) >= 0:
        raise ValueError("La función debe tener valores de diferente signo en los extremos del intervalo.")

    iter_count = 0
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if func(c) == 0:
            return c  # ¡Encontramos la raíz exacta!
        elif func(c) * func(a) < 0:
            b = c
        else:
            a = c

        iter_count += 1
        if iter_count >= max_iter:
            raise ValueError("Máximo número de iteraciones alcanzado. No se pudo encontrar una solución dentro de la tolerancia dada.")

    return (a + b) / 2

# Ejemplo de uso:
def f(x):
    return (0.3) ** abs(x) * sin(4 * x) - tanh(2 * x) + 2

raiz = biseccion(f, -1, 1)
print("La raíz de la función es aproximadamente:", raiz)



