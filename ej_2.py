import sympy as sp

# Definir las variables simbólicas
x = sp.symbols('x')
y = sp.symbols('y')

# Definir las funciones simbólicas
#pol_tray2 = -0.0007202*x**3 + 0.04448*x**2 - 0.9411*x + 8.684
#pol_tray1 = 1.537e-08*x**9 - 2.51e-06*x**8 + 0.0001701*x**7 - 0.006211*x**6 + 0.1324*x**5 - 1.661*x**4 + 11.66*x**3 - 39.41*x**2 + 41.1*x + 0.3241


pol_tray2 = x**2 +x
pol_tray1 = y**3 +3*y


deriv_pol_tray1_x = sp.diff(pol_tray1, y)
deriv_pol_tray2_x = sp.diff(pol_tray2, x)

jacobiano = sp.Matrix([[deriv_pol_tray1_x], [deriv_pol_tray2_x]])


print("Jacobiano:")
print(jacobiano)



#deriv_tray2 = sp.diff(pol_tray2, x)

#print(pol_tray2)
#print(deriv_tray2)

def y(x):
    return pol_tray2(x)








'''
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Puntos')
plt.plot(x, y, color='red', linestyle='-', marker='', label='Recorrido 1era trayectoria')
plt.plot(x_4, y_4, color='green', linestyle='-', marker='', label='Recorrido 2nda trayectoria')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x_4, y_4, label='4 mediciones', color='orange')
plt.title('Grafico de las 2 trayectorias')
plt.legend()
plt.grid(True)
plt.show()'''


