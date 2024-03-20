import numpy as np
import matplotlib.pyplot as plt

# Leer los datos del archivo CSV
data_graf1 = np.loadtxt('mnyo_tp01_2024_1s/mnyo_mediciones.csv')

# Extraer las columnas x e y del primer archivo
x_1 = data_graf1[:, 0]
y_1 = data_graf1[:, 1]

data_ground_truth = np.loadtxt('mnyo_ground_truth.csv')

# Extraer las columnas x e y del segundo archivo
x_t = data_ground_truth[:, 0]
y_t = data_ground_truth[:, 1]



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

##graf 1
ax1.scatter(x_1, y_1, color='blue', label='Puntos')
ax1.plot(x_1, y_1, color='red', linestyle='-', marker='', label='Líneas')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Gráfico de puntos con líneas')
ax1.legend()
ax1.grid(True)

##graf original
ax2.scatter(x_t, y_t, color='blue', label='Puntos')
ax2.plot(x_t, y_t, color='red', linestyle='-', marker='', label='Líneas')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Gráfico original')
ax2.legend()
ax2.grid(True)


plt.tight_layout() 
plt.show()

