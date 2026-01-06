import numpy as np
import matplotlib.pyplot as plt
import math

# 1. Convertir 30 grados a radianes
grados = 30
radianes = np.radians(grados) # o math.radians(grados)

# 2. Calcular el valor de y = sen(30)
valor_y = np.sin(radianes)
def Seno(d,h):
    # 3. Crear los datos para la gráfica
    # Crear una secuencia de números para el eje x (en radianes)
    x_valores = np.linspace(d*np.pi, h*np.pi, 400) # Ejemplo de un intervalo
    y_valores = np.sin(x_valores)

    # 4. Crear la figura y los ejes
    fig, ax = plt.subplots()

    # 5. Graficar la función seno
    ax.plot(x_valores, y_valores, label='y = sen(x)')

    # 6. Marcar el punto específico (x=30 grados, y=sen(30))
    # Para el eje x, usamos la conversión de 30 grados
    ax.plot(grados, valor_y, 'ro', label=f'sen({grados}°) = {valor_y:.2f}') # 'ro' para un punto rojo

    # 7. Personalizar la gráfica
    ax.set_xlabel('x (grados)')
    ax.set_ylabel('y')
    ax.set_title('Gráfica de y = sen(x) con el punto sen(30°)')
    ax.legend()
    ax.grid(True)

    # 8. Mostrar la gráfica
    plt.show()

# 2. Calcular el valor de y = cos(30)
valor_y = np.cos(radianes)
def Coseno(d,h):
    # 3. Crear los datos para la gráfica
    # Crear una secuencia de números para el eje x (en radianes)
    x_valores = np.linspace(d*np.pi, h*np.pi, 400) # Ejemplo de un intervalo
    y_valores = np.sin(x_valores)

    # 4. Crear la figura y los ejes
    fig, ax = plt.subplots()

    # 5. Graficar la función seno
    ax.plot(x_valores, y_valores, label='y = cos(x)')

    # 6. Marcar el punto específico (x=30 grados, y=sen(30))
    # Para el eje x, usamos la conversión de 30 grados
    ax.plot(grados, valor_y, 'ro', label=f'cos({grados}°) = {valor_y:.2f}') # 'ro' para un punto rojo

    # 7. Personalizar la gráfica
    ax.set_xlabel('x (grados)')
    ax.set_ylabel('y')
    ax.set_title('Gráfica de y = cos(x) con el punto cos(30°)')
    ax.legend()
    ax.grid(True)

    # 8. Mostrar la gráfica
    plt.show()

