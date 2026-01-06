import numpy as np
import matplotlib.pyplot as plt
import math

# pto(x,y) o pto(x) donde x e y deben ser números como cadenas
def evPol(pto):
    #print("pto = ",pto)
    if ',' in pto:
        x = float(pto[2:3])
        y = float(pto[4:5])
    else:
        x = float(pto[2:3])
    cad = input('Ingrese la función. Ej:\nx**3-3*x+5\nx**3*y**2-x**2-y**3+8\n: ')
    r = eval(cad)
    #print('Valor: ',r)
    return r

"""
# res = evPol('P(2,3)')
"""

def grPol(d,h):
    tg = input('Gráfico: \n1. Bidimensional\n2.Tridimensional\n: ')
    match(tg):
        case '1':
            cad = input('y = ')
            z=[]
            j = []
            i = d
            while(i<=h):
                xc = cad.replace('x',str(i))
                j.append(i)
                if i<0:
                    rt = eval(xc)*(-1)
                else:
                    rt = eval(xc)
                z.append(rt)
                i+=0.005
            plt.plot(j,z)
            plt.show()
        case '2':
            cad = input('z = ')
            z=[]
            j = []
            i = d
            while(i<=h):
                xc = cad.replace('x',str(i))
                xc = xc.replace('y',str(i))
                j.append(i)
                if i<0:
                    rt = eval(xc)*(-1)
                else:
                    rt = eval(xc)
                z.append(rt)
                i+=0.005
            plt.plot(j,z)
            plt.show()
        case _:
            print("Error en tipo de gráfico.")

def Graf3d():
    import numpy as np
    import matplotlib.pyplot as plt

    # Vamos a generar 50 valores para los ejes X e Y
    r = np.linspace(-8,8)
    s = np.linspace(-8,8)
    X, Y = np.meshgrid(r,s)

    # Ahora generemos el radio de un circulo en el plano XY
    R = np.sqrt(X**2 + Y**2)
    # Generemos una curva de nivel del seno en el eje Z
    Z = np.sin(R)/R
    # Tracemos un objeto entorno fig
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    # Tracemos finalmente la superficie
    ax.plot_surface(X, Y, Z)
    plt.show()

    # Cambiemos la generación de valores en linspace, de -1 a 1
    # Grabar y correr el módulo
    # Cuál es el gráfico si cambiamos a -8, 8?
    # Qué ocurre si los cambiamos a -10, 10?
    # Qué ocurre si Z = np.sin(R)/R + np.cos(R)/R
    # Si ahora lo cambiamos R = np.sqrt(X**2 - Y**2) ?

    # Vamos a trabajar con rago de -8 a 8 y añadirle otros atributos al gráfico

    import matplotlib.pyplot as plt
    import numpy as np

    X, Y = np.meshgrid(np.linspace(-8, 8), 
                       np.linspace(-8, 8))
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R) / R
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot_surface(X, Y, Z, color = "lightgreen", alpha = 0.7, linewidth = 0.4,edgecolor="red",cmap="Spectral_r")
    # fig.colorbar(plot, ax = ax, shrink = 0.5, aspect = 10)
    plt.title("Gráfico de superficie 3D")
    plt.suptitle("Silla de montar con un cono central")
    ax.set_xlabel('Valor en el eje X')
    ax.set_ylabel('Valor en el eje Y')
    ax.set_zlabel('Valor en el eje Z')
    plot = ax.plot_surface(X, Y, Z, cmap = 'bwr')
    fig.colorbar(plot, ax = ax, shrink = 0.5, aspect = 10)
    plt.show()



