# Lenguaje PyCond terminado en su primera versión
# El 12 de diciembre del 2018
# Hora: 16:50
# Version 1.0

import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *

import sys
import os

ruta = 'd:/pyCcon'
sys.path.append(ruta)
import grTrig
import grFunciones




global st, wc, cad, Tvar, jv, jl, js, jc, ji, wxtab,itab, ctype, aX


fname = input("Nombre del programa fuente, incluya la ruta: ")
fn = open(fname)
#fn=open("c:\pyCon\Datos00.txt")
sent = fn.read().split("\n")
fn.close()




def leer_matriz(filas, columnas):
    # Función para leer una matriz
    matriz = []
    print(f"Ingrese los elementos de la matriz ({filas}x{columnas}):")
    for i in range(filas):
        fila = []
        for j in range(columnas):
            elemento = float(input(f"Elemento [{i+1},{j+1}]: "))
            fila.append(elemento)
        matriz.append(fila)
    return np.array(matriz)


def processMatrix():
    # Leer el tamaño de las matrices
    filas1 = int(input("\n\nIngrese el número de filas de la primera matriz: "))
    columnas1 = int(input("Ingrese el número de columnas de la primera matriz: "))

    filas2 = int(input("\nIngrese el número de filas de la segunda matriz: "))
    columnas2 = int(input("Ingrese el número de columnas de la segunda matriz: "))

    # Leer las matrices
    print("\nPrimera matriz:")
    matriz1 = leer_matriz(filas1, columnas1)

    print("\nSegunda matriz:")
    matriz2 = leer_matriz(filas2, columnas2)
    # Las matrices
    print("\nPrimera matriz")
    print(matriz1)
    print("\nSegunda matriz")
    print(matriz2)

    # Suma de matrices
    if matriz1.shape == matriz2.shape:
        suma = matriz1 + matriz2
        print("\nSuma de las matrices:")
        print(suma)
    else:
        print("\nNo se puede sumar: las matrices tienen dimensiones diferentes.")

    # Resta de matrices
    if matriz1.shape == matriz2.shape:
        resta = matriz1 - matriz2
        print("\nResta de las matrices:")
        print(resta)
    else:
        print("\nNo se puede restar: las matrices tienen dimensiones diferentes.")

    # Transpuesta de las matrices
    transpuesta1 = matriz1.T
    transpuesta2 = matriz2.T

    print("\nTranspuesta de la primera matriz:")
    print(transpuesta1)

    print("\nTranspuesta de la segunda matriz:")
    print(transpuesta2)

    # Producto de matrices
    if columnas1 == filas2:
        producto = np.dot(matriz1, matriz2)
        print("\nProducto de las matrices:")
        print(producto)
    else:
        print("\nNo se puede multiplicar: el número de columnas de la primera matriz no coincide con el número de filas de la segunda matriz.")

    # Determinante de una matriz (solo para matrices cuadradas)
    #from decimal import *
    #getcontext().prec = 6
    if filas1 == columnas1:
        determinante1 = np.linalg.det(matriz1)
        xdet1 = +Decimal(determinante1)
        print(f"\nDeterminante de la primera matriz: {xdet1}")
    else:
        print("\nLa primera matriz no es cuadrada, no se puede calcular el determinante.")

    if filas2 == columnas2:
        determinante2 = np.linalg.det(matriz2)
        xdet2 = +Decimal(determinante2)
        print(f"\nDeterminante de la segunda matriz: {xdet2}")
    else:
        print("\nLa segunda matriz no es cuadrada, no se puede calcular el determinante.")

    # Inversa de una matriz (solo para matrices cuadradas y con determinante no nulo)
    if filas1 == columnas1:
        try:
            inversa1 = np.linalg.inv(matriz1)
            print("\nInversa de la primera matriz:")
            print(inversa1)
        except np.linalg.LinAlgError:
            print("\nLa primera matriz no tiene inversa (es singular).")
    else:
        print("\nLa primera matriz no es cuadrada, no se puede calcular la inversa.")

    if filas2 == columnas2:
        try:
            inversa2 = np.linalg.inv(matriz2)
            print("\nInversa de la segunda matriz:")
            print(inversa2)
        except np.linalg.LinAlgError:
            print("\nLa segunda matriz no tiene inversa (es singular).")
    else:
        print("\nLa segunda matriz no es cuadrada, no se puede calcular la inversa.")
    return




"""

def decodeArrayT(jarr,w):
        global cad,aX
        wp = w.split(" ")
        wp[1] = wp[1][:-1].strip()
        cad = cad + "\taX.append("+wp[1]+")\n"
        #cad = cad + "X[jarr]\n"
        jarr+=1
"""

def decodeArrayO(jbrr,w):
        global cad,aX
        if "OImprimir" in w:
                cad = cad + "for i in range(len(aX)):\n\tprint(aX[i])\n"
        else:
                processMatrix()



for i in range(len(sent)):
        sent[i]=sent[i].strip()

 
def decodeVar(ix):
        cname = wc[ix][0:wc[ix].find(":")]
        ct = wc[ix][wc[ix].find(":")+1:wc[ix].find(";")].strip()
        ctype.append(ct)
        Tvar.append(cname.split(","))
        for i in range(len(Tvar)):
                for j in range(len(Tvar[i])):
                        Tvar[i][j] = Tvar[i][j].strip()



def decodeInput(ix,ww):
        #<Expression> --> <Leer> <var> <;>
        #<Expression> --> <Leer> <Msg> <,> <var>

        global cad, itab
        com="'"
        msg=""
        if itab == 1:
                xtab = "\t"
        elif itab == 2:
                xtab = "\t"
                itab = 0
        else:
                xtab = ""
        if "," in ww:
                msg = ww[1:ww.find(",")-1]
                xv = ww[ww.find(",")+1:ww.find(";")]
                xv = xv.lstrip()
        else:
                msg = "Ingresa el dato: "
                xv = ww[0:ww.find(";")]
                xv = xv.lstrip()
        
        cty = ""
        for k in range(len(Tvar)):
                for g in range(len(Tvar[k])):
                        #if xv in Tvar[k]:
                        if xv in Tvar[k]:
                                cty = ctype[k]
                                break
        
        if cty =="":
                print("La variable ",xv, " no está definida")

        if cty=="Entero":
                cty = "int"
        elif cty == "Real":
                cty = "float"
        elif cty == "Cadena":
                cty = ""
        
        if cty == "":
                xc=xv+"="+cty+"input('"+msg+"')\n"
        else:
                xc=xv+"="+cty+"(input('"+msg+"'))\n"
        
        cad = cad + xtab+xc

        #print(cad)




def decodePrint(ix,ww):
        # <Expression> --> <Imprimir> <var> [,<var>,...]
        # <Expression> --> <Imprimir> <Msg> <,> <var> [,<Msg> <,><var>]

        global cad, itab
        comas = ww.split(",")
        msg=[]
        xv=[]
        xcad=""
        if itab == 1:
                xtab = "\t"
        elif itab == 2:
                xtab = "\t"
                itab = 0
        else:
                xtab = ""
        for k in range(len(comas)):
                if "'" in comas[k]:
                        msg.append(comas[k])
                else:
                        xv.append(comas[k])
        if len(comas)==1:
                
                if itab == 1:
                        xtab = "\t"
                else:
                        xtab=""
                
                if len(msg)>0:
                        xcad="print("+msg[0]+")\n"
                else:
                        xcad = "print("+xv[0][:-1]+")\n"
        if len(comas)==2:
                if len(msg)>0:
                        xcad = "print("+msg[0]+","+xv[0][:-1]+")\n"
                else:
                        xcad = "print("+xv[0]+","+ xv[1][:-1]+")\n"
        if len(comas)==3:
                if "'" in comas[0]:
                        msg1 = msg[0]
                        if "'" in comas[1]:
                                msg2 = msg[1]
                                if "'" in comas[2]:
                                        msg3 = msg[2]
                                        xcad="print("+msg1+" , "+msg2+" , "+msg3+")\n"
                                else:
                                        xcad = "print("+msg1+" , "+msg2+" , "+xv[0][:-1]+")\n"
                        else:
                                xcad = "print("+msg1+" , "+xv[0]+" , "+xv[1][:-1]+")\n"
                else:
                        xcad = "print("+xv[0]+" , "+xv[1]+" , "+xv[2][:-1]+")\n"
        elif len(comas) == 4:
                if "'" in comas[0]:
                        msg1 = msg[0]
                        if "'" in comas[1]:
                                msg2 = msg[1]
                                if "'" in comas[2]:
                                        msg3 = msg[2]
                                        xcad="print("+msg1+" , "+msg2+" , "+msg3+")\n"
                                        if "'" in msg[3]:
                                                msg4 = msg[3]
                                                xcad = "print("+msg1+" , "+msg2+" , "+msg3+" , "+msg4+")\n"
                                        else:
                                                cad="print("+msg1+" , "+msg2+" , "+msg3+" , "+xv[0][:-1]+")\n"
                                else:
                                        xcad = "print("+msg1+" , "+msg2+" , "+xv[0]+" , "+xv[1][:-1]+")\n"
                        else:
                                xcad = "print("+msg1+" , "+xv[0]+" , "+xv[1]+" , "+xv[2][:-1]+")\n"
                else:
                        xcad = "print("+xv[0]+" , "+xv[1]+" , "+xv[2]+" , "+xv[3][:-1]+")\n"
                #
                xcad = "print("+xv[0]+" , "+xv[1]+" , "+ xv[2]+" , "+xv[3][:-1]+")\n"
        
        cad=cad+xtab+xcad


####


def decode(t):
        W = []
        for i in range(len(t)):
                W.append(float(t[i]))
        return W


def FRegMultilineal(nv):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    
    #print("Nro de variables: ",nv)
    fname = input("Nombre del archivo de datos, con ruta: ")
    fn = open(fname)
    #fn=open("c:\pyCon\d04.txt")
    datos = fn.read().split("\n")
    fn.close()
    a = []
    Y = []
    X1 = []
    X2 = []
    X3 = []
    tA = []
    A = []

    if nv == 3:
        n = len(datos)
        for i in range(n):
            if "," in datos[i]:
                a = datos[i].split(",")
                Y.append(float(a[0]))
                X1.append(float(a[1]))
                X2.append(float(a[2]))
            else:
                a = datos[i].split("\t")
                Y.append(float(a[0]))
                X1.append(float(a[1]))
                X2.append(float(a[2]))
    else:
        n = len(datos)
        for i in range(n):
            if "," in datos[i]:
                a = datos[i].split(",")
                Y.append(float(a[0]))
                X1.append(float(a[1]))
                X2.append(float(a[2]))
                X3.append(float(a[3]))
            else:
                a = datos[i].split("\t")
                Y.append(float(a[0]))
                X1.append(float(a[1]))
                X2.append(float(a[2]))
                X3.append(float(a[3]))
    # Aqui empieza la insercón
    #import pandas as pd
    import statsmodels.api as sm
    from statsmodels.formula.api import ols # Para una sintaxis más similar a R

    # 1. Crear datos de ejemplo (Variable Dependiente 'y', Independientes 'x1', 'x2')
    #data = {
    #    'y': [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 36, 45, 42, 50, 52, 54, 58, 62, 65, 60, 68, 70, 85],
    #    'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    #    'x2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 13, 18, 17, 20, 23, 25, 22, 28, 30, 35, 34, 38, 40, 42]
    #}
    #df = pd.DataFrame(data)
    if nv == 3:
        data = pd.DataFrame({"Y":Y, "X1":X1, "X2":X2})
        df = pd.DataFrame(data)
        X = sm.add_constant(df[['X1', 'X2']])
        #Y = df('Y')
        model = sm.formula.ols("Y ~ X1 + X2", data=data)
    else:
        data = pd.DataFrame({"Y":Y, "X1":X1, "X2":X2, "X3":X3})
        df = pd.DataFrame(data)
        X = sm.add_constant(df[['X1', 'X2', 'X3']])
        #Y = df('Y')
        model = sm.formula.ols("Y ~ X1 + X2 + X3", data=data)
    # 2. Añadir la constante (intercepto) al modelo (necesario para statsmodels)
    #X = sm.add_constant(df[['x1', 'x2']])
    #y = df['Y']

    # 3. Ajustar el modelo de Regresión Lineal Múltiple
    # Usaremos OLS (Ordinary Least Squares)
    # model = sm.OLS(y, X)
    #model = sm.formula.ols("y ~ x1 + x2", data=data)
    results = model.fit()

    # 4. Imprimir la Tabla ANOVA (usando .summary())
    print("--- Tabla ANOVA y Resumen del Modelo ---")
    print(results.summary()) # Esto ya te da una tabla detallada

    # 5. Imprimir Coeficientes con pandas para un formato más limpio (opcional)
    print("\n--- Tabla de Coeficientes Formateada ---")
    # Extraer la tabla de coeficientes
    coef_df = results.params.to_frame(name='Coeficiente')
    # Añadir info adicional como P-value y t-value si se desea
    coef_df['P-Value'] = results.pvalues
    coef_df['t-Value'] = results.tvalues
    print(coef_df.to_string(float_format="%.4f")) # Formato con 4 decimales

    # 6. Para una tabla ANOVA *pura* (tabla de varianza)
    print("\n--- Tabla ANOVA de Varianza (con sm.stats.anova_lm) ---")
    #anova_table = sm.stats.anova_lm(results, typ=2) # typ=2 es común
    anova_table = sm.stats.anova_lm(results, type = 2)
    print(anova_table.to_string(float_format="%.4f"))

    # Hasta aquí la inserción
    #datos = pd.DataFrame({"Y":Y, "X1":X1})
    #fig, ax = plt.subplots(figsize = (5, 2.5))
    #sns.regplot(x='X1', y = 'Y', data=datos, ax = ax);
    #plt.show()
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    if nv == 3:
        datos = pd.DataFrame({"Y":Y, "X1":X1, "X2":X2})
    else:
        datos = pd.DataFrame({"Y":Y, "X1":X1, "X2":X2, "X3":X3})
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.regplot(x="X1",y ="Y",data = datos, ax = ax);
    sns.regplot(x="X2",y ="Y",data = datos, ax = ax);
    if nv == 4:
        sns.regplot(x="X3",y ="Y",data = datos, ax = ax);

    #Gráfico de dispersión para agrimil.txt

    fig, ax = plt.subplots(2,2,figsize=(10,8))

    ax[0,0].scatter(Y,X1,c='r',marker='*')
    ax[0,0].set_title("Gráfico de dispersión Y vs X1")
    ax[0,0].set_xlabel("X1")
    ax[0,0].set_ylabel("Y")

    ax[0,1].scatter(Y,X2,c='g',marker='o')
    ax[0,1].set_title("Gráfico de dispersión Y vs X2")
    ax[0,1].set_xlabel("X2")
    ax[0,1].set_ylabel("Y")

    if nv == 4:
        ax[1,0].scatter(Y,X3,c='b',marker='+')
        ax[1,0].set_title("Gráfico de dispersión Y vs X3")
        ax[1,0].set_xlabel("X3")
        ax[1,0].set_ylabel("Y")
        j = np.linspace(1,len(Y),len(Y))
        ax[1,1].bar(j,Y,width = 0.8,color='r')
        ax[1,1].set_title("Gráfico de barras de Y")
        ax[1,1].set_xlabel("Valores de Y")
        ax[1,1].set_ylabel("Y")

    plt.show()

    #
    unos = np.ones(n)
    tA = []
    tA.append(unos)
    tA.append(X1)
    tA.append(X2)
    if nv == 4:
        tA.append(X3)
    A = np.transpose(tA)
    tAA = tA@A
    invA = np.linalg.inv(tAA)
    Y = np.array(Y)
    tAY = tA@Y
    beta = invA@tAY
    print("Coef. ",beta)
    sY = 0
    for i in range(n):
        sY = sY + Y[i]
    mY = sY/n
    Yest = []
    if nv == 3:
        for i in range(n):
            Yest.append(beta[0]+beta[1]*X1[i]+beta[2]*X2[i])
    else:
        for i in range(n):
            Yest.append(beta[0]+beta[1]*X1[i]+beta[2]*X2[i]+beta[3]*X3[i])
    SCE = 0
    SCT = 0
    for i in range(n):
        SCE = SCE + (Y[i]-Yest[i])**2
        SCT = SCT + (Y[i]- mY)**2
    SCR = SCT - SCE
    CME = SCE/(n-nv)
    CMT = SCT/(n-1)
    CMR = SCR/(nv-1)
    R2 = 1 - SCE/SCT
    R2j = 1 - CME/CMT
    Fc = CMR/CME
    glr = nv - 1
    gle = n - nv
    glt = n - 1
    ro = R2**0.5
    errTipico = CME**0.5
    print("\n\nEcuación de regresión estimada:\n")
    if nv == 3:
        print("Y = ",beta[0]," + ",beta[1],"X1" , " + " , beta[2],"X2")
    else:
        print("Y = ",beta[0]," + ",beta[1],"X1" , " + " , beta[2],"X2" , " + ", beta[3],"X3")
    print("\n")
    print(f"'Coeficiente de determinación: ' {R2:>8.4f}")
    print(f"'Coeficiente de determinación ajustado: ' {R2j:>8.4f}")
    print(f"'Coeficiente de correlación: ' {ro:>8.4f}")
    print(f"'Error tipico: ' {errTipico:>8.4f}")
    print("\n\nOtra forma de presentación de la tabla ANOVA:\n")
    #print("SCE = ",SCE," SCT = ",SCT)
    #print("R2 = ",R2,"  R2j = ",R2j, "  ro = ",ro)
    """
    SCT = S(Y - Ym)**2   / (n-1)    = CMT      Total
    SCE = S(Y - Yest)**2 / (n-nv-1) = CME
    SCR = SCT - SCE      / nv    = CMR
    CMR = SCR/nv         Debido a la regresion
    CME = SCE/(n-nv-1)  Debido al error
    R**2 = 1 - SCE/SCT
    R**2j = 1 - CME/CMT
    Fc = CMR/CME
    """

    print("\n\n=====> T A B L A   D E L   A N O V A >=====\n")
    print(f"{'Fuente   ':>10s} {'gLib':>5s} {'S.Cuadrados':>12s} {'C. Medio':>12s} {'F. calc':>8s}")
    print(f"{'Regresion':>10s} {glr:>5.0f} {SCR:>12.4f} {CMR:>12.4f} {Fc:>8.4f}")
    print(f"{'Residuales':>10s} {gle:>5.0f} {SCE:>12.4f} {CME:>12.4f}")
    print(f"{'Total':>10s} {glt:>5.0f} {SCT:>12.4f}")
    print("\n\nEso es todo amigos...")
    #sys.exit()
    return




def media(W):
        return sum(W)/len(W)

def mediana(W):
        return np.median(W)

def cuartiles(W):
        return np.quantile(W,0.25), np.quantile(W,0.50), np.quantile(W,0.75)


def cas(W):
        from scipy.stats import skew
        return skew(W)


def var(W):
        return np.round(np.var(W),4)

def cov(W,Z):
        return np.cov(W,Z)

def cor(W,Z):
        
        return np.corrcoef(W,Z)


def regre(Y, X):
        n = len(Y)
        Res = []
        sX = 0
        sY = 0
        sXY = 0
        sX2 = 0
        sY2 = 0
        #sdx2 = 0
        for i in range(n):
            sX = sX + X[i]
            sY = sY + Y[i]
            sXY = sXY + X[i]*Y[i]
            sX2 = sX2 + X[i]*X[i]
            sY2 = sY2 + Y[i]*Y[i]
        mX = sX/n
        mY = sY/n
        vX = (sX2-n*mX*mX)/(n-1)
        vY = (sY2-n*mY*mY)/(n-1)
        #for i in range(n):
        #        sdx2 = sdx2 + (Y[i]-mY)**2
        b1 = (n*sXY-sX*sY)/(n*sX2-sX*sX)
        b0 = mY - b1*mX 
        yest = []
        sce = 0
        scr = 0
        for i in range(n):
                yest.append(b0+b1*X[i])
        for i in range(n):
                sce = sce + (Y[i]-yest[i])**2
        scr = 0
        for i in range(n):
                scr = scr+(yest[i]-mY)**2

        sct = sce + scr
        cmt = sct/(n-1)
        cmr = scr/1
        cme = sce/(n-2)
        Fc = cmr/cme
        R2 = 1-sce/sct
        s2e = cme
        #R2c = 1-(1-R2)*(n-1)/(n-2)
        R2c = 1 - cme/cmt
        #
        ro = np.sqrt(R2)
        errTip = np.sqrt(cme)
        raya = 51*"-"
        xrec =""
        xrec = "Resultados de la estimación Lineal\n"+raya+"\n"
        xrec = xrec+"Fuente    glib      S.cuadr.   Cuad.medio     Fcalc"+"\n"
        xrec = xrec+"Regresión    "+str(1) + "     "+str(np.round(scr,4))+"    "+str(np.round(cmr,4))+"  "+str(np.round(Fc,4))+"\n"
        xrec = xrec+"Errores     "+str(n-2)+"      "+str(np.round(sce,4))+"      "+str(np.round(cme,4))+"\n"
        xrec = xrec+"Totales     "+str(n-1)+"       "+str(np.round(sct,4))+"\n"
        xrec = xrec + raya + "\n\n"
        xrec = xrec + "Coeficiente de determinación: " + str(np.round(R2,4))+"\n"
        xrec = xrec + "Coeficiente de determinación corregido: "+str(np.round(R2c,4))+"\n"
        xrec = xrec + "Coeficiente de correlación: "+str(np.round(ro,4))+"\n"
        xrec = xrec + "Error típico: "+str(np.round(errTip,4)) + "\n\n"
        xrec = xrec + "Coeficientes de la regresión estimados:\n"
        xrec = xrec + "Término independiente: " + str(np.round(b0,4))+"\n"
        xrec = xrec + "Coeficiente de X: " + str(np.round(b1,4))+"\n"
        xrec = xrec + "Ecuación de Regresion estimada: Y = " + str(np.round(b0,4)) + " + " + str(np.round(b1,4))+"X"+ "\n"
        #return xrec
        print(xrec)
        return



def TablaFrec(x):
    import numpy as np
    import pandas as pd
    #fname = 
    #x = [38,376,37,36,398,37, 363,162,190,230,430,328,380,266,365,348,625,860,938, 680,460,864,650,7385,204,438,180,280,350,820,160,250,240,265,480,620,435,382,510,428]
    min = np.min(x)
    max = np.max(x)
    k = 5
    amp = np.round((max-min)/k,decimals=2)
    tablaF = pd.DataFrame(columns=['LimInf','LimSup','frAbs','frAc','frRel','frRelAc'])
    lim = np.arange(min,max+amp,amp)
    linf = lim[:k]
    lsup = lim[1:k+1]
    tablaF['LimInf'] = linf
    tablaF['LimSup'] = lsup
    tablaF['frAbs'] = np.array([np.sum((x>=linf[n]) & (x<=lsup[n])) for n in range(len(linf))])
    tablaF['frAc'] = np.array([np.sum(tablaF['frAbs'].iloc[:n].values) for n in range(1,k+1)])
    tablaF['frRel'] = tablaF['frAbs']/len(x)
    tablaF['frRelAc'] = tablaF['frAc']/len(x)
    print(tablaF)





def decodeArrayT(jarr,w):
        global cad,aX
        if "TabFrec" in w:
            fname = input("Nombre del archivo: ")
            fn = open(fname)
            cad = fn.read()
            fn.close()
            x = []
            if ',' in cad:
                datos = cad.split(',')
                cad = ""
            else:
                datos = cad.split('\n')
                cad = ""
            for r in datos:
                x.append(float(r))
            TablaFrec(x)
        else:
            wp = w.split(" ")
            wp[1] = wp[1][:-1].strip()
            cad = cad + "\taX.append("+wp[1]+")\n"
            #cad = cad + "X[jarr]\n"
            jarr+=1




def FEstad():
        global X, Y
        fname = input("Nombre del archivo de datos, con ruta: ")
        fn = open(fname)
        #fn=open("c:\pyCon\d04.txt")
        datos = fn.read().split("\n")
        fn.close()
        n = len(datos)
        X = []
        Y = []
        #print("n = ",n, "\n",datos)
        if "," in datos[0]:
            for i in range(n):
                xd = datos[i].split(",")
                Y.append(float(xd[0]))
                X.append(float(xd[1]))
        else:
            xd = datos[0].split("\t")
            for i in range(n):
                X.append(float(xd[i]))
            xd = datos[1].split("\t")
            for i in range(n):
                Y.append(float(xd[i]))
        regre(Y,X)
        return






###

def EDescrip(datos,modo):
        global X
        if '\t' in datos:
                zcad = datos.split('\t')
                X = [float(num) for num in zcad]
        elif ',' in datos:
            zcad = datos.split(',')
            X = [float(num) for num in zcad]
        else:
            zcad = datos.split('\n')
            X = []
            for i in zcad:
                X.append(float(i))
        if modo == 0 or modo == 1:
            TablaFrec(X)
        #else:
            s=0;s2 = 0
            for i in X:
                s = s + i
                s2 = s2 + i*i
            mx = s/len(X)
            vx = (s2-len(X)*mx*mx)/(len(X)-1)
            print("Media : ",np.round(s/len(X),decimals=2))
            print("Mediana: ",np.round(np.median(X),decimals=2))
            print("Cuartil 1: ",np.quantile(X,0.25))
            print("Cuartil 2: ",np.quantile(X,0.5))
            print("Cuartil 3: ",np.quantile(X,0.75))
            print("Varianza: ",np.round(vx,decimals=4))
            print("Desv. estandar: ",np.round(np.sqrt(vx),decimals= 2))
            print("Coef.variabilidad: ",np.round(np.sqrt(vx)/mx,decimals=4))
            print("Coeficiente de asimetría: ",np.round(3*(mx - np.median(X))/np.sqrt(vx),decimals=4))
        return



def FEstad00(zwc):
        global X, Y
        nv = int(input("Numero de variables: "))
        if nv == 3 or nv == 4:
            FRegMultilineal(nv)
            return
        if nv == 2:
                FEstad()
                return
        else:
                nopc = int(input("\n\nIngreso de los datos:\n1. Desde el teclado\n2. Desde un archivo\nDigite su opción: "))
                if nopc == 1:
                        n = int(input("Numero de datos: "))
                        datos=[]
                        for i in range(n):
                                datos.append(float(input("Dato "+str(i+1)+": ")))
                        #print("datos   = ",datos)
                        tecla = 1
                        EDescrip(datos,tecla)
                        return
                else:
                        fname = input("Nombre del archivo de datos: ")
                        fn = open(fname)
                        datos = fn.read()
                        fn.close()
                        tecla = 0
                        EDescrip(datos,tecla)
                        return





def decodeIf(ix,ww):
        global cad, itab, jl, jc, ji, wxtab
        ww = ww.strip()
        expr = ""
        wxtab=""
        if "{" in ww and itab == 1:
                wxtab="\t"
        if "{" in ww:
                itab = 1
                xtab = "\t"
        else:
                xtab ="\t"
                itab = 0
        if "entonces" in ww:
                expX = ww[ww.find("ces ")+4:]   # Se debe decodificar
                zww = ww[0:ww.find(" ent")]
                if "<=" in zww:
                        exp1 = zww[0:zww.find("<")]
                        rop = "<="
                        exp2 = zww[zww.find("<=")+2:]
                        expr = exp1+rop+exp2
                        print("   expr = ",expr)
                if "=" in zww:
                        exp1 = zww[0:zww.find("=")]
                        rop = "=="
                        xtab="\t"
                        exp2 = zww[zww.find("=")+1:]            # Era 2
                
                if "!=" in zww:
                        #exp1 = zww[0:zww.find("<")]
                        expr = zww.split("!")
                        exp1 = expr[0].strip()
                        expr = expr[1].strip()[2:]
                        exp2 = expr
                        rop = "!="
                        #exp2 = zww[zww.find(">")+2:]
                        print("exp1 = ",exp1," rop = ",rop," exp2 = ",exp2)
                        expr = exp1+"!="+exp2
                if ">" in zww and "=" not in zww:
                        exp1 = zww[0:zww.find(">")]
                        irop = ">="
                        exp2 = zww[zww.find(">")+2:]
                
                if "<" in zww and "=" not in zww:
                        exp1 = zww[0:zww.find("<")]
                        rop = "<"
                        exp2 = zww[zww.find("<")+1:]
                
                if "=" in zww and "<" not in zww and ">" not in zww:
                        exp1 = zww[0:zww.find("=")]
                        rop = "=="
                        exp2 = zww[zww.find("=")+1:]
                
                if ">" in zww and "=" not in zww:
                        exp1 = zww[0:zww.find(">")]
                        rop = ">"
                        exp2 = zww[zww.find(">")+1:]

                if expr!="":
                        cad = cad + "if "+ expr+":\n"+xtab
                else:
                        cad = cad + wxtab+"if "+ exp1 +rop + exp2 + ":\n"+xtab
                #Decodificando expX
                if "Leer" in expX:
                        jl+=1
                        zwc=expX[5:]
                        itab = 2
                        decodeInput(jl,zwc)
                elif "Imprimir" in expX:
                        ji+=1
                        zwc=expX[9:]
                        itab = 2
                        decodePrint(ji,zwc)
                else:
                        jc+=1
                        zwc = expX
                        decodeCalc(jc,zwc)
        #Añadido
        else:
                ww = ww[:-1]
                cad = cad + "if " + ww + ":\n"     #+xtab



def decodeElse(ix,ww):
        global cad, itab, wxtab
        ww=ww.strip()
        
        if "{" in ww:
                itab = 1
                xtab = "\t"
                cad = cad + "else " + ":\n"


                        
def decodeFor(ix,ww):
        global cad, itab, wxtab
        ww=ww.strip()
        wxtab =""
        if "{" in ww and itab == 1:
                wxtab = "\t"
        if "{" in ww:
                itab = 1
                xtab = "\t"
        else:
                itab = 0
                xtab = ""
        
        xbl=ww.split(" ")
        if "veces" in ww:
                ntimes = ww[0:ww.find(" ")]
                cad = cad + "for i in range("+ntimes+"):\n"
        elif "incr" in ww:
                cad = cad + "for "+xbl[0] +" in "+ "range(" + xbl[2]+","+ xbl[4]+","+ xbl[6]+"):\n"       #+xtab
        else:
                cad = cad + wxtab+"for "+xbl[0] +" in "+ "range(" + xbl[2]+","+ xbl[4]+"+1):\n"      #+xtab



              
def decodeWhile(ix,ww):
        global cad, itab
        exp1 = exp2 = rop = ""
        ww = ww.strip()
        if "{" in ww:
                itab = 1
        else:
                itab = 0
        
        ww = ww[0:ww.find("{")-1]
        if "<=" in ww:
                exp1 = ww[0:ww.find("<")]
                rop = "<="
                exp2 = ww[ww.find("<=")+2:]
        if "==" in ww:
                exp1 = ww[0:ww.find("=")]
                rop = "=="
                exp2 = ww[ww.find("=")+2:]
        
        if "<>" in ww:
                exp1 = ww[0:ww.find("<")]
                rop = "!="
                exp2 = ww[ww.find(">")+2:]
        
        if ">" in ww and "=" not in ww:
                exp1 = ww[0:ww.find(">")]
                irop = ">="
                exp2 = ww[ww.find(">")+2:]
        
        if "<" in ww and "=" not in ww:
                exp1 = ww[0:ww.find("<")]
                rop = "<"
                exp2 = ww[ww.find("<")+1:]
        
        if "=" in ww and "<" not in ww and ">" not in ww:
                exp1 = ww[0:ww.find("=")]
                rop = "=="
                exp2 = ww[ww.find("=")+1:]
        
        if ">" in ww and "=" not in ww:
                exp1 = ww[0:ww.find(">")]
                rop = ">"
                exp2 = ww[ww.find(">")+1:]
        
        cad = cad + "while "+ exp1 +rop + exp2 + ":\n"



def Graficos(sent):
        global cad, ejeX
        obj = compile(cad,"ice01","exec")
        exec(obj)
        gcod = sent[:3]
        if gcod != "GEv" and gcod != "G3d" and gcod != "GPo" and gcod != "GSC" and gcod != "G3a":
                #
                sent = sent[:len(sent)-1]
                #cad = cad + sent+'\n'
                lista = sent[3:].split(',')
                n = len(lista)
                x = []
                for i in range(n):
                        if i == 0:
                                x.append(float(lista[i][1:]))
                        elif i==n-1:
                                lg = len(lista[i])
                                x.append(float(lista[i][:lg-1]))
                        else:
                                x.append(float(lista[i]))
                if len(ejeX) > 0:
                        j = ejeX
                        ejeX=[]
                else:
                        j = np.linspace(1,n,n)
                #j = np.linspace(1,n,n)
                #print("x = ",x)
        match(gcod):
                case "GLi":
                        plt.plot(j,x)
                        plt.title("GRAFICO DE LINEA")
                        plt.show()
                case "GBa":
                        colores = ["red","green","blue","cyan"]
                        fig, ax = plt.subplots()
                        barras = ax.bar(j,x,color=colores)
                        for barra in barras:
                                altura = barra.get_height()
                                ax.text(barra.get_x() + barra.get_width()/2.,
                                        altura+0.5,
                                        f'{altura}',
                                        ha = 'center',va = 'bottom')
                        #plt.bar(j,x,color=colores)
                        ax.set_title("GRAFICO DE BARRAS")
                        ax.set_ylabel("Valor")
                        ax.set_xlabel("Categorías")
                        plt.ylim(0,max(x)+5)
                        plt.show()
                case "GPi":
                        colores = ["orange","cyan","brown","grey","skyblue","green","red"]
                        #explode = [0.0,0.0,0.0,0.0,0.0,0.1,0.2]
                        explo = []
                        for i in range(len(x)-1):
                                explo.append(0.0)
                        explo.append(0.2)
                        plt.pie(x,labels=j,explode=explo,colors=colores,autopct="%1.1f%%",shadow=True)
                        plt.title("GRAFIO DE TORTA")
                        plt.show()
                case "GSe":
                        a1 = sent[3:][1:]
                        lg = len(a1)
                        a2 = a1[:lg-1]
                        #print("... ",a2)
                        a2 = a2.split(',')
                        desde = float(a2[0])
                        hasta = float(a2[1])
                        grTrig.Seno(desde,hasta)
                        #print(a2, " , ",desde, " , ",hasta)
                case "GCo":
                        a1 = sent[3:][1:]
                        lg = len(a1)
                        a2 = a1[:lg-1]
                        a2 = a2.split(',')
                        desde = float(a2[0])
                        hasta = float(a2[1])
                        grTrig.Coseno(desde,hasta)
                        #print(a2, " , ",desde, " , ",hasta)
                case "GCu":
                        cad = sent[4:len(sent)-1]
                        #print('cad ',cad)
                        desde = float(cad.split(',')[0])
                        hasta = float(cad.split(',')[1])
                        print("desde = ",desde," hasta = ",hasta)
                        grFunciones.grPol(desde,hasta)
                case "GSC":
                        grFunciones.GrafSenCos(sent)
                case "GPo":
                        grFunciones.grPolinomials()
                case "G3d":
                        grFunciones.Graf3d(sent)
                case "G3a":
                    grFunciones.Graf3d(sent)
                case "GEv":
                        if ',' in sent:
                                x = sent[4:5]
                                y = sent[6:7]
                                pto = 'P('+x+','+y+')'
                                #print(pto)
                                res = grFunciones.evPol(pto)
                                cad = ""
                                #print("x = ",x, ' y = ',y, ' sent = ',sent)
                        else:
                                x = sent[4:5]
                                pto = 'P('+x+')'
                                #print(pto)
                                res = grFunciones.evPol(pto)
                                #print("Evaluación: ',res)
                                #print("x = ",x, ' sent = ',sent)
                                #res = grFunciones.evPol(pto)
                                cad = ""
                        print('Evaluación: ',res,"\n\n")
                case _:
                        print("Error en tipo de gráfico")



def decodeCalc(ix,ww):
        global cad,itab, wxtab
        ww = ww[:-1]
        ifun = 0
        kk = 0
        if itab == 1:
                xtab="\t"
        else:
                xtab = ""
        # pte1 Contiene la variable que recibe un valor o expresión
        # pte2 Contiene la variable o expresión
        # ls1  Contiene el valor hasta donde llega la variable
        # ww   Contiene la variable, la función u operador y la expresión o valor
        if "regre(" in ww:
                ls1 = ww.find("regre(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                opr = "regre"
                ifun = 0
                kk = 1
                pte2 = ww[ls2:]
                ww = pte1 + opr + pte2
        if "raizc(" in ww:
                ls1 = ww.find("raizc(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                ifun = 1
                opr = "np.sqrt"
                pte2 = ww[ls2:]
                # En prueba
                ww = pte1 + opr + pte2
                # Termina la prueba
        if "entero(" in ww:
                ls1 = ww.find("entero(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                ifun = 1
                opr = "int"
                pte2 = ww[ls2:]
                # En prueba
                ww = pte1 + opr + pte2
                # Termina la prueba
        if "exp(" in ww:
                ls1 = ww.find("exp(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                ifun = 1
                opr = "np.exp"
                pte2 = ww[ls2:]
                ww = pte1 + opr + pte2
        if "pi()" in ww:
                ls1 = ww.find("pi(")
                pte1 = ww[0:ls1]
                ls2 = ww.find(")")
                ifun = 1
                opr = "np.pi"
                pte2 = ww[ls2+1:]
                ww = pte1 + opr + pte2
        if "redondear(" in ww:
                ls1 = ww.find("redondear(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                opr = "np.round("
                ifun = 1
                pte2 = ww[ls2+1:]
                ww = pte1 + opr + pte2
        if "pot(" in ww:
                ls1 = ww.find("pot(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                opr = "np.power"
                ifun = 1
                pte2 = ww[ls2:]
                ww = pte1 + opr + pte2
        if "log10(" in ww:
                ls1 = ww.find("log10(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                opr = "np.log10"
                ifun = 1
                pte2 = ww[ls2:]
                ww =pte1 + opr + pte2
        if "ln(" in ww:
                ls1 = ww.find("ln(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                opr = "np.log"
                ifun = 1
                pte2 = ww[ls2:]
                ww =pte1 + opr + pte2
        if "fac(" in ww:
                ls1 = ww.find("fac(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                opr = "math.factorial"
                ifun = 1
                pte2 = ww[ls2:]
                ww =pte1 + opr + pte2
                #print("facccc = ",ww)
        if "sen(" in ww:
                ls1 = ww.find("sen(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                opr = "np.sin"
                ifun = 1
                pte2 = ww[ls2:-1]
                ww = pte1 + opr + pte2+'*np.pi/180)'
        if "cos(" in ww:
                ls1 = ww.find("cos(")
                pte1 = ww[0:ls1]
                ls2 = ww.find("(")
                opr = "np.cos"
                ifun = 1
                pte2 = ww[ls2:-1]
                ww = pte1 + opr + pte2+'*np.pi/180)'
        if "aleat(" in ww:
                ls1 = ww.find("aleat(")
                pte1 = ww[0:ls1]
                ls2 = ww.find(")")
                opr = "np.random.rand()"
                ifun = 1
                pte2 = ww[ls2+1:]
                ww = pte1 + opr + pte2
        else:
                ls1 = ww.find("=")
                pte1 = ww[0:ls1]
                pte2 = ww[ls1+1:]
                opr = "="
                ww = pte1 + opr + pte2
        cad = cad + wxtab + xtab+ww+"\n"




st=[]
wc=[]
Tvar=[]
wxtab=""
ctype =[]
jv = -1
jl = -1
ji = -1
jc = -1
jr = -1
jm = 0
js = 0
itab = 0
xtab = ""
aX = []
jarr = -1
jbrr = 0
ejeX = []
cad="import numpy as np\n"
cad = cad + "import math\n"
cad = cad + "import matplotlib.pyplot as plt\n"

        
for i in range(len(sent)):
        w=sent[i][:1]
        if w == "D" or w == "#" or w=="" or w=="{" or w=="}":
                if w=="D":
                        cad = cad
                else:
#                        cad=cad+sent[i]+"\n"
                        if w == "{":
                                itab = 1
                        if w == "}":
                                itab = 0
        #s
        elif w=="V":
                st.append(sent[i][0:3])
                wc.append(sent[i][4:])
                jv+=1
                decodeVar(jv)
        elif w=="L":
                st.append(sent[i][0:4])
                wc.append(sent[i][5:])
                zwc=sent[i][5:]
                jl+=1
                decodeInput(jl,zwc)
        elif w=="I":
                st.append(sent[i][0:8])
                wc.append(sent[i][9:])
                zwc=sent[i][9:]
                ji+=1
                decodePrint(ji,zwc)
        elif w=="S":
                st.append(sent[i][0:2])
                wc.append(sent[i][3:])
                zwc = sent[i][3:]
                js+=1
                decodeIf(js,zwc)
        elif w=="C":
                st.append(sent[i][0:14])
                wc.append(sent[i][15:])
                zwc = sent[i][15:]
                js+=1
                decodeElse(js,zwc)                
        elif w=="R":
                if "variando" in sent[i]:
                        st.append(sent[i][:16])
                        wc.append(sent[i][17:sent[i].find("{")-1])
                        zwc=sent[i][17:]
                        jr+=1
                        decodeFor(jr,zwc)
                else:
                        st.append(sent[i][:8])
                        wc.append(sent[i][9:])
                        zwc=sent[i][7:]
                        jr+=1
                        decodeFor(jr,zwc)
        
        elif w=="M":
                st.append(sent[i][:9])
                wc.append(sent[i][10:])
                jm+=1
                zwc = sent[i][8:]
                decodeWhile(jm,zwc)
        elif w=="E":
                zwc = sent[i][6:]
                zwc = zwc[:-2]
                #jf+=1
                FEstad00(zwc)
                #FEstad()
        elif w=="G":
                Graficos(sent[i])
        elif w == "X":
                wc = sent[i][:-2]
                ejeX = wc[2:].split(",")
        elif w == "T":
                jarr+=1
                decodeArrayT(jarr,sent[i])
        elif w == "O":
                jbrr+=1
                decodeArrayO(jbrr,sent[i])
        elif w == "P":
                processMatrix()
        else:
                st.append("x")
                wc.append(sent[i])
                zwc = sent[i]
                jc+=1
                decodeCalc(jc,zwc)

"""
if len(aX)>0:
        cad = cad + "if len(aX)>0:\n\tfor i in range(len(aX)):\tprint(aX[i])"
        print(cad)
else:
        print(cad)
"""

print(cad+"\n\n\n\n\n\n")
obj = compile(cad,"ice01","exec")
exec(obj)


