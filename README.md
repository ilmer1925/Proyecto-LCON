# Proyecto-LCON
(En este espacio se encuentra el compilador IceCompilador01 escrito en Python 3.13 que se usará como procesador de todos los programas fuentes escitos en el lenguaje LCON totalmente en español.  También en este repositorio se encuentran todos los archivo, script o módulos que se usan en el compilador. También se encuentran las librerías de terceros, necesarios para potenciar la performance del lenguaje.)

El proyecto LCON consiste en la implementación de un nuevo lenguaje de programación al cual le llamamos LCON que al ser procesado por el IceCompilador01, que chequeará la sintaxis del programa fuente, si no tiene errores lo traducira al lenguaje Python para ser ejecutado línea por línea como un intérprete. 

El programa fuente debe ser codificado usando cualquier editor de textos. Yo he usado el Block de notas, accesorio del Windows, por ejemplo. La extensión debe ser "txt".

![ej00](https://github.com/user-attachments/assets/298306c3-1f58-45f9-bab4-f6809ee4ddc6)


## ¿QUÉ ES LO QUE DEBE DESCARGAR A SU EQUIPO? ##

(Ante todo debe crear una carpeta de trabajo que contendrá todo los archivos usados en este proyecto)

1. El archivo IceCompilador.py,
2. El manual del lenguaje LCON, Lenguaje LCON.docx,
3. Los archivos (programa fuente con extensión "txt") usados en el manual,
4. Los archivos de datos (con extensión "txt") usados por los programas fuentes.



Para la ejecución del programa fuente se requiere de tres archivos:
- El archivo compilador IceCompilador01.py, que debe estar abierto en todas las sesiones
- El archivo que contiene el programa fuente, cuyo nombre debe ser ingresado con toda la extensión ("txt")
- El archivo de datos, si así lo requiere el programa fuente (que también debe haberse grabado en un editor con extendión "txt").


<img width="215" height="333" alt="image" src="https://github.com/user-attachments/assets/5b44b737-b244-40db-9a0a-fbbd7dacffe9" />

## ENTONCES ##
PASO 1: Ejecutar el compilador
PASO 2: Pedirá nombre del programa fuente. Se debe ingresar la ruta y nombre del archivo
PASO 3: Si el programa fuente requiere de datos, el compilador pedirá que se ingrese el nombre del archivo, incluyendo la ruta.

En el paso 3, el compilador puede pedir que los datos sean ingresados desde el teclado.

Para realizar estos tres pasos no se requiere conocer nada respecto del lenguaje Python pue de lo que se trata es de aprender el lenguaje LCON. Sólo que para ello se debe tener instalado el Python. Por esta razón explicaremos en detalle la forma de uso.



# ¿CÓMO USARLO? EN DETALLE #

- Disponer de un equipo con sistema operativo al Windows.
- Que tengan instalados la última versión del Python, o por lo menos la versión 3.13.
- Que tenga instalados, vía el Shell de Windows (Windows PowerShell), todas las librerías contenidas en el archivo ListaLiberias.txt, aquí los incluímos líneas abajo.
- La instalación del Python (última versión) así como las librerías, se explica detalladamente en el anexo 3 del manual del lenguaje ( ## LenguajeLCon.docx ## )
- La importación de todas las librerías de terceros, no debe ser preocupación del usuario pues ya se cargan cuando se carga el compilador.
- Las librerías no necesitan descargarlas; deben ser instaladas en el entorno del Windows PowerShell del Windows usando el PIP

¿CÓMO HAGO ESTO?; ES DECIR, CÓMO INSTALO UNA LIBRERÍA, POR EJEMPLO numpy?

1. Hacer clic con el botón derecho del mouse en el botón izquierdo de la barra de tareas <Inicio> del Windows
2. Seleccionar Windows PowerShell
3. En la línea habilitada (prompt) digitar: pip install numpy
   Este sería el caso para instalar el numpy.
   De la misma forma se hace para instalar las otras librería.

El pip se instala automáticamente cuando se instala el Python,

Si por alguna razón se obtiene el mensaje que dice que pip no está instalada, es más sencillo desinstalar el Python y volver a hacerlo pero teniendo cuidado de activar la casilla relativo al pip durante el proceso de instalación.


## AHORA YA PUEDE USAR EL LCON PARA COMPILAR Y EJEUCTAR LOS PROGRAMAS FUENTES EN LCON ##

Repetimos lo que dijimos antes:

1. Cargar el Python (para una ejecución directa su icono debe estar en la barra de tareas). Esto se explica en el anexo del Manual del LCON.
1. Cargar a memoria el IceCompilador01.py usando: <File> - <Open ...> ubicar la carpeta donde descargó el compilador.
2. Ejecutarlo usando <Run> - <Run module>
3. Cuando pida el nombre del programa fuente, ingrese la ruta y el archivo (por ejemplo: d:/progs/EjPrg05.txt)
4. Si pidiera nombre del archivo de datos, digitar la ruta con el nombre del archivo(por ejemplo: d:/progs/ahorro.txt


## USO DE LAS LIBRERÍAS PROPIAS Y DE TERCEROS ##

## Librerías contenidas en el compilador, que se importan pero no requieren ser instaladas:
# grFunciones #
# grTrig #


## Librerías que vienen con la instalación del Python, que no requieren ser instaladas pero el compilador ya los importa usando: import xxx ##

# os #
# sys #
# math #


## Archivos de terceros necesario para el uso del IceCompilador01 que deben ser instaladas en el Windows PowerShell para después ser importadas en el lenguaje ##


# numpy #
# decimal #
# mandas #
# matplotlib #
# scipy #
# seaborn #
# plotnine #
# bokeh #
# sklearn #
# statsmodels #
# warnings #


## Recuerde que: ##

Durante la instalación del Python, debe tener cuidado de marcar las casillas (opciones) donde se proponen añadir el path al sistema y lo referido al PIP. Si no hay mensaje respecto a este último es porque ya lo instala.

## Cómo instalar estas librerías: ##

Estando en el Windows PowerShell:
pip install xxxx
donde xxxx debe ser cada uno de los nombres anteriores



## Notas adicionales: ##

## N1: ##
    Si desea ejecutar una de las aplciaciones *.py debe chequear si usa archivos de datos (*.txt) y si éstos ya están en el repositorio correspondiente, debe descargarlos y usarlos en su equipo.

## N2: ##
    Si desea ejecutar un programa fuente en LCON, debe verificar si tanto el programa fuente (en *.txt) como los archivos de datos (*.txt) se encuentran en sus repositorios correspondientes para luego descargarlos y usarlos en su equipo.

## N3: ##

    No se olvide de crear una carpeta particular hacia donde debe descargar el compilador, los programas fuente usados en el Manual de LCON y los datos requeridos.




