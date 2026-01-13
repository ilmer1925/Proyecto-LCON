# Proyecto-LCON
(En este espacio se encuentra el compilador IceCompilador01 escrito en Putyon 3.13 que se usará como procesador de todos los programas fuentes escitos en el lenguje LCON totalmente en español.  También en este repositorio se encuentran todos los archivo, scrit o módulos que se usan en el compilador. También se ecnuentran las librerías de terceros.)

El proyecto LCON consiste en la implementación de un nuevo lenguaje de programación al cual le llamamos LCON que al ser procesado por el IceCompilador01, chequeará la sintaxis fuente y si no tiene errores dicho programa será traducido al lenguaje Python y ser ejecutado.

El programa fuente debe ser codificado en cualquier editor de textos. Yo he usado el block de notas del Windows, por ejemplo. La extensión debe ser "txt".

![ej00](https://github.com/user-attachments/assets/298306c3-1f58-45f9-bab4-f6809ee4ddc6)


Para la ejecución del programa fuente se requiere de tres archivo:
- El archivo compilador que debe estar abierto en toda la sesión
- El archivo que contiene el programa fuente
- El archivo de datos, si así lo requiere el programa fuente.

Los datos requeridos por el programa fuente deben ser ingresados desde el teclado o usando un archivo de datos, los cuales contendrán a los datos y cuya extensión será "txt".

<img width="215" height="333" alt="image" src="https://github.com/user-attachments/assets/5b44b737-b244-40db-9a0a-fbbd7dacffe9" />


¿CÓMO USARLO?

- Disponer de un equipo que como sistema operativo a Windows.
- Que tengan instalados la última versión del Python.
- Que tenga instalados, vía el Shell de Windows, todas las librerías contenidas en el archivo ListaLiberias.txt
- La instalación del Python (última versión) así como las librerías, se explica detalladamente en el anexo 3 del manual del lenguaje (LenguajeLCon.docx)
- La importación de todas las librerías de terceros, no debe ser preocupación del usuario pues ya se cargan cuando se carga el compilador.
- Las librerías no necesitan descargarlas; en el Shell de Windows se instala usando PIP

¿CÓMO INSTALO UNA LIBRERÍA, POR EJEMPLO numpy?

1. Hacer clic con el botón derecho del mouse en el botón izquierdo de la barra de tareas <Inicio> del Windows
2. Seleccionar Windows PowerChell
3. En la línea habilitada digitar: pip install numpy
   Este sería el caso para instalar el numpy. de igual manera se hace con las otras librería.

El pip se instala automáticament cuando se instala el Python,

Si por alguna razón se obtiene que dice que pip no está instalada, es más sencillo desinstalar el Python y volver a hacerlo pero teniendo cuidado de activar la casilla relativo al pip


AHORA YA PUEDE USAR EL LCON PARA EJEUCTAR LOS PROGRAMAS FUENTES EN LCON

1. Cargar a memoria el IceCompilador01.py usando: <File> - <Open ...> ubicar la carpeta hacia donde descargó el compilador.
2. Ejecutarlo usando <Run> - <Run module>
3. Cuando pida el nombre del programa fuente, ingrese la ruta y el archivo (por ejemplo: d:/progs/EjPrg05.txt)
4. Si pidiera nombre del archivo de datos, digitar la ruta con el nombre del archivo(por ejemplo: d:/progs/ahorro.txt


## USO DE LAS LIBRERÍAS PROPIAS Y DE TERCEROS ##

## Librerías contenidas en el compilador, que se importan pero no requierenser instaladas:
grFunciones
grTrig


## Librerías que vienen con la instalación del Python ##

os
sys
math


## Archivos de terceros necesario para el uso del IceCompilador01 que deben ser instaladas en el Windows PowerShell para después ser importadas en el lenguaje ##


numpy
decimal
mandas
matplotlib
scipy
seaborn
plotnine
bokeh
sklearn
statsmodels
warnings


## Recuerde que: ##

Durante la instalación del Python, debe tener cuidado de marcar las casillas (opciones) donde se proponen añadir el path al sistema y lo referido al PIP. Si no hay mensaje respecto a este último es porque ya lo instala.

## Cómo instalar estas librerías: ##

Estando en el Windows PowerShell:
pip install xxxx
donde xxxx debe ser cada uno de los nombres anteriores


## Notas adicionales: ##

## N1: ##
    Si desea ejecutar una de las aplciaciones *.py debe chequear si usa archivos de datos (*.txt) y si éstgos ya están en el repositorio correspondiente para descargarlos y usarlos en su equipo.

## N2": ##
    Si desea ejecutar un programa fuente en LCON, debe verificar si tanto el programa fuente (en *.txt) como los archivos de datos (*.txt) se encuentran en sus repositorios correspondientes para luego descargarlos y usarlos en su equipo.



