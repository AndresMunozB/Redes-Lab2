# -*- coding: utf-8 -*-
"""
Created on Sun May  8 06:28:03 2016

@author: Francisco
"""
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from numpy import arange, linspace
from scipy.fftpack import fft, ifft

#==============================================================================
# Qué hace la función?: Trabajo con los datos de audio, obtengo la cantidad de datos y determino el tiempo
# Parámetros de entrada: Matriz con los datos de la amplitud del audio
# Parámetros de salida: Vector con la señal a trabajar, el largo de la señal y un vector con los tiempos de la señal
#==============================================================================
def obtener_datos_1(info):
    señal = info[:,0]#obtengo el vector de solo un canal de audio
    largo_señal = len(señal)#obtengo el largo de la señal
    largo_señal = float(largo_señal)
    tiempo = float(largo_señal)/float(rate)#genero el tiempo total del audio
    x = arange(0,tiempo,1.0/float(rate))#genero un vector de 0 hasta tiempo con intervalos del porte de la frecuencia
    return señal,largo_señal,x
    

#==============================================================================
# Qué hace la función?: Grafica el la amplitud del audio con respecto al tiempo
# Parámetros de entrada: Vector con la señal (EJE Y) y vector con el tiempo (EJE X)
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_audio_respecto_tiempo(señal,x):
    plt.plot(x,señal,"--")
    plt.title("Audio con respecto al tiempo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.show()
    
    
#==============================================================================
# Qué hace la función?: Grafica el la amplitud del audio con respecto al tiempo
# Parámetros de entrada: Vector con la señal (EJE Y) y vector con el tiempo (EJE X)
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_audio_respecto_tiempo2(señal,x):
    plt.plot(x,señal,"--")
    plt.title("Audio con respecto al tiempo (sin ruido)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [dB]")
    plt.show()

#==============================================================================
# Qué hace la función?: Genera la transformada de fourier de una señal dada
# Parámetros de entrada: Vector con la señal a transformar y el largo de la señal
# Parámetros de salida: Vector con la amplitud transformada y un vector con las frecuencias
#==============================================================================    
def obtener_transformada_fourier(señal,largo_señal):
    transFour = fft(señal,largo_señal)#eje Y
    transFourN = transFour/largo_señal#eje y normalizado
    
    aux = linspace(0.0,1.0,largo_señal/2+1)#obtengo las frecuencias
    xfourier = rate/2*aux#genero las frecuencias dentro del espectro real
    yfourier = transFourN[0.0:largo_señal/2+1]#genero la parte necesaria para graficar de la transformada
    return xfourier,yfourier
    

#==============================================================================
# Qué hace la función?: Grafica la transformada de fourier de una función
# Parámetros de entrada: -vector de amplitudes (EJE Y) y vector con frecuencias (EJE X)
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_transformada_1(xfourier,yfourier):
    plt.plot(xfourier,abs(yfourier))
    plt.title("Amplitud respecto a la frecuencia (fft)")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud [dB]")
    plt.show()


#==============================================================================
# Qué hace la función?: Grafica el espectograma de una señal
# Parámetros de entrada: -vector de amplitudes y frecuencia muestral
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_espectograma(señal,rate):
    plt.specgram(señal, Fs = rate)#primer parametro es la señal leida y el segudno es la frecuencia de muestreo
    plt.title("Espectograma audio")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.show()
    
    
#==============================================================================
# Qué hace la función?: Grafica el espectograma de una señal
# Parámetros de entrada: -vector de amplitudes y frecuencia muestral
# Parámetros de salida: NINGUNO
#==============================================================================    
def graficar_espectograma2(señal,rate):
    plt.specgram(señal, Fs = rate)#primer parametro es la señal leida y el segudno es la frecuencia de muestreo
    plt.title("Espectograma audio (sin ruido)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.show()
    

#==============================================================================
# Qué hace la función?: Aplica la transformada de fourier inversa a una señal
# Parámetros de entrada: Señal con amplitudes y largo de la señal
# Parámetros de salida: Vector con la transformada inversa
#==============================================================================
def aplicar_inversa_fourier(yfourier,largo_señal):
    inverTransFour = ifft(yfourier*largo_señal,largo_señal)
    return inverTransFour
    

#==============================================================================
# Qué hace la función?: Aplica filtro FIR a una señal y genera vectores para graficar
# Parámetros de entrada: Señal de audio y frecuencia de muestreo
# Parámetros de salida: vector "x" e "y" del nuevo audio 
#==============================================================================
def aplicar_filtro(señal,rate):
    n = 60000
    #Los siguientes cortes de obtienen después de analizar la transformada de fourier y realizar pruebas.
    lowcut = 1850 #frecuencia de corte inicial 
    highcut = 2030 #frecuencia de corte de final
    nyq = (1/2)*rate #frecuencia de Nyquist. Numero de muestras por unidad de tiempo (debe ser 1/(2*frecuencia de muestreo)) 
    lowcut_aux = lowcut/nyq #Cada frecuencia en corte debe estar entre 0 y NYQ .
    highcut_aux = highcut/nyq #Cada frecuencia en corte debe estar entre 0 y NYQ 
    #Filtro paso bajo
    a = signal.firwin(n, cutoff = lowcut_aux, window = 'blackmanharris')
    #Filtro paso alto con espectro invertido
    b = - signal.firwin(n, cutoff = highcut_aux, window = 'blackmanharris'); 
    b[n/2] = b[n/2] + 1
    #Combinando en un filtro paso banda
    d = - (a+b);d[n/2] = d[n/2] + 1
    
    #se aplica el filtro
    y = lfilter(d, [1.0],  señal)
    
    
    tiempo = len(y)/float(rate)
    #generamos vector de tiempo
    largo_señal = len(señal)#obtengo el largo de la señal
    largo_señal = float(largo_señal)
    tiempo = float(largo_señal)/float(rate)#genero el tiempo total del audio
    x = arange(0,tiempo,1.0/float(rate))#genero un vector de 0 hasta tiempo con intervalos del porte de la frecuencia
    return x,y

#==============================================================================
# inicio del codigo a ejecutar
# PARA GENERAR LOS GRAFICOS ES NECESARIO DESCOMENTAR LINEAS DE CODIGO!!!!!
#==============================================================================

# PUNTO 1, IMPORTAR LA SEÑAL DE AUDIO
rate,info=read("beacon.wav")

# PUNTO 2, GRAFICAR ESPECTOGRAMA
# obtener datos:
señal,largo_señal,x = obtener_datos_1(info)
#GRAFICO LA FUNCION DEL AUDIO CON RESPECTO AL TIEMPO
graficar_audio_respecto_tiempo(señal,x)

#GRAFICO ESPECTOGRAMA
graficar_espectograma(señal,rate)

print(señal)
#PUNTO 3, SOBRE EL AUDIO EN EL DOMINIO DE SU FRECUENCIA:
# la señal se trabaja igual a como llego (sin realizar transformada de fourier)
# solo se usa la transforma de fourier para analizar las frecuencias y obtener los valores de corte
xfourier,yfourier = obtener_transformada_fourier(señal,largo_señal)
#GRAFICO LA FUNCION DE LA AMPLITUD CON RESPECTO A LA FRECUENCIA
graficar_transformada_1(xfourier,yfourier)
# APLICAR FILTRO FIR, PROBAR DISTINTOS PARAMETROS
x2,y = aplicar_filtro(señal,rate)

#GRAFICAR AUDIO CON RESPECTO AL TIEMPO (SIN RUIDO)
graficar_audio_respecto_tiempo2(y,x2)


#GRAFICO ESPECTOGRAMA (SIN RUIDO)
graficar_espectograma2(y,rate)

#ESCRIBIR AUDIO
señal[:,0] = y 
señal[:,1] = y
write("sinRuido.wav",rate,señal)




