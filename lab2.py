# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:06:02 2018
Integrantes:
    Diego Mellis - 18.663
    Andrés Muñoz - 19.646.487-5
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
# Función: En base a los datos que entrega beacon.wav se obtiene			   
# los datos de la señal, la cantidad de datos que esta tiene, y el tiempo que  
# que dura el audio.														   
# Parámetros de entrada: Matriz con los datos de la amplitud del audio.
# Parámetros de salida: Vector con la señal a trabajar, el largo de la señal y 
# un vector con los tiempos de la señal.
#==============================================================================
def getDatos(info,rate):
	#Datos del audio.
	signal = info[:,0]
	#print(signal)
	#Largo de todos los datos.
	len_signal = len(signal)
	#Transformado a float.
	len_signal = float(len_signal)
	#Duración del audio.
	time = len_signal/float(rate)
	#print(time)
	#Eje x para el gráfico, de 0 a la duración del audio.
	t = linspace(0, time, len_signal)
	#print(t)
	return signal, len_signal, t

#=============================================================================
# Función: Grafica los datos del audio en función del tiempo.
# Parámetros de entrada: El arreglo con los datos del tiempo y los datos de la
# señal.
# Parámetros de salida: Ninguno, pero se muestra un gráfico por pantalla.
#=============================================================================
def graphTime(t, signal):
	plt.plot(t,signal)
	plt.title("Audio con respecto al tiempo")
	plt.xlabel("Tiempo [s]")
	plt.ylabel("Amplitud [dB]")
	savefig("Audio con respecto al tiempo.png")
	plt.show()

#=============================================================================
# Función: Función que se encarga de obtener la transformada de Fourier de los
# datos de la señal.
# Parámetros de entrada: Un arreglo con los datos de la señal, y el largo de 
# este arreglo.
# Parámetros de salida: Dos arreglos, uno con los valores del eje x y otro con
# los valores del eje y.
#=============================================================================
def fourierTransformation(signal, len_signal):
	fourierT = fft(signal)
	fourierNorm = fourierT/len_signal
	xfourier = np.fft.fftfreq(len(fourierNorm),1/rate)
	return xfourier, fourierNorm

#===============================================================================
# Función: Obtiene la anti-transformada de fourier del arreglo yfourier.
# Parámetros de entrada: Arreglo yfourier y el largo de este arreglo.
# Parametros de salida: Un arreglo con los valores de la anti transformada de 
# Fourier.
#===============================================================================
def getInverseFourier(yfourier,len_freq):
		fourierTInv = ifft(yfourier)*len_freq
		return fourierTInv

#==============================================================================
# Función: Grafica el espectograma de una señal
# Parámetros de entrada: signal y frecuencia muestral
# Parámetros de salida: Ninguno, pero se muestra un gráfico.
#==============================================================================    
def graphSpecgram(signal,rate):
    plt.specgram(signal, Fs = rate)#primer parametro es la señal leida y el segudno es la frecuencia de muestreo
    plt.title("Espectograma audio")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.show()

#==============================================================================
# Función: Aplica filtro FIR a una señal y genera vectores para graficar
# Parámetros de entrada: Señal de audio y frecuencia de muestreo
# Parámetros de salida: vector "x" e "y" del nuevo audio 
#==============================================================================
def appFilter(signal,rate):
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
    y = lfilter(d, [1.0],  signal)
    
    
    time = len(y)/float(rate)
    #generamos vector de tiempo
    len_signal = len(signal)#obtengo el largo de la señal
    len_signal = float(len_signal)
    time = float(len_signal)/float(rate)#genero el tiempo total del audio
    x = arange(0,tiempo,1.0/float(rate))#genero un vector de 0 hasta tiempo con intervalos del porte de la frecuencia
    return x,y
