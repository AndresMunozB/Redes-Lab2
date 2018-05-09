# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:06:02 2018
Integrantes:
    Diego Mellis - 18.663.454-3
    Andrés Muñoz - 19.646.487-5
"""

from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from numpy import arange, linspace
from pylab import savefig
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
	senal = info[:,0]
	#print(signal)
	#Largo de todos los datos.
	len_signal = len(senal)
	#Transformado a float.
	len_signal = float(len_signal)
	#Duración del audio.
	time = len_signal/float(rate)
	#print(time)
	#Eje x para el gráfico, de 0 a la duración del audio.
	t = linspace(0, time, len_signal)
	#print(t)
	return senal, len_signal, t

#=============================================================================
# Función: Grafica los datos del audio en función del tiempo.
# Parámetros de entrada: El arreglo con los datos del tiempo y los datos de la
# señal.
# Parámetros de salida: Ninguno, pero se muestra un gráfico por pantalla.
#=============================================================================
def graphTime(t, senal,text,figura):
	plt.plot(t,senal)
	plt.title(text)
	plt.xlabel("Tiempo [s]")
	plt.ylabel("Amplitud [dB]")
	savefig(figura)
	plt.show()

#=============================================================================
# Función: Función que se encarga de obtener la transformada de Fourier de los
# datos de la señal.
# Parámetros de entrada: Un arreglo con los datos de la señal, y el largo de 
# este arreglo.
# Parámetros de salida: Dos arreglos, uno con los valores del eje x y otro con
# los valores del eje y.
#=============================================================================
def fourierTransformation(senal, len_signal):
	fourierT = fft(senal)
	fourierNorm = fourierT/len_signal
	xfourier = np.fft.fftfreq(len(fourierNorm),1/rate)
	return xfourier, fourierNorm

#===============================================================================
# Función: Grafica la transformada de Fourier, usando los arreglos de la función
# anterior.
# Parámetros de entrada: arreglo con los valores del eje x y arreglo con los 
# valores del eje y.
# Parámetros de salida: Ninguno, se muestra un gráfico por pantalla.
#===============================================================================
def graphTransformation(xfourier,yfourier,text,figura):
    
	plt.title(text)
	plt.xlabel("Frecuencia [Hz]")
	plt.ylabel("Amplitud [dB]")
	plt.plot(xfourier,abs(yfourier))
	savefig(figura)
	plt.show()

#==============================================================================
# Función: Grafica el espectograma de una señal
# Parámetros de entrada: signal y frecuencia muestral
# Parámetros de salida: Ninguno, pero se muestra un gráfico.
#==============================================================================    
def graphSpecgram(senal,rate,text,figura):
    plt.specgram(senal, Fs = rate)#primer parametro es la señal leida y el segudno es la frecuencia de muestreo
    plt.title(text)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    savefig(figura)
    plt.show()

#===============================================================================
# Función: Obtiene la anti-transformada de fourier del arreglo yfourier.
# Parámetros de entrada: Arreglo yfourier y el largo de este arreglo.
# Parametros de salida: Un arreglo con los valores de la anti transformada de 
# Fourier.
#===============================================================================
def getInverseFourier(yfourier,len_freq):
		fourierTInv = ifft(yfourier)*len_freq
		return fourierTInv

#===============================================================================
# Función: Grafica la anti-transformada de fourier.
# Parámetros de entrada: Arreglo del tiempo y arreglo de la anti-transformada
# de Fourier.
# Parámetros de salida: Ninguno, se muestra un gráfico por pantalla.
#===============================================================================
def graphWithInverse(time, invFourier,text,figura):
	plt.title(text)
	plt.xlabel("Tiempo[s] (IFFT)")
	plt.ylabel("Amplitud [dB]")
	plt.plot(time,invFourier)
	savefig(figura)
	plt.show()


#==============================================================================
# Función: Aplica filtro FIR paso bajo a una señal y genera vectores para graficar
# Parámetros de entrada: Señal de audio y frecuencia de muestreo
# Parámetros de salida: vector "x" e "y" del nuevo audio 
#==============================================================================
def appLowFilter(senal, rate):
    nyquist = rate/2
    lowcut = 1200
    lowcut2 = lowcut/nyquist
    numtaps = 1201
    filteredLow = signal.firwin(numtaps,cutoff = lowcut2, window = 'hamming' )
    filtered = lfilter(filteredLow,1.0,senal)
    len_signal = len(senal)#obtengo el largo de la señal
    len_signal = float(len_signal)
    time = float(len_signal)/float(rate)#genero el tiempo total del audio
    x = arange(0,time,1.0/float(rate))
    return x,filtered

#==============================================================================
# Función: Aplica filtro FIR paso alto a una señal y genera vectores para graficar
# Parámetros de entrada: Señal de audio y frecuencia de muestreo
# Parámetros de salida: vector "x" e "y" del nuevo audio 
#==============================================================================
def appHighFilter(senal, rate):
    nyquist = rate/2
    highcut = 8000
    highcut2 = highcut/nyquist
    numtaps = 8001
    filteredHigh = - signal.firwin(numtaps,cutoff = highcut2, window = 'hamming', pass_zero = False)
    filtered = lfilter(filteredHigh,1.0,senal)
    len_signal = len(senal)#obtengo el largo de la señal
    len_signal = float(len_signal)
    time = float(len_signal)/float(rate)#genero el tiempo total del audio
    x = arange(0,time,1.0/float(rate))
    return x,filtered

#==============================================================================
# Función: Aplica filtro FIR paso banda a una señal y genera vectores para graficar
# Parámetros de entrada: Señal de audio y frecuencia de muestreo
# Parámetros de salida: vector "x" e "y" del nuevo audio 
#==============================================================================
def appBandPassFilter(senal,rate):
    
    nyquist = rate/2
    lowcut = 3000
    lowcut2 = lowcut/nyquist
    highcut = 8000
    highcut2 = highcut/nyquist
    numtaps = 8001
    filteredBandPass = - signal.firwin(numtaps,cutoff = [lowcut2,highcut2], window = 'hamming', pass_zero = False)
    filtered = lfilter(filteredBandPass,1.0,senal)
    len_signal = len(senal)#obtengo el largo de la señal
    len_signal = float(len_signal)
    time = float(len_signal)/float(rate)#genero el tiempo total del audio
    x = arange(0,time,1.0/float(rate))
    return x,filtered


""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""BLOQUE PRINCIPAL"""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Obtenemos los datos del audio.
rate,info = read("beacon.wav")
senal, largo_senal, t = getDatos(info,rate)

#Los graficamos
graphTime(t,senal,"Audio con respecto al tiempo", "Audio con respecto al tiempo.png")

#Sacamos la transformada de Fourier y graficamos.
xfourier,yfourier = fourierTransformation(senal,len(senal))

#Graficamos la transformada y el espectograma.
graphTransformation(xfourier, yfourier, "Transformada de fourier audio original", "Transformada de fourier audio original.png")

graphSpecgram(senal,rate,"Espectograma audio original", "Espectograma audio original.png")

#En esta sección se aplican los filtros.

#FILTRO PASO BAJO
x1,filtradoLow = appLowFilter(senal,rate)
#Gráficos
graphTime(x1,filtradoLow,"Audio con filtro paso bajo", "Audio con filtro paso bajo.png")
xfourier2, yfourier2 = fourierTransformation(filtradoLow,len(filtradoLow))
graphTransformation(xfourier2,yfourier2,"Transformada Fourier y filtro paso bajo", "Transformada Fourier y filtro paso bajo.png")
graphSpecgram(filtradoLow,rate,"Espectograma audio filtro paso bajo","Espectograma audio filtro paso bajo.png")


#FILTRO PASO ALTO
x2,filtradoHigh = appHighFilter(senal,rate)
graphTime(x2,filtradoHigh,"Audio con filtro paso alto", "Audio con filtro paso alto.png")
xfourier3, yfourier3 = fourierTransformation(filtradoHigh,len(filtradoHigh))
graphTransformation(xfourier3,yfourier3,"Transformada Fourier y filtro paso alto", "Transformada Fourier y filtro paso alto.png")
graphSpecgram(filtradoHigh,rate,"Espectograma audio filtro paso alto","Espectograma audio filtro paso alto.png")

#FILTRO PASO BANDA
x3,filtradoBandPass = appBandPassFilter(senal,rate)
graphTime(x3,filtradoBandPass,"Audio con filtro paso banda", "Audio con filtro paso banda.png")
xfourier4, yfourier4 = fourierTransformation(filtradoBandPass,len(filtradoBandPass))
graphTransformation(xfourier4,yfourier4,"Transformada Fourier y filtro paso banda", "Transformada Fourier y filtro paso banda.png")
graphSpecgram(filtradoBandPass,rate,"Espectograma audio filtro paso banda","Espectograma audio filtro paso banda.png")

write("sinRuidoLow.wav",rate,filtradoLow.astype('int16'))
write("sinRuidoHigh.wav",rate,filtradoHigh.astype('int16'))
write("sinRuidoBandPass.wav",rate,filtradoHigh.astype('int16'))