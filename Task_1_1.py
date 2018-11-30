
## Mocking Bot - Task 1.1: Note Detection

#  Instructions
#  ------------
#
#  This file contains Main function and note_detect function. Main Function helps you to check your output
#  for practice audio files provided. Do not make any changes in the Main Function.
#  You have to complete only the note_detect function. You can add helper functions but make sure
#  that these functions are called from note_detect function. The final output should be returned
#  from the note_detect function.
#
#  Note: While evaluation we will use only the note_detect function. Hence the format of input, output
#  or returned arguments should be as per the given format.
#  
#  Recommended Python version is 2.7.
#  The submitted Python file must be 2.7 compatible as the evaluation will be done on Python 2.7.
#  
#  Warning: The error due to compatibility will not be entertained.
#  -------------


## Library initialisation

# Import Modules
# DO NOT import any library/module
# related to Audio Processing here
import numpy as np
import math
import wave
import os
import matplotlib.pyplot as plt
import struct
from scipy.signal import get_window
from numpy import mean, sqrt, square, arange
from scipy.fftpack import fft, ifft
tol = 1e-14          


# Teams can add helper functions
# Add all helper functions here

############################### Your Code Here ##############################################
def peakDetection(mX, t):
	"""
	Detect spectral peak locations
	mX: magnitude spectrum, t: threshold
	returns ploc: peak locations
	"""

	thresh = np.where(np.greater(mX[1:-1],t), mX[1:-1], 0); # locations above threshold
	next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     # locations higher than the next one
	prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
	ploc = thresh * next_minor * prev_minor                 # locations fulfilling the three criteria
	ploc = ploc.nonzero()[0] + 1                            # add 1 to compensate for previous steps
	return ploc
def peakInterp(mX, pX, ploc):
	"""
	Interpolate peak values using parabolic interpolation
	mX, pX: magnitude and phase spectrum, ploc: locations of peaks
	returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
	"""

	val = mX[ploc]                                          # magnitude of peak bin
	lval = mX[ploc-1]                                       # magnitude of bin at left
	rval = mX[ploc+1]                                       # magnitude of bin at right
	iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
	ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
	ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks by linear interpolation
	return iploc, ipmag, ipphase


def f0Twm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0):
	"""
	Function that wraps the f0 detection function TWM, selecting the possible f0 candidates
	and calling the function TWM with them
	pfreq, pmag: peak frequencies and magnitudes,
	ef0max: maximum error allowed, minf0, maxf0: minimum  and maximum f0
	f0t: f0 of previous frame if stable
	returns f0: fundamental frequency in Hz
	"""
	if (minf0 < 0):                                  # raise exception if minf0 is smaller than 0
		raise ValueError("Minimum fundamental frequency (minf0) smaller than 0")

	if (maxf0 >= 16000):                             # raise exception if maxf0 is bigger than 10000Hz
		raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")

	if (pfreq.size < 3) & (f0t == 0):                # return 0 if less than 3 peaks and not previous f0
		return 0

	f0c = np.argwhere((pfreq>minf0) & (pfreq<maxf0))[:,0] # use only peaks within given range
	if (f0c.size == 0):                              # return 0 if no peaks within range
		return 0
	f0cf = pfreq[f0c]                                # frequencies of peak candidates
	f0cm = pmag[f0c]                                 # magnitude of peak candidates

	if f0t>0:                                        # if stable f0 in previous frame
		shortlist = np.argwhere(np.abs(f0cf-f0t)<f0t/2.0)[:,0]   # use only peaks close to it
		maxc = np.argmax(f0cm)
		maxcfd = f0cf[maxc]%f0t
		if maxcfd > f0t/2:
			maxcfd = f0t - maxcfd
		if (maxc not in shortlist) and (maxcfd>(f0t/4)): # or the maximum magnitude peak is not a harmonic
			shortlist = np.append(maxc, shortlist)
		f0cf = f0cf[shortlist]                         # frequencies of candidates

	if (f0cf.size == 0):                             # return 0 if no peak candidates
		return 0

	f0, f0error = TWM_p(pfreq, pmag, f0cf)        # call the TWM function with peak candidates

	if (f0>0) and (f0error<ef0max):                  # accept and return f0 if below max error allowed
		return f0
	else:
		return 0

def TWM_p(pfreq, pmag, f0c):
	"""
	Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
	[better to use the C version of this function: UF_C.twm]
	pfreq, pmag: peak frequencies in Hz and magnitudes,
	f0c: frequencies of f0 candidates
	returns f0, f0Error: fundamental frequency detected and its error
	"""

	p = 0.5                                          # weighting by frequency value
	q = 1.4                                          # weighting related to magnitude of peaks
	r = 0.5                                          # scaling related to magnitude of peaks
	rho = 0.33                                       # weighting of MP error
	Amax = max(pmag)                                 # maximum peak magnitude
	maxnpeaks = 10                                   # maximum number of peaks used
	harmonic = np.matrix(f0c)
	ErrorPM = np.zeros(harmonic.size)                # initialize PM errors
	MaxNPM = min(maxnpeaks, pfreq.size)
	for i in range(0, MaxNPM) :                      # predicted to measured mismatch error
		difmatrixPM = harmonic.T * np.ones(pfreq.size)
		difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
		FreqDistance = np.amin(difmatrixPM, axis=1)    # minimum along rows
		peakloc = np.argmin(difmatrixPM, axis=1)
		Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
		PeakMag = pmag[peakloc]
		MagFactor = 10**((PeakMag-Amax)/20)
		ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
		harmonic = harmonic+f0c

	ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
	MaxNMP = min(maxnpeaks, pfreq.size)
	for i in range(0, f0c.size) :                    # measured to predicted mismatch error
		nharm = np.round(pfreq[:MaxNMP]/f0c[i])
		nharm = (nharm>=1)*nharm + (nharm<1)
		FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
		Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
		PeakMag = pmag[:MaxNMP]
		MagFactor = 10**((PeakMag-Amax)/20)
		ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

	Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)  # total error
	f0index = np.argmin(Error)                       # get the smallest error
	f0 = f0c[f0index]                                # f0 with the smallest error

	return f0, Error[f0index]
def dftAnal(x, w, N):
	"""
	Analysis of a signal using the discrete Fourier transform
	x: input signal, w: analysis window, N: FFT size 
	returns mX, pX: magnitude and phase spectrum
	"""

	
	hN = (N//2)+1                                           # size of positive spectrum, it includes sample 0
	hM1 = (w.size+1)//2                                     # half analysis window size by rounding
	hM2 = w.size//2                                         # half analysis window size by floor
	fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
	w = w / sum(w)                                          # normalize analysis window
	xw = x*w                                                # window the input sound
	fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
	fftbuffer[-hM2:] = xw[:hM2]        
	X = fft(fftbuffer)                                      # compute FFT
	absX = abs(X[:hN])                                      # compute ansolute value of positive side
	absX[absX<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
	mX = 20 * np.log10(absX)                                # magnitude spectrum of positive frequencies in dB
	X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            # for phase calculation set to 0 the small values
	X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0            # for phase calculation set to 0 the small values         
	pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
	return mX, pX



def note_detect(audio_file):

	#   Instructions
	#   ------------
	#   Input   :   audio_file -- a single test audio_file as input argument
	#   Output  :   Detected_Note -- String corresponding to the Detected Note
	#   Example :   For Audio_1.wav file, Detected_Note = "A4"

	Detected_Note = ""

	
	
	
		
	# Add your code here
	sound_file = audio_file

	file_length = sound_file.getnframes()

	frame_chunk=1
	file_length = sound_file.getnframes()
	
	while True:

		if(frame_chunk>file_length):
			break
		
		frame_chunk=frame_chunk*2


	fs=sound_file.getframerate()
	sound = np.zeros(file_length)
	
	for i in range(file_length):
		data = sound_file.readframes(1)
		data = struct.unpack("<h", data)
		sound[i] = int(data[0])
	
 	sound = np.divide(sound, float(2**15))



	
	
	arr=[]
	rms1=0.00

	




		
	M=40001
	window='blackman'
	t=-300
	H=500
	w=get_window(window,M)
	fs=44100
	x=sound[int(0.09*fs):int(0.09*fs+M)]
	frame_chunk=1
	while True:
		if(frame_chunk>len(x)):
			break
	
		frame_chunk=frame_chunk*2
		
	minf0=10
	maxf0=16000-1
	f0et=1
	
	N=frame_chunk
	hN = N//2                                                  # size of positive spectrum
	hM1 = int(math.floor((w.size+1)/2))                        # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                            # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                             # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM1))                             # add zeros at the end to analyze last sample
	pin = hM1                                                  # init sound pointer in middle of anal window          
	pend = x.size - hM1                                        # last sample to start a frame
	fftbuffer = np.zeros(N)                                    # initialize buffer for FFT
	w = w / sum(w)                                             # normalize analysis window
	f0 = []                                                    # initialize f0 output
	f0t = 0                                                    # initialize f0 track
	f0stable = 0                                               # initialize f0 stable
	while pin<pend:             
		x1 = x[pin-hM1:pin+hM2]                                  # select frame
		mX, pX = dftAnal(x1, w, N)                           # compute dft           
		ploc = peakDetection(mX, t)                           # detect peak locations   
						
		iploc, ipmag, ipphase = peakInterp(mX, pX, ploc)      # refine peak values
		ipfreq = fs * iploc/N                                    # convert locations to Hez
		f0t = f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
		if ((f0stable==0)&(f0t>0)) \
				or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
			f0stable = f0t                                         # consider a stable f0 if it is close to the previous one
		else:
			f0stable = 0
		f0 = np.append(f0, f0t)                                  # add f0 to output array
		pin += H  



	f0.sort()
	
	freq=np.mean(f0[len(f0)-9:len(f0)-1])
		





		
	A4 = 440	
	C0 = A4*pow(2, -4.75)
	name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
	
	h = round(12*math.log(abs(freq)/C0)/math.log(2))
	octave = h // 12

	n = h % 12
	Detected_Note = name[int(n)] + str(octave)

	
	return Detected_Note


############################### Main Function ##############################################

if __name__ == "__main__":

	#   Instructions
	#   ------------
	#   Do not edit this function.

	# code for checking output for single audio file
	path = os.getcwd()
	
	file_name = path + "\Task_1.1_Audio_files\Audio_1.wav"
	audio_file = wave.open(file_name)

	Detected_Note = note_detect(audio_file)

	print("\n\tDetected Note = " + str(Detected_Note))

	# code for checking output for all audio files
	x = raw_input("\n\tWant to check output for all Audio Files - Y/N: ")
	
	if x == 'Y':

		Detected_Note_list = []

		file_count = len(os.listdir(path + "\Task_1.1_Audio_files"))

		for file_number in range(1, file_count):

			file_name = path + "\Task_1.1_Audio_files\Audio_"+str(file_number)+".wav"
			audio_file = wave.open(file_name)

			Detected_Note = note_detect(audio_file)
			
			Detected_Note_list.append(Detected_Note)

		print("\n\tDetected Notes = " + str(Detected_Note_list))
	
	
