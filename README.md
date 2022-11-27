# 46500-Probabilistic-Methods-in-Wind-Energy

### Action Items:

Data selection:

	1) Seperate turbines
	
	2) Cut off data before and after failure 
	
	3) Cut out Idling (RPM > something)
	
	4) Train-Test split 
	
		a) Train: only normal behavior
		
		b) Test: both normal and failure behavior

Target variable: (Trial and Error)

	A) Power? (Most interesting for wind turbines)
	
	B) Each a model for all phases. If the difference between the phases is too big -> failure)
	
		1) Temperature
		
		2) Voltage
		
		3) Current
		
		4) Active power

Feature selection:

	A) Correlation
	
		1) Compute correlation coefficients between target variable and features
		
		2) Take the N features with biggest correlation
		
	B) Educated guess
	
		1) Temperatures: internal and external
		
		2) Power curve: power, rotation speed, ambient wind speed
		
		3) ...?
	
Time scale:

	1) Compute auto-correlation and "Integral timescale"
	
	2) Calculate moving average over the last "Integral timescale"

Model:

	A) Long short-term memory with tensorflow (maybe you get it running)
	
	B) Artificial neural network with sklearn (like Exercise6_1)
	
	C) Random Forest with sklearn

Training, Validation, ...	

Deadline extension: Saturday 03/12 23:59pm



### Some Links that I found helpful:
##### How to install tensorflow:
- https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/

But sometimes there are issues.
- https://stackoverflow.com/questions/46568913/tensorflow-import-error-no-module-named-tensorflow

Maybe it works better to use the example environment from the DL exercise.

### Some good introductions:
A very narrative introduction about the theoretical background.
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/


A well documented example code that might help us.
- https://towardsdatascience.com/lstm-recurrent-neural-networks-how-to-teach-a-network-to-remember-the-past-55e54c2ff22e

### Documentations of Tensorflow:
- https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#args_1
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
- https://www.tensorflow.org/api_docs/python/tf/estimator/WarmStartSettings
