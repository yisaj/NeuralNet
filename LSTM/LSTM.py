import numpy as np
import random

data = [
	{'x': [1,1], 'y': [-1,-1,1,-1,1]},
	{'x': [-1,1], 'y': [1,1,1,1,1]},
	{'x': [1,-1], 'y': [-1,-1,-1,-1,-1]}
	]

def cross_entropy_cost(a, y):
	#where a is the output of the network and y is the desired output
	return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
	
def cross_entropy_cost_prime(a, y):
	return y / a - (1 - y) / (1 - a)

class LSTM:
	def __init__(self, layers):
		self.layers = layers
		self.initialize_network_values()
		
	def initialize_network_values(self):
	#Creates the initial arrays for the biases, weights, cells, outputs, and inputs
		#biases
		self.b_a = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.b_i = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.b_f = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.b_o = [np.random.randn(y, 1) for y in self.layers[1:]]
		
		#weights
		self.w_ax = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
		self.w_ah = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_ix = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
		self.w_ih = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_ic = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_fx = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
		self.w_fh = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_fc = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_ox = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
		self.w_oh = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_oc = [np.random.randn(y, 1) for y in self.layers[1:]]
		
		#cells
		self.cells = [np.random.randn(y, 1) for y in self.layers[1:]]
		
		#outputs
		self.outputs = [np.random.randn(y, 1) for y in self.layers[1:]]
		
		#inputs
		self.inputs = [np.random.randn(y, 1) for y in self.layers]
		
		#errors
		self.errors = []
		
	def feed_forward(self, x):
	#Runs input x through the network. Outputs h are stored in self.output and saved for the next run through
		x = np.array(x)[np.newaxis].transpose()
		self.inputs[0] = x
		for n in range(len(self.layers) - 1):
			a = np.tanh(np.dot(self.w_ax[n], self.inputs[n]) + self.w_ah[n] * self.outputs[n])
			i = np.tanh(np.dot(self.w_ix[n], self.inputs[n]) + self.w_ih[n] * self.outputs[n] + self.w_ic[n] * self.cells[n] + self.b_i[n])
			f = np.tanh(np.dot(self.w_fx[n], self.inputs[n]) + self.w_fh[n] * self.outputs[n] + self.w_fc[n] * self.cells[n] + self.b_f[n])
			o = np.tanh(np.dot(self.w_ox[n], self.inputs[n]) + self.w_oh[n] * self.outputs[n] + self.w_oc[n] * self.cells[n] + self.b_o[n])
			self.cells[n] = a * i
			self.cells[n] = self.cells[n] * f
			self.outputs[n] = self.cells[n] * o
			self.inputs[n + 1] = self.cells[n] * o
		return self.outputs
		
	def train(self, data, epochs, eta):
		self.output_history = []
		np.random.shuffle(data)
		for n in xrange(epochs):
			#feeds each input in the input through the network, compiling a list of each final output
			for x in data[n]['x']:
				self.output_history.append(self.feed_forward(x))
			#calculates error at the output layer
			output_error = cross_entropy_cost_prime(self.outputs[-1], np.array(data[n]['y'][data[n]['x'].index(x)])[np.newaxis].T) * (1 - np.tan(self.outputs[-1]) ** 2)
			output_layer_err = [output_error]
			#calculates error for each layer in the output plane
			for j in xrange(len(self.layers) - 1):
				output_layer_err.insert(0, [np.dot(self.w_ax[-1 - j].T, output_layer_err[-1 - j]) * (1 - np.tan(self.inputs[-2 - j]) ** 2)])
			self.errors.insert(0, output_layer_err)
			for i in xrange(len(self.output_history)) - 1):
				layer_err = [self.w_ah[-1 - i] * self.errors[-1 - i][-1] * (1 - np.tan(self.output_history[-2 - j][-1]) ** 2)]
				for k in xrange(len(self.layers) - 1):
					ext_w = np.append(self.w_ax[-1 - k].T, self.w_ah[-1 - k], axis=1)
					layer_err.insert(0, ext_w)
				self.error.insert(0, layer_err)
		return self.errors