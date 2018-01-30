import numpy as np
import random

def tanh(x):
	return np.tanh(x)
	
def tanh_prime(x):
	return 1 - np.tanh(x) ** 2

def logistic(x):
	return 1 / (1 + np.exp(-x))
	
def logistic_prime(x):
	return logistic(x) * logistic(-x)
	
def cross_entropy_cost(a, y):
	#where a is the output of the network and y is the desired output
	return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
	
def cross_entropy_cost_prime(a, y):
	return a - y
	
def quadratic_cost(a, y):
	0.5 * np.linalg.norm(a - y) ** 2

class LSTM:
	def __init__(self, layers):
		self.layers = layers
		#outputs
		self.outputs = [np.random.randn(y, 1) for y in self.layers[1:]]
		#cells
		self.c = [np.random.randn(y, 1) for y in self.layers[1:]]
		#biases
		self.b_a = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.b_i = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.b_f = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.b_o = [np.random.randn(y, 1) for y in self.layers[1:]]
		#weights
		self.w_ah = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_ih = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_fh = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_oh = [np.random.randn(y, 1) for y in self.layers[1:]]

		self.w_ic = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_fc = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.w_oc = [np.random.randn(y, 1) for y in self.layers[1:]]
		
		self.w_ax = [np.random.randn(y, x) for y, x in zip(self.layers[1:], self.layers[:-1])]
		self.w_ix = [np.random.randn(y, x) for y, x in zip(self.layers[1:], self.layers[:-1])]
		self.w_fx = [np.random.randn(y, x) for y, x in zip(self.layers[1:], self.layers[:-1])]
		self.w_ox = [np.random.randn(y, x) for y, x in zip(self.layers[1:], self.layers[:-1])]

	def activation(self, input, index):
		return tanh(np.dot(self.w_ax[index], input) + self.w_ah[index] * self.outputs[index] + self.b_a[index])

	def input_gate(self, input, index):
		return tanh(np.dot(self.w_ix[index], input) + self.w_ih[index] * self.outputs[index] + self.w_ic[index] * self.c[index] + self.b_i[index])
		
	def forget_gate(self, input, index):
		return tanh(np.dot(self.w_fx[index], input) + self.w_fh[index] * self.outputs[index] + self.w_fc[index] * self.c[index] + self.b_f[index])
	
	def output_gate(self, input, index):
		return tanh(np.dot(self.w_ox[index], input) + self.w_oh[index] * self.outputs[index] + self.w_oc[index] * self.c[index] + self.b_o[index])

	def feed_forward(self, input):
		input = np.array(input)[np.newaxis].T
		for i in xrange(len(self.layers) - 1):
			self.c[i] = self.c[i] * self.forget_gate(input, i) + self.activation(input, i) * self.input_gate(input, i)
			self.outputs[i] = tanh(self.c[i]) * self.output_gate(input, i)
			input = self.outputs[i]
		return self.outputs
		
	def train(self, data, duration, eta):
		for k in xrange(duration):
			pass

	def backprop(self, target):
		self.w_errs = [[(self.output_history[-1][-1] - target) * self.outputs[-1][-1]]]
		self.b_errs = [[self.output_history[-1][-1] - target]]
		
		for l in xrange(len(self.layers - 2)):
			self.w_errs[0].insert(0, self.)