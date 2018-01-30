from __future__ import division 
import numpy as np
import random

def cross_entropy_cost(a, y):
	#where a is the output of the network and y is the desired output
	return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
	
def cross_entropy_cost_prime(a, y):
	return a - y

class FFN:
	def __init__(self, layers):
		self.layers = layers
		self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
		self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
		
	def feed_forward(self, input):
		input = np.array(input)[np.newaxis].T
		self.inputs = [input]
		self.inputs.append(np.dot(self.weights[0], input) + self.biases[0])
		self.outputs = [np.tanh(self.inputs[1])]
		for j in xrange(len(self.layers) - 2):
			self.inputs.append(np.dot(self.weights[j + 1], self.outputs[j]) + self.biases[j + 1])
			self.outputs.append(np.tanh(self.inputs[j + 2]))
		return self.outputs[-1]
		
	def train(self, data, epochs, eta):
	#data should be a list of objects where each object has two properties 'x' and 'y', where x is the input to the network and y is the target output value
		for i in xrange(epochs):
			np.random.shuffle(data)
			for datum in data:
				self.feed_forward(datum['x'])
				delta = self.outputs[-1] - np.array(datum['y'])[np.newaxis].T
				self.w_errs = [delta * self.outputs[-1]]
				self.b_errs = [delta]
				for k in xrange(len(self.layers) - 2):
					self.w_errs.insert(0, np.dot(self.weights[-k - 1].T, self.w_errs[-k - 1]) * (1 - np.tanh(self.inputs[-k - 2]) ** 2) * self.inputs[-k - 2])
					self.b_errs.insert(0, np.dot(self.weights[-k - 1].T, self.b_errs[-k - 1]) * (1 - np.tanh(self.inputs[-k - 2]) ** 2))
				for l in xrange(len(self.w_errs)):
					self.weights[l] = self.weights[l] - eta * self.w_errs[l]
					self.biases[l] = self.biases[l] - eta * self.b_errs[l]
			print (i + 1) / epochs