import numpy as np

top = 0
def if_max(x, y):
	global top
	if max(np.abs(x)) >= top:
		top = max(np.abs(x))
		print str(max(np.abs(x))) + ' - ' + y

def f(x):
	if_max(x, 'f')
	return 1.0 / (1.0 + np.exp(-x))

class RN:
	def __init__(self, layers, unfold_length):
		self.layers = layers
		self.unfold_length = unfold_length
		self.weights = [np.append(np.random.randn(y, x), np.random.randn(y, y), axis=1) for y, x in zip(layers[1:], layers[:-1])]
		self.biases = [np.random.randn(y, 1) for y in layers[1:]]
		self.input_prime = [[np.random.randn(y, 1) for y in layers[1:]]]
		self.net = []
		
	def feed(self, input):
		if len(self.input_prime) > self.unfold_length:
			self.input_prime.pop(0)
			self.net.pop(0)	
		input = [np.append(input, self.input_prime[-2][0], axis=0)]
		self.input_prime.append([0 for y in self.layers[1:]])
		self.net.append([])
		for i in xrange(len(self.layers) - 1):
			self.net[-1].append(np.dot(self.weights[i], input[i]) + self.biases[i])
			y = f(self.net[-1][i])
			input_prime[-1][i] = y
			if i == len(self.layers) - 2:
				input.append(y)
				break
			input.append(np.append(y, self.input_prime[-2][i + 1], axis=0))
		return input[-1]
		
	def backprop(self, target, eta):
		"""This one the errs are backwards, they're insert in the order of end to front of network.'"""
		self.err_o = self.input_prime[-1][-1] - target
		self.err = [[np.dot(self.weights[-i - 1], self.err_o) * f_prime(self.net[-1][-2])]]
		for i in xrange(len(self.layers[2:])):
			self.err[0].insert(0, np.dot(self.weights[-i - 1].T, self.err[0][-i - 1]) * f_prime(self.net[-1][-i - 2]))
		for i in xrange(self.unfold_length):
			self.err.insert(0, [])
			