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
	
def f_prime(x):
	#if_max(x, 'fp')
	return (1.0 - f(x)) * f(x)

class RN:
	def __init__(self, layers, unfold_length):
		self.layers = layers
		self.unfold_length = unfold_length
		self.cweights = [np.random.randn(y, x) for y, x in zip(layers[1:], layers[:-1])]
		self.hweights = [np.random.randn(y, y) for y in layers[1:-1]]
		self.biases = [np.random.randn(y, 1) for y in layers[1:]]
		self.cinput = []
		self.hinput = [[np.random.randn(y, 1) for y in layers[1:-1]]]
		self.cnet = []
		self.hnet = []
		
	def feed(self, input):
		if len(self.hinput) > self.unfold_length:
			self.hinput.pop(0)
			self.cinput.pop(0)
			self.cnet.pop(0)
			self.hnet.pop(0)
		self.cinput.append([input])
		self.hinput.append([])
		self.cnet.append([])
		self.hnet.append([])
		for i in xrange(len(self.layers) - 1):
			self.cnet[-1].append(np.dot(self.cweights[i], self.cinput[-1][i]))
			if i < len(self.layers) - 2:
				self.hnet[-1].append(np.dot(self.hweights[i], self.hinput[-2][i]))
				y = f(self.cnet[-1][-1] + self.hnet[-1][-1] + self.biases[i])
				self.cinput[-1].append(y)
				self.hinput[-1].append(y)
			else:
				y = f(self.cnet[-1][-1] + self.biases[i])
				self.cinput[-1].append(y)
		return self.cinput[-1][-1]
		
	def backprop(self, target, eta):
		self.err_o = self.cinput[-1][-1] - target
		self.cweights[-1] -= eta * np.dot(self.err_o, self.cinput[-1][-2].T)
		self.err = [[np.dot(self.cweights[-1].T, self.err_o) * f_prime(self.cnet[-1][-2])]]
		for i in xrange(len(self.layers[1:-2])):
			self.err[0].insert(0, np.dot(self.cweights[-i - 2].T, self.err[-1][-i - 1]) * f_prime(self.cnet[-1][-i - 3]))
		if len(self.hnet) - 2 < self.unfold_length:
			unfold_times = len(self.hnet) - 2
		else:
			unfold_times = self.unfold_length - 2
		for j in xrange(unfold_times):
			self.err.insert(0, [])
			self.err[0].append(np.dot(self.hweights[-1].T, self.err[-j - 1][-1]) * f_prime(self.hnet[-j - 2][-1]))
			for k in xrange(len(self.layers[1:-2])):
				self.err[-j - 2].insert(0, (np.dot(self.cweights[-k - 2].T, self.err[-j - 2][-k - 1]) + np.dot(self.hweights[-k - 2], self.err[-j - 1][-k - 2])) * f_prime(self.cnet[-j - 2][-k - 3] + self.hnet[-j - 1][-k - 2]))
		self.cdeltaw = []
		self.hdeltaw = []
		for l in xrange(len(self.err[-1])):
			self.cdeltaw.append(np.dot(self.err[-1][l], self.cinput[-1][l].T))
			self.hdeltaw.append(np.dot(self.err[-1][l], self.hinput[-2][l].T))
		for m in xrange(len(self.err[:-1])):
			for n in xrange(len(self.err[-1])):
				self.cdeltaw = self.cdeltaw + np.dot(self.err[m + 1][n], self.cinput[m + 1][n].T)
				self.hdeltaw = self.hdeltaw + np.dot(self.err[m + 1][n], self.hinput[m][n].T)
		for o in xrange(len(self.cdeltaw)):
			self.cdeltaw[o] *= eta
			self.hdeltaw[o] *= eta
			self.cweights[o] -= self.cdeltaw[o]
			self.hweights[o] -= self.hdeltaw[o]
			
	def train(self, traindata, epochs, eta, testdata = None):
		"""Something's wrong with the backprop function! Causing overflow in the f_prime function."""
		for i in xrange(epochs):
			for datum in traindata:
				self.feed(datum['x'])
				self.backprop(datum['y'], eta)
			total_err = 0
			if testdata:
				for datum in testdata:
					total_err += self.feed(datum['x']) - datum['y']
				print "error - " + str(total_err / len(testdata))