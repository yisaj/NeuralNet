import numpy as np

top = 0
def if_max(x, y):
	global top
	if max(np.abs(x)) > top:
		top = max(np.abs(x))
		print str(max(np.abs(x))) + ' - ' + y

def f(x):
	if_max(x, 'f')
	return 1.0 / (1.0 + np.exp(-x))
	
def f_prime(x):
	#if_max(x, 'fp')
	return (1.0 - f(x)) * f(x)

def softmax(x):
	return np.exp(x) / (1 + np.exp(np.sum(x)))
	
alpha_cypher = [' ','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','.',',','\'','\"',':',';','-','(',')','!','?','\n','/']

def alpha_encode(letter):
	x = [0.0] * 76
	x[alpha_cypher.index(letter)] = 1.0
	return np.array(x)[np.newaxis].T
	
def create_alpha_data(string):
	data = []
	for i in xrange(len(string) - 1):
		datum = {}
		datum['x'] = alpha_encode(string[i])
		datum['y'] = alpha_encode(string[i + 1])
		data.append(datum)
	return data

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
		self.err = [self.cinput[-1][-1] - target]
		self.err.insert(0, np.dot(self.cweights[-1].T, self.err[-1]) * f_prime(self.cnet[-1][-2]))
		if len(self.cnet) < self.unfold_length:
			backsteps = len(self.cnet) - 1
		else:
			backsteps = self.unfold_length - 1

		for i in xrange(backsteps):
			self.err.insert(0, np.dot(self.hweights[-1].T, self.err[-i - 2]) * f_prime(self.hnet[-i - 2][-1]))
		self.cweights[-1] -= eta * np.dot(self.err[-1], self.cinput[-1][-2].T)
		cdeltaw = np.dot(self.err[-2], self.cinput[-1][-3].T)
		hdeltaw = np.dot(self.err[-2], self.hinput[-2][0].T)
		for j in xrange(len(self.err[:-2])):
			cdeltaw += np.dot(self.err[-j - 3], self.cinput[-j - 2][-3].T)
			hdeltaw += np.dot(self.err[-j - 3], self.hinput[-j - 3][0].T)
		cdeltaw /= len(self.err[:-1])
		hdeltaw /= len(self.err[:-1])
		#Calculate the delta ws and AVERAGE them
		self.cweights[-2] -= eta * cdeltaw
		self.hweights[0] -= eta * hdeltaw
			
	def train(self, traindata, epochs, eta, testdata = None):
		for i in xrange(epochs):
			for datum in traindata:
				self.feed(datum['x'])
				self.backprop(datum['y'], eta)
			total_err = 0
			if testdata:
				for datum in testdata:
					total_err += self.feed(datum['x']) - datum['y']
				print "error - " + str(total_err / len(testdata))
	
	def alpha_loop(self, input, times):
		input = alpha_encode(input)
		result = ""
		for i in xrange(times):
			self.feed(input)
			probability = 0
			softmax_arrray =
			threshold = np.random.rand()
			for j in xrange(76):
				if softmaxarray[j] <= threshold:
					
			
			
			letter = self.cinput[-1][-1].T.tolist()[0].index(np.max(self.cinput[-1][-1]))
			result += alpha_cypher[letter]
			input = alpha_encode(alpha_cypher[letter])
		return result