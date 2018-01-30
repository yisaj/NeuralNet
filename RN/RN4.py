from __future__ import division
import numpy as np

top = 0
def if_max(x, y):
	global top
	if max(np.abs(x)) > top:
		top = max(np.abs(x))
		print str(max(np.abs(x))) + ' - ' + y

def f(x):
	#x = np.clip(x, -700, 36.7)
	if_max(x, 'f')
	return 1.0 / (1.0 + np.exp(-x))
	
def f_prime(x):
	#x = np.clip(x, -700, 36.7)
	if_max(x, 'fp')
	return (1.0 - f(x)) * f(x)

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
	def __init__(self, inp, hid, out, unfold):
		self.inp = inp
		self.hid = hid
		self.out = out
		self.unfold = unfold
		self.w_ci = np.random.randn(hid, inp)
		self.w_cc = np.random.randn(hid, hid)
		self.w_oc = np.random.randn(out, hid)
		self.biases = [np.random.randn(hid, 1), np.random.randn(out, 1)]
		self.nets = [[np.zeros((x, 1)) for x in [hid, out]]]
		self.ys = [[np.zeros((x, 1)) for x in [inp, hid, out]]]
		self.err = None
		
	def feed(self, input, drop_ci, drop_cc, drop_oc, drop_bc, drop_bo):
		if len(self.nets) > self.unfold:
			self.nets.pop(0)
			self.ys.pop(0)
		self.nets.append([])
		self.ys.append([])
		self.ys[-1].append(input)
		self.nets[-1].append(np.dot(self.w_ci * drop_ci, input) + np.dot(self.w_cc * drop_cc, self.ys[-2][1]) + self.biases[0] * drop_bc)
		self.ys[-1].append(f(self.nets[-1][-1]))
		self.nets[-1].append(np.dot(self.w_oc * drop_oc, self.ys[-1][1]) + self.biases[1] * drop_bo)
		self.ys[-1].append(f(self.nets[-1][-1]))
		return(self.ys[-1][-1])
		
	def backprop(self, target, eta, drop_ci, drop_cc, drop_oc, drop_bc, drop_bo):
		self.err = self.ys[-1][-1] - target
		self.w_oc -= eta * np.dot(self.err, self.ys[-1][1].T) * drop_oc
		self.biases[1] -= eta * self.err * drop_bo
		if len(self.ys) - 1 < self.unfold:
			unfold = len(self.ys) - 1
		else:
			unfold = self.unfold
		self.err = np.dot(self.w_oc.T, self.err) * f_prime(self.nets[-1][0])
		deltaw_ci = np.dot(self.err, self.ys[-1][0].T)
		deltaw_cc = np.dot(self.err, self.ys[-2][1].T)
		deltab_c = self.err
		for i in xrange(unfold - 1):
			self.err = np.dot(self.w_cc.T, self.err) * f_prime(self.nets[-i - 2][0])
			deltaw_ci += np.dot(self.err, self.ys[-i - 2][0].T)
			deltaw_cc += np.dot(self.err, self.ys[-i - 3][1].T)
			deltab_c += self.err
		#print deltaw_cc[0][0]
		self.w_ci -= eta * deltaw_ci / unfold * drop_ci
		self.w_cc -= eta * deltaw_cc / unfold * drop_cc
		self.biases[0] -= eta * deltab_c / unfold * drop_bc
		
	def train(self, traindata, epochs, eta, testletter, drop, testdata = None):
		j = 0
		for i in xrange(epochs):
			for datum in traindata:
				j += 1
				drop_ci = np.random.binomial(1, drop, self.w_ci.shape)
				drop_cc = np.random.binomial(1, drop, self.w_cc.shape)
				drop_oc = np.random.binomial(1, drop, self.w_oc.shape)
				drop_bc = np.random.binomial(1, drop, self.biases[0].shape)
				drop_bo = np.random.binomial(1, drop, self.biases[1].shape)
				self.feed(datum['x'], drop_ci, drop_cc, drop_oc, drop_bc, drop_bo)
				self.backprop(datum['y'], eta, drop_ci, drop_cc, drop_oc, drop_bc, drop_bo)
				if j % 100 == 0:
					print self.alpha_loop(testletter, 30)
				
	def alpha_loop(self, input, times):
		input = alpha_encode(input)
		result = ""
		for i in xrange(times):
			drop_ci = np.random.binomial(1, 1, self.w_ci.shape)
			drop_cc = np.random.binomial(1, 1, self.w_cc.shape)
			drop_oc = np.random.binomial(1, 1, self.w_oc.shape)
			drop_bc = np.random.binomial(1, 1, self.biases[0].shape)
			drop_bo = np.random.binomial(1, 1, self.biases[1].shape)
			self.feed(input, drop_ci, drop_cc, drop_oc, drop_bc, drop_bo)
			letter = self.ys[-1][-1].T.tolist()[0].index(np.max(self.ys[-1][-1]))
			result += alpha_cypher[letter]
			input = alpha_encode(alpha_cypher[letter])
		return result