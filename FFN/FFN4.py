from __future__ import division
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
	
alpha_cypher = [' ','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
	
class FFN:
	def __init__(self, layers):
		self.layers = layers
		self.weights = [np.random.randn(y, x) * 0.01 for y, x in zip(layers[1:], layers[:-1])]
		self.biases = [np.random.randn(y, 1) * 0.01 for y in layers[1:]]
		
	def feed(self, input, dropconnect_wmatrices, dropconnect_bmatrices):
		self.input = [input]
		self.net = []
		for i in xrange(len(self.layers) - 1):
			y = f(np.dot(self.weights[i] * dropconnect_wmatrices[i], self.input[i]) + self.biases[i] * dropconnect_bmatrices[i])
			self.input.append(y)
		return self.input[-1]
		
	def backprop(self, target, eta, dropconnect_wmatrices, dropconnect_bmatrices):
		self.err = [(self.input[-1] - target) * (1.0 - self.input[-1]) * self.input[-1]]
		self.weights[-1] -= eta * np.dot(self.err[-1], self.input[-2].T) * dropconnect_wmatrices[-1]
		self.biases[-1] -= eta * self.err[0] * dropconnect_bmatrices[-1]
		for i in xrange(len(self.layers) - 2):
			self.err.insert(0, np.dot(self.weights[-i - 1].T, self.err[-i - 1]) * (1.0 - self.input[-i - 2]) * self.input[-i - 2])
			self.weights[-i - 2] -= eta * np.dot(self.err[-i - 2], self.input[-i - 3].T) * dropconnect_wmatrices[-i - 2]
			self.biases[-i - 2] -= eta * self.err[-i - 2] * dropconnect_bmatrices[-i - 2]
	
	def test(self, testdata):
		yes = 0
		no = 0
		yescorr = 0
		nocorr = 0
		choose = []
		for datum in testdata:
			output = self.feed(datum['x'], [1] * len(self.weights), [1] * len(self.biases))
			if datum['y'][0] > datum['y'][1]:
				yes += 1
				if output[0] > output[1]:
					yescorr += 1
					choose.append(datum['text'])
			else:
				no += 1
				if output[0] < output[1]:
					nocorr += 1
				if output[0] > output[1]:
					choose.append(datum['text'])
		print "YES - " + str(yescorr) + " / " + str(yes) + " - "  + str(yescorr / yes)
		print "NO - " + str(nocorr) + " / " + str(no) + " - " + str(nocorr / no)
		print "TOTAL - " + str(yescorr + nocorr) + " / " + str(yes + no) + " - " + str((yescorr + nocorr) / (yes + no))
		return choose
	
	def train(self, data, epochs, eta, dropconnect_probability, testdata = None):
		for i in xrange(epochs):
			np.random.shuffle(data)
			for datum in data:
				dropconnect_wmatrices = []
				dropconnect_bmatrices = []
				for weight in self.weights:
					dropconnect_wmatrices.append(np.random.binomial(1, dropconnect_probability, weight.shape))
				for bias in self.biases:
					dropconnect_bmatrices.append(np.random.binomial(1, dropconnect_probability, bias.shape))

				self.feed(datum['x'], dropconnect_wmatrices, dropconnect_bmatrices)
				self.backprop(datum['y'], eta, dropconnect_wmatrices, dropconnect_bmatrices)
			if (testdata):
				self.test(testdata)
			
	def get_words(self, data):
		selected = []
		for word in data:
			output = self.feed(word['x'], [1] * len(self.weights), [1] * len(self.biases))
			if output[0] > output[1]:
				selected.append(word['text'])
		return selected
		
def create_vocabs(articles):
	vocabs = []
	for article in articles:
		vocab = []
		article = np.sum(np.sum(article))
		for word in article:
			if not word in vocab:
				vocab.append(word)
		vocabs.append(vocab)
	return vocabs
		
def create_data(text):
	vocabs = create_vocabs(text)
	data = []
	for i in xrange(len(vocabs)):
		body = text[i][2]
		for j in text[i][3:-2]:
			body += j
		article = np.sum(text[i][0] + text[i][1] + body)
		body = np.sum(body)
		for word in vocabs[i]:
			if len(word) < 1:
				print i
			x = [0.0] * 9
			if word in text[i][0][0]:
				x[0] = 1.0
			if word in text[i][1][0]:
				x[1] = 1.0
			if word in body:
				x[2] = 1.0
			if word[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
				x[3] = 1.0
			for sentence in np.sum(text[i][:-1]):
				if sentence[0] == word:
					x[4] = 1.0
			x[5] = len(word)
			x[6] = article.count(word)
			x[7] = vocabs[i].index(word)
			x[8] = 0
			for letter in word: #don't use alpha-cypher
				if letter.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
					x[8] += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(letter.upper())
			y = [0.0] * 2
			if word in text[i][-1][-1]:
				y[0] = 1.0
			else:
				y[1] = 1.0
			data.append({'x':np.array(x)[np.newaxis].T,'y':np.array(y)[np.newaxis].T,'text':word})
	return data