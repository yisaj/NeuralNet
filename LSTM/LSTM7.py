from __future__ import division
import numpy as np
import random
import os
import time

top = 0
def softmax(x):
	a = np.zeros((len(x),1)) + np.sum(np.abs(x))
	return np.exp(x) / np.exp(a)

def if_max(x, y):
	global top
	if max(np.abs(x)) >= top:
		top = max(np.abs(x))
		print str(max(np.abs(x))) + ' - ' + y

def f(x):
	if_max(x, 'f')
	x = np.clip(x, -300, 300)
	return np.clip(1.0 / (1.0 + np.exp(-x)), 0.000001, 0.999999)

def f_prime(x):
	if_max(x, 'fp')
	return np.clip((1.0 - f(x)) * f(x), 0.000001, 0.5)

def g(x):
	if_max(x, 'g')
	x = np.clip(x, -300, 300)
	return np.clip(4.0 / (1.0 + np.exp(-x)) - 2.0, -1.999999, 1.999999)

def g_prime(x):
	if_max(x, 'gp')
	return np.clip((1.0 - f(x)) * f(x) * 4.0, 0.000001, 2.0)

def h(x):
	if_max(x, 'h')
	x = np.clip(x, -300, 300)
	return np.clip(2.0 / (1.0 + np.exp(-x)) - 1.0, -1.999999, 1.999999)

def h_prime(x):
	if_max(x, 'hp')
	return np.clip((1.0 - f(x)) * f(x) * 2.0, 0.000001, 1.0)

alpha_cypher = [' ','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','.',',','\'','\"',':',';','-','(',')','!','?','\n','/']

binary_cypher = [
[0,0,0,0,0,0,0],[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1],[1,1,0,0,0,0,0],[1,0,1,0,0,0,0],[1,0,0,1,0,0,0],
[1,0,0,0,1,0,0],[1,0,0,0,0,1,0],[1,1,1,0,0,1,1],[1,1,1,0,0,0,0],[1,1,0,1,0,0,0],[1,1,0,0,1,0,0],[1,1,0,0,0,1,0],[1,1,0,0,0,0,1],[1,1,1,1,0,0,0],[1,1,1,0,1,0,0],[1,1,1,0,0,1,0],
[0,1,0,0,1,1,0],[1,1,1,1,1,0,0],[1,1,1,1,0,1,0],[1,1,1,1,0,0,1],[1,1,1,1,1,1,0],[1,1,1,1,1,0,1],[1,1,1,1,1,1,1],[0,1,1,1,1,1,1],[1,0,1,1,1,1,1],[1,1,0,1,1,1,1],[1,1,1,0,1,1,1],
[1,1,1,1,0,1,1],[0,0,1,1,1,1,1],[0,1,0,1,1,1,1],[0,1,1,0,1,1,1],[0,1,1,1,0,1,1],[0,1,1,1,1,0,1],[0,1,1,1,1,1,0],[0,0,0,1,1,1,1],[0,0,1,0,1,1,1],[0,0,1,1,0,1,1],[0,0,1,1,1,0,1],
[0,0,1,1,1,1,0],[0,0,0,0,1,1,1],[0,0,0,1,0,1,1],[0,0,0,1,1,0,1],[0,0,0,1,1,1,0],[0,0,0,0,0,1,1],[0,0,0,0,1,0,1],[0,0,0,0,1,1,0],[0,0,1,1,0,0,0],[0,0,0,1,1,0,0],[0,1,0,0,0,1,0],
[0,1,0,1,0,0,0],[0,0,0,1,0,1,0],[1,1,0,0,1,1,0],[0,1,1,0,1,0,1],[0,1,0,1,1,0,0],[1,0,1,1,1,0,0],[0,0,1,1,0,1,0],[1,0,0,0,0,0,1],[1,1,0,0,1,1,1],[1,0,0,0,0,1,1],[1,1,1,0,0,0,1],
[1,0,0,0,1,1,1],[1,0,0,1,0,1,1],[1,1,0,1,0,0,1],[0,1,0,1,0,1,1],[1,1,0,1,0,1,0],[0,1,1,1,1,0,0],[0,1,0,1,1,1,0],[0,1,1,1,0,1,0]
]

def alpha_encode_bible_compact(string):
	x = []
	y = []
	for i in xrange(len(string) - 1):
		x.append(np.array(binary_cypher[alpha_cypher.index(string[i])])[np.newaxis].T)
		y.append(np.array(binary_cypher[alpha_cypher.index(string[i + 1])])[np.newaxis].T)
	return [{'x': x, 'y': y}]
	
def alpha_encode_bible(string):
	x = []
	y = []
	for i in xrange(len(string) - 1):
		x.append(alpha_encode(string[i]))
		y.append(alpha_encode(string[i + 1]))
	return [{'x': x, 'y': y}]

def alpha_encode_data(data):
	result = []
	for datum in data:
		set = {'x': [], 'y': []}
		for x in datum[0]:
			a = [0] * 76
			a[alpha_cypher.index(x)] = 1
			set['x'].append(np.array(a)[np.newaxis].T)
		for y in datum[1]:
			b = [0] * 76
			b[alpha_cypher.index(y)] = 1
			set['y'].append(np.array(b)[np.newaxis].T)
		result.append(set)
	return result
	
def alpha_decode(binary):
	return alpha_cypher[binary.tolist().index(max(binary).tolist())]

def alpha_encode(alpha):
	result = [0.0] * 76
	result[alpha_cypher.index(alpha)] = 1.0
	return np.array(result)[np.newaxis].T
	
class LSTM:
	def __init__(self, inp, hid, out):
		self.hid = hid
		#weights from last layer of hidden cells to output layer
		self.w_oc = np.random.randn(out, hid)
		self.b_o = np.random.randn(out, 1)
		#initial cell states and biases for each gate and cell output
		self.s = np.zeros((hid, 1))
		self.b_s = np.random.randn(hid, 1)	
		self.b_u = np.random.randn(hid, 1)	
		self.b_f = np.random.randn(hid, 1)	
		self.b_n = np.random.randn(hid, 1)	
		#weights from each hidden unit to other hidden units in their layer and their gates
		self.w_ch = np.random.randn(hid, hid)
		self.w_uh = np.random.randn(hid, hid)
		self.w_fh = np.random.randn(hid, hid)
		self.w_nh = np.random.randn(hid, hid)
		#weights from each hidden layer to the next one
		self. w_cc = np.random.randn(hid, inp)
		self. w_uc = np.random.randn(hid, inp)
		self. w_fc = np.random.randn(hid, inp)
		self. w_nc = np.random.randn(hid, inp)
		#outputs for t = -1
		self.h = np.random.randn(hid, 1)
		self.h_prime = np.random.randn(hid, 1)
		#initial partial derivatives for cell, input gate, and forget gate deltas
		self.parw_c = np.zeros((hid, inp + hid))
		self.parw_n = np.zeros((hid, inp + hid))
		self.parw_f = np.zeros((hid, inp + hid))
		self.parb_c = np.zeros((hid, 1))
		self.parb_n = np.zeros((hid, 1))
		self.parb_f = np.zeros((hid, 1))
		#intermediate values
		self.net_c = np.zeros((hid, 1))
		self.net_u = np.zeros((hid, 1))
		self.net_f = np.zeros((hid, 1))
		self.net_n = np.zeros((hid, 1))
		self.y_c = np.zeros((hid, 1))
		self.y_u = np.zeros((hid, 1))
		self.y_f = np.zeros((hid, 1))
		self.y_n = np.zeros((hid, 1))
		self.y_s = np.zeros((hid, 1))
		
	def feed(self, input):
		self.input = input
		self.net_c = np.dot(self.w_cc, input) + np.dot(self.w_ch, self.h)
		self.net_u = np.dot(self.w_uc, input) + np.dot(self.w_uh, self.h)
		self.net_f = np.dot(self.w_fc, input) + np.dot(self.w_fh, self.h)
		self.net_n = np.dot(self.w_nc, input) + np.dot(self.w_nh, self.h)
		
		self.y_c = g(self.net_c + self.b_s)
		self.y_u = f(self.net_u + self.b_u)
		self.y_f = f(self.net_f + self.b_f)
		self.y_n = f(self.net_n + self.b_n)
		
		self.s = self.y_c * self.y_n + self.s * self.y_f
		self.y_s = h(self.s)
		
		self.h_prime = self.h
		self.h = self.y_s * self.y_u
		self.net_o = np.dot(self.w_oc, self.h)
		self.y_o = f(self.net_o + self.b_o)
		
	def train(self, data, epochs, eta):
		for i in xrange(epochs):	
			for datum in data:
				for k in xrange(len(datum['x'])):
					self.feed(datum['x'][k])
					
					self.err_o = f_prime(self.net_o) * (self.y_o - datum['y'][k])
					self.w_oc -= eta * np.dot(self.err_o, self.h.T)
					self.b_o -= eta * self.err_o
					
					self.err_u = f_prime(self.net_u) * h(self.s) * np.dot(self.w_oc.T, self.err_o)
					self.w_uc -= eta * np.dot(self.err_u, self.input.T)
					self.w_uh -= eta * np.dot(self.err_u, self.h_prime.T)
					self.b_u -= eta * self.err_u
						
					self.e_s = self.y_u * h_prime(self.s) * np.dot(self.w_oc.T, self.err_o)

					self.parw_c = self.parw_c * self.y_f + np.dot(g_prime(self.net_c) * self.y_n, np.append(self.input, self.h_prime)[np.newaxis])
					self.parw_n = self.parw_n * self.y_f + np.dot(g(self.net_c) * f_prime(self.net_n), np.append(self.input, self.h_prime)[np.newaxis])
					self.parw_f = self.parw_f * self.y_f + np.dot(h(self.s) * f_prime(self.net_f), np.append(self.input, self.h_prime)[np.newaxis])
					self.parb_c = self.parb_c * self.y_f + g_prime(self.net_c) * self.y_n
					self.parb_n = self.parb_n * self.y_f + g(self.net_c) * f_prime(self.net_n)
					self.parb_f = self.parb_f * self.y_f + h(self.s) * f_prime(self.net_f)

					self.w_cc += eta * self.e_s * self.parw_c[:,self.hid:]
					self.w_nc += eta * self.e_s * self.parw_n[:,self.hid:]
					self.w_fc += eta * self.e_s * self.parw_f[:,self.hid:]
					self.w_ch += eta * self.e_s * self.parw_c[:,:self.hid]
					self.w_nh += eta * self.e_s * self.parw_n[:,:self.hid]
					self.w_fh += eta * self.e_s * self.parw_f[:,:self.hid]
					self.b_s += eta * self.e_s * self.parb_c
					self.b_n += eta * self.e_s * self.parb_n
					self.b_f += eta * self.e_s * self.parb_f
	
	def alpha_loop_compact(self, input, times):
		result = ''
		input = np.array(binary_cypher[alpha_cypher.index(input)])[np.newaxis].T
		for i in xrange(times):
			self.feed(input)
			if np.round(self.h[-1]).T.tolist()[0] in binary_cypher:
				result += alpha_cypher[binary_cypher.index(np.round(self.h[-1]).T.tolist()[0])]
				input = np.round(self.h[-1])
			else:
				result += '_'
				input = self.h[-1]
		return result
				
	def alpha_loop(self, input, times):
		result = ''
		input = alpha_encode(input)
		for i in xrange(times):
			self.feed(input)
			result += alpha_decode(self.y_o)
			input = alpha_encode(alpha_decode(self.y_o))
		return result
		
def process_article(directory):
	articles = []
	for subdir, dirs, files in os.walk(directory):
		n = 0
		for file in files:
			article = open(os.path.join(directory, file), 'r').read()
			article = article.replace(',', '')
			article = article.replace('?', '')
			article = article.replace('!', '')
			article = article.replace(';', '')
			article = article.replace(':', '')
			article = article.split("\n\n")
			article[0] = article[0][6:]
			article[-1] = article[-1][8:]
			for i in xrange(len(article)):
				article[i] = article[i].split('. ')
				for j in xrange(len(article[i])):
					if '.' in article[i][j]:
						article[i][j] = article[i][j][:-1]
					article[i][j] = article[i][j].split(' ')
			articles.append(article)
			n += 1
	articles = remove_repeats(articles)
	return articles

def create_vocab(words):
	result = []
	for word in words:
		if not word in result:
			result.append(word)
	return result

def create_data(articles):
	data = []
	for article in articles:
		vocab = create_vocab(np.sum(np.sum(article)))
		body = np.sum(article[2:-1])
		maxcount = max([np.sum(np.sum(article[:-1])).count(x) for x in vocab])
		maxlen = max([len(x) for x in vocab])
		for word in vocab:
			if type(body[0]) == type([]):
				body = np.sum(body)
			x = [0.0] * 8
			if word in article[0][0]:
				x[0] = 1.0
			if word in article[1][0]:
				x[1] = 1.0
			if word in body:
				x[2] = 1.0
			x[3] = np.sum(np.sum(article[:-1])).count(word)
			if len(word) > 0:
				if word[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
					x[4] = 1.0
			for sentence in np.sum(article[:-1]):
				if word == sentence[0]:
					x[5] = 1.0
					break
			x[6] = len(word)
			x[7] = vocab.index(word)
			if word in article[-1][0]:
				y = [1.0]
			else:
				y = [0.0]
			data.append({'x': np.array(x)[np.newaxis].T, 'y': np.array(y)[np.newaxis].T,'text': word})
	return data
	
def repeat_words(articles, threshold):
	vocab = {}
	distinct = []
	repeat = []
	for i in xrange(len(articles)):
		for word in articles[i][-1][0]:
			if word in vocab:
				vocab[word] += 1
			if not word in vocab:
				vocab[word] = 1
	for i in vocab:
		if vocab[i] <= threshold:
			distinct.append(i)
		if vocab[i] > threshold:
			repeat.append(i)
	return [distinct, repeat]

cut_words = ['to', 'from', 'would', 'after', 'a', 'an', 'not', 'that', 'and', 'have', 'which', 'The', 'are', 'due', 'been', 'will', 'is', 'it', 'in', 'the', 'has', 'because', 'for', 'be', 'by', 'on', 'about', 'of', 'there', 'but', 'with', 'as', 'at']
	
def remove_repeats(articles):
	for a in xrange(len(articles)):
		for p in xrange(len(articles[a])):
			for s in xrange(len(articles[a][p])):
				for word in cut_words:
					for i in xrange(articles[a][p][s].count(word)):
						articles[a][p][s].remove(word)
	return articles
	