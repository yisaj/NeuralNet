from __future__ import division
from numbapro import cuda, jit, int32, float64, void, vectorize
import numpy as np
import math
import random
import os
import scipy.special
import time

@jit
def expit(x):
	return 1 / (1 + np.exp(x))

def f(x):
	x = np.clip(x, -300, 300)
	return expit(x)

def f_prime(x):
	return (1 - expit(x)) * expit(x)

def g(x):
	x = np.clip(x, -300, 300)
	return 4 * expit(x) - 2

def g_prime(x):
	return (1 - expit(x)) * expit(x) * 4

def h(x):
	x = np.clip(x, -300, 300)
	return 2 * expit(x) - 1

def h_prime(x):
	return (1 - expit(x)) * expit(x) * 2

alpha_cypher = [' ','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','.',',','\'','\"',':',';','-','(',')','!','?']

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
			a = [0] * 74
			a[alpha_cypher.index(x)] = 1
			set['x'].append(np.array(a)[np.newaxis].T)
		for y in datum[1]:
			b = [0] * 74
			b[alpha_cypher.index(y)] = 1
			set['y'].append(np.array(b)[np.newaxis].T)
		result.append(set)
	return result
	
def alpha_decode(binary):
	return alpha_cypher[binary.tolist().index(max(binary).tolist())]

def alpha_encode(alpha):
	result = [0] * 74
	result[alpha_cypher.index(alpha)] = 1
	return np.array(result)[np.newaxis].T
	
class LSTM:
	def __init__(self, inp, hid, out):
		self.hid = hid
		#weights from last layer of hidden cells to output layer
		self.w_oc = np.random.randn(out, hid)
		self.b_o = np.random.randn(out, 1)
		#initial cell states and biases for each gate and cell output
		self.s = np.random.randn(hid, 1)
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
		timer = True
		for i in xrange(epochs):	
			for datum in data:
				for k in xrange(len(datum['x'])):
					if timer == True:
						start = time.clock()
						
					self.feed(datum['x'][k])
					
					self.err_o = self.y_o - datum['y'][k]
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
					if timer == True:
						stop = time.clock()
						lag = len(datum['x']) * (stop - start)
						minutes = int(lag / 60)
						seconds = int(lag - (minutes * 60))
						timer = False
					print str((k + 1) / len(datum['x']) * 100) + '% - ' + str(minutes) + 'm ' + str(seconds) + 's estimated'
	
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
	articles = {}
	for subdir, dirs, files in os.walk(directory):
		n = 0
		for file in files:
			article = open(os.path.join(directory, file), 'r').read()
			article = article.replace('.', '')
			article = article.replace(',', '')
			article = article.replace('?', '')
			article = article.replace('!', '')
			article = article.replace(';', '')
			article = article.replace(':', '')
			article = article.split("\n\n")
			article[0] = article[0][6:]
			article[-1] = article[-1][8:]
			for i in xrange(len(article)):
				article[i] = article[i].split(' ')
			articles[n] = article
			n += 1
	return articles
	
def repeat_words(articles):
	vocab = {}
	distinct = []
	repeat = []
	for i in xrange(len(articles)):
		for word in articles[i][-1]:
			if word in vocab:
				vocab[word] += 1
			if not word in vocab:
				vocab[word] = 1
	for i in vocab:
		if vocab[i] <= 1:
			distinct.append(i)
		if vocab[i] > 1:
			repeat.append(i)
	return [distinct, repeat]