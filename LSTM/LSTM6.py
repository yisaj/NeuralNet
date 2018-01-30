from __future__ import division
from decimal import *
import numpy as np
import random
import time

getcontext().prec = 10

top = 0
def if_max(x, y):
	global top
	if max(np.abs(x)) >= top:
		top = max(np.abs(x))
		print str(max(np.abs(x))) + ' - ' + y

def f(x):
	if_max(x, 'f')
	return 1 / (1 + np.exp(-x))
	
def f_prime(x):
	if_max(x, 'fp')
	return (1 - f(x)) * f(x)
	
def g(x):
	if_max(x, 'g')
	return 4 / (1 + np.exp(-x)) - 2
	
def g_prime(x):
	if_max(x, 'gp')
	return (1 - f(x)) * f(x) * 4

def r(x):
	if_max(x, 'h')
	return 2 / (1 + np.exp(-x)) - 1
	
def r_prime(x):
	if_max(x, 'hp')
	return (1 - f(x)) * f(x) * 2

alpha_cypher = [' ','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','.',',','\'','\"',':',';','-','(',')','!','?']

binary_cypher = [[Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(0),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(1)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(1)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(1)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(0),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(0)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(1)], [Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(0),Decimal(1),Decimal(1)], [Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(1),Decimal(0),Decimal(1)], [Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0)], [Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(0),Decimal(1),Decimal(0),Decimal(0),Decimal(0),Decimal(0),Decimal(1)], [Decimal(0),Decimal(0),Decimal(1),Decimal(1),Decimal(1),Decimal(1),Decimal(1),Decimal(1)]]

def alpha_encode_bible(string):
	x = []
	y = []
	for i in xrange(len(string) - 1):
		x.append(np.array(binary_cypher[alpha_cypher.index(string[i])])[np.newaxis].T)
		y.append(np.array(binary_cypher[alpha_cypher.index(string[i + 1])])[np.newaxis].T)
	return [{'x': x, 'y': y}]
	
def alpha_encode(letter):
	return binary_cypher[aplha_cypher.index(letter)]
	
def alpha_decode(binary):
	return alpha_cypher[binary_cypher.index(binary.T.tolist()[0])]
		
class LSTM:
	def decrand(self, y, x):
		array = np.random.randn(y, x).astype(Decimal)
		for i in xrange(len(array)):
			for j in xrange(len(array[i])):
				array[i][j] = Decimal(array[i][j])
		return array
	
	def deczeros(self, y, x):
		array = np.zeros((y, x)).astype(Decimal)
		for i in xrange(len(array)):
			for j in xrange(len(array[i])):
				array[i][j] = Decimal(array[i][j])
		return array

	def __init__(self, input, hidden, output):
		self.hidden = hidden
		
		self.w_oc = self.decrand(output, hidden)
		self.b_o = self.decrand(output, 1)
		
		self.s = self.decrand(hidden, 1)
		self.b_s = self.decrand(hidden, 1)
		self.b_u = self.decrand(hidden, 1)
		self.b_f = self.decrand(hidden, 1)
		self.b_n = self.decrand(hidden, 1)
		
		self.w_ch = self.decrand(hidden, hidden)
		self.w_uh = self.decrand(hidden, hidden)
		self.w_fh = self.decrand(hidden, hidden)
		self.w_nh = self.decrand(hidden, hidden)
		
		self.w_cc = self.decrand(hidden, input)
		self.w_uc = self.decrand(hidden, input)
		self.w_fc = self.decrand(hidden, input)
		self.w_nc = self.decrand(hidden, input)
		
		self.h = [self.decrand(y, 1) for y in [input, hidden, output]]
		self.h_prime = [self.decrand(y, 1) for y in [input, hidden, output]]
		
		self.parw_c = self.deczeros(hidden, input + hidden)
		self.parw_n = self.deczeros(hidden, input + hidden)
		self.parw_f = self.deczeros(hidden, input + hidden)
		self.parb_c = self.deczeros(hidden, 1)
		self.parb_n = self.deczeros(hidden, 1)
		self.parb_f = self.deczeros(hidden, 1)
		
	def feed(self, input):
		self.h_prime[0]= self.h[0]
		self.h[0] = input
		self.net_c = np.dot(self.w_cc, self.h[0]) + np.dot(self.w_ch, self.h[1])
		self.net_u = np.dot(self.w_uc, self.h[0]) + np.dot(self.w_uh, self.h[1])
		self.net_f = np.dot(self.w_fc, self.h[0]) + np.dot(self.w_fh, self.h[1])
		self.net_n = np.dot(self.w_nc, self.h[0]) + np.dot(self.w_nh, self.h[1])
		
		self.y_c = g(self.net_c + self.b_s)
		self.y_u = f(self.net_u + self.b_u)
		self.y_f = f(self.net_f + self.b_f)
		self.y_n = f(self.net_n + self.b_n)
		
		self.s = self.y_c * self.y_n + self.s * self.y_f
		self.y_s = r(self.s)
		self.h_prime[1] = self.h[1]
		self.h[1] = self.y_s
		
		self.net_o = np.dot(self.w_oc, self.h[1])
		self.y_o = f(self.net_o + self.b_o)
		self.h_prime[2] = self.h[2]
		self.h[2] = self.y_o
		
	def backprop(self, target, eta):
		self.err_o = self.y_o - target
		self.w_oc -= eta * np.dot(self.err_o, self.h[1].T)
		self.b_o -= eta * self.err_o
		
		self.err_u = f_prime(self.net_u) * r(self.s) * np.dot(self.w_oc.T, self.err_o)
		self.w_uc -= eta * np.dot(self.err_u, self.h[0].T)
		self.w_uh -= eta * np.dot(self.err_u, self.h_prime[1].T)
		self.b_u -= eta * self.err_u
		
		self.e_s = self.y_u * r_prime(self.s) * np.dot(self.w_oc.T, self.err_o)
		self.parw_c = self.parw_c * self.y_f + np.dot(g_prime(self.net_c) * self.y_n, np.append(self.h[0], self.h_prime[1])[np.newaxis])
		self.parw_n = self.parw_n * self.y_f + np.dot(g(self.net_c) * f_prime(self.net_n), np.append(self.h[0], self.h_prime[1])[np.newaxis])
		self.parw_f = self.parw_f * self.y_f + np.dot(r(self.s) * f_prime(self.net_f), np.append(self.h[0], self.h_prime[1])[np.newaxis])
		self.parb_c = self.parb_c * self.y_f + g_prime(self.net_c) * self.y_n
		self.parb_n = self.parb_n * self.y_f + g(self.net_c) * f_prime(self.net_n)
		self.parb_f = self.parb_f * self.y_f + r(self.s) * f_prime(self.net_f)
		
		self.w_cc += eta * self.e_s * self.parw_c[:,self.hidden:]
		self.w_nc += eta * self.e_s * self.parw_c[:,self.hidden:]
		self.w_fc += eta * self.e_s * self.parw_c[:,self.hidden:]
		self.w_ch += eta * self.e_s * self.parw_c[:,:self.hidden]
		self.w_nh += eta * self.e_s * self.parw_c[:,:self.hidden]
		self.w_fh += eta * self.e_s * self.parw_c[:,:self.hidden]
		self.b_s += eta * self.e_s * self.parb_c
		self.b_n += eta * self.e_s * self.parb_n
		self.b_f += eta * self.e_s * self.parb_f
		
	def train(self, data, epochs, eta):
		for i in xrange(epochs):
			for datum in data:
				for j in xrange(len(datum['x'])):
					self.feed(datum['x'][j])
					self.backprop(datum['y'][j], eta)
					print (j + 1) / len(datum['x']) * 100
					
	def alpha_loop_compact(self, input, times):
		result = ''
		input = np.array(binary_cypher[alpha_cypher.index(input)])[np.newaxis].T
		for i in xrange(times):
			self.feed(input)
			output = self.h[2]
			for j in xrange(len(output)):
				output[j][0] = output[j][0].to_integral_value()		
			if output.T.tolist()[0] in binary_cypher:
				result += alpha_cypher[binary_cypher.index(output.T.tolist()[0])]
				input = output
			else:
				result += '_'
				input = self.h[2]
		return result