#!/usr/bin/python

import tensorflow as tf
import random
import csv

class SparseNN:

	VAL_SPLIT = 0.1
	BATCH_SIZE = 32

	INPUT_WIDTH = 2
	MAX_HIDDEN_WIDTH = 1
	
	FULL_CONN_EDGES_ENCODER = []
	FULL_CONN_EDGES_DECODER = []

	input = None
	encoder = None
	output = None
	label = None

	train_input = None
	train_label = None
	val_input = None
	val_label = None

	sess = None
	loss = None
	train_step = None

	mse = None

	def __init__(self, input_width, max_hidden_width):
		self.INPUT_WIDTH = input_width
		self.MAX_HIDDEN_WIDTH = max_hidden_width

		for i in range(self.INPUT_WIDTH):
			for j in range(self.MAX_HIDDEN_WIDTH):
				self.FULL_CONN_EDGES_ENCODER.append([j, i])
				self.FULL_CONN_EDGES_DECODER.append([i, j])

		self.sess = tf.InteractiveSession()
	
	@staticmethod
	def __weight_variable(shape, edges):
		initial = tf.Variable(tf.truncated_normal([len(edges)], stddev=0.1))
		return tf.SparseTensor(
			indices=edges,
			shape=shape,
			values=initial)

	@staticmethod
	def __bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def define_model(self, encoder_edges=FULL_CONN_EDGES_ENCODER, decoder_edges=FULL_CONN_EDGES_DECODER):
		#print edges
		self.input = tf.placeholder(tf.float32, shape=[None, self.INPUT_WIDTH])
		input_trans = tf.transpose(self.input)

		# Encoder
		encoder_weight = SparseNN.__weight_variable([self.MAX_HIDDEN_WIDTH, self.INPUT_WIDTH], encoder_edges)
		encoder_bias = SparseNN.__bias_variable([self.MAX_HIDDEN_WIDTH, 1])
		encoder = tf.sparse_tensor_dense_matmul(encoder_weight, input_trans) + encoder_bias
		encoder = tf.nn.sigmoid(encoder)
		self.encoder = encoder

		# Decoder
		decoder_weight = SparseNN.__weight_variable([self.INPUT_WIDTH, self.MAX_HIDDEN_WIDTH], decoder_edges)
		decoder_bias = SparseNN.__bias_variable([self.INPUT_WIDTH, 1])
		decoder = tf.sparse_tensor_dense_matmul(decoder_weight, encoder) + decoder_bias
		decoder = tf.nn.sigmoid(decoder)

		self.output = decoder
		self.label = self.input
		label_trans = tf.transpose(self.label)

		self.loss = tf.reduce_mean(tf.reduce_mean(tf.square(self.output - label_trans), axis=0))
		self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

		self.sess.run(tf.global_variables_initializer())

	def define_data(self, input_data):
		split = int(len(input_data) * self.VAL_SPLIT)
		self.val_input = input_data[0: split]
		self.train_input = input_data[split: ]

	def __shuffle_train_data(self):
		index = range(len(self.train_input))
		random.shuffle(index)
		new_train_input = []
		for i in range(len(index)):
			new_train_input.append(self.train_input[i])
		self.train_input = new_train_input

	def train(self):
		self.mse = 999999999.9
		for iter in range(10000):
			for i in range(0, len(self.train_input) - self.BATCH_SIZE, self.BATCH_SIZE):
				self.train_step.run(
					feed_dict={self.input: self.train_input[i: i + self.BATCH_SIZE]})
			newMse = self.loss.eval(feed_dict={self.input: self.val_input})
			print newMse
			if newMse >= self.mse:
				break

			self.mse = newMse			
			self.__shuffle_train_data()

	def getMse(self):
		return self.mse

	def getCode(self):
		code = self.encoder.run(feed_dict={self.input: self.train_input})
		return code.tolist()

if __name__ == '__main__':

	## Read request

	file = open('request', 'r')
	parameters = file.read().split('\n')
	data_filename = parameters[0]
	input_width = int(parameters[1])
	hidden_width = int(parameters[2])
	edge_count = int(parameters[3])
	edges = []
	for i in range(edge_count):
		edge = [int(intStr) for intStr in parameters[4 + i].split(' ')]
		edges.append(edge)
	
	saveOrNot = (parameters[4 + edge_count] == '1')
	if saveOrNot:
		outputFilename = parameters[5 + edge_count]

	## Read input data
	inputDataFile = open(data_filename, 'rb')
	inputData = []
	csvReader = csv.reader(inputDataFile, delimiter=' ')
	for row in csvReader:
		inputData.append(row)

	## Build model
	sparse_nn = SparseNN(input_width, max_hidden_width=hidden_width)
	sparse_nn.define_model(edges)
	sparse_nn.define_data(inputData)
	sparse_nn.train()

	## Output MSE
	replyFile = open('reply', 'w')
	replyFile.write(str(sparse_nn.getMse()))

	## Output data
	if saveOrNot:
		code = sparse_nn.getCode()
		csvWriter = csv.writer(outputFilename, 'wb')
		csvWriter.write(code)
	
'''
if __name__ == '__main__':
	## Solve classic XOR problem
	sparse_nn = SparseNN(2, max_hidden_width=2)
	#sparse_nn.define_model(edges=[[[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0], [0, 1]]])
	sparse_nn.define_model()
	sparse_nn.define_data([[0, 0], [0, 1], [1, 0], [1, 1]] * 1000)
	sparse_nn.train()
'''
