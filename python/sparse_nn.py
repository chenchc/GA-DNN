import tensorflow as tf
import random
import csv
import re
from joblib import Parallel, delayed

class SparseNN:

	VAL_SPLIT = 0.1
	BATCH_SIZE = 128
	LR = 0.01

	INPUT_WIDTH = 1
	OUTPUT_WIDTH = 1
	MAX_HIDDEN_DEPTH = 1
	MAX_HIDDEN_WIDTH = 1
	
	FULL_CONN_EDGES = []

	input = None
	output = None
	label = None

	train_input = None
	train_label = None
	val_input = None
	val_label = None

	sess = None
	loss = None
	train_step = None
	accuracy = None

	mse = 9999999.9
	final_accuracy = 0.0
	parameter_count = 0

	def __init__(self, input_width, output_width, max_hidden_depth, max_hidden_width):
		self.INPUT_WIDTH = input_width
		self.OUTPUT_WIDTH = output_width
		self.MAX_HIDDEN_DEPTH = max_hidden_depth
		self.MAX_HIDDEN_WIDTH = max_hidden_width

		prev_width = self.INPUT_WIDTH
		for i in range(self.MAX_HIDDEN_DEPTH):
			full_conn_layer_edges = []
			for j in range(self.MAX_HIDDEN_WIDTH[i]):
				for k in range(prev_width):
					full_conn_layer_edges.append([j, k])
			self.FULL_CONN_EDGES.append(full_conn_layer_edges)
			prev_width = self.MAX_HIDDEN_WIDTH[i]

		full_conn_layer_edges = []
		for i in range(self.OUTPUT_WIDTH):
			for j in range(prev_width):
				full_conn_layer_edges.append([i, j])
		self.FULL_CONN_EDGES.append(full_conn_layer_edges)

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

	def define_model(self, edges=FULL_CONN_EDGES):
		self.parameter_count = reduce(lambda x, y: x + y, [len(layer) for layer in edges])
		self.input = tf.placeholder(tf.float32, shape=[None, self.INPUT_WIDTH])
		input_trans = tf.transpose(self.input)

		prev_width = self.INPUT_WIDTH
		prev_tensor = input_trans
		for i in range(self.MAX_HIDDEN_DEPTH + 1):
			if i < self.MAX_HIDDEN_DEPTH:
				cur_width = self.MAX_HIDDEN_WIDTH[i]
			else:
				cur_width = self.OUTPUT_WIDTH
			
			hidden_weight = SparseNN.__weight_variable([cur_width, prev_width], edges[i])
			hidden_bias = SparseNN.__bias_variable([cur_width, 1])
			hidden = tf.sparse_tensor_dense_matmul(hidden_weight, prev_tensor) + hidden_bias
			if i < self.MAX_HIDDEN_DEPTH:
				hidden = tf.nn.sigmoid(hidden)
			#print prev_tensor.get_shape(), hidden_weight.get_shape(), hidden_bias.get_shape(), hidden.get_shape()
				
			prev_width = cur_width
			prev_tensor = hidden
		
		self.output = prev_tensor
		self.label = tf.placeholder(tf.float32, shape=[None, self.OUTPUT_WIDTH])
		label_trans = tf.transpose(self.label)

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, label_trans, dim=0))
		self.train_step = tf.train.AdamOptimizer(self.LR).minimize(self.loss)

		correct_prediction = tf.equal(tf.argmax(self.output, 0), tf.argmax(label_trans, 0))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.sess.run(tf.global_variables_initializer())

	def define_data(self, input_data, label_data):
		split = int(len(input_data) * self.VAL_SPLIT)
		self.val_input = input_data[0: split]
		self.val_label = label_data[0: split]
		self.train_input = input_data[split: ]
		self.train_label = label_data[split: ]

	def __shuffle_train_data(self):
		index = range(len(self.train_input))
		random.shuffle(index)
		new_train_input = []
		new_train_label = []
		for i in range(len(index)):
			new_train_input.append(self.train_input[i])
			new_train_label.append(self.train_label[i])
		self.train_input = new_train_input
		self.train_label = new_train_label

	def train(self):
		for iter in range(10000):
			for i in range(0, len(self.train_input) - self.BATCH_SIZE, self.BATCH_SIZE):
				self.train_step.run(
					feed_dict={self.input: self.train_input[i: i + self.BATCH_SIZE], 
					self.label: self.train_label[i: i + self.BATCH_SIZE]})
			newMse = self.loss.eval(feed_dict={self.input: self.val_input, self.label: self.val_label})
			if newMse >= self.mse:
				break
			self.mse = newMse
			self.final_accuracy = self.accuracy.eval(feed_dict={self.input: self.val_input, self.label: self.val_label})
			print 'Accuracy: ' + str(self.final_accuracy)
			print 'Loss: ' + str(self.mse)
			self.__shuffle_train_data()

	def getLoss(self):
		return self.mse

	def getAccuracy(self):
		return self.final_accuracy

	def getParameterCount(self):
		return self.parameter_count

def getTopologySequences():
	sequences = [[]] * 3

	shrink_I30O20 = open('data/shrink_I30O20', 'r').read().split('\n')
	shrink_I40O30 = open('data/shrink_I40O30', 'r').read().split('\n')
	for i in range(3):
		sequences[i] = [shrink_I30O20[i]] + sequences[i]
		jobindex = int(re.search('J[123]', shrink_I30O20[i]).group(0)[1])
		sequences[i] = [shrink_I40O30[jobindex - 1]] + sequences[i]
		jobindex = int(re.search('J[123]', shrink_I40O30[jobindex - 1]).group(0)[1])
		sequences[i] = ['I49O40-' + str(jobindex)] + sequences[i]

	return sequences

def getEdges(sequence):
	edges = []
	for i in range(3):
		topology_file = open('data/topology_' + sequence[i], 'r').read().split('\n')
		edge_num = int(topology_file[2])
		edges_layer = [[int(node) for node in edge.split(' ')] for edge in topology_file[3: 3 + edge_num]]
		edges.append(edges_layer)

	edges_layer_last = [[i, j] for i in range(10) for j in range(20)]
	edges.append(edges_layer_last)

	return edges

def doSparse(topology_sequence, output, i):
	topology_sequence = topology_sequences[i]
	edges = getEdges(topology_sequence)
	sparse_nn = SparseNN(49, 10, max_hidden_depth=3, max_hidden_width=[40, 30, 20])
	sparse_nn.define_model(edges)
	sparse_nn.define_data(trainFeatures, labels)
	sparse_nn.train()
	output.write('[Sparse-connected #' + str(i) + ']\n')
	output.write('Files: ' + ' - '.join(topology_sequence) + '\n')
	output.write('Parameter count: ' + str(sparse_nn.getParameterCount()) + '\n')
	output.write('Accuracy: ' + str(sparse_nn.getAccuracy()) + '\n')

if __name__ == '__main__':
	output = open('data/Report.txt', 'w')
	
	## Prepare data
	trainFeaturesFile = open('mnist.csv', 'rb')
	trainFeaturesReader = csv.reader(trainFeaturesFile)
	trainFeatures = [row for row in trainFeaturesReader]
	labelsFile = open('mnist_label.csv', 'rb')
	labelsReader = csv.reader(labelsFile)
	labels = [row for row in labelsReader]

	## Run baseline fully-connected MLP
	sparse_nn = SparseNN(49, 10, max_hidden_depth=3, max_hidden_width=[40, 30, 20])
	sparse_nn.define_model()
	sparse_nn.define_data(trainFeatures, labels)
	sparse_nn.train()
	output.write('[Fully-connected Baseline]\n')
	output.write('Parameter count: ' + str(sparse_nn.getParameterCount()) + '\n')
	output.write('Accuracy: ' + str(sparse_nn.getAccuracy()) + '\n')

	topology_sequences = getTopologySequences()
	for i in range(3):
		doSparse(topology_sequences[i], output, i)
			
