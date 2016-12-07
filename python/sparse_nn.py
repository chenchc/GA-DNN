import tensorflow as tf
import random

class SparseNN:

	VAL_SPLIT = 0.1
	BATCH_SIZE = 32

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

	def __init__(self, input_width, output_width, max_hidden_depth=1, max_hidden_width=10):
		self.INPUT_WIDTH = input_width
		self.OUTPUT_WIDTH = output_width
		self.MAX_HIDDEN_DEPTH = max_hidden_depth
		self.MAX_HIDDEN_WIDTH = max_hidden_width

		prev_width = self.INPUT_WIDTH
		for i in range(self.MAX_HIDDEN_DEPTH):
			full_conn_layer_edges = []
			for j in range(self.MAX_HIDDEN_WIDTH):
				for k in range(prev_width):
					full_conn_layer_edges.append([j, k])
			self.FULL_CONN_EDGES.append(full_conn_layer_edges)
			prev_width = self.MAX_HIDDEN_WIDTH

		full_conn_layer_edges = []
		for i in range(self.OUTPUT_WIDTH):
			for j in range(self.MAX_HIDDEN_WIDTH):
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
		#print edges
		self.input = tf.placeholder(tf.float32, shape=[None, self.INPUT_WIDTH])
		input_trans = tf.transpose(self.input)

		prev_width = self.INPUT_WIDTH
		prev_tensor = input_trans
		for i in range(self.MAX_HIDDEN_DEPTH + 1):
			if i < self.MAX_HIDDEN_DEPTH:
				cur_width = self.MAX_HIDDEN_WIDTH
			else:
				cur_width = self.OUTPUT_WIDTH
			
			hidden_weight = SparseNN.__weight_variable([cur_width, prev_width], edges[i])
			hidden_bias = SparseNN.__bias_variable([cur_width, 1])
			hidden = tf.sparse_tensor_dense_matmul(hidden_weight, prev_tensor) + hidden_bias
			if i < self.MAX_HIDDEN_DEPTH:
				hidden = tf.nn.relu(hidden)
			#print prev_tensor.get_shape(), hidden_weight.get_shape(), hidden_bias.get_shape(), hidden.get_shape()
				
			prev_width = cur_width
			prev_tensor = hidden
		
		self.output = prev_tensor
		self.label = tf.placeholder(tf.float32, shape=[None, self.OUTPUT_WIDTH])
		label_trans = tf.transpose(self.label)

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, label_trans, dim=0))
		self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

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
		for iter in range(100):
			for i in range(0, len(self.train_input) - self.BATCH_SIZE, self.BATCH_SIZE):
				self.train_step.run(
					feed_dict={self.input: self.train_input[i: i + self.BATCH_SIZE], 
					self.label: self.train_label[i: i + self.BATCH_SIZE]})
			print self.output.eval(feed_dict={self.input: self.val_input[0:4], self.label: self.val_label[0:4]})
			print self.accuracy.eval(feed_dict={self.input: self.val_input, self.label: self.val_label})
			print self.loss.eval(feed_dict={self.input: self.val_input, self.label: self.val_label})
			self.__shuffle_train_data()

if __name__ == '__main__':
	## Solve classic XOR problem
	sparse_nn = SparseNN(2, 2, max_hidden_depth=1, max_hidden_width=8)
	#sparse_nn.define_model(edges=[[[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0], [0, 1]]])
	sparse_nn.define_model()
	sparse_nn.define_data([[0, 0], [0, 1], [1, 0], [1, 1]] * 1000, [[0, 1], [1, 0], [1, 0], [0, 1]] * 1000)
	sparse_nn.train()
