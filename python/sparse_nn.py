import tensorflow as tf

class SparseNN:

	INPUT_WIDTH = 1
	OUTPUT_WIDTH = 1
	MAX_HIDDEN_DEPTH = 1
	MAX_HIDDEN_WIDTH = 1
	
	FULL_CONN_EDGES = []

	input = None
	label = None

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
	
	@staticmethod
	def __weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	@staticmethod
	def __bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def define_model(self, edges=FULL_CONN_EDGES):
		#print edges
		input = tf.placeholder(tf.float32, shape=[self.INPUT_WIDTH, None])

		prev_width = self.INPUT_WIDTH
		prev_tensor = input
		for i in range(self.MAX_HIDDEN_DEPTH + 1):
			if i < self.MAX_HIDDEN_DEPTH:
				cur_width = self.MAX_HIDDEN_WIDTH
			else:
				cur_width = self.OUTPUT_WIDTH

			hidden_weight = tf.sparse_to_dense(
				sparse_indices=edges[i], 
				sparse_values=SparseNN.__weight_variable([len(edges[i])]), 
				output_shape=[cur_width, prev_width])
			hidden_bias = SparseNN.__bias_variable([cur_width])
			hidden = tf.matmul(hidden_weight, prev_tensor) + hidden_bias
				
			prev_width = cur_width
			prev_tensor = hidden
		
		output = prev_tensor
		label = tf.placeholder(tf.float32, shape=[self.OUTPUT_WIDTH, None])

if __name__ == '__main__':
	sparse_nn = SparseNN(768, 10)
	sparse_nn.define_model()
