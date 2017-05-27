import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, input_size, hidden_size, session):
        '''Initializes a new instance of the VariationalAutoencoder class.'''
        self.session = session
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights = {}
        self.transfer_fn = tf.nn.softplus

        self.__initialize()
        self.__build()

    def __initialize(self):
        '''Initializes the weights needed to build the computational graph.'''
        weights = {}

        weights['weights_1'] = tf.get_variable('weights_1',[self.input_size, self.hidden_size])
        weights['bias_1'] = tf.Variable(tf.zeros([self.hidden_size], dtype=tf.float32))
        weights['weights_2'] = tf.Variable(tf.zeros([self.hidden_size, self.input_size], dtype=tf.float32))
        weights['bias_2'] = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32))

        self.weights = weights

    def get_optimizer(self):
        '''Returns the optimizer for this instance.'''
        return self.optimizer

    def get_loss(self):
        '''Returns the loss function for this instance.'''
        return self.loss_fn

    def get_weights_and_biases(self):
        '''Returns the weights and biases of this instance.'''
        return self.weights

    def get_internal_representation(self):
        '''Returns the internal, embedded representation variables.'''
        return self.hidden

    def batch_fit(self, input):
        '''Fits the model to the given batch.'''
        loss, _ = self.session.run((self.loss_fn, self.optimizer),
                                   feed_dict={self.input: input})
        return loss

    def transform(self, input):
        return self.session.run(self.hidden, feed_dict={self.input: input})

    def __build(self):
        '''Builds the computational graph.'''
        self.input = tf.placeholder(tf.float32, [None, self.input_size])

        hidden_1_result = tf.matmul(self.input, self.weights['weights_1'])
        self.hidden = self.transfer_fn(tf.add(hidden_1_result,
                                              self.weights['bias_1']))

        reconstruction_result = tf.matmul(self.hidden, self.weights['weights_2'])
        self.reconstruction = tf.add(reconstruction_result, self.weights['bias_2'])

        diff = tf.subtract(self.reconstruction, self.input)
        self.loss_fn = 0.5 * tf.reduce_sum(tf.pow(diff, 2.0))
        self.optimizer_fn = tf.train.AdamOptimizer(learning_rate=0.001)
        self.optimizer = self.optimizer_fn.minimize(self.loss_fn)

class VariationalAutoencoder(object):
    def __init__(self, input_size, hidden_size, session):
        '''Initializes a new instance of the VariationalAutoencoder class.'''
        self.session = session
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights = {}

        self.__initialize()
        self.__build()

    def __initialize(self):
        '''Initializes the weights needed to build the computational graph.'''
        weights = {}

        weights['weights_1'] = tf.get_variable('weights_1',[self.input_size, self.hidden_size])
        weights['log_sigma_weights_1'] = tf.get_variable('log_sigma_weights_1', [self.input_size, self.hidden_size])
        weights['bias_1'] = tf.Variable(tf.zeros([self.hidden_size], dtype=tf.float32))
        weights['log_sigma_bias_1'] = tf.Variable(tf.zeros([self.hidden_size], dtype=tf.float32))
        weights['weights_2'] = tf.Variable(tf.zeros([self.hidden_size, self.input_size], dtype=tf.float32))
        weights['bias_2'] = tf.Variable(tf.zeros([self.input_size], dtype=tf.float32))

        self.weights = weights

    def get_optimizer(self):
        '''Returns the optimizer for this instance.'''
        return self.optimizer

    def get_loss(self):
        '''Returns the loss function for this instance.'''
        return self.loss_fn

    def get_weights(self):
        '''Returns the weights of this instance.'''
        return self.weights

    def get_internal_representation(self):
        '''Returns the internal, embedded representation variables.'''
        return self.z

    def batch_fit(self, input):
        '''Fits the model to the given batch.'''
        loss, _ = self.session.run((self.loss_fn, self.optimizer),
                                   feed_dict={self.input: input})
        return loss

    def transform(self, input):
        return self.session.run(self.z_mean, feed_dict={self.input: input})

    def partial_fit(self, X):
        loss, opt = self.session.run((self.loss_fn, self.optimizer), feed_dict={self.input: X})
        return loss

    def transform(self, X):
        return self.session.run(self.z_mean, feed_dict={self.input: X})

    def __build(self):
        '''Builds the computational graph.'''
        self.input = tf.placeholder(tf.float32, [None, self.input_size])

        hidden_1_result = tf.matmul(self.input, self.weights['weights_1'])
        self.z_mean = tf.add(hidden_1_result, self.weights['log_sigma_bias_1'])

        log_sigma_result = tf.matmul(self.input, self.weights['log_sigma_weights_1'])
        self.z_log_sigma_sq = tf.add(log_sigma_result, self.weights['log_sigma_bias_1'])

        eps = tf.random_normal(tf.stack([tf.shape(self.input)[0], self.hidden_size]), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        y_result = tf.matmul(self.z, self.weights['weights_2'])
        self.y = tf.add(y_result, self.weights['bias_2'])

        reconstruction_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.y, self.input), 2.0))
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq \
                                           - tf.square(self.z_mean) \
                                           - tf.exp(self.z_log_sigma_sq), 1)

        self.loss_fn = tf.reduce_mean(reconstruction_loss + latent_loss)
        self.optimizer_fn = tf.train.AdamOptimizer(learning_rate=0.001)
        self.optimizer = self.optimizer_fn.minimize(self.loss_fn)
