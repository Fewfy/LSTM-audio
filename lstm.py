# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import ipdb
from tensorflow.python.ops import rnn_cell


class Model():
    
    def __init__(self, args, infer=False):
        if infer:
            args.batch_size = 1
            args.seq_length = 1
            
        self.args = args
        cell = rnn_cell.BasicLSTMCell(args.run_size, state_is_tuple=False)
        
        cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)
        
        self.cell = cell
        
        self.input_data = tf.placeholder(tf.float32, [None, args.seq_length, 2])
        self.target_data = tf.placeholder(tf.float32, [None, args.seq_length, 2])
        
        # learning rate
        self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate')
        self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)
        
        output_size = 5 # 2 mu, 2 sigma and 1 corr
        
        # Embedding for the spatial coordinates
        with tf.variable_scope('coordinate_embedding'):
            embedding_w = tf.get_variable('embedding_w', [2, args.embdding_size])
            embedding_b = tf.get_variable('embedding_b', [args.embedding_size])
            
        with tf.variable_scope('rnnlm'):
            output_w = tf.get_variable('output_w', [args.rnn_size, output_size], initializer=tf.truncated_normal_initializer(stddev=0.01), trainable=True)
            output_b = tf.get_variable('output_b', [output_size], initializer=tf.constant_initializer(0.01), trainable=True)
            
        # Split inputs according to sequences
        inputs = tf.split(1, args.seq_length, self.input_data)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        
        # Embed the input spatial points into the embedding space
        embedded_inputs = []
        for x in inputs:
            embedded_x = tf.nn.relu(tf.add(x, embedding_w), embedding_b)
            embedded_inputs.append(embedded_x)
            
        outputs, last_state = tf.nn.seq2seq.rnn_decoder(embedded_inputs, self.initial_state, cell, loop_function=None, scope='rnnlm')
        # Concatening the ouptus from the RNN decoder and reshape it to ?xargs.rnn_size
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        
        # Apply the output linear layer
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        # Store the final LSTM cell state after the input data has been feeded
        self.final_state = last_state
        
        # reshape target data so that it aligns with predictions
        flat_target_data = tf.reshape(self.target_data, [-1, 2])
        # Extracting the x-coordinates and y-coordinates from the target data
        [x_data, y_data] = tf.split(1, 2, flat_target_data)
        
        def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
            normx = tf.subtract(x, mux)
            normy = tf.subtract(y, muy)
            # Calculate sx*sy
            sxsy = tf.multiply(sx, sy)
            
            # Calculate the expotienal factor
            z = tf.square(tf.divide(normx, sx)) + tf.square(tf.divide(normy, sy)) - 2*tf.divide(tf.multiply(normx, normy), sxsy)
            negRho = 1 - tf.square(rho)
            result = tf.exp(tf.divide(-z, 2*negRho))
            denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
            result = tf.divide(result, denom)
            self.result = result
            return result
        
        def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
            result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
            epsilon = 1e-20
            result = -tf.log(tf.maximum(result0, epsilon))
            return tf.reduce_mean(result)
        
        def get_coef(output):
            z = output
            z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(1, 5, z)
            
            z_sx = tf.exp(z_sx)
            z_sy = tf.exp(z_sy)
            z_corr = tf.tanh(z_corr)
            
            return [z_mux, z_muy, z_sx, z_sy, z_corr]
        
        [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(output)
        self.output = output
        self.mux = o_mux
        self.muy = o_muy
        self.sx = o_sx
        self.sy = o_sy
        self.corr = o_corr
        
        lossfunc = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)
        self.cost = tf.divide(lossfunc, (args.batch_size * args.seq_length))
        tvars = tf.trainable_variables()
        
        l2 = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars)
        self.cost = self.cost + l2
        
        self.gradients = tf.gradients(self.cost, tvars)
        grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)
        optimizer = tf.train.RMSPropOptimizer(self.lr)
        
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    def sample(self, sess, traj, true_traj, num=10):
        def sample_gaussian_2d(mux, muy, sx, sy, rho):
            mean = [mux, muy]
            cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]
        
        state = sess.run(self.cell.zero_state(1, tf.float32))
        
        for pos in traj[:-1]:
            data = np.zeros((1, 1, 2), dtype=np.float32)
            data[0, 0, 0] = pos[0]
            data[0, 0, 1] = pos[1]
            
            feed = {self.input_data: data, self.initial_state: state}
            
            [state] = sess.run([self.final_state], feed)
        
        ret = traj
        
        last_pos =  traj[-1]
        
        prev_data = np.zeros((1, 1, 2), dtype=np.float32)
        prev_data[0, 0, 0] = last_pos[0]
        prev_data[0, 0, 1] = last_pos[1]
        
        prev_target_data = np.reshape(true_traj[traj.shape[0]], (1, 1, 2))
        
        for t in range(num):
            feed = {self.input_data: prev_data, self.initial_state: state, self.target_data: prev_target_data}
            
            [o_mux, o_muy, o_sx, o_sy, o_corr, state, cost] = sess.run([self.mux, self.muy, self.sx, self.sy, self.corr, self.final_state, self.cost], feed)
            next_x, next_y = sample_gaussian_2d(o_mux[0][0], o_muy[0][0], o_sx[0][0], o_sy[0][0], o_corr[0][0])
            ret = np.vstack((ret, [next_x, next_y]))
            
            prev_data[0, 0, 0] = next_x
            prev_data[0, 0, 1] = next_y
            
        return ret
        
    