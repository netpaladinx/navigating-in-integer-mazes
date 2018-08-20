from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

import task

default_hparams = tf.contrib.training.HParams(
    n_nodes=10000,  # 0 ~ 9999
    n_dims=50,
    n_type_rels=10,
    learning_rate=0.0001,
    max_itrs=2000,
    batch_size=20
)

EPSILON = 1e-8


def l1_nor(v, axis, epsilon=EPSILON):
    rd_max = tf.reduce_max(v, axis=axis, keepdims=True)
    fl = tf.minimum(0., rd_max - epsilon)
    v = v - fl
    rd_sum = tf.reduce_sum(v, axis=axis, keepdims=True)
    return v / rd_sum

class Model(object):
    def __init__(self, maze, hparams=default_hparams):
        self.hparams = hparams
        self.maze = maze

        self.tf_graph = tf.Graph()
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True

        with self.tf_graph.as_default():
            self._build_model()

    def _build_model(self):
        hp = self.hparams

        self.input_pl = tf.placeholder(tf.int32, [None, 2], name='input')  # bs x 2
        self.src, self.dst = tf.split(self.input_pl, 2, axis=1)  # bs x 1, bs x 1
        self.batch_size = tf.shape(self.input_pl)[0]

        self.src = tf.squeeze(self.src, axis=1)  # bs
        self.dst = tf.squeeze(self.dst, axis=1)  # bs

        self.node_emb = None
        self.weight_inp_src = tf.get_variable('weight_inp_src', shape=[hp.n_dims, hp.n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        self.weight_inp_dst = tf.get_variable('weight_inp_dst', shape=[hp.n_dims, hp.n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        self.bias_inp = tf.get_variable('bias_inp', shape=[hp.n_dims], initializer=tf.zeros_initializer())  # n_dims
        self.weight_sta = tf.get_variable('weight_sta', shape=[hp.n_dims, hp.n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        self.weight_emb = tf.get_variable('weight_emb', shape=[hp.n_dims, hp.n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        self.bias_hid = tf.get_variable('bias_hid', shape=[hp.n_dims], initializer=tf.zeros_initializer())  # n_dims
        self.weight_rel = tf.get_variable('weight_rel', shape=[hp.n_type_rels, hp.n_dims], initializer=tf.variance_scaling_initializer())  # n_type_rels x n_dims
        self.bias_rel = tf.get_variable('bias_rel', shape=[hp.n_type_rels], initializer=tf.zeros_initializer())  # n_type_rels

        self.src_emb = self._get_embs(self.src)  # bs x n_dims
        self.dst_emb = self._get_embs(self.dst)  # bs x n_dims
        self.context = tf.tanh(tf.matmul(self.src_emb, self.weight_inp_src) + tf.matmul(self.dst_emb, self.weight_inp_dst) + self.bias_inp)  # bs x n_dims

        self.focus_init = tf.one_hot(self.src, hp.n_nodes)  # bs x n_nodes
        self.state_init = tf.expand_dims(self.focus_init, axis=-1) * tf.expand_dims(self.context, axis=1)  # bs x n_nodes x n_dims

        self.focus_1, self.state_1 = self._jump(self.focus_init, self.state_init)
        self.focus_2, self.state_2 = self._jump(self.focus_1, self.state_1)
        self.focus_3, self.state_3 = self._jump(self.focus_2, self.state_2)
        self.focus_4, self.state_4 = self._jump(self.focus_3, self.state_3)  # bs x n_nodes, bs x n_nodes x n_dims
        
        dst_idx_flattened = tf.range(0, self.batch_size) * hp.n_nodes + self.dst
        self.prediction_prob = tf.gather(tf.reshape(self.focus_4, [-1]), dst_idx_flattened)  # bs

        self.loss = tf.reduce_mean(-tf.log(self.prediction_prob + 1./hp.n_nodes*EPSILON))
        self.hit_top1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.focus_4, axis=1, output_type=tf.int32), self.dst), tf.float32))
        self.hit_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.focus_4, self.dst, 5), tf.float32))
        self.hit_top10 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.focus_4, self.dst, 10), tf.float32))
        self.hit_top20 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.focus_4, self.dst, 20), tf.float32))

        self.global_step = tf.train.create_global_step()
        self.train_op = tf.train.AdamOptimizer(learning_rate=hp.learning_rate).minimize(loss=self.loss, global_step=self.global_step)
        
        self.init_op = tf.global_variables_initializer()

    def _get_embs(self, inp):
        hp = self.hparams
        with tf.device('/cpu:0'):
            if self.node_emb is None:
                self.node_emb = tf.get_variable('node_emb', shape=[hp.n_nodes, hp.n_dims], initializer=tf.truncated_normal_initializer(stddev=0.01))  # n_nodes x n_dims
            return tf.nn.embedding_lookup(self.node_emb, inp)

    def _jump(self, focus, state):
        '''
        focus: bs x n_nodes (from)
        state: bs x n_nodes (from) x n_dims
        '''
        hp = self.hparams

        hidden = tf.tanh(tf.tensordot(state, self.weight_sta, [[2], [0]]) + tf.tensordot(self.node_emb, self.weight_emb, [[1], [0]]) + self.bias_hid) # bs x n_nodes x n_dims
        hidden_emb = tf.reshape(tf.transpose(hidden, perm=[1, 0, 2]), [hp.n_nodes, -1])  # n_nodes x (bs*n_dims)
        hidden_from = tf.gather(hidden_emb, self.maze.edges_from()) # n_edges x (bs*n_dims)
        hidden_to = tf.gather(hidden_emb, self.maze.edges_to())  # n_edges x (bs*n_dims)
        
        n2n_dims = tf.reshape(hidden_from * hidden_to, [self.maze.num_edges(), -1, hp.n_dims])  # n_edges x bs x n_dims
        n2n_rels = tf.nn.softplus(tf.tensordot(n2n_dims, self.weight_rel, [[2], [1]]) + self.bias_rel) * tf.expand_dims(tf.cast(self.maze.edges_rels(), tf.float32), axis=1)  # n_edges x bs x n_type_rels
        trans = tf.reduce_sum(n2n_rels, axis=2)  # n_edges x bs

        bs = tf.shape(trans)[1]
        trans_rd_sum = tf.segment_sum(trans, self.maze.edges_from()) # n_nodes (from) x bs
        trans_rd_sum = tf.gather(trans_rd_sum, self.maze.edges_from())  # n_edges x bs
        trans_nor = trans / trans_rd_sum  # n_edges x bs
        
        msg_sent = hidden * tf.expand_dims(focus, axis=2)  # bs x n_nodes (from) x n_dims
        msg_emb = tf.reshape(tf.transpose(msg_sent, perm=[1, 0, 2]), [hp.n_nodes, -1])  # n_nodes (from) x (bs*n_dims)
        msg_from = tf.reshape(tf.gather(msg_emb, self.maze.edges_from()), [self.maze.num_edges(), -1, hp.n_dims])  # n_edges x bs x n_dims
        
        msg_to = msg_from * tf.expand_dims(trans_nor, axis=2)  # n_edges x bs x n_dims
        msg_recv = tf.transpose(tf.unsorted_segment_sum(msg_to, self.maze.edges_to(), hp.n_nodes), perm=[1, 0, 2])  # bs x n_nodes (to) x n_dims

        focus_from = tf.gather(tf.transpose(focus), self.maze.edges_from())  # n_edges x bs
        focus_to = focus_from * trans_nor  # n_edges x bs
        focus = tf.transpose(tf.unsorted_segment_sum(focus_to, self.maze.edges_to(), hp.n_nodes))  # bs x n_nodes (to)
        state = state + msg_recv  # bs x n_nodes (to) x n_dims

        return focus, state

    def train(self, FLAGS):
        hp = self.hparams
        
        with tf.Session(graph=self.tf_graph, config=self.tf_config) as sess:
            sess.run(self.init_op)

            n_itrs = 0
            n_examples = 0
            while n_itrs < hp.max_itrs:
                samples = self.maze.sampler(hp.batch_size)
                _, loss, hit_top1, hit_top5, hit_top10, hit_top20 = \
                    sess.run([self.train_op, self.loss, self.hit_top1, self.hit_top5, self.hit_top10, self.hit_top20], 
                             feed_dict={self.input_pl: samples})
                
                n_itrs += 1
                n_examples += hp.batch_size
                if (n_itrs + 1) % FLAGS.print_freq == 0:
                    print('n_itrs: %d | n_examples: %d | loss: %.8f | hit_top1: %.8f | hit_top5: %.8f | hit_top10: %.8f | hit_top20: %.8f' % 
                          (n_itrs, n_examples, loss, hit_top1, hit_top5, hit_top10, hit_top20))

if __name__ == '__main__':
    tf.flags.DEFINE_integer("print_freq", 1, "Frequency of printing")
    FLAGS = tf.flags.FLAGS

    maze = task.IntegerMaze()
    model = Model(maze)
    model.train(FLAGS)