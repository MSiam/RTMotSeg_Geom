"""
This class will take control of the whole process of training or testing Segmentation models
"""

import tensorflow as tf

from models import *
from train import *
from test import *
from utils.misc import timeit

import os
#import pdb
import pickle
from utils.misc import calculate_flops

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Agent:
    """
    Agent will run the program
    Choose the type of operation
    Create a model
    reset the graph and create a session
    Create a trainer or tester
    Then run it and handle it
    """

    def __init__(self, args):
        self.args = args
        self.mode = args.mode

        # Get the class from globals by selecting it by arguments
        self.model = globals()[args.model]
        self.operator = globals()[args.operator]

        self.sess = None

    @timeit
    def build_model(self, x_in=None, flow_in=None):
        if self.mode == 'train' or self.mode == 'overfit':  # validation phase
            with tf.variable_scope('network') as scope:
                self.model = self.model(self.args)
                self.model.build()

            #            print('Building Train Network')
            #            with tf.variable_scope('network') as scope:
            #                self.train_model = self.model(self.args, phase=0)
            #                self.train_model.build()
            #
            #            print('Building Test Network')
            #            with tf.variable_scope('network') as scope:
            #                scope.reuse_variables()
            #                self.test_model = self.model(self.args, phase=1)
            #                self.test_model.build()
        else:  # inference phase
            print('Building Test Network')
            with tf.variable_scope('network') as scope:
                self.train_model = None
                self.model = self.model(self.args)
                self.model.build(x_in, flow_in)
                # calculate_flops()

    @timeit
    def run(self):
        """
        Initiate the Graph, sess, model, operator
        :return:
        """
        print("Agent is running now...\n\n")

        # Reset the graph
        tf.reset_default_graph()

        # Create the sess
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        # Create Model class and build it
        if self.mode == 'test_opt':
            x_in, flow_in = self.optimized_data_loader()

        # Create Model class and build it
        if self.mode != 'test_opt':
            with self.sess.as_default():
                self.build_model()
        else:
            with self.sess.as_default():
                self.build_model(x_in, flow_in)

        # Create the operator
        self.operator = self.operator(self.args, self.sess, self.model, self.model)

        if self.mode == 'train_n_test':
            print("Sorry this mode is not available for NOW")
            exit(-1)
            # self.train()
            # self.test()
        elif self.mode == 'train':
            self.train()
        elif self.mode == 'overfit':
            self.overfit()
        elif self.mode == 'inference':
            self.inference()
        elif self.mode == 'inference_pkl':
            self.load_pretrained_weights(self.sess, 'pretrained_weights/linknet_weights.pkl')
            self.test(pkl=True)
        elif self.mode == 'debug':
            self.debug()
        elif self.mode == 'test':
            self.test()
        elif self.mode == 'test_opt':
            self.inference(True)
        else:
            print("This mode {{{}}}  is not found in our framework".format(self.mode))
            exit(-1)

        self.sess.close()
        print("\nAgent is exited...\n")

    def load_pretrained_weights(self, sess, pretrained_path):
        print('############### START Loading from PKL ##################')
        with open(pretrained_path, 'rb') as ff:
            pretrained_weights = pickle.load(ff, encoding='latin1')

            print("Loading pretrained weights of resnet18")
            # all_vars = tf.trainable_variables()
            # all_vars += tf.get_collection('mu_sigma_bn')
            all_vars = tf.all_variables()
            for v in all_vars:
                if v.op.name in pretrained_weights.keys():
                    if str(v.shape) != str(pretrained_weights[v.op.name].shape):
                        print(v.shape)
                        print(pretrained_weights[v.op.name].shape)
                        print("Oh goooddd!!!")
                        exit(0)
                    assign_op = v.assign(pretrained_weights[v.op.name])
                    sess.run(assign_op)
                    print(v.op.name + " - loaded successfully, size ", pretrained_weights[v.op.name].shape)
            print("All pretrained weights of resnet18 is loaded")

    def train(self):
        try:
            self.operator.train()
            self.operator.finalize()
        except KeyboardInterrupt:
            self.operator.finalize()

    def test(self, pkl=False):
        try:
            self.operator.test(pkl)
        except KeyboardInterrupt:
            pass

    def overfit(self):
        try:
            self.operator.overfit()
            self.operator.finalize()
        except KeyboardInterrupt:
            self.operator.finalize()

    def debug(self):
        self.load_pretrained_weights(self.sess, 'pretrained_weights/linknet_weights.pkl')
        try:
            self.operator.debug_layers()
        except KeyboardInterrupt:
            pass

    def inference(self, opt=False):
        try:
            if opt:
                self.operator.test_optimized()
            else:
                self.operator.test_inference()
        except KeyboardInterrupt:
            pass

    def optimized_data_loader(self):
        import numpy as np
        import scipy.misc
        with tf.device('/cpu:0'):
            self.data_x = np.load(self.args.data_dir + "X_val.npy")
            self.flow_x = np.load(self.args.data_dir + "Flo_val.npy")

            self.data_x_new = np.zeros((self.data_x.shape[0], self.args.img_height, self.args.img_width, 3),
                                       dtype=np.uint8)
            for i in range(self.data_x.shape[0]):
                self.data_x_new[i] = scipy.misc.imresize(self.data_x[i], (self.args.img_height, self.args.img_width))
            self.flow_x_new = np.zeros((self.flow_x.shape[0], self.args.img_height, self.args.img_width, 3),
                                       dtype=np.uint8)
            for i in range(self.flow_x.shape[0]):
                self.flow_x_new[i] = scipy.misc.imresize(self.flow_x[i], (self.args.img_height, self.args.img_width))

            self.data_x = self.data_x_new
            self.flow_x = self.flow_x_new

            print(self.data_x.shape)
            print(self.data_x.dtype)
            print("DATA ITERATOR HERE!!")

            self.features_placeholder = tf.placeholder(tf.float32, self.data_x.shape)
            self.flow_placeholder = tf.placeholder(tf.float32, self.flow_x.shape)

            dataset = tf.contrib.data.Dataset.from_tensor_slices(self.features_placeholder)
            dataset2 = tf.contrib.data.Dataset.from_tensor_slices(self.flow_placeholder)

            dataset = dataset.batch(self.args.batch_size)
            dataset2 = dataset2.batch(self.args.batch_size)

            self.iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                                            dataset.output_shapes)
            self.iterator2 = tf.contrib.data.Iterator.from_structure(dataset2.output_types,
                                                            dataset2.output_shapes)
 
            self.next_batch = self.iterator.get_next()
            self.next_batch_flow = self.iterator2.get_next()

            self.training_init_op = self.iterator.make_initializer(dataset)
            self.training_init_op2 = self.iterator2.make_initializer(dataset2)

            self.sess.run(self.training_init_op, feed_dict={self.features_placeholder: self.data_x})
            self.sess.run(self.training_init_op2, feed_dict={self.flow_placeholder: self.flow_x})

        return self.next_batch, self.next_batch_flow
