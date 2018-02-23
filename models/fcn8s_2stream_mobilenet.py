from models.basic.basic_model import BasicModel
from models.encoders.mobilenet import MobileNet
from layers.convolution import conv2d_transpose, conv2d, atrous_conv2d, depthwise_separable_conv2d
from utils.img_utils import decode_labels
from utils.misc import _debug

import tensorflow as tf
import numpy as np

class FCN8s2StreamMobileNet(BasicModel):
    """
    FCN8s with MobileNet as an encoder Model Architecture
    """

    def __init__(self, args):
        super().__init__(args)
        # init encoder
        self.encoder = None
        # init network layers

    def build(self):
        print("\nBuilding the MODEL...")
        self.init_input()
        self.init_network()
        self.init_output()
        self.init_train()
        self.init_summaries()
        print("The Model is built successfully\n")

    def init_input(self):
        with tf.name_scope('input'):
            self.x_pl = tf.placeholder(tf.float32,
                                       [self.args.batch_size, self.params.img_height, self.params.img_width, 3])
            self.flo_pl = tf.placeholder(tf.float32,
                                       [self.args.batch_size, self.params.img_height, self.params.img_width, 3])
            self.y_pl = tf.placeholder(tf.int32, [self.args.batch_size, self.params.img_height, self.params.img_width])

            if self.params.weighted_loss:
                self.wghts = np.zeros((self.args.batch_size, self.params.img_height, self.params.img_width), dtype= np.float32)
            self.is_training = tf.placeholder(tf.bool)

    def init_summaries(self):
        with tf.name_scope('pixel_wise_accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pl, self.out_argmax), tf.float32))

        with tf.name_scope('segmented_output'):
            input_summary = tf.cast(self.x_pl, tf.uint8)
            flow_summary = tf.cast(self.flo_pl, tf.uint8)
            # labels_summary = tf.py_func(decode_labels, [self.y_pl, self.params.num_classes], tf.uint8)
            preds_summary = tf.py_func(decode_labels, [self.out_argmax, self.params.num_classes], tf.uint8)
            self.segmented_summary = tf.concat(axis=2, values=[input_summary,flow_summary,
                                                               preds_summary])  # Concatenate row-wise

        # Every step evaluate these summaries
        if self.loss is not None:
            with tf.name_scope('train-summary'):
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('pixel_wise_accuracy', self.accuracy)

        self.merged_summaries = tf.summary.merge_all()

        # Save the best iou on validation
        self.best_iou_tensor = tf.Variable(0.0, trainable=False, name='best_iou')
        self.best_iou_input = tf.placeholder('float32', None, name='best_iou_input')
        self.best_iou_assign_op = self.best_iou_tensor.assign(self.best_iou_input)

    def init_network(self):
        """
        Building the Network here
        :return:
        """

        # Init MobileNet as an encoder
        self.app_encoder = MobileNet(x_input=self.x_pl, num_classes=self.params.num_classes, prefix='app_',
                                 pretrained_path=self.args.pretrained_path, mean_path= self.args.data_dir+'mean.npy',
                                 train_flag=self.is_training, width_multipler=1.0, weight_decay=self.args.weight_decay)
        self.motion_encoder = MobileNet(x_input=self.flo_pl, num_classes=self.params.num_classes, prefix='mot_',
                                 pretrained_path=self.args.pretrained_path, mean_path= self.args.data_dir+'flo_mean.npy',
                                 train_flag=self.is_training, width_multipler=1.0, weight_decay=self.args.weight_decay)

        # Build Encoding part
        self.app_encoder.build()
        self.motion_encoder.build()
        self.feed2= tf.multiply(self.app_encoder.conv3_2, self.motion_encoder.conv3_2)
        self.width_multiplier= 1.0
        self.conv4_1 = depthwise_separable_conv2d('conv_ds_6_1', self.feed2, width_multiplier=self.width_multiplier,
                                                  num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                  stride=(1, 1), activation=tf.nn.relu6,
                                                  batchnorm_enabled=True, is_training=self.is_training,
                                                  l2_strength=self.args.weight_decay)
        _debug(self.conv4_1)
        self.conv4_2 = depthwise_separable_conv2d('conv_ds_7_1', self.conv4_1, width_multiplier=self.width_multiplier,
                                                  num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                  stride=(2, 2), activation=tf.nn.relu6,
                                                  batchnorm_enabled=True, is_training=self.is_training,
                                                  l2_strength=self.args.weight_decay)
        _debug(self.conv4_2)
        self.conv5_1 = depthwise_separable_conv2d('conv_ds_8_1', self.conv4_2, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu6,
                                                      batchnorm_enabled=True, is_training=self.is_training,
                                                      l2_strength=self.args.weight_decay)
        _debug(self.conv5_1)
        self.conv5_2 = depthwise_separable_conv2d('conv_ds_9_1', self.conv5_1, width_multiplier=self.width_multiplier,
                                                  num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                  stride=(1, 1), activation=tf.nn.relu6,
                                                  batchnorm_enabled=True, is_training=self.is_training,
                                                  l2_strength=self.args.weight_decay)
        _debug(self.conv5_2)
        self.conv5_3 = depthwise_separable_conv2d('conv_ds_10_1', self.conv5_2,
                                                  width_multiplier=self.width_multiplier,
                                                  num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                  stride=(1, 1), activation=tf.nn.relu6,
                                                  batchnorm_enabled=True, is_training=self.is_training,
                                                  l2_strength=self.args.weight_decay)
        _debug(self.conv5_3)
        self.conv5_4 = depthwise_separable_conv2d('conv_ds_11_1', self.conv5_3,
                                                  width_multiplier=self.width_multiplier,
                                                  num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                  stride=(1, 1), activation=tf.nn.relu6,
                                                  batchnorm_enabled=True, is_training=self.is_training,
                                                  l2_strength=self.args.weight_decay)
        _debug(self.conv5_4)
        self.conv5_5 = depthwise_separable_conv2d('conv_ds_12_1', self.conv5_4,
                                                  width_multiplier=self.width_multiplier,
                                                  num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                  stride=(1, 1), activation=tf.nn.relu6,
                                                  batchnorm_enabled=True, is_training=self.is_training,
                                                  l2_strength=self.args.weight_decay)
        _debug(self.conv5_5)
        self.conv5_6 = depthwise_separable_conv2d('conv_ds_13_1', self.conv5_5,
                                                  width_multiplier=self.width_multiplier,
                                                  num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                  stride=(2, 2), activation=tf.nn.relu6,
                                                  batchnorm_enabled=True, is_training=self.is_training,
                                                  l2_strength=self.args.weight_decay)
        _debug(self.conv5_6)
        self.conv6_1 = depthwise_separable_conv2d('conv_ds_14_1', self.conv5_6,
                                                  width_multiplier=self.width_multiplier,
                                                  num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                  stride=(1, 1), activation=tf.nn.relu6,
                                                  batchnorm_enabled=True, is_training=self.is_training,
                                                  l2_strength=self.args.weight_decay)
        _debug(self.conv6_1)
        # Pooling is removed.
        self.score_fr = conv2d('conv_1c_1x1_1', self.conv6_1, num_filters=self.params.num_classes, l2_strength=self.args.weight_decay,
                               kernel_size=(1, 1))

        self.feed1= self.conv4_2

        # Build Decoding part
        with tf.name_scope('upscore_2s'):
            self.upscore2 = conv2d_transpose('upscore2', x=self.score_fr,
                                             output_shape=self.feed1.shape.as_list()[0:3] + [
                                                 self.params.num_classes], batchnorm_enabled=self.args.batchnorm_enabled, is_training= self.is_training,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.args.weight_decay, bias=self.args.bias)
            _debug(self.upscore2)

            self.score_feed1 = conv2d('score_feed1', x=self.feed1, batchnorm_enabled=self.args.batchnorm_enabled, is_training= self.is_training,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1), bias= self.args.bias,
                                      l2_strength=self.args.weight_decay)
            _debug(self.score_feed1)
            self.fuse_feed1 = tf.add(self.score_feed1, self.upscore2)

        with tf.name_scope('upscore_4s'):
            self.upscore4 = conv2d_transpose('upscore4', x=self.fuse_feed1,
                                             output_shape=self.feed2.shape.as_list()[0:3] + [
                                                 self.params.num_classes], batchnorm_enabled=self.args.batchnorm_enabled, is_training= self.is_training,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.args.weight_decay, bias=self.args.bias)
            _debug(self.upscore4)
            self.score_feed2 = conv2d('score_feed2', x=self.feed2, batchnorm_enabled=self.args.batchnorm_enabled, is_training= self.is_training,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1), bias=self.args.bias,
                                      l2_strength=self.args.weight_decay)
            _debug(self.score_feed2)
            self.fuse_feed2 = tf.add(self.score_feed2, self.upscore4)

        with tf.name_scope('upscore_8s'):
            self.upscore8 = conv2d_transpose('upscore8', x=self.fuse_feed2,
                                             output_shape=self.x_pl.shape.as_list()[0:3] + [self.params.num_classes], is_training= self.is_training,
                                             kernel_size=(16, 16), stride=(8, 8), l2_strength=self.args.weight_decay, bias=self.args.bias)
            _debug(self.upscore8)
        self.logits = self.upscore8

