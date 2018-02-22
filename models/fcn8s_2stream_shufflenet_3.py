from models.basic.basic_model import BasicModel
from models.encoders.shufflenet import ShuffleNet
from layers.convolution import conv2d_transpose, conv2d, atrous_conv2d
from utils.misc import get_vars_underscope, _debug
import numpy as np
import tensorflow as tf
from utils.img_utils import decode_labels
import pdb

class FCN8s2StreamShuffleNet3(BasicModel):
    """
    FCN8s with ShuffleNet as an encoder Model Architecture
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
        if self.args.data_mode=='experiment':
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

        # Init ShuffleNet as an encoder
        self.app_encoder = ShuffleNet(x_input=self.x_pl, num_classes=self.params.num_classes, prefix='app_',
                                  pretrained_path=self.args.pretrained_path, train_flag=self.is_training,
                                  batchnorm_enabled=self.args.batchnorm_enabled, num_groups=self.args.num_groups,
                                  weight_decay=self.args.weight_decay, bias=self.args.bias, mean_path= self.args.data_dir+'mean.npy')

        self.motion_encoder = ShuffleNet(x_input=self.flo_pl, num_classes=self.params.num_classes, prefix='mot_',
                                  pretrained_path=self.args.pretrained_path, train_flag=self.is_training,
                                  batchnorm_enabled=self.args.batchnorm_enabled, num_groups=self.args.num_groups,
                                  weight_decay=self.args.weight_decay, bias=self.args.bias, mean_path= self.args.data_dir+'flo_mean.npy')


        # Build Encoding part
        self.app_encoder.build()
        self.motion_encoder.build()
        self.combined_score= tf.multiply(self.app_encoder.stage2, self.motion_encoder.stage2)
        _debug(self.combined_score)

        self.stage3 = self.app_encoder.stage(self.combined_score, stage=3, repeat=7)
        _debug(self.stage3)

        self.stage4 = self.app_encoder.stage(self.stage3, stage=4, repeat=3)
        _debug(self.stage4)

        self.feed1= self.stage3
        self.feed2= self.combined_score
        # First Experiment is to use the regular conv2d
        self.score_fr = conv2d('combined_conv_1c_1x1', self.stage4, num_filters=self.params.num_classes, l2_strength=self.args.weight_decay,
                               kernel_size=(1, 1))
        _debug(self.score_fr)

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
        #import pdb; pdb.set_trace()
