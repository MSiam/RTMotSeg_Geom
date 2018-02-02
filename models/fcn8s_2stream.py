from models.basic.basic_model import BasicModel
from models.encoders.VGG import VGG16
from models.encoders.mobilenet import MobileNet
from layers.convolution import conv2d_transpose, conv2d
import numpy as np
import tensorflow as tf
from utils.misc import _debug
from utils.img_utils import decode_labels

class FCN8s2Stream(BasicModel):
    """
    FCN8s Model Architecture
    """

    def __init__(self, args):
        super().__init__(args)
        # init encoder
        self.encoder = None
        # init network layers
        self.upscore2 = None
        self.score_feed1 = None
        self.fuse_feed1 = None
        self.upscore4 = None
        self.score_feed2 = None
        self.fuse_feed2 = None
        self.upscore8 = None

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

        # Init a VGG16 as an encoder
        self.app_encoder = VGG16(x_input=self.x_pl, prefix= 'app_',
                             num_classes=self.params.num_classes,
                             pretrained_path=self.args.pretrained_path,
                             train_flag=self.is_training,
                             reduced_flag=False,
                             weight_decay=self.args.weight_decay, mean_path= self.args.data_dir+'mean.npy')
        self.motion_encoder = VGG16(x_input=self.flo_pl, prefix= 'mot_',
                             num_classes=self.params.num_classes,
                             pretrained_path=self.args.pretrained_path,
                             train_flag=self.is_training,
                             reduced_flag=False,
                             weight_decay=self.args.weight_decay, mean_path= self.args.data_dir+'flo_mean.npy')

        # Build Encoding part
        self.app_encoder.build()
        self.motion_encoder.build()
        self.combined_score= tf.multiply(self.app_encoder.score_fr, self.motion_encoder.score_fr)
        # Build Decoding part
        with tf.name_scope('upscore_2s'):
            self.upscore2 = conv2d_transpose('upscore2', x=self.combined_score,
                                             output_shape=self.app_encoder.feed1.shape.as_list()[0:3] + [
                                                 self.params.num_classes], batchnorm_enabled=self.args.batchnorm_enabled,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.app_encoder.wd, bias=self.args.bias)

            self.app_score_feed1 = conv2d('app_score_feed1', x=self.app_encoder.feed1, batchnorm_enabled=self.args.batchnorm_enabled,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1), bias= self.args.bias,
                                      l2_strength=self.app_encoder.wd)
            self.app_score_feed1 = tf.nn.relu(self.app_score_feed1)
            self.mot_score_feed1 = conv2d('mot_score_feed1', x=self.motion_encoder.feed1, batchnorm_enabled=self.args.batchnorm_enabled,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1), bias= self.args.bias,
                                      l2_strength=self.motion_encoder.wd)
            self.mot_score_feed1 = tf.nn.relu(self.mot_score_feed1)
            self.score_feed1= tf.multiply(self.app_score_feed1, self.mot_score_feed1)
            self.fuse_feed1 = tf.add(self.score_feed1, self.upscore2)

        with tf.name_scope('upscore_4s'):
            self.upscore4 = conv2d_transpose('upscore4', x=self.fuse_feed1,
                                             output_shape=self.app_encoder.feed2.shape.as_list()[0:3] + [
                                                 self.params.num_classes], batchnorm_enabled=self.args.batchnorm_enabled,
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.app_encoder.wd, bias=self.args.bias)

            self.app_score_feed2 = conv2d('app_score_feed2', x=self.app_encoder.feed2, batchnorm_enabled=self.args.batchnorm_enabled,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1), bias=self.args.bias,
                                      l2_strength=self.app_encoder.wd)
            self.app_score_feed2 = tf.nn.relu(self.app_score_feed2)
            self.mot_score_feed2 = conv2d('mot_score_feed2', x=self.motion_encoder.feed2, batchnorm_enabled=self.args.batchnorm_enabled,
                                      num_filters=self.params.num_classes, kernel_size=(1, 1), bias=self.args.bias,
                                      l2_strength=self.motion_encoder.wd)
            self.mot_score_feed2 = tf.nn.relu(self.mot_score_feed2)
            self.score_feed2= tf.multiply(self.app_score_feed2, self.mot_score_feed2)

            self.fuse_feed2 = tf.add(self.score_feed2, self.upscore4)

        with tf.name_scope('upscore_8s'):
            self.upscore8 = conv2d_transpose('upscore8', x=self.fuse_feed2,
                                             output_shape=self.x_pl.shape.as_list()[0:3] + [self.params.num_classes],
                                             kernel_size=(16, 16), stride=(8, 8), l2_strength=self.app_encoder.wd, bias=self.args.bias)

        self.logits = self.upscore8
