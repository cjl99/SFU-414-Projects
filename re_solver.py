import tensorflow as tf
if float(tf.__version__[0:3])>=2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import data
import os
import re_cnn as cnn
import time
from datetime import datetime

class Solver(object):
    def __init__(self, train=True, common_params=None, solver_params=None, net_params=None, dataset_params=None):

        self.device_id = int(common_params['gpus']) # 0
        self.image_size = int(common_params['image_size']) #256
        self.batch_size = int(common_params['batch_size'])
        self.num_gpus = 1

        self.learning_rate = float(solver_params['learning_rate'])
        self.moment = float(solver_params['moment']) #?
        self.max_steps = int(solver_params['max_iterators'])
        self.train_dir = str(solver_params['train_dir'])
        # don't know
        self.lr_decay = float(solver_params['lr_decay'])
        self.decay_steps = int(solver_params['decay_steps'])

        self.train = train # ?
        self.cnn = cnn.Model(train=train, common_params=common_params, net_params=net_params)
        self.dataset = data.Dataset(common_params=common_params, dataset_params=dataset_params)

    def train_model(self):
    # with tf.device('/gpu:' + str(self.device_id)):
        #
        self.global_step = tf.get_variable(name = 'global_step',
                                           shape = [],
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)
        self.decay_lr = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                   global_step=self.global_step,
                                                   decay_steps=self.decay_steps,
                                                   decay_rate=self.lr_decay,
                                                   staircase=True) # ? don't know here
        self.opt = tf.train.AdamOptimizer(learning_rate=self.decay_lr, beta2=0.99)

        with tf.name_scope('graph') as scope:
            new_loss, self.total_loss = self.construct_graph(scope)
            # self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            grads = self.opt.compute_gradients(new_loss)

            # self.summaries.append(tf.summary.scalar('learning_rate', self.decay_lr))

            # for grad, var in grads:
            #     if grad is not None:
            #         self.summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            apply_gradient_op = self.opt.apply_gradients(grads, global_step=self.global_step)

            # for var in tf.trainable_variables():
            #     self.summaries.append(tf.summary.histogram(var.op.name, var))

            variable_averages = tf.train.ExponentialMovingAverage(0.999, self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            train_op = tf.group(apply_gradient_op, variables_averages_op)
            saver = tf.train.Saver(write_version=1)
            saver1 = tf.train.Saver()
            # summary_op = tf.summary.merge(self.summaries)
            init =  tf.global_variables_initializer()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            sess.run(init)
            # saver1.restore(sess, './models/model.ckpt')
            # nilboy
            # summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            for step in range(self.max_steps):
                # start_time = time.time()
                # t1 = time.time()
                data_l, gt_ab_313 = self.dataset.generate_batches()
                # t2 = time.time()
                _, loss_value = sess.run([train_op, self.total_loss], feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313})
                                                                                 # self.prior_boost_nongray:prior_boost_nongray})
                # duration = time.time() - start_time
                # t3 = time.time()
                # print('io: ' + str(t2 - t1) + '; compute: ' + str(t3 - t2))
                # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                # if step % 1 == 0:
                #     num_examples_per_step = self.batch_size * self.num_gpus
                #     examples_per_sec = num_examples_per_step / duration
                #     sec_per_batch = duration / self.num_gpus
                #
                #     format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                #                   'sec/batch)')
                #     print(format_str % (datetime.now(), step, loss_value,
                #                         examples_per_sec, sec_per_batch))
                #
                # if step % 10 == 0:
                #     summary_str = sess.run(summary_op, feed_dict={self.data_l: data_l, self.gt_ab_313: gt_ab_313,
                #                                                   self.prior_boost_nongray: prior_boost_nongray})
                #     summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 500 == 0:
                    checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
            return


    def construct_graph(self, scope):
        self.data_l = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 1])
        self.gt_ab_313 = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 512])
        # self.prior = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 512])

        self.conv8_313 = self.cnn.conv_net(data_l=self.data_l)
        self.new_loss, self.g_loss = self.cnn.loss_update(upSample=self.conv8_313,
                                                          gt_ab_313=self.gt_ab_313)
                                                          # prior_boost_nongray=self.prior)
        # show
        tf.summary.scalar('new_loss', self.new_loss)
        tf.summary.scalar('total_loss', self.g_loss)

        #?
        # self.update = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss)
        return self.new_loss, self.g_loss