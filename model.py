import tensorflow as tf


class MAMLModel(object):
    def __init__(self, name, sess, inner_lr, meta_lr, train_lr):
        self.name = name
        with tf.variable_scope(self.name):
            self.inner_lr = inner_lr
            self.meta_lr = meta_lr
            self.train_lr = train_lr
            # self.grad_steps = 2  # 1
            self.weights = self.build_model()  # 返回网络权重
            self.train_build_ops()  # 元训练阶段
            self.sess = sess
            # self.test_sgd_ops()  # 元测试阶段
            self.test_adam_ops()  # 元测试阶段

    def build_model(self):  # 搭建网络模型
        self.inputs1 = tf.placeholder(shape=[None, 72, 28, 32, 2], dtype=tf.float32, name='inputs1')
        self.labels1 = tf.placeholder(shape=[None, 72, 28, 32, 2], dtype=tf.float32, name='labels1')
        self.inputs2 = tf.placeholder(shape=[None, 72, 28, 32, 2], dtype=tf.float32, name='inputs2')
        self.labels2 = tf.placeholder(shape=[None, 72, 28, 32, 2], dtype=tf.float32, name='labels2')
        conv_initializer = tf.contrib.layers.xavier_initializer()
        weights = {}

        weights['conv1'] = tf.get_variable('conv1w', [3, 3, 3, 2, 2], dtype=tf.float32, initializer=conv_initializer)
        weights['b1'] = tf.get_variable('conv1b', [2], initializer=tf.zeros_initializer())
        weights['conv2'] = tf.get_variable('conv2w', [3, 3, 3, 2, 8], dtype=tf.float32, initializer=conv_initializer)
        weights['b2'] = tf.get_variable('conv2b', [8], initializer=tf.zeros_initializer())
        weights['conv3'] = tf.get_variable('conv3w', [3, 3, 3, 8, 16], dtype=tf.float32, initializer=conv_initializer)
        weights['b3'] = tf.get_variable('conv3b', [16], initializer=tf.zeros_initializer())
        weights['conv4'] = tf.get_variable('conv4w', [3, 3, 3, 16, 32], dtype=tf.float32, initializer=conv_initializer)
        weights['b4'] = tf.get_variable('conv4b', [32], initializer=tf.zeros_initializer())
        weights['conv5'] = tf.get_variable('conv5w', [3, 3, 3, 32, 16], dtype=tf.float32, initializer=conv_initializer)
        weights['b5'] = tf.get_variable('conv5b', [16], initializer=tf.zeros_initializer())
        weights['conv6'] = tf.get_variable('conv6w', [3, 3, 3, 16, 8], dtype=tf.float32, initializer=conv_initializer)
        weights['b6'] = tf.get_variable('conv6b', [8], initializer=tf.zeros_initializer())
        weights['conv7'] = tf.get_variable('conv7w', [3, 3, 3, 8, 2], dtype=tf.float32, initializer=conv_initializer)
        weights['b7'] = tf.get_variable('conv7b', [2], initializer=tf.zeros_initializer())
        # 卷积
        weights['conv8'] = tf.get_variable('conv8w', [3, 3, 3, 2, 2], dtype=tf.float32, initializer=conv_initializer)
        weights['b8'] = tf.get_variable('conv8b', [2], initializer=tf.zeros_initializer())
        # 反卷积
        weights['conv_T'] = tf.get_variable('conv_Tw', [3, 3, 3, 2, 2], dtype=tf.float32, initializer=conv_initializer)
        weights['b_T'] = tf.get_variable('conv_Tb', [2], initializer=tf.zeros_initializer())
        # 残差块1
        weights['conv9'] = tf.get_variable('conv9w', [3, 3, 3, 2, 8], dtype=tf.float32, initializer=conv_initializer)
        weights['b9'] = tf.get_variable('conv9b', [8], initializer=tf.zeros_initializer())
        weights['conv10'] = tf.get_variable('conv10w', [3, 3, 3, 8, 16], dtype=tf.float32, initializer=conv_initializer)
        weights['b10'] = tf.get_variable('conv10b', [16], initializer=tf.zeros_initializer())
        weights['conv11'] = tf.get_variable('conv11w', [3, 3, 3, 16, 2], dtype=tf.float32, initializer=conv_initializer)
        weights['b11'] = tf.get_variable('conv11b', [2], initializer=tf.zeros_initializer())
        # 残差块2
        weights['conv12'] = tf.get_variable('conv12w', [3, 3, 3, 2, 8], dtype=tf.float32, initializer=conv_initializer)
        weights['b12'] = tf.get_variable('conv12b', [8], initializer=tf.zeros_initializer())
        weights['conv13'] = tf.get_variable('conv13w', [3, 3, 3, 8, 16], dtype=tf.float32, initializer=conv_initializer)
        weights['b13'] = tf.get_variable('conv13b', [16], initializer=tf.zeros_initializer())
        weights['conv14'] = tf.get_variable('conv14w', [3, 3, 3, 16, 2], dtype=tf.float32, initializer=conv_initializer)
        weights['b14'] = tf.get_variable('conv14b', [2], initializer=tf.zeros_initializer())
        # 残差块3
        weights['conv15'] = tf.get_variable('conv15w', [3, 3, 3, 2, 8], dtype=tf.float32, initializer=conv_initializer)
        weights['b15'] = tf.get_variable('conv15b', [8], initializer=tf.zeros_initializer())
        weights['conv16'] = tf.get_variable('conv16w', [3, 3, 3, 8, 16], dtype=tf.float32, initializer=conv_initializer)
        weights['b16'] = tf.get_variable('conv16b', [16], initializer=tf.zeros_initializer())
        weights['conv17'] = tf.get_variable('conv17w', [3, 3, 3, 16, 2], dtype=tf.float32, initializer=conv_initializer)
        weights['b17'] = tf.get_variable('conv17b', [2], initializer=tf.zeros_initializer())
        return weights

    def forward_propagation(self, x, weights):  # 我的要改成卷积操作 必须有strides[0] = strides[4] = 1
        for i in range(7):
            x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv{}'.format(i+1)], strides=[1, 1, 1, 1, 1], padding='SAME')
                                 + weights['b{}'.format(i+1)])
        x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv8'], strides=[1, 2, 2, 2, 1], padding='SAME') + weights['b8'])
        x = tf.nn.leaky_relu(tf.nn.conv3d_transpose(x, weights['conv_T'], strides=[1, 2, 2, 2, 1], padding='SAME', output_shape=tf.shape(self.inputs1)) + weights['b_T'])
        y = x
        x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv9'], strides=[1, 1, 1, 1, 1], padding='SAME') + weights['b9'])
        x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv10'], strides=[1, 1, 1, 1, 1], padding='SAME') + weights['b10'])
        x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv11'], strides=[1, 1, 1, 1, 1], padding='SAME') + weights['b11'])
        x = tf.nn.leaky_relu(tf.add(x, y))
        y = x
        x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv12'], strides=[1, 1, 1, 1, 1], padding='SAME') + weights['b12'])
        x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv13'], strides=[1, 1, 1, 1, 1], padding='SAME') + weights['b13'])
        x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv14'], strides=[1, 1, 1, 1, 1], padding='SAME') + weights['b14'])
        x = tf.nn.leaky_relu(tf.add(x, y))
        y = x
        x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv15'], strides=[1, 1, 1, 1, 1], padding='SAME') + weights['b15'])
        x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv16'], strides=[1, 1, 1, 1, 1], padding='SAME') + weights['b16'])
        x = tf.nn.leaky_relu(tf.nn.conv3d(x, weights['conv17'], strides=[1, 1, 1, 1, 1], padding='SAME') + weights['b17'])
        output = tf.nn.leaky_relu(tf.add(x, y), name='output')
        return output

    def train_build_ops(self):
        loss_support = tf.losses.mean_squared_error(self.labels1, self.forward_propagation(self.inputs1, self.weights))
        # loss_query = tf.losses.mean_squared_error(self.labels2, self.forward_propagation(self.inputs2, self.weights))

        # opt = tf.train.AdamOptimizer(self.inner_lr)
        # self.train_op = opt.minimize(loss_support)

        grads = tf.gradients(loss_support, list(self.weights.values()))
        # 40为裁剪规约数 b = a*clip_norm/max(l2范数, clip_norm)， _值为全局规约数，即2范数
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        # self.norm = _
        grads = dict(zip(self.weights.keys(), grads))
        updated_weights = dict(zip(self.weights.keys(), [weight_value-self.inner_lr*grads[key] for key, weight_value in self.weights.items()]))  # 参数更新

        # for i in range(self.grad_steps-1):  # 再更新两次权重
        #     loss_support = tf.losses.mean_squared_error(self.labels1, self.forward_propagation(self.inputs1, updated_weights))
        #     # loss_query = tf.losses.mean_squared_error(self.labels2, self.forward_propagation(self.inputs2, updated_weights))
        #     grads = tf.gradients(loss_support, list(updated_weights.values()))
        #     grads, _ = tf.clip_by_global_norm(grads, 1.0)
        #     grads = dict(zip(updated_weights.keys(), grads))
        #     updated_weights = dict(zip(updated_weights.keys(), [weight_value-self.inner_lr*grads[key] for key, weight_value in updated_weights.items()]))

        # 三次梯度下降后再次计算MSE
        loss_support = tf.losses.mean_squared_error(self.labels1, self.forward_propagation(self.inputs1, updated_weights))
        loss_query = tf.losses.mean_squared_error(self.labels2, self.forward_propagation(self.inputs2, updated_weights))
        self.loss_support = loss_support
        self.loss_query = loss_query
        optimizer = tf.train.AdamOptimizer(self.meta_lr)  # 0.001
        self.metatrain_op = optimizer.minimize(loss_query)  # run这个节点可以返回loss，使用查询集更新初始参数

    # def test_sgd_ops(self):
    #     # 在训练集上进行fine-tune，在测试集上进行评估
    #     self.loss_train = tf.losses.mean_squared_error(self.labels_train, self.forward_propagation(self.inputs_train, self.weights))
    #     self.loss_test= tf.losses.mean_squared_error(self.labels_test, self.forward_propagation(self.inputs_test, self.weights))
    #     grads = tf.gradients(self.loss_train, list(self.weights.values()))
    #     grads = dict(zip(self.weights.keys(), grads))
    #     # 更新权重
    #     self.copy = [tf.assign(self.weights[key], weight_value-self.inner_lr*grads[key]) for key, weight_value in self.weights.items()]

    def test_adam_ops(self):
        # 在训练集上进行fine-tune，在测试集上进行评估
        self.loss_train_adam = tf.losses.mean_squared_error(self.labels1, self.forward_propagation(self.inputs1, self.weights))
        self.loss_test_adam = tf.losses.mean_squared_error(self.labels2, self.forward_propagation(self.inputs2, self.weights))
        optimizer_test = tf.train.AdamOptimizer(self.train_lr)
        self.metatest_op = optimizer_test.minimize(self.loss_train_adam)

    def train(self, x_support, y_support, x_query, y_query):
        #  run metatrain_op这个节点可以返回loss_query，使用查询集更新初始参数
        loss_support, loss_query, _, = self.sess.run([self.loss_support, self.loss_query, self.metatrain_op],
                                                     feed_dict={self.inputs1: x_support,
                                                                self.labels1: y_support,
                                                                self.inputs2: x_query,
                                                                self.labels2: y_query})
        return loss_support, loss_query

    def test(self, x_train, y_train, x_test, y_test):
        _, test_loss, train_loss = self.sess.run([self.metatest_op, self.loss_test_adam, self.loss_train_adam],
                                                 feed_dict={self.inputs1: x_train,
                                                            self.labels1: y_train,
                                                            self.inputs2: x_test,
                                                            self.labels2: y_test})
        return train_loss, test_loss
