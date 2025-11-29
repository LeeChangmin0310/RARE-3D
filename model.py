import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        """
        state_size: (H, W, C)  예) (100, 120, 4)
        action_size: 가능한 액션 개수
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        H, W, C = state_size
        glorot = tf.glorot_uniform_initializer()

        with tf.variable_scope(self.name):

            # ----- Placeholders -----
            self.inputs_ = tf.placeholder(tf.float32, [None, H, W, C], name="inputs")
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name="IS_weights")
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            # =====================================================
            # 1) CONV1: 8x8, stride 4, filters 32
            # =====================================================
            with tf.variable_scope("conv1"):
                w1 = tf.get_variable(
                    "w",
                    shape=[8, 8, C, 32],
                    initializer=glorot
                )
                b1 = tf.get_variable(
                    "b",
                    shape=[32],
                    initializer=tf.zeros_initializer()
                )
                conv1 = tf.nn.conv2d(
                    self.inputs_,
                    w1,
                    strides=[1, 4, 4, 1],
                    padding="VALID"
                )
                conv1 = tf.nn.bias_add(conv1, b1)
                self.conv1_out = tf.nn.elu(conv1, name="conv1_out")

            # =====================================================
            # 2) CONV2: 4x4, stride 2, filters 64
            # =====================================================
            with tf.variable_scope("conv2"):
                w2 = tf.get_variable(
                    "w",
                    shape=[4, 4, 32, 64],
                    initializer=glorot
                )
                b2 = tf.get_variable(
                    "b",
                    shape=[64],
                    initializer=tf.zeros_initializer()
                )
                conv2 = tf.nn.conv2d(
                    self.conv1_out,
                    w2,
                    strides=[1, 2, 2, 1],
                    padding="VALID"
                )
                conv2 = tf.nn.bias_add(conv2, b2)
                self.conv2_out = tf.nn.elu(conv2, name="conv2_out")

            # =====================================================
            # 3) CONV3: 4x4, stride 2, filters 128
            # =====================================================
            with tf.variable_scope("conv3"):
                w3 = tf.get_variable(
                    "w",
                    shape=[4, 4, 64, 128],
                    initializer=glorot
                )
                b3 = tf.get_variable(
                    "b",
                    shape=[128],
                    initializer=tf.zeros_initializer()
                )
                conv3 = tf.nn.conv2d(
                    self.conv2_out,
                    w3,
                    strides=[1, 2, 2, 1],
                    padding="VALID"
                )
                conv3 = tf.nn.bias_add(conv3, b3)
                self.conv3_out = tf.nn.elu(conv3, name="conv3_out")

            # =====================================================
            # 4) Flatten: (None, 4, 5, 128) → (None, 2560)
            # =====================================================
            with tf.variable_scope("flatten"):
                flat_dim = 4 * 5 * 128
                self.flatten = tf.reshape(self.conv3_out, [-1, flat_dim])

            # -----------------------------------------------------
            # Helper dense (순수 TF1)
            # -----------------------------------------------------
            def dense(x, units, name, activation=None):
                in_dim = x.get_shape().as_list()[1]
                with tf.variable_scope(name):
                    W = tf.get_variable(
                        "W",
                        shape=[in_dim, units],
                        initializer=glorot
                    )
                    b = tf.get_variable(
                        "b",
                        shape=[units],
                        initializer=tf.zeros_initializer()
                    )
                    z = tf.matmul(x, W) + b
                    if activation is not None:
                        return activation(z)
                    else:
                        return z

            # Value stream V(s)
            self.value_fc = dense(self.flatten, 512, "value_fc", activation=tf.nn.elu)
            self.value = dense(self.value_fc, 1, "value")

            # Advantage stream A(s, a)
            self.advantage_fc = dense(self.flatten, 512, "advantage_fc", activation=tf.nn.elu)
            self.advantage = dense(self.advantage_fc, self.action_size, "advantages")

            # Aggregation: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
            mean_advantage = tf.reduce_mean(self.advantage, axis=1, keepdims=True)
            self.output = self.value + (self.advantage - mean_advantage)

            # Q(s, a_selected)
            self.Q = tf.reduce_sum(self.output * self.actions_, axis=1)

            # PER용 absolute error
            self.absolute_errors = tf.abs(self.target_Q - self.Q)

            # Loss with importance sampling weights
            self.loss = tf.reduce_mean(
                self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q)
            )

            # Optimizer
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
