"""
A version of Proximal Policy Optimization (PPO) using multi agents and 3-layer CNN.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

Dependencies:
tensorflow 1.14.0
gym 0.9.2
"""
import tensorflow as tf
import numpy as np
import gym, threading, queue
import pickle
import time


EP_MAX = 5000
EP_LEN = 500
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 10            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
GAME = 'MsPacman-v0'
S_DIM, A_DIM = 88 * 80 * 1, 9         # state and action dimension
mspacman_color = np.array([210, 164, 74]).mean()


# Preprocess the image input
def preprocess(obs):
    img = obs[1:176:2, ::2]              # crop and downsize
    img = img.sum(axis=2)                # to greyscale
    img[img == mspacman_color] = 0       # Improve contrast
    img = (img // 3 - 128).astype(np.int64)
    img = img.reshape(88, 80, 1)
    return img.astype(np.float).ravel()


def cnn_model(x, trainable):
    # Input Layer
    input_layer = tf.reshape(x, [-1, 88, 80, 1])
    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[8, 8],
        padding="same",
        strides=4,
        activation=tf.nn.relu,
        trainable=trainable)

    # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[4, 4],
        padding="same",
        strides=2,
        activation=tf.nn.relu,
        trainable=trainable)

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        trainable=trainable)

    # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #
    # pool2_flat = tf.reshape(pool2, [-1, 22 * 20 * 64])
    flat_layer = tf.reshape(conv3, shape=[-1, 64*11*10])
    output = tf.layers.dense(flat_layer, 200, tf.nn.relu, trainable=trainable)
    # dropout = tf.layers.dropout(inputs=l1, rate=0.5, training=trainable)

    return output


def epsilon_greedy(i):
    if np.random.rand() < 0.1:
        return np.random.randint(A_DIM)     # random action
    else:
        return i                            # optimal action


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        lc = cnn_model(self.tfs, trainable=True)
        self.v = tf.layers.dense(lc, 1, trainable=True)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
        oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)  # shape=(None, )
        ratio = pi_prob / oldpi_prob
        surr = ratio * self.tfadv  # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1].ravel(), data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available
        self.saver.save(self.sess, 'models/cnn_ai_models/test_model')

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            la = cnn_model(self.tfs, trainable)
            a_prob = tf.layers.dense(la, A_DIM, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def choose_action(self, s):
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})
        a = np.argmax(prob_weights[0])
        a = epsilon_greedy(a)
        # a = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return a

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset()
            img = preprocess(s)
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(img)
                s_, r, done, _ = self.env.step(a)
                if done:
                    r = -10
                buffer_s.append(img)
                buffer_a.append(a)
                buffer_r.append(r-1)
                img = preprocess(s_)
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = self.ppo.get_v(img)

                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, None]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break

                    if done:
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)


if __name__ == '__main__':
    start_time = time.time()
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)

    print('Training time:', time.time() - start_time)

    with open('results/cnn_ai.pickle', 'wb') as fp:
        pickle.dump(GLOBAL_RUNNING_R, fp)

    # plot reward change and test
    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.ion()
    # plt.savefig('cnn_ai.jpg')
    # plt.show()
    # env = gym.make(GAME)
    # total = 0
    # for i in range(100):
    #     s = env.reset()
    #     x = preprocess(s)
    #     reward = 0
    #     done = None
    #     while done is not True:
    #         action = GLOBAL_PPO.choose_action(x)
    #         time.sleep(0.01)
    #         s, r, done, info = env.step(action)
    #         x = preprocess(s)
    #         reward += r
    #         env.render()
    #     total += reward
    # print('Average reward: ', total / 100)
