"""
This program will test the trained ai_robots on playing Ms Pacman
"""
import tensorflow as tf
import gym
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

GAME = 'MsPacman-v0'
mspacman_color = np.array([210, 164, 74]).mean()
# Choose trained model
saved_model_dir = ['models/nn_ai_models',
                   'models/diff_nn_ai_models',
                   'models/cnn_ai_models',
                   'models/pretrain_cnn_ai_models',
                   'models/lstm_cnn_ai_models',
                   'models/simple_ppo_ai_models'][1]
selected_model = saved_model_dir.split('/')[1]
print('You selected model: ', selected_model)

# Preprocess the image input
def preprocess(obs):
    img = obs[1:176:2, ::2]                # crop and downsize
    img = img.sum(axis=2)                  # to greyscale
    img[img == mspacman_color] = 0         # Improve contrast
    img = (img // 3 - 128).astype(np.int64)
    img = img.reshape(88, 80, 1)
    return img.astype(np.float).ravel()


# load meta graph and restore weights
sess = tf.Session()
saver = tf.train.import_meta_graph(saved_model_dir + '/test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint(saved_model_dir))


# for n in tf.get_default_graph().as_graph_def().node:
#     print(n.name)

# Start playing many episodes
env = gym.make(GAME)
total = 0
episodes = 1
video = []
im = 0


if saved_model_dir == 'models/diff_nn_ai_models':
    for i in range(episodes):
        s = env.reset()
        im = Image.fromarray(s, 'RGB')
        img = preprocess(s)
        prev_img = img
        obs = img - prev_img
        reward = 0
        done = None
        while done is not True:
            prob = sess.run('pi/dense_1/Softmax:0', feed_dict={'state:0': obs[None, :]})  # load in the trained policy
            # select action w.r.t the actions prob
            action = np.random.choice(range(prob.shape[1]), p=prob[0])
            # time.sleep(0.01)
            s, r, done, _ = env.step(action)
            saved_image = Image.fromarray(s, 'RGB')
            video.append(saved_image)
            prev_img = img
            img = preprocess(s)
            obs = img - prev_img
            reward += r
            env.render()
        total += reward
    print('Average reward: ', total / episodes)
    im.save('results/' + selected_model + '.gif', save_all=True, append_images=video)
else:
    for i in range(episodes):
        s = env.reset()
        im = Image.fromarray(s, 'RGB')
        obs = preprocess(s)
        reward = 0
        done = None
        while done is not True:
            prob = sess.run('pi/dense_1/Softmax:0', feed_dict={'state:0': obs[None, :]})    # load in the trained policy
            action = np.random.choice(range(prob.shape[1]), p=prob[0])           # select action w.r.t the actions prob
            time.sleep(0.01)
            s, r, done, info = env.step(action)                                  # Move one step
            saved_image = Image.fromarray(s, 'RGB')
            video.append(saved_image)
            obs = preprocess(s)
            reward += r
            env.render()
        total += reward
    print('Average reward: ', total / episodes)
    im.save('results/' + selected_model + '.gif', save_all=True, append_images=video)
