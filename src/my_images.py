import tensorflow as tf
import gym
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# GAME = 'MsPacman-v0'
# mspacman_color = np.array([210, 164, 74]).mean()
#
# # Preprocess the image input
# def preprocess(obs):
#     img = obs[1:176:2, ::2]                # crop and downsize
#     img = img.sum(axis=2)                  # to greyscale
#     img[img == mspacman_color] = 0         # Improve contrast
#     img = (img // 3 - 128).astype(np.int64)
#     img = img.reshape(88, 80, 1)
#     return img
#
#
# env = gym.make(GAME)
# s = env.reset()
# prev_img = preprocess(s)
# s_, r, done, _ = env.step(1)
# img = preprocess(s_)
# im = Image.fromarray(img - prev_img, 'RGB')
# im.show()

import imageio

gif_original = 'results/random.gif'
gif_speed_up = 'results/speed_up.gif'

gif = imageio.mimread(gif_original)

imageio.mimsave(gif_speed_up, gif, fps=30)