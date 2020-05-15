"""
Test on random robots playing Ms Pacman
"""
import gym
import time
from PIL import Image
from array2gif import write_gif

env = gym.make('MsPacman-v0')
total = 0
video = []

for i in range(1):
    s = env.reset()
    im = Image.fromarray(s, 'RGB')
    r, done = None, None
    ep_r = 0
    while done is not True:
        # action = env.action_space.sample()
        action = 3
        s, r, done, _ = env.step(action)
        saved_image = Image.fromarray(s, 'RGB')
        video.append(saved_image)
        ep_r += r
        env.render()
    total += ep_r

print('Average score for random bot: ', total / 1)
im.save('results/greedy.gif', save_all=True, append_images=video)
