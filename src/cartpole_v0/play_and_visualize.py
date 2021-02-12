import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm

def select_action(state):

    with torch.no_grad():
        output=policy_net(state)
        # print(output)
        # print(policy_net(state).max(1)[1].view(1, 1))
        #exit()
        return policy_net(state).max(1)[1].view(1, 1)

class DQN(nn.Module):

    def __init__(self, h, w, outputs, model_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(4, MODEL_SIZE),
                                    nn.BatchNorm1d(MODEL_SIZE),
                                    nn.ReLU()
                                    )

        self.layer2 = nn.Sequential(nn.Linear(MODEL_SIZE, MODEL_SIZE*2),
                                    nn.BatchNorm1d(MODEL_SIZE*2),
                                    nn.ReLU()
                                    )
        self.layer3 = nn.Sequential(nn.Linear(MODEL_SIZE*2, MODEL_SIZE*4),
                                    nn.BatchNorm1d(MODEL_SIZE*4),
                                    nn.ReLU()
                                    )

        self.head = nn.Linear(MODEL_SIZE*4, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.layer1(x.float())
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.head(x)
        return x


env = gym.make('CartPole-v0').unwrapped


n_actions = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_SIZE=512
policy_net = DQN(None, None, 2, model_size=512).to(device)
policy_net.eval()



for i in range(200):
    env.reset()
    current_state = torch.tensor([env.state], device=device)
    #env.render()
    done=False
    t=0
    policy_net.load_state_dict(torch.load(f'models/linear{i}.pth'))
    while not done:
        action = select_action(current_state)
        env.render()
        # screen = env.render(mode='rgb_array')#.transpose((2, 0, 1))
        # plt.imshow(screen)
        # plt.title(f'Episode {i}')
        # plt.show()
        current_state, reward, done, _ = env.step(action.item())
        current_state = torch.tensor([current_state], device=device).float()
        t+=1

    print(f'Episode length: {t}')
