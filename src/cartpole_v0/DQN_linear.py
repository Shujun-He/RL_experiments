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

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        #self.position = 0

    def push(self, *args):
        """Saves a transition."""
        #if len(self.memory) < self.capacity:
        self.memory.append(None)
        self.memory[-1] = Transition(*args)
        #self.position = (self.position + 1) % self.capacity
        if len(self.memory) > self.capacity:
            self.memory=self.memory[-self.capacity:]


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

BATCH_SIZE = 16
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 1
MODEL_SIZE = 512
#MEMORYCAPACITY = 2*TARGET_UPDATE*200
STEPS=None
#MEMORYCAPACITY=int(10*BATCH_SIZE*10)
MEMORYCAPACITY=int(1e8)
# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions, model_size=MODEL_SIZE).to(device)
target_net = DQN(screen_height, screen_width, n_actions, model_size=MODEL_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(),weight_decay=1e-5, lr=1e-2)
memory = ReplayMemory(MEMORYCAPACITY)

def update_lr(optimizer, factor=1e-1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor



steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        policy_net.eval()
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    plt.plot(np.ones(len(durations_t))*195,'r-')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.ylim([0,220])
    plt.pause(0.001)  # pause a bit so that plots are updated

    plt.savefig('linear_results.png')

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

class History_Dataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data=data

    def __getitem__(self,idx):
        return self.data[idx]

def optimize_model():
    policy_net.train()
    if len(memory) < BATCH_SIZE:
        return
    #if STEPS is not None:
    batches=len(memory)//BATCH_SIZE
    #else:
        #batches=STEPS
    #batches=10
    for i in tqdm(range(batches)):
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))


        #dataset=torch.utils.data

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        # next_state_batch = torch.cat(batch.next_state_batch)
        #
        # print(state_batch.shape)
        # print(next_state_batch.shape)
        # exit()

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # print(state_batch.shape)
        # print(action_batch.shape)
        # print(reward_batch.shape)
        # exit()


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

num_episodes = 400
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    #exit()
    current_state = torch.tensor([env.state], device=device)

    for t in tqdm(count()):
        # Select and perform an action
        action = select_action(current_state)
        next_state, reward, done, _ = env.step(action.item())

        next_state = torch.tensor([next_state], device=device).float()

        cart_pos_delta = (torch.abs(current_state[0,0])-torch.abs(next_state[0,0]))/2.4
        cart_vel_delta = (torch.abs(current_state[0,1])-torch.abs(next_state[0,1]))/2.4
        car_angle_delta = (torch.abs(current_state[0,2])-torch.abs(next_state[0,2]))/0.209
        car_ang_vel_delta = (torch.abs(current_state[0,3])-torch.abs(next_state[0,3]))/0.209

        #reward = reward + cart_pos_delta + car_angle_delta
        reward = cart_pos_delta + car_angle_delta + cart_vel_delta + car_ang_vel_delta
        #reward = 1


        reward = torch.tensor([reward], device=device).float()
        #next_state = torch.tensor([next_state], device=device)

        # Observe new state
        if done:
            next_state = None


        # Store the transition in memory
        memory.push(current_state, action, next_state, reward)

        # Move to the next state
        current_state = next_state

        # Perform one step of the optimization (on the target network)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
        elif t > 200:
            # episode_durations.append(t + 1)
            # plot_durations()
            # break
            pass
    optimize_model()
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(target_net.state_dict(),f'models/linear{i_episode}.pth')

    if i_episode%20==0:
        BATCH_SIZE*=2
        BATCH_SIZE=np.clip(BATCH_SIZE,0,2048)
    if i_episode%60==0:
        update_lr(optimizer)


print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
#plt.savefig('conv_results.png')
