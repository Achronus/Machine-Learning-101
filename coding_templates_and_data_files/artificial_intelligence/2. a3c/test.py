# Testing the Agent
import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
import time
from collections import deque

def test(rank, params, shared_model):
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name, video=True)
    env.seed(params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.eval()
    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    start_time = time.time()
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)
        value, action_value, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(action_value)
        action = prob.max(1)[1].data.numpy()
        state, reward, done, _ = env.step(action[0, 0])
        reward_sum += reward
        if done:
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60) # 60 seconds break to allow the other agents to test the environment
        state = torch.from_numpy(state)
        
