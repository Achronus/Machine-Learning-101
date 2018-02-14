# Training the AI
import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank) # desynchronize the agents, making their seeds unique
    env = create_atari_env(params.env_name)
    env.seed(params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    state = env.reset() # numpy array of: 1 * 42 * 42 (1 = channel; 42 = dimensions of image)
    state = torch.from_numpy(state) # Changes numpy array to torch tensor
    done = True # True for when game has finished
    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps):
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            state, reward, done = env.step(action.numpy()) # Playing the action
            done = (done or episode_length >= params.max_episode_length) # Prevent agent from getting stuck in a state
            reward = max(min(reward, 1), -1) # Clamp the reward
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                # Stop exploration and move onto next step
                break
        R = torch.zeros(1, 1) # Cumulative reward
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1) # Generalised Advantage Estimation: A(a,s) = Q(a,s) - V(s)
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i] # R = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(n-1) * r_(n-1) + gamma^nb_steps * V(last_state)
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2) # Q*(a*,s) = V*(s)
            TD = rewards[i] + params.gamma * values[i+1].data - values[i].data # Temporal Difference
            gae = gae * params.gamma * params.tau + TD # gae = sum_i (gamma*tau) ^i * TD(i)
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i] # policy_loss = -sum_i log(pi_i)*gae + 0.01*H_i
        optimizer.zero_grad()
        # Backward Propagation
        (policy_loss + 0.5 * value_loss).backward()
        # Prevents gradient from taking large values - value stays between 0 and 40
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()