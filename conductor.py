from env import SnakeEnvironment, ACTIONS
from agent import Agent
import random
import numpy as np
import torch

# conducts the training process by facilitating communication between env and agent
class Conductor:

    @staticmethod
    def train(episodes, gamma, epsilon_start, epsilon_decay, batch_size, target_update_freq):
        env = SnakeEnvironment()
        agent = Agent()
        epsilon = epsilon_start
        train_steps = 0
        scores = []

        for episode in range(episodes):
            score = 0
            env.reset()
            state = env.startState()
            done = False
            
            if episode % 10 == 0:
                print(f"episode: {episode}\nLast 3 scores: {scores[-3:]}")

            while not done:

                # choose action with epsilon greedy policy
                if random.random() < epsilon:
                    action = random.choice(ACTIONS)
                else:
                    Q_values = agent.online_net.forward(torch.tensor(state, dtype=torch.float32))
                    action = np.argmax(Q_values.detach().numpy())

                # perform action
                next_state, reward, done = env.step(action)

                score += reward

                # store the transition in the agents buffer
                agent.replay_buffer.add(state, action, reward, next_state, done)

                # update state
                state = next_state

                # train on a mini batch
                if agent.replay_buffer.is_ready():
                    # returned as torch tensors
                    states, actions, rewards, next_states, dones = agent.replay_buffer.sample(batch_size)

                    # calculate the td target on all transitions
                    # this uses the second state in the transition to bootstrap from
                    next_qvals = agent.target_net.forward(next_states)
                    next_max_qval, next_max_qval_action = torch.max(next_qvals, dim=1)
                    td_target = rewards + gamma * (1 - dones) * next_max_qval

                    # calculate the Q values of the first state in the transition
                    qvals = agent.online_net.forward(states)

                    # we only know information about the taken action
                    # cannot use the other Q values in the loss calculation
                    # 2 argument indexing zips together equal index values,
                    # then takes each position in the matrix corresponding to a pair
                    taken_action_qval = qvals[torch.arange(batch_size), actions]

                    # loss + optimize
                    loss = torch.nn.MSELoss()(taken_action_qval, td_target)
                    agent.optimizer.zero_grad()
                    loss.backward()
                    agent.optimizer.step()

                    train_steps += 1

                # update the target net
                if train_steps > 0 and train_steps % target_update_freq == 0:
                    agent.update_target_net()

            epsilon *= epsilon_decay
            scores.append(score)

            torch.save(agent.online_net.state_dict(), 'net_params/online_net.pth')

        return scores