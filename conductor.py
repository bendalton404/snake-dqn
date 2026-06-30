from env import SnakeEnvironment, ACTIONS
from agent import Agent_MLP, Agent_CNN

import torch

# conducts the training process by facilitating communication between env and agent
class Conductor:

    @staticmethod
    def train(episodes, gamma, epsilon_start, epsilon_decay, batch_size, target_update_steps):
        env = SnakeEnvironment()
        agent = Agent_MLP(target_update_steps=target_update_steps)
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
                torch.save(agent.online_net.state_dict(), 'net_params/online_net.pth')

            while not done:

                # choose action with epsilon greedy policy
                action = agent.choose_action(state, epsilon)

                # perform action
                next_state, reward, done = env.step(action)

                score += reward

                # store the transition in the agents buffer
                agent.store_transition(state, action, reward, next_state, done)

                # update state
                state = next_state

                agent.train_mini_batch(batch_size=batch_size, gamma=gamma)

            epsilon *= epsilon_decay
            scores.append(score)

        return scores
    
    
    @staticmethod
    def view():
        env = SnakeEnvironment()
        agent = Agent_MLP(target_update_steps=1000)
        agent.load_online_net()
        done = False
        env.reset()
        state = env.startState()
        while not done:
            # play greedily
            action = agent.choose_action(state, 0)
            next_state, reward, done = env.step(action)
            env.printBoard()
            state = next_state
