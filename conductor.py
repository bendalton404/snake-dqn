from env import SnakeEnvironment, ACTIONS
from agent import Agent_MLP, Agent_CNN



# conducts the training process by facilitating communication between env and agent
class Conductor:

    @staticmethod
    def train(episodes, gamma, epsilon_start, epsilon_decay, batch_size, target_update_steps):
        env = SnakeEnvironment()
        agent = Agent_CNN(target_update_steps=target_update_steps)
        epsilon = epsilon_start
        scores = []

        for episode in range(episodes):
            score = 0
            env.reset()
            state = env.encode_state_for_cnn()
            done = False
            
            if episode > 0 and episode % 10 == 0:
                print(f"episode: {episode}\nLast 3 scores: {scores[-3:]}")
                agent.save_online_net()

            while not done:

                # choose action with epsilon greedy policy
                action = agent.choose_action(state, epsilon)

                # perform action
                reward, done = env.step(action)
                next_state = env.encode_state_for_cnn()

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
        state = env.encode_state_for_mlp()
        while not done:
            # play greedily
            action = agent.choose_action(state, 0)
            reward, done = env.step(action)
            next_state = env.encode_state_for_mlp()
            env.printBoard()
            state = next_state


if __name__ == "__main__":
    Conductor.view()