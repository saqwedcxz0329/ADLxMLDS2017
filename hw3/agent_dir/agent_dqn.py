import numpy as np

from agent_dir.agent import Agent
from environment import Environment
from agent_dir.deep_q_network import DeepQNetwork

seed = 11037

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        observation_space = self.env.get_observation_space()
        action_space = self.env.get_action_space()
        n_features = observation_space.shape[0] * observation_space.shape[1] * observation_space.shape[2]
        n_actions = action_space.n

        self.model = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      )

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        total_episodes = 100000
        step = 0
        self.env.seed(seed)

        for i in range(total_episodes):
            cur_obs = self.env.reset()
            self.init_game_setting()
            done = False
            #playing one game
            while(not done):
                # RL choose action based on observation
                action = self.make_action(cur_obs)

                # RL take action and get next observation and reward
                next_obs, reward, done, info = self.env.step(action)

                cur_flat_obs = cur_obs.reshape(-1)
                next_flat_obs = next_obs.reshape(-1)
                self.model.store_transition(cur_flat_obs, action, reward, next_flat_obs)

                if (step > 200) and (step % 5 == 0):
                    self.model.train()

                # swap observation
                cur_obs = next_obs

                step += 1


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()

