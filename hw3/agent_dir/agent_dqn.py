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
                      reward_decay=0.99,
                      e_greedy=0.9,
                      replace_target_iter=1000,
                      memory_size=10000,
                      batch_size=32,
                      )

        self.model_folder = args.models_dir
        self.store_model_name = args.store_dqn_model_name
        
        if args.trained_dqn_model_name is not None:
            self.trained_model_name = args.trained_dqn_model_name
            self.model.restore(self.model_folder, self.trained_model_name)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.model.restore(self.model_folder, self.trained_model_name)

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
        total_episodes = 100000000
        start_learning_step = 10000
        step = 0
        avg_rs = np.zeros(30)
        eps_rs_list = []
        best_avg_rs = -100
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
                eps_rs_list.append(reward)
                self.model.store_transition(cur_flat_obs, action, reward, next_flat_obs)

                if (step > start_learning_step) and (step % 4 == 0):
                    self.model.train()

                # swap observation
                cur_obs = next_obs

                step += 1
            episode_reward = sum(eps_rs)
            avg_rs[i%30] = episode_reward
            eps_rs_list = []
            print('Run %d episodes, reward: %d, avg_reward: %.3f' % (i, episode_reward, np.mean(avg_rs)))
            with open('reward_dqn.txt', 'a') as reward_file:
                reward_file.write('{},{}\n'.format(i, episode_reward))
            if(step > start_learning_step and np.mean(avg_rs) > best_avg_rs):
                best_avg_rs = np.mean(avg_rs)
                self.model.save(self.model_folder, self.store_model_name, i)


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

