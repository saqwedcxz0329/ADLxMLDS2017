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
        n_actions = 2  # right:0 left:1


        self.model_folder = args.models_dir
        self.store_model_name = args.store_dqn_model_name
        self.reward_file_name = args.reward_file_name
        
        self.model = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.99,
                      e_greedy=0.95,
                      replace_target_iter=1000,
                      memory_size=10000,
                      batch_size=32,
                      e_greedy_increment=0.95/1e6
                      )
        if args.trained_dqn_model_name is not None:
            self.trained_model_name = args.trained_dqn_model_name
            self.model.restore(self.model_folder, self.trained_model_name)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.model.epsilon = 0.9
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
        online_net_update_freq = 4
        traget_net_update_freq = 1000

        step = 0
        avg_rs = np.zeros(100)
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
                actual_action, action = self.make_action(cur_obs, test=False)

                # RL take action and get next observation and reward
                next_obs, reward, done, info = self.env.step(actual_action)

                cur_flat_obs = cur_obs.reshape(-1)
                next_flat_obs = next_obs.reshape(-1)
                eps_rs_list.append(reward)
                self.model.store_transition(cur_flat_obs, action, reward, next_flat_obs, done)

                # swap observation
                cur_obs = next_obs
                step += 1

                if (step > start_learning_step):
                    if (step % online_net_update_freq == 0):
                        self.model.train()
                    if (step % traget_net_update_freq == 0):
                        self.model.update_target_net()
                self.model.update_epsilon()

            episode_reward = sum(eps_rs_list)
            avg_rs[i%100] = episode_reward
            eps_rs_list = []
            print('Run %d episodes, reward: %d, avg_reward: %.3f, epsilon: %.3f, step: %d' % (i, episode_reward, np.mean(avg_rs), self.model.epsilon, step))
            with open(self.reward_file_name, 'a') as reward_file:
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
        observation = observation.reshape(-1)
        action = self.model.make_action(observation)
        actual_action = self._transfer_to_actual_action(action)
        if test:
            return actual_action
        else:
            return actual_action, action

    def _transfer_to_actual_action(self, action):
        if action == 0:
            actual_action = 2 # Right
        elif action == 1:
            actual_action = 3 # Left
        else:
            raise ValueError('Ivalid action!!')
        return actual_action
