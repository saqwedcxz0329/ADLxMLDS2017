import numpy as np
import scipy.misc

from agent_dir.agent import Agent
from environment import Environment
from agent_dir.policy_network import PolicyNetwork

seed = 11037

def prepro(o, image_size=[80, 80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32), axis=2)

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        
        n_features = 80 * 80 * 1
        self.n_actions = 3  # stop:0 up:1 down:2

        self.model = PolicyNetwork(self.n_actions,
                                   n_features,
                                   learning_rate=0.0001,
                                   reward_decay=0.9)

        self.model_folder = args.models_dir
        self.store_model_name = args.store_pg_model_name
        self.reward_file_name = args.reward_file_name

        if args.trained_pg_model_name is not None:
            self.trained_model_name = args.trained_pg_model_name
            self.model.restore(self.model_folder, self.trained_model_name)

        if args.test_pg:
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
        self.prev_obs = None

    def train(self, env):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        total_episodes = 100000
        env.seed(seed)
        
        avg_vt = np.zeros(30)
        
        best_avg_reward = -100
        counter = 0
        for i in range(total_episodes):
            cur_obs = env.reset()
            self.init_game_setting()
            done = False
            #playing one game
            while(not done):
                actual_action, action, gray_state = self.make_action(cur_obs, test=False)
                
                cur_obs, reward, done, info = env.step(actual_action)

                self.model.store_transition(gray_state, action, reward)

            episode_reward = sum(self.model.ep_rs)
            if 'running_reward' not in globals():
                running_reward = episode_reward
            else:
                running_reward = running_reward * 0.99 + episode_reward * 0.01

            vt = self.model.train()
            avg_vt[i % 30] = episode_reward
            print('Run %d episodes, reward: %d, avg_reward: %.3f' % (i, episode_reward, np.mean(avg_vt)))
            with open(self.reward_file_name, 'a') as reward_file:
                reward_file.write('{},{}\n'.format(i, episode_reward))
            if counter >= 30 and np.mean(avg_vt) > best_avg_reward:
                self.model.save(self.model_folder, self.store_model_name, i)
                best_avg_reward = np.mean(avg_vt)
            counter += 1
            
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        state = observation - self.prev_obs if self.prev_obs is not None else observation
        self.prev_obs = observation

        gray_state = prepro(state).reshape(-1)
        if np.random.uniform() < 0.95:
            action = self.model.make_action(gray_state)
        else:
            action = np.random.randint(0, self.n_actions)
        # action = self.model.make_action(gray_state)
        actual_action = self._transfer_to_actual_action(action)
        if test:
            return actual_action
        else:
            return actual_action, action, gray_state
            
    def _transfer_to_actual_action(self, action):
        if action == 0:
            actual_action = 0
        elif action == 1:
            actual_action = 2 # Up
        elif action == 2:
            actual_action = 3 # Down
        else:
            raise ValueError('Ivalid action!!')
        return actual_action
