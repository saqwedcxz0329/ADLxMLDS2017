import numpy as np
import scipy.misc

from agent_dir.agent import Agent
from environment import Environment
from agent_dir.policy_network import PolicyNetwork

seed = 11037
model_path = './models'
model_name = 'model'

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

        self.model = PolicyNetwork(env.get_action_space().n,
                                   n_features,
                                   learning_rate=0.0001,
                                   reward_decay=0.9)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.model.restore(model_path, model_name)

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


    def train(self, env):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        total_episodes = 10000
        env.seed(seed)
        for i in range(total_episodes):
            cur_state = env.reset()
            prev_state = None
            self.init_game_setting()
            done = False
            #playing one game
            while(not done):
                state = cur_state - prev_state if prev_state is not None else cur_state
                action = self.make_action(state, test=True)

                prev_state = cur_state
                cur_state, reward, done, info = env.step(action)
                
                pre_gray_state = prepro(prev_state).reshape(-1)
                self.model.store_transition(pre_gray_state, action, reward)

            episode_reward = sum(self.model.ep_rs)
            if 'running_reward' not in globals():
                running_reward = episode_reward
            else:
                running_reward = running_reward * 0.99 + episode_reward * 0.01
            print('Run %d episodes, reward: %d' % (i, running_reward))

            vt = self.model.train()

            self.model.save(model_path, model_name)        



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
        observation = prepro(observation).reshape(-1)
        return self.model.make_action(observation)
        # return self.env.get_random_action()

