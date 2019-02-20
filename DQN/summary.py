import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import os
import time

from os.path import expanduser
home = expanduser("~")

# update frequency
FREQUENCY = 1

EPISODE_MIN = 0

STEP_MIN_M = -1

TIME_MIN_M = 0

STEP_MIN_F = 3
STEP_MAX_F = 12

TIME_MIN_F = 0
TIME_MAX_F = 0.025

REWARD_MIN_F = 0
REWARD_MAX_F = 1



class summary:
    def __init__(
            self,
            # which summaries to display: ['sumiz_step', 'sumiz_time', 'sumiz_reward', 'sumiz_average_reward', 'sumiz_epsilon']
            summary_types = [],
            # the optimal step count of the optimal policy
            step_goal = 0,
            # the maximum reward for the optimal policy
            reward_goal = 0,
            # maximum exploitation value
            epsilon_goal = 0,
            # the 'focus' section graphs only between the start and end focus index. Somes useful for detail comparasion
            start_focus = 0,
            end_focus = 0,
            # desired name for file
            NAME = "default_image",
            # file path to save graph. i.e "/Desktop/Py/Scenario_Comparasion/Maze/Model/"
            SAVE_PATH = home,

            EPISODE_MAX = 100,

            STEP_MAX_M = 500,

            TIME_MAX_M = 120,

            REWARD_MIN_M = -100,

            REWARD_MAX_M = 100
    ):

        self.summary_types = summary_types
        self.step_goal = step_goal
        self.reward_goal = reward_goal
        self.epsilon_goal = epsilon_goal
        self.start_focus = start_focus
        self.end_focus = end_focus
        self.general_filename = NAME
        self.save_path = SAVE_PATH
        self.EPISODE_MAX = EPISODE_MAX
        self.STEP_MAX_M = STEP_MAX_M
        self.TIME_MAX_M = TIME_MAX_M
        self.REWARD_MIN_M = REWARD_MIN_M
        self.REWARD_MAX_M = REWARD_MAX_M
        self.to_summarize = True

        self.step_summary = []
        self.time_summary = []
        self.reward_summary = []
        self.epsilon_summary = []
        self.average_reward_summary = []

        # quick break
        if not self.summary_types:
            self.to_summarize = False
            return

        # initialize the number of main axis
        self.num_main_axes = 0

        self.is_average_reward_axes = False


        # determines number of graph we want to plot in the figure
        for element in self.summary_types:
            if element == 'sumiz_step':
                self.num_main_axes += 1
            if element == 'sumiz_time':
                self.num_main_axes += 1
            if element == 'sumiz_reward':
                self.num_main_axes += 1
            if element == 'sumiz_epsilon':
                self.num_main_axes += 1
            if element == 'sumiz_average_reward':
                self.num_main_axes += 1
                self.is_average_reward_axes = True

        if not self.num_main_axes:
            self.to_summarize = False
            return

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if len(os.listdir(home + self.save_path) ) != 0:
        #     filelist = [ f for f in os.listdir(home + self.save_path) if f.endswith(".png") ]
        #     for f in filelist:
        #         os.remove(os.path.join(home + self.save_path, f))
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.num_focus_axes = 0
        for element in self.summary_types:
            if element == 'sumiz_step':
                self.num_focus_axes += 1
            if element == 'sumiz_time':
                self.num_focus_axes += 1
            if element == 'sumiz_epsilon':
                self.num_focus_axes += 1

        if self.step_goal != 0:
            self.average_reward_goal = self.reward_goal/float(self.step_goal)
        else:
            self.average_reward_goal = 0

    def summarize(
        self,
        # for the current iteration
        episode_count,
        # an array that records steps taken in each episode. Index indicates episode
        step_count = 1,
        # an array that records the operation time for each episode
        time_count = 0,
        # an array that records total reward collected in each episode
        reward_count = 0,
        # epsilon greedy value
        epsilon_value = 0, 
        e_greedy_formula = 'insert formula'
    ):
        self.update(step_count, time_count, reward_count, epsilon_value)

        if not self.to_summarize:
            return


        # generate summary graphs
        if episode_count % FREQUENCY == 0:
            fig1 = plt.figure(figsize=(5, 10)) # ploting normally takes ~0.5 seconds
            i = 1
            for element in self.summary_types:
                if element == 'sumiz_step':
                    ax1 = fig1.add_subplot(self.num_main_axes, 1, i)
                    plt.axis([EPISODE_MIN,self.EPISODE_MAX,STEP_MIN_M,self.STEP_MAX_M])
                    ax1.plot(range(len(self.step_summary)),self.step_summary)
                    ax1.plot(range(len(self.step_summary)),np.repeat(self.step_goal, len(self.step_summary)), 'r:')
                    ax1.set_title('Number of steps taken in each episode')
                    ax1.set_xlabel('Episode')
                    ax1.set_ylabel('Steps taken')
                    i += 1

                if element == 'sumiz_time':
                    ax2 = fig1.add_subplot(self.num_main_axes, 1, i)
                    plt.axis([EPISODE_MIN,self.EPISODE_MAX,TIME_MIN_M,self.TIME_MAX_M])
                    ax2.plot(range(len(self.time_summary)),self.time_summary)
                    ax2.set_title('Execution time in each episode')
                    ax2.set_xlabel('Episode')
                    ax2.set_ylabel('Execution time (s)')
                    i += 1

                if element == 'sumiz_reward':
                    ax3 = fig1.add_subplot(self.num_main_axes, 1, i)
                    plt.axis([EPISODE_MIN,self.EPISODE_MAX,self.REWARD_MIN_M,self.REWARD_MAX_M])
                    ax3.plot(range(len(self.reward_summary)),self.reward_summary)
                    ax3.plot(range(len(self.reward_summary)), np.repeat(self.reward_goal, len(self.reward_summary)), 'r:')
                    ax3.set_title('Reward in each episode')
                    ax3.set_xlabel('Episode')
                    ax3.set_ylabel('Reward')
                    i += 1

                if element == 'sumiz_epsilon':
                    ax4 = fig1.add_subplot(self.num_main_axes, 1, i)
                    plt.axis([EPISODE_MIN,self.EPISODE_MAX, 0, 1])
                    ax4.plot(range(len(self.epsilon_summary)),self.epsilon_summary, label = e_greedy_formula)
                    ax4.plot(range(len(self.epsilon_summary)), np.repeat(self.epsilon_goal, len(self.epsilon_summary)), 'r:')
                    ax4.set_title('Epsilon Greedy')
                    ax4.set_xlabel('Episode \n ' + e_greedy_formula)
                    ax4.set_ylabel('Epsilon')
                    i += 1

                if element == 'sumiz_average_reward':
                    ax5 = fig1.add_subplot(self.num_main_axes, 1, i)
                    ax5.plot(range(len(self.average_reward_summary)),self.average_reward_summary)
                    ax5.plot(range(len(self.average_reward_summary)), np.repeat(self.average_reward_goal, len(self.average_reward_summary)), 'r:')
                    ax5.set_title('Reward in each episode per step')
                    ax5.set_xlabel('Episode')
                    ax5.set_ylabel('Reward per step')
                    i += 1
            
            plt.tight_layout()
            fig1.savefig(home + self.save_path + self.general_filename +  ".png")
            plt.close(fig1)

        if not self.num_focus_axes or self.start_focus == self.end_focus:
            return

        # generate index-focused summary graph
        if episode_count % FREQUENCY == 0 and episode_count > self.start_focus and episode_count < self.end_focus:
            fig2 = plt.figure(figsize=(5, 5))
            i = 1
            for element in self.summary_types:
                if element == 'sumiz_step':
                    ax1 = fig2.add_subplot(self.num_focus_axes, 1, i)
                    plt.axis([self.start_focus,self.end_focus,STEP_MIN_F,STEP_MAX_F])
                    ax1.plot(range(self.start_focus, min(episode_count, self.end_focus)), self.step_summary[self.start_focus:min(episode_count, self.end_focus)])
                    ax1.plot(range(self.start_focus, min(episode_count, self.end_focus)), np.repeat(self.step_goal, min(episode_count, self.end_focus) - self.start_focus), 'r:')
                    ax1.set_title('Number of steps taken in each episode')
                    ax1.set_xlabel('Episode')
                    ax1.set_ylabel('Steps taken')
                    i += 1

                if element == 'sumiz_time':
                    ax2 = fig2.add_subplot(self.num_focus_axes, 1, i)
                    plt.axis([self.start_focus,self.end_focus,TIME_MIN_F,TIME_MAX_F])
                    ax2.plot(range(self.start_focus, min(episode_count, self.end_focus)), self.time_summary[self.start_focus:min(episode_count, self.end_focus)])
                    ax2.set_title('Execution time in each episode')
                    ax2.set_xlabel('Episode')
                    ax2.set_ylabel('Execution time (s)')
                    i += 1

                if element == 'sumiz_epsilon':
                    ax3 = fig2.add_subplot(self.num_focus_axes, 1, i)
                    ax3.plot(range(self.start_focus, min(episode_count, self.end_focus)), self.epsilon_summary[self.start_focus:min(episode_count, self.end_focus)])
                    ax3.plot(range(self.start_focus, min(episode_count, self.end_focus)), np.repeat(self.epsilon_goal, min(episode_count, self.end_focus) - self.start_focus), 'r:')
                    ax3.set_title('Epsilon Greedy')
                    ax3.set_xlabel('Episode')
                    ax3.set_ylabel('Epsilon')
                    i += 1

            plt.tight_layout()
            fig2.savefig(home + self.save_path + self.general_filename +"_focused_summary.png")
            plt.close(fig2)
    def update(self, step_count, time_count, reward_count, epsilon_value):

        self.step_summary.append(step_count)
        self.time_summary.append(time_count)
        self.reward_summary.append(reward_count)
        self.epsilon_summary.append(epsilon_value)

        if  not self.is_average_reward_axes:
            return

        # check for divide by zero error
        if step_count == 0:
            print("Step array contains zero(s). reward-per-step graph will be omitted.")
            self.average_reward_summary.append(0)
        else:
            # find average reward in each episode
            self.average_reward_summary.append(reward_count / float(step_count))

    def display_parameters(self, intial_epsilon = None, max_epsilon = None, learning_rate = None, reward_decay = None, memory_size = None):
        print('='*40)
        print('{}{}{}'.format('='*11, ' Hyper-parameters ', '='*11))
        print('='*40)
        print('{}{}'.format(' Starting Epsilon: ', intial_epsilon))
        print('{}{}'.format(' Maximum Epsilon: ', max_epsilon))
        print('{}{}'.format(' Learning Rate (Alpha): ', learning_rate))
        print('{}{}'.format(' Reward Decay (Gamma): ', reward_decay))
        print('{}{}'.format(' Memory Size: ', memory_size))
        print('='*40)
