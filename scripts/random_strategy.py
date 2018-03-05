'''
This script is responsible for generating episodes using random strategy, including using random strategy to play with AI
'''

from zertz.ZertzGame import ZertzGame
import numpy as np
import pickle

RESULT_PATH = 'results/random_AI_episode.txt'

class RandomStrategy(ZertzGame):
    def __init__(self):
        ZertzGame.__init__(self)

    def get_random_action(self):
        '''
        Returns a random valid action
        :return:
        '''
        actions = self.get_valid_actions()
        index = np.random.choice(len(actions), size=1)[0]
        random_action = actions[index]
        return random_action

    def collect_random_episode(self):
        '''
        Generates an episode that both players following the random strategy
        TODO: add a cap to the length of trajectories
        :return: episode: (s1, a1, p1), (s2, a2, p2)...]. Winner is the last player that made a move
        '''
        episode = []
        state = self.get_current_state()

        while self.get_game_ended() == 0:
            action = self.get_random_action()
            player = self.cur_player
            episode.append((state, action, player))
            state = self.get_next_state(action)

        return episode

    def collect_random_ai_episode(self):
        '''
        Generates an episode that using random strategy to play with AI
        We are player 1 and the AI is player 0
        :return: (s1, a1, p1), (s2, a2, p2)...]. Winner is the last player that made a move
        '''
        episode = []
        state = self.get_current_state()
        print "Zertz game starts!" # TODO: print out the settings of the game like board size, etc.

        while self.get_game_ended() == 0:
            player = self.cur_player
            print '-----------------------'
            print 'Current player is player %i' %player

            if player == 1:
                action = self.get_random_action()
                print 'Random player action: ', action
                episode.append((state, action, player))
                state = self.get_next_state(action)

            else:
                assert player == 0
                action = raw_input('Please enter AI action: ') # TODO: the action is a string and need to be transform into a tuple
                print 'AI action ', action
                episode.append((state, action, player))
                state = self.get_next_state(action)

        winner = 0 if self.get_game_ended() == 1 else 1
        print 'Game ends! Player %i wins!' %self.cur_player
        return episode

def write_episode(path, episode):
    '''
    Writes the generated episode into a txt file
    :param path:
    :param episode:
    :return:
    '''
    with open(path, 'wb') as f:
        pickle.dump(episode, f)

if __name__ == '__main__':
    strategy = RandomStrategy()
    episode = strategy.collect_random_ai_episode()
    write_episode(RESULT_PATH, episode)


