#---------------------------------------------
#

from zertz.ZertzLogic import Board
from zertz.ZertzGame import ZertzGame
from zertz.ZertzPlayer import Player
import numpy as np

class RandomStrategy(ZertzGame):
    def __init__(self):
        ZertzGame.__init__(self)

    def get_random_action(self):
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
                action = raw_input('Please enter AI action: ')
                print 'AI action ', action
                episode.append((state, action, player))
                state = self.get_next_state(action)

        winner = self.cur_player
        print 'Game ends! Player %i wins!' %self.cur_player
        return episode

if __name__ == '__main__':
    strategy = RandomStrategy()
    episode = strategy.collect_random_ai_episode()

