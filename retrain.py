'''
This script optimizes the neural network via retraining
'''
from collections import deque
import random
import numpy as np

from mcts import MCTS
from selfplay import SelfPlay, Arena
import time

class Coach(object):
    def __init__(self, game, model, config):
        self.game = game
        self.model = model
        self.config = config
        self.prev_model = self.model.__class__(self.game, self.config)

    def learn(self):
        for i in range(self.config.num_iters):
            self_play = SelfPlay(self.game, self.model)
            examples = self_play.generate_play_data()
            for _ in range(self.config.num_episodes):
                examples += self_play.generate_play_data()
            examples = self.examples_to_array(examples)
            examples = self.shuffle_examples(examples)

            # Step 1. Keep a copy of the current model
            self.model.save_checkpoint(filename='temp.pth.tar')
            self.prev_model.load_checkpoint(filename='temp.pth.tar')

            # Step 2. Training the model
            prev_mcts = MCTS(self.game, self.prev_model, self.config.c_puct, self.config.num_sims)
            self.model.train(examples)
            new_mcts = MCTS(self.game, self.model, self.config.c_puct, self.config.num_sims)

            # Step 3. Evaluate the model
            print 'PITTING AGAINST PREVIOUS VERSION'
            arena = Arena(self.game, new_mcts, prev_mcts)
            # Player 1 is the optimized player
            player1_win, player2_win, draw = arena.play_matches(self.config.arena_games)
            print 'NEW MODEL/PREV MODEL WINS : %d / %d ; DRAWS : %d' % (player1_win, player2_win, draw)

            if ((player1_win * 1.0) / self.config.arena_games) > self.config.arena_threshold:
                print 'ACCEPTING NEW MODEL'
                self.model.save_checkpoint(filename=self.getCheckpointFile(i))
                self.model.save_checkpoint(filename='best.pth.tar')
            else:
                print 'REJECTING NEW MODEL'
                self.model.load_checkpoint(filename='temp.pth.tar')

    def examples_to_array(self, list_of_examples):
        np_board = np.array([ne[0] for ne in list_of_examples])
        np_pi_put = np.array([ne[1] for ne in list_of_examples])
        np_pi_cap = np.array([ne[2] for ne in list_of_examples])

        np_pi = np.concatenate((np_pi_put, np_pi_cap), axis=1)

        np_v = np.array([ne[3] for ne in list_of_examples])
        np_mask = np.array([ne[4] for ne in list_of_examples])
        return (np_board, np_pi, np_v)

    def shuffle_examples(self, examples):
        order = np.random.permutation(len(examples[0]))
        for array in examples:
            array = array[order, ...]
        return examples

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

class Individual(Coach):
    # Iteratively generate examples with self play and then train on those examples
    def __init__(self, game, model, config):
        self.game = game
        self.model = model
        self.config = config
        self.example_buffer = deque(maxlen=self.config.buffer_size)

    def learn(self):
        for i in range(self.config.num_iters):
            print 'Staring the %i th iteration...' %i
            # Step 1. Generate training examples by self play with current model
            self_play = SelfPlay(self.game, self.model)
            new_examples = []
            for j in range(self.config.num_episodes):
                start = time.time()
                new_examples += self_play.generate_play_data()
                now = time.time() - start

                if j % 100 == 0:
                    print 'Time to generate an episode = %i s' %now

            random.shuffle(new_examples)
            self.example_buffer.extend(new_examples)
            training_examples = self.examples_to_array(self.example_buffer)

            # Step 2. Train the model
            self.model.train(training_examples, i)
            self.model.save_checkpoint(filename=self.getCheckpointFile(i))

