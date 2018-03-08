'''
This script optimizes the neural network via retraining
'''
from mcts import MCTS
from selfplay import SelfPlay, Arena

class Coach(object):
    def __init__(self, game, model, config):
        self.game = game
        self.model = model
        self.config = config
        self.pmodel = self.model.__class__(self.game, self.config)

        self.mcts = MCTS(self.game, self.model, self.config.c_puct, self.config.num_sims)

    def learn(self):
        for i in range(self.config.num_iters):
            # TODO: double check if examples have been shuffled
            self_play = SelfPlay(self.game, self.model)
            print('.')
            examples = self_play.generate_play_data()
            print('.')

            # Step 1. Keep a copy of the current model
            self.model.save_checkpoint(filename='temp.pth.tar')
            self.pmodel.load_checkpoint(filename='temp.pth.tar')

            # Step 2. Training the model
            pmcts = MCTS(self.game, self.pmodel, self.config.c_puct, self.config.num_sims)
            self.model.train(examples)
            nmcts = MCTS(self.game, self.model, self.config.c_puct, self.config.num_sims)


            # Step 3. Evaluate the model
            print 'PITTING AGAINST PREVIOUS VERSION'
            arena = Arena(self.game, nmcts, pmcts)
            # Player 1 is the optimized player
            player1_win, player2_win, draw = arena.play_matches(self.config.arena_games)
            print 'NEW/PREV WINS : %d / %d ; DRAWS : %d' % (player1_win, player2_win, draw)

            if (player1_win * 1.0) / self.config.arena_games > self.config.arena_threshold:
                print 'ACCEPTING NEW MODEL'
                self.model.save_checkpoint(filename=self.getCheckpointFile(i))
                self.model.save_checkpoint(filename='best.pth.tar')
            else:
                print 'REJECTING NEW MODEL'
                self.model.load_checkpoint(filename='temp.pth.tar')


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'


