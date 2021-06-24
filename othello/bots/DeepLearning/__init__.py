import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random, os
from othello.OthelloUtil import getValidMoves
from othello.bots.DeepLearning.OthelloModel import OthelloModel
from othello.OthelloGame import OthelloGame

class BOT():

    def __init__(self, board_size, model_name=None, *args, **kargs):
        self.board_size=board_size
        self.model = OthelloModel( input_shape=(self.board_size, self.board_size) )
        try:
            if model_name == None:
                self.model.load_weights()
            else:
                self.model.load_weights(model_name)
            print('model loaded')
        except:
            print('no model exist')
            pass
        
        self.collect_gaming_data=False
        self.history=[]
    
    def getAction(self, game, color):
        predict = self.model.predict( game )
        valid_positions=getValidMoves(game, color)
        valids=np.zeros((game.size), dtype='int')
        valids[ [i[0]*game.n+i[1] for i in valid_positions] ]=1
        predict*=valids
        # position = np.argmax(predict) # play
        score = sorted(enumerate(predict), key=lambda p:p[1], reverse=True) # train
        position = random.choice(score[:3])[0]

        if self.collect_gaming_data:
            tmp=np.zeros_like(predict)
            tmp[position]=1.0
            self.history.append([np.array(game.copy()), tmp, color])
        
        position=(position//game.n, position%game.n)
        return position
    
    def save_train_history_figure(self, fileName):
        plt.figure()
        plt.plot(self.train_history.history['loss'])
        plt.plot(self.train_history.history['val_loss'])
        plt.title('Train History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('othello/bots/DeepLearning/train_history/' +  fileName)
        print('train history figure saved')

    def save_train_history_csv(self, gen, fileName="othello/bots/DeepLearning/train_history/val_loss.csv"):
        if os.path.isfile(fileName):
            loss_dict = pd.read_csv(fileName).to_dict('list')
        else:
            loss_dict = {
                "generation": list(),
                "last_validation_loss": list(),
            }
        loss_dict['generation'].append(gen)
        loss_dict['last_validation_loss'].append(self.train_history.history['val_loss'][-1])
        pd.DataFrame(loss_dict).to_csv(fileName, index=False)
        print('train history csv saved')

    def save_train_history(self, fileName):
        print(self.train_history.history['loss'])
        print(self.train_history.history['val_loss'])
        self.save_train_history_figure(fileName)
        self.save_train_history_csv(gen=fileName.split('.')[0])

    def gen_data(self, bot, rival_bot, args):
        def getSymmetries(board, pi):
            # mirror, rotational
            pi_board = np.reshape(pi, (len(board), len(board)))
            l = []
            for i in range(1, 5):
                for j in [True, False]:
                    newB = np.rot90(board, i)
                    newPi = np.rot90(pi_board, i)
                    if j:
                        newB = np.fliplr(newB)
                        newPi = np.fliplr(newPi)
                    l += [( newB, list(newPi.ravel()) )]
            return l
        bot.history=[]
        history=[]
        game=OthelloGame(bot.board_size)
        game.play(bot, rival_bot, verbose=args['verbose'])
        for step, (board, probs, player) in enumerate(bot.history):
            sym = getSymmetries(board, probs)
            for b,p in sym:
                history.append([b, p, player])
        bot.history.clear()
        if rival_bot != bot:
            for step, (board, probs, player) in enumerate(rival_bot.history):
                sym = getSymmetries(board, probs)
                for b,p in sym:
                    history.append([b, p, player])
        rival_bot.history.clear()
        game_result=game.isEndGame()
        return [(x[0],x[1]) for x in history if (game_result==0 or x[2]==game_result)]

    def self_play_train(self, args):
        self.collect_gaming_data=True
        data=[]
        for i in range(args['num_of_generate_data_for_train']):
            if args['verbose']:
                print('self playing', i+1)
            data+=self.gen_data(self, self, args)
        
        self.collect_gaming_data=False
        self.train_history = self.model.fit(data, batch_size = args['batch_size'], epochs = args['epochs'])
        self.model.save_weights()

    def play_train(self, rival_bot, args):
        self.collect_gaming_data=True
        data=[]
        for i in range(args['num_of_generate_data_for_train']):
            if args['verbose']:
                print('playing', i+1)
            # 各執一次黑子
            data+=self.gen_data(self, rival_bot, args)
            data+=self.gen_data(rival_bot, self, args)
        
        self.collect_gaming_data=False
        self.train_history = self.model.fit(data, batch_size = args['batch_size'], epochs = args['epochs'])
        self.model.save_weights()