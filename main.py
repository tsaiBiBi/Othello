from othello import OthelloGame
from othello.bots.DeepLearning import BOT
import time, os
import pandas as pd

class Human:
    def getAction(self, game, color):
        print('input coordinate:', end='')
        coor=input()
        return (int(coor[1])-1, ord(coor[0])-ord('A'))

class Result:
    def __init__(self, generation, color, rival):
        self.num = 0
        self.color = color
        self.rival = rival
        self.generation = generation
        self.board = {
            "black": 0,
            "white": 0,
            "tie": 0,
        }
    
    def record(self, winner):
        self.num += 1
        if winner == OthelloGame.WHITE:
            self.board['white'] += 1
        elif winner == OthelloGame.BLACK:
            self.board['black'] += 1
        elif winner == 0:
            self.board['tie'] += 1

    def save_result(self, fileName="othello/bots/DeepLearning/train_history/result.csv"):
        odds = ( self.board[self.color] / self.num ) * 100

        if os.path.isfile(fileName):
            table_dict = pd.read_csv(fileName).to_dict('list')
        else:
            table_dict = {
                "generation": list(),
                "bot": list(),
                "odds": list(),
                "rival": list(),
                "black": list(),
                "white": list(),
                "tie": list()
            }
        table_dict["generation"].append(self.generation)
        table_dict["rival"].append(self.rival)
        table_dict["bot"].append(self.color)
        table_dict["odds"].append(odds)
        table_dict["white"].append(self.board["white"])
        table_dict["black"].append(self.board["black"])
        table_dict["tie"].append(self.board["tie"])
        pd.DataFrame(table_dict).to_csv(fileName, index=False)

def test(bot, rival_bot, generation, rival):
    game_result = Result(generation=generation, color='white', rival=rival)
    for _ in range(5):
        game=OthelloGame(BOARD_SIZE)
        game_result.record(game.play(black=rival_bot, white=bot))
    game_result.save_result()

    game_result = Result(generation=generation, color='black', rival=rival)
    for _ in range(5):
        game=OthelloGame(BOARD_SIZE)
        game_result.record(game.play(black=bot, white=rival_bot))
    game_result.save_result()


BOARD_SIZE=8

# bot = BOT(board_size=BOARD_SIZE)
bot_org = BOT(board_size=BOARD_SIZE, model_name='model_8x8_org.h5')
bot_random = BOT(board_size=BOARD_SIZE, model_name='model_8x8_random.h5')
bot_pre = BOT(board_size=BOARD_SIZE, model_name='model_8x8_save.h5')
bot_data200 = BOT(board_size=BOARD_SIZE, model_name='model_8x8_data200.h5')
bot_data100 = BOT(board_size=BOARD_SIZE, model_name='model_8x8_data100.h5')
bot_data50 = BOT(board_size=BOARD_SIZE, model_name='model_8x8_data50.h5')

args={
    'num_of_generate_data_for_train': 10, # 8
    'epochs': 10, # 10
    'batch_size': 4, # 4
    'verbose': False # True
}
train_times = 3
for i in range(train_times):
    print("# train_", i)
    bot = BOT(board_size=BOARD_SIZE)
    bot.self_play_train(args)
    bot.save_train_history(fileName="self_play_train_" + str(i) + '.jpg')

# bot.play_train(rival_bot=bot_pre, args=args)
# bot.play_train(rival_bot=bot_random, args=args)
# bot.play_train(rival_bot=bot_org, args=args)
# bot.play_train(rival_bot=bot_data200, args=args)
# bot.play_train(rival_bot=bot_data100, args=args)

test(bot=bot, rival_bot=bot_pre, generation="race_vs_save", rival="pre")
test(bot=bot, rival_bot=bot_random, generation="race_vs_random", rival="pre")
test(bot=bot, rival_bot=bot_org, generation="race_vs_org", rival="pre")
test(bot=bot, rival_bot=bot_data200, generation="race_vs_data200", rival="data200")
test(bot=bot, rival_bot=bot_data100, generation="race_vs_data100-32*3", rival="data100")
test(bot=bot, rival_bot=bot_data100, generation="race_vs_data50", rival="data100")