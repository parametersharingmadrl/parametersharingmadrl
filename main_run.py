
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Provide 1 argument")
    arg = sys.argv[1]
    
    game_list = [
                 "pursuit", 
                 "waterworld", 
                 "multiwalker"
                ]
    if arg not in game_list:
        raise Exception("Input a valid game. Choose from {}".format(game_list))
        
    if arg == "pursuit":
        import sisl_games.pursuit.test
    
    if arg == "waterworld":
        import sisl_games.waterworld.test
    
    if arg == "multiwalker":
        import sisl_games.multiwalker.test
