import random
class Player:
    def __init__(self, player, n):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "red" if your player will
        play as Red, or the string "blue" if your player will play
        as Blue.
        """
        self.n = n 
        self.board = [[0 for i in range (n)]for j in range(n)]
        self.last_action = ()
        self.player = player
        self.id = 0
        self.op = 0
        self.turn_no = 0

        if (self.player == 'blue'):
            self.id = 1
            self.op = 2
        else:
            self.id = 2
            self.op = 1
        # put your code here

    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """ 
        
        

        return self.choose_turn( self.board, self.player)


    
    def check_diamonds(self, board, captured, y, x, y_new, x_new, y_check, x_check, y_checker, x_checker):

        if max(y_checker, y_new, y_check) <= self.n - 1 and min(y_checker, y_new, y_check) >= 0:
            if max(x_checker, x_new, x_check) <= self.n - 1 and min(x_checker, x_new, x_check) >= 0:

                if board[y][x] == board[y_new][x_new]:

                    if board[y_check][x_check] == board[y_checker][x_checker]:
                        if board[y_check][x_check] != 0 and board[y][x] != board[y_check][x_check] :
                            captured.append([(y_check, x_check), (y_checker, x_checker)])
                


    #call this to check if a dimoand was present after last turn, where i and j are the last action indices
    def validate_diamonds(self, i, j):
        self.captured = []
        captured_index = []

        captured_index = [[i+2, j-1, i+1, j-1, i+1, j], [i-2, j+1, i-1, j, i-1, j+1], [i+1, j-2, i+1, j-1, i, j-1],
        [i+1, j+1, i+1, j, i, j+1], [i-1, j+2, i-1, j+1, i, j+1], [i-1, j-1, i, j-1, i-1, j] , 
        [i, j+1, i+1, j, i-1, j+1],[i, j-1, i+1, j-1, i-1, j], [ i+1, j, i, j+1, i+1, j-1],
        [i+1, j-1, i, j-1, i+1, j], [i-1, j, i, j-1, i-1, j+1], [i-1, j+1, i-1, j, i, j+1]]


        for list in captured_index:
            self.check_diamonds(self.board, self.captured, i, j, list[0], list[1], list[2], list[3], list[4], list[5])


        for m in self.captured:
            for n in m:
                self.board[n[0]][n[1]] = 0


    def turn(self, player, action):
        
        self.Update_board(player,action, self.board)
        self.last_action = action

        if("STEAL" != self.last_action[0]):
            i = self.last_action[1]
            j = self.last_action[2]
            self.validate_diamonds(i, j)
        

        
       
        self.turn_no +=1

    


    def choose_turn(self, board, player):
        

        
        not_placed = True
        while not_placed:
            
            num1 = random.randint(0,self.n - 1)
            num2 = random.randint(0,self.n - 1)

            if self.n % 2 != 0:
                if num1 == self.n //2 and num2 == self.n //2 and self.turn_no == 0:

                    continue

            if board[num1][num2] == 0:
            
                if player == 'red':
                    board[num1][num2] = 1
                    return ("PLACE", num1,num2)
                else:
                    board[num1][num2] = 2
                    return ("PLACE", num1,num2)
            else:
                continue
    

                

    
       
    def Update_board(self, player, action, Board):
        
        if(action[0] == "PLACE"):

            x = action[1]
            y = action[2]

            if(player == "red"):
                Board[x][y] = 1

            else:
                Board[x][y] = 2

        elif(action[0] == "STEAL"):

            x = self.last_action[1]
            y = self.last_action[2]
            
            Board[x][y] = 0
            Board[y][x] = self.op



        




        
