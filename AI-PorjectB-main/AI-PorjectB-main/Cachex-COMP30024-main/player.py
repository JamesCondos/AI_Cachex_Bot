import numpy as np


# Code from Assignment 1, repurposed for Assignment 2 (go to 174 to see player implementation)

class Graph:
    #class definition
    def __init__(self, num):
        self.noRows = num
        self.noHex = num*num
        self.AdjList = {}
        self.NodeList = []
        self.placed = []
        self.id = 0


    #method for adding edge to graph
    def addEdge(self, n1, n2):  
        newNodes = []
        Node1 = tuple(n1)
        Node2 = tuple(n2)

        #if the node is already in already in adjacency list
        if Node1 in self.AdjList:
            #it saves nodes in a newNodes lists
            newNodes.extend(self.AdjList[Node1])
            newNodes.append(Node2)
            #allocates it in dictionary (adjacency list)
            self.AdjList[Node1] = newNodes
        else:
            # add node to adjacency list
            self.AdjList[Node1] = [Node2]

    #creates node list
    def createNodes(self):

        for i in range(0,self.noRows):
            for j in range(0,self.noRows):
                self.NodeList.append([i,j])
    
    #creats list of srounding possible nodes to compare to
    def createSurroundingList(self,node):
        x = node[0]
        y = node[1]
        lis = [[x+1,y],[x,y+1],[x-1,y+1],[x-1,y],
        [x,y-1],[x+1,y-1]]
        return lis


        
    #creates graph
    def createGraph(self, board, id, opponent):
        self.AdjList = {}
        self.NodeList = []
        self.id = id

        #creates node list
        self.createNodes()

        # for each node in node list compare to itself
        for node1 in self.NodeList:
            for node2 in self.NodeList:
                
                if(board[node2[0]][node2[1]] != opponent and board[node1[0]][node1[1]] != opponent):

                    #create possible surrounding nodes
                    surroundingNodes = self.createSurroundingList(node1)

                    # if surrounding nodes exists add edge 
                    if( node2 in surroundingNodes):
                        self.addEdge(node1, node2)
        
        for n in range(self.noRows):
        
            if(opponent == 1):
                if(board[n][0] != opponent):
                    self.addEdge([-1,self.noRows/2],  [n, 0])
                    self.addEdge([n, 0], [-1,self.noRows/2])

                if(board[n][self.noRows-1] != opponent):
                    self.addEdge( [self.noRows+1,self.noRows/2], [n, self.noRows-1])
                    self.addEdge( [n, self.noRows-1], [self.noRows+1,self.noRows/2])
               
    

            if(opponent == 2):
                if(board[0][n] != opponent):
                    self.addEdge([self.noRows/2,-1], [0,n])
                    self.addEdge([0,n], [self.noRows/2,-1]) 

                if(board[(self.noRows-1)][n] != opponent):
                    self.addEdge( [self.noRows/2,self.noRows+1] , [(self.noRows-1),n])
                    self.addEdge(  [(self.noRows-1),n], [self.noRows/2,self.noRows+1] )

                



    def printGraph(self):
        print(self.AdjList)
    


    #reconstructs path from path taken by appending to a list to the left
    def reconstructPath(self, start, current, pathtaken):

        reconst_path = []
        while pathtaken[tuple(current)] != current:
            reconst_path.append(current)
            current = pathtaken[tuple(current)]
        reconst_path.append(start)
        return reconst_path[1:-1]
    

    #get distance between nodes
    def distanceNode(self, node1, node2):
        distance = np.sqrt( (node1[0] - node2[0])**2 + (node1[1] - node2[1])**2 )
        
        distance += self.checkifplaced(node2)
        return distance
        
    def preferredStart(self, node):

        if self.id == 1:
            index = 1
        else:
            index = 0

        if node[index] == int(self.noRows/2):
            return True

    #get heuristic (currently using linear heuristic funciton)
    def getHeurestic(self, node):

        h = 0

        if self.preferredStart(node):
            return -10

        if self.id == 1:
            h = self.noRows - node[0]
        else:
            h = self.noRows - node[1]

        h += self.checkifplaced(node)
        

        return h

    #gets F function which is g + h functions 
    def getF(self, node, g):
        F = g[node] + self.getHeurestic(node)
        return F


   

    # checkign if current player has already placed node
    def checkifplaced(self, neighbour):
        num = 0

        for node in self.placed:
            if neighbour == tuple(node):
                
                num = num - 1
        
        
        return num
    
    # finds lowest F value in Openset
    def lowestF(self, Expandlist, fFunction):
        currentNode = None
        lowest = None

        for node in Expandlist:
            f = fFunction[tuple(node)]
            
            #print(node)
            #print(f)
            if lowest == None:
                currentNode = node
                lowest = f

            elif lowest > f:
                currentNode = node
        
        
        return currentNode
    
    #sets functions to infinite
    def setadjInfinite(self,adj):
        newadj = {}
        for node in adj:
            newadj[node] = 100
        return newadj


    def tracetoFurtherstNode(self,start):
        searching = True

        while(searching):
            

            for neighbour in self.AdjList[start]:
                if(neighbour in self.placed):
                    return

        return

    #A star Search
    #Pseudocode 
    #inspired by En.wikipedia.org. 2022. 
    # A* search algorithm - 
    # Wikipedia. [online] Available at: <https://en.wikipedia.org/wiki/A*_search_algorithm#cite_note-Felner2011-15> [Accessed 1 April 2022].
    def aStarSearch(self, start, end, placed):

        #realStart = start

        #if(trace):
        #    start = self.tracetoFurtherstNode(start)
        
        self.placed = placed

        #creates Priority queue of nodes that are needed to be expanded upon
        Expandlist = [start]
        hashStart = tuple(start)

        #saved path taken
        pathTaken = {}

        #functions defs
        gFunction = {}
        fFunction = {}

        #starts path taken at start
        pathTaken[hashStart] = start

        #sets up functions
        gFunction = self.setadjInfinite(self.AdjList)
        fFunction = self.setadjInfinite(self.AdjList)
        gFunction[hashStart] = 0
        fFunction[hashStart] = self.getHeurestic(start)

        current = None
        
        # loops to find optiumum route using heurestic function
        # all wieghts between nodes are equal
        while(Expandlist):

            #sets current to lowest F      
            current = self.lowestF(Expandlist, fFunction)
            
            #found end returns path
            if current == end:
                return self.reconstructPath(start, current, pathTaken)
                
            #removes from open set as it is now closed
            Expandlist.remove(current)
            hashCurrent = tuple(current)
            
            #loops through every connecting node and finds if better paths exists
            for neighbour in self.AdjList[hashCurrent]:
                
              
                total = gFunction[hashCurrent] + 1 #distance will always be 1

                
                #ensures that it doesn't revist visted nodes and takes better gFunction node
                if total < gFunction[neighbour]:    
                    pathTaken[neighbour] = current
                    gFunction[neighbour] = total
                    fFunction[neighbour] = total + self.getHeurestic(neighbour)
                    if neighbour not in Expandlist:
                        Expandlist.append(list(neighbour))

        #if(trace):
            #return self.aStarSearch(realStart,end,self.placed,False)
        #else:
        return " didn't find "



class Player:

    def __init__(self, player, n):

        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "red" if your player will
        play as Red, or the string "blue" if your player will play
        as Blue.
        """
        # put your code here
        self.n = n
        self.board = [[0 for i in range (n)]for j in range(n)]
        self.graph = Graph(self.n)
        self.myPlaced = []
        self.opPlaced = []
        self.last_action = ()
        self.player = player
        self.turncount = 0
        
        if (self.player == 'red'):
            self.id = 1
            self.op = 2
            self.startend = [self.n/2,self.n+1],[self.n/2,-1]
            self.startendOp = [self.n+1,self.n/2],[-1,self.n/2]
        else:
            self.id = 2
            self.op = 1
            self.startend = [self.n+1,self.n/2],[-1,self.n/2]
            self.startendOp = [self.n/2,self.n+1],[self.n/2,-1]
            


    
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
            Board[y][x] = 2
    


    
    def Minimax_decision(self):


        if(self.turncount == 0 and self.id == 2):
            self.Update_board("blue",("STEAL",  ), self.board)
            return ("STEAL",  )
        
        print(self.turncount)

        self.graph.createGraph(self.board, self.id, self.op)
        playerPath = self.graph.aStarSearch(self.startend[1],self.startend[0], self.myPlaced)

        if(self.turncount != 0):
            self.graph.createGraph(self.board, self.op, self.id)
            opPath = self.graph.aStarSearch(self.startendOp[1],self.startendOp[0], self.opPlaced)
        else:
            opPath = playerPath

        Value = {}
        currentBoard = self.board


        for Step in playerPath: 
                if((currentBoard[Step[0]][Step[1]] != self.id) and (Step in opPath)):
                    
                    #print('THIS IS TURN',self.turn)
                    if(self.turncount == 0 and self.n%2 != 0):
                        #print(self.n//2)
                        if Step[0] == self.n//2 and Step[1] == self.n//2:
                            continue
                        
                    Value[(Step[0],Step[1])] = self.Minimax_value(currentBoard,1,0)
            
        placement = max(Value, key=Value.get)
        print(placement)

        
        return ("PLACE", placement[0], placement[1])
        
            
   
    
    def Evaluation(self, board_state):
        
        self.graph.createGraph(board_state, self.op, self.id)
        Path = self.graph.aStarSearch(self.startendOp[1],self.startendOp[0], self.opPlaced)
        minusnum = 0
        
        for step in Path:
            if(board_state[step[0]][step[1]] != self.op):
                minusnum += 1
        
        self.graph.createGraph(board_state, self.op, self.id)
        Path = self.graph.aStarSearch(self.startendOp[1],self.startendOp[0], self.opPlaced)
        plusnum = 0

        for step in Path:
            if(board_state[step[0]][step[1]] != self.id):
                plusnum += 1

        evalScore = plusnum - minusnum
        


        return evalScore
        

    def Minimax_value(self, board_state, mmState, depth):


        if depth == 3:

            return self.Evaluation(board_state)

        elif mmState == 1:
            
            self.graph.createGraph(board_state, self.op, self.id)
            sucessorStates = self.graph.aStarSearch(self.startendOp[1],self.startendOp[0], self.opPlaced)
            
            lowestValue = 1
            
            for place in sucessorStates:

                if(board_state[place[0]][place[1]] != self.op):
                    mmState == 0
                    value = self.Minimax_value(board_state, mmState, depth+1)
                    
                    if (value < lowestValue):
                        lowestValue = value
                
            return lowestValue

        else:

            self.graph.createGraph(board_state, self.id, self.op)
            sucessorStates = self.graph.aStarSearch(self.startend[1],self.startend[0], self.myPlaced)
            
            highestValue = -1

            for place in sucessorStates:

                if(board_state[place[0]][place[1]] != self.id):
                    mmState = 1
                    value = self.Minimax_value(board_state, mmState, depth+1)
                    
                    if (value > highestValue):
                        highestValue = value
                
            return highestValue
        
        
            
         


    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        # put your code here

        return self.Minimax_decision()


            
       
    


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

        captured_index = [[i+2, j-1, i+1, j-1, i+1, j], [i-2, j+1, i-1, j, i-1, j+1], [i+1, j-2, i+1, j-1, i, j-1],
        [i+1, j+1, i+1, j, i, j+1], [i-1, j+2, i-1, j+1, i, j+1], [i-1, j-1, i, j-1, i-1, j] , 
        [i, j+1, i+1, j, i-1, j+1],[i, j-1, i+1, j-1, i-1, j], [ i+1, j, i, j+1, i+1, j-1],
        [i+1, j-1, i, j-1, i+1, j], [i-1, j, i, j-1, i-1, j+1], [i-1, j+1, i-1, j, i, j+1]]


        for list in captured_index:
            self.check_diamonds(self.board, self.captured, i, j, list[0], list[1], list[2], list[3], list[4], list[5])


        for m in self.captured:
            for n in m:
                #print(self.myPlaced)
                self.board[n[0]][n[1]] = 0

                if [n[0],n[1]] == self.myPlaced:
                    self.myPlaced.remove([n[0],n[1]])

    
    def turn(self, player, action):

        
        self.turncount += 1

        self.Update_board(player,action, self.board)
        self.graph.createGraph(self.board, self.id, self.op)

        if("STEAL" == action[0] and player == self.op):
            self.opPlaced.append([self.last_action[2],self.last_action[1]])
        elif():
            self.opPlaced.append([action[1],action[2]])

        self.last_action = action

        if("STEAL" != self.last_action[0]):
            self.validate_diamonds(self.last_action[1], self.last_action[2])
            

        

        print(self.board)
        """
        Called at the end of each player's turn to inform this player of 
        their chosen action. Update your internal representation of the 
        game state based on this. The parameter action is the chosen 
        action itself. 
        
        Note: At the end of your player's turn, the action parameter is
        the same as what your player returned from the action method
        above. However, the referee has validated it at this point.
        """
        # put your code here


