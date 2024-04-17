
'''

    Sokoban assignment


The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

You are NOT allowed to change the defined interfaces.
In other words, you must fully adhere to the specifications of the 
functions, their arguments and returned values.
Changing the interfacce of a function will likely result in a fail 
for the test of your code. This is not negotiable! 

You have to make sure that your code works with the files provided 
(search.py and sokoban.py) as your code will be tested 
with the original copies of these files. 

Last modified by 2022-03-27  by f.maire@qut.edu.au
- clarifiy some comments, rename some functions
  (and hopefully didn't introduce any bug!)

'''

# You have to make sure that your code works with 
# the files provided (search.py and sokoban.py) as your code will be tested 
# with these files
from ast import Num
from itertools import filterfalse
import search 
import sokoban


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (10755012, 'Kenzie', 'Haigh'), (1081425, 'Luke', 'Whitton'), (11132833, 'Emma', 'Wu') ]
    #raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def taboo_cells(warehouse):
    
    #KENZIE HAIGH

    '''  
    Identify the taboo cells of a warehouse. A "taboo cell" is by definition
    a cell inside a warehouse such that whenever a box get pushed on such 
    a cell then the puzzle becomes unsolvable. 
    
    Cells outside the warehouse are not taboo. It is a fail to tag an 
    outside cell as taboo.
    
    When determining the taboo cells, you must ignore all the existing boxes, 
    only consider the walls and the target  cells.  
    Use only the following rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of 
             these cells is a target.
    
    @param warehouse: 
        a Warehouse object with the worker inside the warehouse

    @return
       A string representing the warehouse with only the wall cells marked with 
       a '#' and the taboo cells marked with a 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''
    ##         "INSERT YOUR CODE HERE"

    #possibly should be rewritten using if x in y instead of the for loops to find stuff

    #need to add a function that will check if a taboo cell is unreachable anyway --> DONE!


    wall_patterns = [((1,0),(-1,0)), #hoz
                    ((0,1),(0,-1))  #vert
                     ]
    
    corner_patterns = [((1,0),(0,1)),
                      ((1,0),(0,-1)),
                      ((-1,0),(0,1)),
                      ((-1,0),(0,-1)),
                       ]
    

    
    def is_corner(warehouse, loc): 
        for pattern in corner_patterns:
                search_pattern_output = ((loc[0] + pattern[0][0],loc[1] + pattern[0][1]), (loc[0] + pattern[1][0],loc[1] + pattern[1][1]))
                applicable_walls = 0
            
                for search_loc in search_pattern_output:
                    for wall_loc in warehouse.walls:
                    
                        if search_loc == wall_loc:
                            applicable_walls = applicable_walls + 1
                            
                            #do checks for obj here
                        
                            if applicable_walls == 2:
                                return (True, search_pattern_output, pattern)
                        
                            break
        return (False, None, None)
    
    def taboo_warehouse_display(warehouse, taboo_corners, taboo_straights):

        X,Y = zip(*warehouse.walls) # pythonic version of the above
        x_size, y_size = 1+max(X), 1+max(Y)
        
        vis = [[" "] * x_size for y in range(y_size)]
        
        for (x,y) in taboo_straights:
            vis[y][x] = "X" 
   
        for (x,y) in taboo_corners:
            vis[y][x] = "X"     #kinda not needed -> the taboo_straights overwrite it but same result at the end lol
            
        for (x,y) in warehouse.walls:
            vis[y][x] = "#"

        return "\n".join(["".join(line) for line in vis])
    
    
    def check_inside_warehouse(inlist):
        temp = []
    #outside of warehouse loop
        for taboo_cell in inlist:
            up = False
            down = False
            left = False
            right = False
            for wall in warehouse.walls:        #these might be mislabeled -> doesnt matter, all that matters is that all 4 are found
                if taboo_cell[0] < wall[0] and wall[1] == taboo_cell[1]:
                   right = True
                elif taboo_cell[0] > wall[0] and wall[1] == taboo_cell[1]:
                   left = True
            
                if taboo_cell[1] < wall[1] and wall[0] == taboo_cell[0]:
                   up = True
                elif taboo_cell[1] > wall[1] and wall[0] == taboo_cell[0]:
                   down = True
               
            if up and down and left and right:
                 temp.append(taboo_cell)
             
        return temp       
    ##############################################################################

    taboo_corner_cell_list = []
    taboo_straight_cell_list = []
    corner_cell_list = []

    ###############################################################################

    #rule 1
    for cell in warehouse.walls:
        resp = is_corner(warehouse, cell)
        if resp[0]:
            safe = False
            dx = resp[2][0][0]
            dy = resp[2][1][1]
            taboo_cell = (cell[0] + dx, cell[1] + dy)
            corner_cell_list.append((cell, resp[1]))
            
            #add loop here to check for Ts and Xs

            # T check
            neg_x_cell = (cell[0] - resp[2][0][0],cell[1] - resp[2][0][1])
            neg_y_cell = (cell[0] - resp[2][1][0],cell[1] - resp[2][1][1])
            neg_x = False
            neg_y = False
            
            for wall in warehouse.walls:
                if wall == neg_x_cell:
                    neg_x = True
                if wall == neg_y_cell:
                    neg_y = True
                    
            #HANDLE T AND X CORNERS
                    
            if neg_x and neg_y:#x                               not sure if this will ever be utilised, inspection of example workshipfiles dont have any *should* work. Might be good to test before sub
                cell_1 = (cell[0] - dx, cell[1] + dy) #neg x
                cell_2 = (cell[0] + dx, cell[1] - dy) #neg y
                cell_3 = (cell[0] - dx, cell[1] - dy) #both
                
                if cell_1 not in warehouse.targets:
                     taboo_corner_cell_list.append(cell_1) 
                
                if cell_2 not in warehouse.targets:
                     taboo_corner_cell_list.append(cell_2) 
                     
                if cell_3 not in warehouse.targets:
                     taboo_corner_cell_list.append(cell_3)   
                     

                corner_cell_list.append((cell, (neg_x_cell, resp[1][1])))     #neg x
                corner_cell_list.append((cell, (resp[1][0], neg_y_cell)))     #neg y
                corner_cell_list.append((cell, (neg_x_cell, neg_y_cell)))     #both
                
            elif neg_x:#T
                 neg_x_taboo_cell = (cell[0] - dx, cell[1] + dy)
                 #check if in targetlost

                 if neg_x_taboo_cell not in warehouse.targets:
                     taboo_corner_cell_list.append(neg_x_taboo_cell) 
                     
                 corner_cell_list.append((cell, (neg_x_cell, resp[1][1])))
                 

            elif neg_y:#T
                neg_y_taboo_cell = (cell[0] + dx, cell[1] - dy)
                
                if neg_y_taboo_cell not in warehouse.targets:
                     taboo_corner_cell_list.append(neg_y_taboo_cell) 
                
                corner_cell_list.append((cell, (resp[1][0], neg_y_cell)))



            #end T X check

            for obj in warehouse.targets:
                    if taboo_cell == obj:
                        safe = True
                        break
                    
            if safe == False:
                   taboo_corner_cell_list.append(taboo_cell)
                
    #should have found all corner cells by now            

    ####################################################################################################
    
    #rule 2

    #will account for T and X when the corner checker does
    
    #array_builder
    corner_neighbour = []
    
    for entry in corner_cell_list:
        corner_neighbour.append((entry[0], entry[1][0])) 
        corner_neighbour.append((entry[0], entry[1][1])) 
        
    #remove duplicates - should be any but just to be safe
    corner_neighbour = list(dict.fromkeys(corner_neighbour))    

    #executer
    for corner_nub in corner_neighbour:
        dx = corner_nub[1][0] - corner_nub[0][0] 
        dy = corner_nub[1][1] - corner_nub[0][1]
        
        cur_loc = corner_nub[1]
        side_1_loc = (cur_loc[0] + dy, cur_loc[1] + dx)
        side_2_loc = (cur_loc[0] - dy, cur_loc[1] - dx)
        
        side_1_state = False
        side_2_state = False
        
        side_1 = []
        side_2 = []
    
        if side_1_loc in warehouse.targets:
            side_1_state = True
        if side_2_loc in warehouse.targets:
            side_2_state = True    
            
        eft = True    
        counter = 0
        for nub_loc in corner_neighbour:
           if nub_loc[1] == cur_loc or nub_loc[0] == cur_loc :
               counter = counter + 1
               
                 
        
        if counter > 1:
            eft = False
            
        while eft:
            cur_loc = (cur_loc[0] + dx, cur_loc[1] + dy)
            side_1_loc = (side_1_loc[0] + dx, side_1_loc[1] + dy)
            side_2_loc = (side_2_loc[0] + dx, side_2_loc[1] + dy)
            
            if side_1_loc in warehouse.targets:
                side_1_state = True
            if side_2_loc in warehouse.targets:
                side_2_state = True  
                
            for nub_loc in corner_neighbour:
                if nub_loc[1] == cur_loc:
                    eft = False
                    break
                
            if (cur_loc[0] + dx, cur_loc[1] + dy) not in warehouse.walls:
                eft = False
                    
            side_1.append(side_1_loc)
            side_2.append(side_2_loc)
        
        if not side_1_state:                                        #and not ((cur_loc[0] + dx) + dy, (cur_loc[1] + dy) + dx) in warehouse.walls:
            taboo_straight_cell_list.extend(side_1)
        if not side_2_state:                                        #and not ((cur_loc[0] + dx) - dy, (cur_loc[1] + dy) - dx) in warehouse.walls:
            taboo_straight_cell_list.extend(side_2)
            
                     
    #####################################################################################
                     
    #remove dupliates from taboo_straight_cell_list

    taboo_straight_cell_list = list(dict.fromkeys(taboo_straight_cell_list))        
    
    #remove out of bounds taboo cells

    #remove all occurances with negitive numbers
    temp = []
    
    X,Y = zip(*warehouse.walls) # stolen from the __str__
    x_size, y_size = 1+max(X), 1+max(Y)
    
    for each in taboo_straight_cell_list:
        if (each[0] > 0) and (each[0] < x_size) and (each[1] > 0) and (each[1] < y_size):
             temp.append(each)
             
    taboo_straight_cell_list = temp
    
   
             
    taboo_corner_cell_list = check_inside_warehouse(taboo_corner_cell_list)        
    taboo_straight_cell_list = check_inside_warehouse(taboo_straight_cell_list)  
    #finishing
    
    returnable_value = taboo_warehouse_display(warehouse, taboo_corner_cell_list, taboo_straight_cell_list)
    
    print(returnable_value)
    
    return returnable_value
    print('EOF')

    #raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SokobanPuzzle(search.Problem):

    

    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. 
    
    '''
    
    #
    #         "INSERT YOUR CODE HERE"
    #
    #     Revisit the sliding puzzle and the pancake puzzle for inspiration!
    #
    #     Note that you will need to add several functions to 
    #     complete this class. For example, a 'result' method is needed
    #     to satisfy the interface of 'search.Problem'.
    #
    #     You are allowed (and encouraged) to use auxiliary functions and classes

    
    def __init__(self, warehouse):
        raise NotImplementedError()

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        """
        raise NotImplementedError

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):

    #KENZIE HAIGH    

    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Impossible', if one of the action was not valid.
           For example, if the agent tries to push two boxes at the same time,
                        or push a box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    
    ##         "INSERT YOUR CODE HERE"

    worker_x = warehouse.worker[0]
    worker_y = warehouse.worker[1]
    
    directions = {
        'Left': (-1, 0),
        'Right': (1, 0),
        'Up': (0, -1),
        'Down': (0, 1),
    }
    
    box_locations = (warehouse.boxes).copy() #pointers

    for action in action_seq:
        
        movement = directions[action]
        dx = movement[0]
        dy = movement[1]
        
        worker_x_move = worker_x + dx
        worker_y_move = worker_y + dy
        
        box_x_move = worker_x + (dx * 2)
        box_y_move = worker_y + (dy * 2)
        
        

        #check if move makes it go into a wall
        if (worker_x_move, worker_y_move) in warehouse.walls:
            return "Impossible"
        
        #check if moves a box
        elif (worker_x_move, worker_y_move) in box_locations:
            
            #check that the box move doesnt move it into a wall
            if (box_x_move, box_y_move) in warehouse.walls:
                return "Impossible"
            if (box_x_move, box_y_move) in box_locations:       #this is implimented because i assume you cant push two boxes at the same time
                return "Impossible" 
            else:
                box_locations.remove((worker_x_move,worker_y_move))
                box_locations.append((box_x_move   ,box_y_move   ))
                
        #if code reaches this point we should be good to update locations
        
        worker_x = worker_x_move
        worker_y = worker_y_move
        
    #convert it back into the warehouse thingy
    
    #grabbed from sokoban __str__

    X,Y = zip(*warehouse.walls) # pythonic version of the above
    x_size, y_size = 1+max(X), 1+max(Y)
        
    vis = [[" "] * x_size for y in range(y_size)]
        # can't use  vis = [" " * x_size for y ...]
        # because we want to change the characters later
    for (x,y) in warehouse.walls:
            vis[y][x] = "#"
    for (x,y) in warehouse.targets:
            vis[y][x] = "."
        # if worker is on a target display a "!", otherwise a "@"
        # exploit the fact that Targets has been already processed
    if vis[worker_y][worker_x] == ".": # Note y is worker[1], x is worker[0] #WHY THE HELL IS IT BACK TO FRONT!!!!!!
            vis[worker_y][worker_x] = "!"
    else:
            vis[worker_y][worker_x] = "@"
        # if a box is on a target display a "*"
        # exploit the fact that Targets has been already processed
    for (x,y) in box_locations:                                     #was self.boxes but we have a boxes copy so we use that
        if vis[y][x] == ".": # if on target
                vis[y][x] = "*"
        else:
                vis[y][x] = "$"
    warehouse_obj =  "\n".join(["".join(line) for line in vis])   #was return 
    
    return warehouse_obj #need to test that this doesnt need a cast
        
        
        
        

    print('EOF')
    
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban(warehouse):
    '''
    This function analyses the given warehouse.
    It returns the two items. The first item is an action sequence solution. 
    The second item is the total cost of this action sequence.
    
    @param 
     warehouse: a valid Warehouse object

    @return
    
        If puzzle cannot be solved 
            return 'Impossible', None
        
        If a solution was found, 
            return S, C 
            where S is a list of actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
            C is the total cost of the action sequence C

    '''
    
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

