
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

        for (x,y) in warehouse.walls:
            vis[y][x] = "#"
            
        for (x,y) in taboo_corners:
            vis[y][x] = "X" 
            
        for (x,y) in taboo_straights:
            vis[y][x] = "Y" 

        return "\n".join(["".join(line) for line in vis])
    
    def solve_rule_2(warehouse):
            # Variables to hold corner positions and taboo cells
        corners = set()
        taboo_straight_cell_list = set()

        # Identify corners
        for x, y in warehouse.walls:
            if is_corner(warehouse, (x, y))[0]:
                corners.add((x, y))

        # Check each corner for possible taboo line stretches
        for corner in corners:
            # Check in each direction from corner
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # right, left, up, down
            for dx, dy in directions:
                cells_between_corners = []
                has_target = False
                target_positions = []
                current_x, current_y = corner
                while True:
                    current_x += dx
                    current_y += dy
                    # Check if next position is another corner or if it goes out of wall boundaries
                    if (current_x, current_y) not in warehouse.walls or (current_x, current_y) in corners:
                        break
                    # Check if there's a target in this stretch
                    if (current_x, current_y) in warehouse.targets:
                        has_target = True
                        target_positions.append((current_x, current_y))
                    cells_between_corners.append((current_x, current_y))

                # If no targets were found in this wall stretch, mark adjacent cells as taboo
                if has_target:
                    for wall_x, wall_y in cells_between_corners:
                        # Determine adjacent positions next to the wall
                        adjacent_positions = [(wall_x + dx, wall_y + dy), (wall_x - dx, wall_y - dy)]
                        for adj_x, adj_y in adjacent_positions:
                            # Mark as taboo if it's not a target, not a wall, not a corner
                            if (adj_x, adj_y) not in target_positions and (adj_x, adj_y) not in warehouse.walls and (adj_x, adj_y) not in corners:
                                taboo_straight_cell_list.add((adj_x, adj_y))
                else:
                    # If no targets, all adjacent cells are taboo
                    for wall_x, wall_y in cells_between_corners:
                        adjacent_positions = [(wall_x + dx, wall_y + dy), (wall_x - dx, wall_y - dy)]
                        for adj_x, adj_y in adjacent_positions:
                            if (adj_x, adj_y) not in warehouse.walls and (adj_x, adj_y) not in corners:
                                taboo_straight_cell_list.add((adj_x, adj_y))

        return list(taboo_straight_cell_list)
    
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
                
                for obj in warehouse.targets:
                     taboo_corner_cell_list.append(cell_1) 
                
                for obj in warehouse.targets:
                     taboo_corner_cell_list.append(cell_2) 
                     
                for obj in warehouse.targets:
                     taboo_corner_cell_list.append(cell_3)   
                     

                corner_cell_list.append((cell, (neg_x_cell, resp[1][1])))     #neg x
                corner_cell_list.append((cell, (resp[1][0], neg_y_cell)))     #neg y
                corner_cell_list.append((cell, (neg_x_cell, neg_y_cell)))     #both
                
            elif neg_x:#T
                 neg_x_taboo_cell = (cell[0] - dx, cell[1] + dy)
                 #check if in targetlost

                 for obj in warehouse.targets:
                     taboo_corner_cell_list.append(neg_x_taboo_cell) 
                     
                 corner_cell_list.append((cell, (neg_x_cell, resp[1][1])))
                 

            elif neg_y:#T
                neg_y_taboo_cell = (cell[0] + dx, cell[1] - dy)
                
                for obj in warehouse.targets:
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
    for each in corner_neighbour:
        OBJ_side_1 = False
        OBJ_side_2 = False
        
        end_loc = None
        
        dx = each[0][0] - each[1][0]
        dy = each[0][1] - each[1][1]
        
        obj_cell_1_start = (each[1][0] + dy, each[1][1] + dx)
        obj_cell_2_start = (each[1][0] - dy, each[1][1] - dx)
        
        temp_list_1 = []
        temp_list_2 = []
        
        #check eitherside of the corner half
        for target in warehouse.targets:
            
            #if either has an OBJ mark that side as OBJ
            if target == obj_cell_1_start:
                OBJ_side_1 = True
            if target == obj_cell_2_start:
                OBJ_side_2 = True
            #else OBJ false status current    
        
        END = False
        new_cell = each[1]
        while not END: 
      
        #move to next cell
            new_cell = (new_cell[0] + dx, new_cell[1] + dy)
            
            obj_cell_1 = (new_cell[1] + dy, new_cell[1] + dx)
            obj_cell_2 = (new_cell[1] - dy, new_cell[1] - dx)
        
        #check cell exists
            found_cell = False
            for cell in warehouse.walls:
                if cell == new_cell:
                    found_cell = True
                    
                    #check eitherside of it for an obj
                    for target in warehouse.targets:
                                #if either has an OBJ mark that side as OBJ
                         if target == obj_cell_1:
                               OBJ_side_1 = True
                         if target == obj_cell_2:
                               OBJ_side_2 = True
                                #if true mark the correct OBJ
                    #check if cell is a corner half
                    for corner in corner_neighbour[1]:
                        
                        if corner == new_cell:
                                end_loc = new_cell    
                                END = True
                                #return side lists for sides that dont have OBJ == True
                        
                    #add cells to temp_list
                    if END != True:
                        temp_list_1.append(obj_cell_1)
                        temp_list_2.append(obj_cell_2)
                        
            if found_cell == False:
                END = True 

        if OBJ_side_1 != True:
            taboo_straight_cell_list.extend(temp_list_1)
            
        if OBJ_side_2 != True:
            taboo_straight_cell_list.extend(temp_list_2)
            
            #if cell doesnt exist remove "each" from corner_neighbour
            #return side lists for sides that dont have OBJ == True

        #remove the found corner from list
        if end_loc != None:
            for each in corner_neighbour:
                 if each[1] == end_loc:
                     corner_neighbour.remove(each) 
                     
    #remove dupliates from taboo_straight_cell_list
    
    taboo_straight_cell_list = solve_rule_2(warehouse)

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
    
    box_locations = (warehouse.boxes).copy() #fucking pointers

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

