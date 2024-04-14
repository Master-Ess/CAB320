
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

    wall_patterns = [((1,0),(-1,0)), #hoz
                    ((0,1),(0,-1))  #vert
                     ]
    
    corner_patterns = [((1,0),(0,1)),
                      ((1,0),(0,-1)),
                      ((-1,0),(0,1)),
                      ((-1,0),(0,-1)),
                       ]
    

    
    def is_corner(warehouse, loc): #i missed up indentation
        for each in corner_patterns:
                search_pattern = ((loc[0] + each[0][0],loc[1] + each[0][1]), (loc[0] + each[1][0],loc[1] + each[1][1]))
                applicable_walls = 0
            
                for search_loc in search_pattern:
                    for wall_loc in warehouse.walls:
                    
                        if search_loc == wall_loc:
                            applicable_walls = applicable_walls + 1
                            
                            #do checks for obj here
                        
                            if applicable_walls == 2:
                                return (True, search_pattern, each)
                        
                            break
        return (False, None, None)

    taboo_corner_cell_list = []
    taboo_straight_cell_list = []
    corner_cell_list = []
    
    #doesnt account for T or X -- need to fix

    #rule 1
    for cell in warehouse.walls:
        resp = is_corner(warehouse, cell)
        if resp[0]:
            safe = False
            dx = resp[2][0][0]
            dy = resp[2][1][1]
            taboo_cell = (cell[0] + dx, cell[1] + dy)
            corner_cell_list.append((cell, resp[1]))
            
            #add quick loop here to check for Ts and Xs

            for obj in warehouse.targets:
                    if taboo_cell == obj:
                        safe = True
                        break
                    
            if safe == False:
                   taboo_corner_cell_list.append(taboo_cell)
                
    #should have found all corner cells by now            

    #rule 2

    #doesnt account for T or X -- need to fix
    
    #array_builder
    corner_neighbour = []
    
    for entry in corner_cell_list:
        corner_neighbour.append((entry[0], entry[1][0])) 
        corner_neighbour.append((entry[0], entry[1][1])) 
    
    #executer
    for each in corner_neighbour:
        OBJ_side_1 = False
        OBJ_side_2 = False
        
        end_loc = None
        
        dx = each[0][0] - each[1][0]
        dy = each[0][1] - each[1][1]
        
        obj_cell_1_start = (new_cell[1][0] + dy, new_cell[1][1] + dx)
        obj_cell_2_start = (new_cell[1][0] - dy, new_cell[1][1] - dx)
        
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
            
            obj_cell_1 = (new_cell[1][0] + dy, new_cell[1][1] + dx)
            obj_cell_2 = (new_cell[1][0] - dy, new_cell[1][1] - dx)
        
        #check cell exists
            found_cell = False
            for cell in warehouse.cells:
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
            taboo_straight_cell_list.append(temp_list_1)
            
        if OBJ_side_2 != True:
            taboo_straight_cell_list.apped(temp_list_2)
            
            #if cell doesnt exist remove "each" from corner_neighbour
            #return side lists for sides that dont have OBJ == True

        #remove them from the list   both start and end corner if possible



             
        
    

        
    
    print('EOF')

    raise NotImplementedError()

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

