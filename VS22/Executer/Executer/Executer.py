
#import gui_sokoban
import mySokobanSolver
import sanity_check
import search
import sokoban

dyn_file_path = "./warehouses/warehouse_custom.txt"

WH = sokoban.Warehouse

WH.load_warehouse(WH, dyn_file_path)

#wh1_modified soloution = ['Left', 'Up', 'Up', 'Up']

mySokobanSolver.taboo_cells(WH)

#print(mySokobanSolver.check_elem_action_seq(WH, ['Right', 'Up', 'Up', 'Left', 'Left', 'Left', 'Up', 'Left', 'Down','Down','Up', 'Left', 'Left', 'Down', 'Down', 'Right', 'Right', 'Up', 'Up','Right', 'Right','Right', 'Right', 'Down','Down', 'Left', 'Up','Right','Up',]))

print('EOC')