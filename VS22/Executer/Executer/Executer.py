
#import gui_sokoban
import mySokobanSolver
import sanity_check
import search
import sokoban

dyn_file_path = "./warehouses/warehouse_03.txt"

WH = sokoban.Warehouse

WH.load_warehouse(WH, dyn_file_path)

mySokobanSolver.taboo_cells(WH)

print('EOC')