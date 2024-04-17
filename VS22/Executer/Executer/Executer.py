#import gui_sokoban
import mySokobanSolver
import sanity_check
import search
import sokoban

dyn_file_path = "./warehouses/warehouse_test.txt"
warehouse = sokoban.Warehouse()
warehouse.load_warehouse(dyn_file_path)

# If you want to check taboo cells, you can uncomment the following:
#taboo_result = mySokobanSolver.taboo_cells(warehouse)
#print("Taboo cells:\n", taboo_result)

solution, cost = mySokobanSolver.solve_weighted_sokoban(warehouse)

if solution == 'Impossible':
    print("No solution found for the puzzle.")
else:
    print("Solution found:")
    print("Actions: ", solution)
    print("Cost: ", cost)

print('End of Computation')