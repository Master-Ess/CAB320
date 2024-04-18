#import gui_sokoban
import mySokobanSolver
import sanity_check
import search
import sokoban
import time

dyn_file_path = "./warehouses/warehouse_47.txt"
warehouse = sokoban.Warehouse()
warehouse.load_warehouse(dyn_file_path)

# If you want to check taboo cells, you can uncomment the following:
#taboo_result = mySokobanSolver.taboo_cells(warehouse)
#print("Taboo cells:\n", taboo_result)

tic = time.perf_counter()

solution, cost = mySokobanSolver.solve_weighted_sokoban(warehouse)

toc = time.perf_counter()
print(f"Completed Sokaban in {toc - tic:0.4f} seconds")

if solution == 'Impossible':
    print("No solution found for the puzzle.")
else:
    print("Solution found:")
    print("Actions: ", solution)
    print("Cost: ", cost)

print('End of Computation')