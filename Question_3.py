import random
import math
import heapq

#TASK 1
#Setup maze 
rows = 6
cols = 6

# Convert a node into its coordinates
def node_to_coordinates(node):
    #node -> (x, y) | x=col, y=row 
    col = node // cols   # row
    row = node % cols    # col
    return col, row

# Convert coordinates into its node
def coordinates_to_node(x, y):
    return x * rows + y

# Generate a random maze
start_node = random.randint(0, 11) 
goal_node  = random.randint(24, 35)
possible   = []     #Nodes excluding start and goal nodes
for n in range(36):
    if n!=start_node and n!=goal_node:
        possible.append(n)
barriers   = random.sample(possible, 4)

print(f"Start: {start_node}, Goal: {goal_node}, Barriers: {barriers}")


def print_maze(path=None):
    path_set = set(path) if path else set()
    print("\n    ", end="")
    for c in range(cols): print(f"  {c} ", end="")
    print()
    for r in range(rows):
        print(f" {r} |", end="")
        for c in range(cols):
            node = coordinates_to_node(c, r)
            if node in barriers:       print("  # ", end="")
            elif node == start_node:   print("  S ", end="")
            elif node == goal_node:    print("  G ", end="")
            elif node in path_set:     print("  * ", end="")
            else:                      print(f" {node:2d} ", end="")
        print("|")
    print()

print_maze()



#TASK 2
#Get sorted vertical, horizontal and diagonal neighbors
def get_neighbors(node, barriers):
    x,y=node_to_coordinates(node)
    

    neighbors =[]

    for dx in [-1, 0, 1] :          #Directions that we can go
        for dy in [-1, 0, 1]:       
            if dx==0 and dy==0:
                continue
            nx=x+dx                 #Coordinates that we are looking, but havent stepped yet
            ny=y+dy                 #Coordinates that we are looking, but havent stepped yet

            if 0 <= nx < cols and 0 <= ny < rows:
                nb=coordinates_to_node(nx, ny)      #New node that we gonna stepped into
                if nb not in barriers:              #No moves are allowed through barriers
                    neighbors.append(nb)        
    return sorted(neighbors) 

#Edge costs between 2 nodes
def euclidean_cost(a,b):
    x1, y1 = node_to_coordinates(a)
    x2, y2 = node_to_coordinates(b)
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

#Depth limited search for path and cost
def dls(current, goal, depth, visited_log, barriers, path, cost):
    visited_log.append(current)
    if current == goal:
        return path, cost
    if depth <= 0:
        return None
    for nb in get_neighbors(current, barriers):
        if nb not in path:
            result = dls (
                nb, goal, depth - 1, 
                visited_log, barriers,
                path + [nb],
                cost + euclidean_cost(current, nb)
            )
            if result:
                return result
    return None

#IDDFS execution
all_visited = []
final_path=None
final_cost=None

for depth in range(36):
    visited_log=[]
    result=dls(start_node,goal_node,depth, visited_log,barriers, [start_node], 0.0)
    all_visited.extend(visited_log)
    if result:
        final_path, final_cost=result
        break

print(f"Visited nodes ({len(all_visited)}): {all_visited}")
print(f"Time: {len(all_visited)} minutes")
print(f"Path: {final_path}")
print(f"Path cost: {final_cost:.4f}")

print_maze(final_path)

# TASK 3
# Chebyshev distance heuristic
def chebyshev(node, goal):
    """h(n,g) = max(|nx-gx|, |ny-gy|)"""
    nx, ny = node_to_coordinates(node)
    gx, gy = node_to_coordinates(goal)
    return max(abs(nx - gx), abs(ny - gy))

# TASK 4
# Best First Search using Chebyshev heuristic
def best_first_search(start, goal, barriers, heuristic=None):
    if heuristic is None:
        heuristic = lambda n: chebyshev(n, goal)

    visited_set  = set()
    visited_list = []
    came_from    = {start: None}
    cost_so_far  = {start: 0.0}

    # Min-heap: (heuristic_value, node)
    heap = [(heuristic(start), start)]

    while heap:
        _, current = heapq.heappop(heap)

        if current in visited_set:
            continue
        visited_set.add(current)
        visited_list.append(current)

        if current == goal:
            # Reconstruct path by tracing came_from
            path, node = [], goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, cost_so_far[goal], visited_list, len(visited_list)

        for nb in get_neighbors(current, barriers):
            if nb not in visited_set:
                new_cost = cost_so_far[current] + euclidean_cost(current, nb)
                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost
                    came_from[nb] = current
                heapq.heappush(heap, (heuristic(nb), nb))

    return None, None, visited_list, len(visited_list)

# Run Best First Search
bfs_path, bfs_cost, bfs_visited, bfs_time = best_first_search(
    start_node, goal_node, barriers
)

print(f"\n[Best First Search]")
print(f"Visited nodes ({len(bfs_visited)}): {bfs_visited}")
print(f"Time: {bfs_time} minutes")
print(f"Path: {bfs_path}")
print(f"Path cost: {bfs_cost:.4f}")
print_maze(bfs_path)


# TASK 5
# Repeat for 3 random mazes and analyse results
from statistics import mean, variance

def run_maze(start, goal, barriers_list):
    """Run both algorithms on one maze and return results."""

    # IDDFS
    av, fp, fc = [], None, None
    for depth in range(36):
        vl = []
        result = dls(start, goal, depth, vl, barriers_list, [start], 0.0)
        av.extend(vl)
        if result:
            fp, fc = result
            break
    id_time     = len(av)
    id_path_len = len(fp) if fp else 0

    # Best First Search
    bfs_p, bfs_c, bfs_v, bfs_t = best_first_search(start, goal, barriers_list)
    bfs_path_len = len(bfs_p) if bfs_p else 0

    return {
        "iddfs": {"time": id_time, "path": fp,    "cost": fc,    "path_len": id_path_len},
        "bfs":   {"time": bfs_t,   "path": bfs_p, "cost": bfs_c, "path_len": bfs_path_len},
    }


print(f"\n{'='*55}")
print("  TASK 5 — THREE RANDOM MAZES")
print(f"{'='*55}")

id_times, id_lens = [], []
bf_times, bf_lens = [], []

for i in range(1, 4):
    # Generate fresh random maze
    s = random.randint(0, 11)
    g = random.randint(24, 35)
    p = [n for n in range(36) if n != s and n != g]
    b = random.sample(p, 4)

    print(f"\n  Maze {i}: Start={s}, Goal={g}, Barriers={b}")
    res = run_maze(s, g, b)

    print(f"  [IDDFS]  Path:{res['iddfs']['path']}  "
          f"Cost:{res['iddfs']['cost']:.4f}  Time:{res['iddfs']['time']} min")
    print(f"  [BFS]    Path:{res['bfs']['path']}    "
          f"Cost:{res['bfs']['cost']:.4f}   Time:{res['bfs']['time']} min")

    id_times.append(res["iddfs"]["time"])
    id_lens.append(res["iddfs"]["path_len"])
    bf_times.append(res["bfs"]["time"])
    bf_lens.append(res["bfs"]["path_len"])

# Mean and variance
print(f"\n{'='*55}")
print("  TASK 5 — STATISTICS")
print(f"{'='*55}")
print(f"\n  IDDFS:")
print(f"    Time  — mean: {mean(id_times):.2f},  variance: {variance(id_times):.2f}")
print(f"    Path  — mean: {mean(id_lens):.2f},   variance: {variance(id_lens):.2f}")
print(f"\n  Best First Search:")
print(f"    Time  — mean: {mean(bf_times):.2f},  variance: {variance(bf_times):.2f}")
print(f"    Path  — mean: {mean(bf_lens):.2f},   variance: {variance(bf_lens):.2f}")

print("""
  ANALYSIS:
  ┌────────────────┬─────────────────────┬─────────────────────┐
  │ Property       │ IDDFS               │ Best First Search   │
  ├────────────────┼─────────────────────┼─────────────────────┤
  │ Complete       │ Yes (finite space)  │ Yes (finite space)  │
  │ Optimal        │ Yes (exhaustive)    │ No (heuristic only) │
  │ Time Complexity│ O(b^d) exponential  │ O(b^m) faster       │
  └────────────────┴─────────────────────┴─────────────────────┘
""")
# TASK 6 — Three distance measures for BFS
# Define our distance functions
def manhattan(node, goal):
    nx, ny = node_to_coordinates(node)
    gx, gy = node_to_coordinates(goal)
    return abs(nx - gx) + abs(ny - gy)

def euclidean_h(node, goal):
    nx, ny = node_to_coordinates(node)
    gx, gy = node_to_coordinates(goal)
    return math.sqrt((nx - gx)**2 + (ny - gy)**2)

def octile(node, goal):
    nx, ny = node_to_coordinates(node)
    gx, gy = node_to_coordinates(goal)
    dx, dy = abs(nx - gx), abs(ny - gy)
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

HEURISTICS = {
    "Manhattan": lambda n, g: manhattan(n, g),
    "Euclidean": lambda n, g: euclidean_h(n, g),
    "Octile":    lambda n, g: octile(n, g),
}

# Run comparison on 3 random mazes
for i in range(1, 4):
    # 1. Generate a single maze for this trial
    s = random.randint(0, 11)
    g = random.randint(24, 35)
    p = [n for n in range(36) if n != s and n != g]
    b = random.sample(p, 4)
    
    print(f"\nTrial {i}: Start={s}, Goal={g}, Barriers={b}")

    # 2. Run IDDFS (The baseline)
    av, fp, fc = [], None, None
    for depth in range(36):
        vl = []
        result = dls(s, g, depth, vl, b, [s], 0.0)
        av.extend(vl)
        if result:
            fp, fc = result
            break
    print(f"[IDDFS]   Cost: {fc:.4f} | Time: {len(av)} min")

    # 3. Run Best First Search with EACH distance measure
    for name, hfn in HEURISTICS.items():
        # Pass the specific heuristic function to your BFS
        bp, bc, bv, bt = best_first_search(s, g, b, heuristic=lambda n: hfn(n, g))
        
        print(f"[BFS-{name:9}] Cost: {bc:.4f} | Time: {bt:2} nodes")



















