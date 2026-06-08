"""
Maze Solver using A* Search
Syntecxhub AI Internship - Week 1, Project 1
============================================
Represents a maze as a 2D grid, solves it using the A* algorithm
with either Manhattan or Euclidean heuristic, visualizes the result,
and handles unreachable goal cases gracefully.
"""

import heapq
import math
import time


# ─────────────────────────────────────────────
#  MAZE DEFINITION
#  0 = open path  |  1 = wall
# ─────────────────────────────────────────────

MAZE = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

START = (0, 0)    # (row, col)
GOAL  = (14, 14)  # (row, col)


# ─────────────────────────────────────────────
#  HEURISTIC FUNCTIONS
# ─────────────────────────────────────────────

def manhattan(a: tuple, b: tuple) -> float:
    """Sum of absolute differences in row and column."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple, b: tuple) -> float:
    """Straight-line distance between two cells."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ─────────────────────────────────────────────
#  A* ALGORITHM
# ─────────────────────────────────────────────

def astar(maze: list, start: tuple, goal: tuple, heuristic=manhattan) -> dict:
    """
    A* search algorithm.

    Parameters
    ----------
    maze      : 2D list of ints (0=open, 1=wall)
    start     : (row, col) tuple
    goal      : (row, col) tuple
    heuristic : callable(a, b) -> float

    Returns
    -------
    dict with keys:
        path       : list of (row,col) from start to goal, or []
        visited    : set of visited nodes
        open_order : list of nodes in expansion order (for visualization)
        cost       : total path cost (steps)
        found      : bool
    """
    rows = len(maze)
    cols = len(maze[0])

    # Priority queue entries: (f_score, counter, node)
    # counter breaks ties so tuples never compare node-to-node
    counter = 0
    open_heap = []
    heapq.heappush(open_heap, (0, counter, start))

    came_from = {}            # node → parent
    g_score = {start: 0}     # cost from start to node
    open_set = {start}        # fast membership check
    visited = set()           # nodes popped from open list
    open_order = []           # expansion order for visualization

    print(f"\n{'='*50}")
    print(f"  A* Search | Heuristic: {heuristic.__name__}")
    print(f"  Start: {start}  →  Goal: {goal}")
    print(f"  Grid: {rows}×{cols} | Walls: {sum(cell==1 for row in maze for cell in row)}")
    print(f"{'='*50}")

    step = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)
        open_set.discard(current)

        if current in visited:
            continue

        visited.add(current)
        open_order.append(current)
        step += 1

        h = heuristic(current, goal)
        g = g_score[current]

        if step <= 5 or step % 10 == 0:
            print(f"  Step {step:3d} | Node {current} | g={g:.1f}  h={h:.2f}  f={g+h:.2f} | "
                  f"Visited: {len(visited)}  Open: {len(open_set)}")

        # ── Goal reached ──
        if current == goal:
            path = _reconstruct(came_from, current)
            print(f"\n  ✓ Goal reached at step {step}!")
            print(f"  Path length : {len(path)} cells")
            print(f"  Total cost  : {g_score[goal]:.1f} steps")
            print(f"  Nodes visited: {len(visited)}")
            return {
                "path": path, "visited": visited,
                "open_order": open_order, "cost": g_score[goal], "found": True
            }

        # ── Expand neighbors (4-directional) ──
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue          # out of bounds
            if maze[nr][nc] == 1:
                continue          # wall
            if neighbor in visited:
                continue          # already expanded

            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_new = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f_new, counter, neighbor))
                open_set.add(neighbor)

    # ── No path found ──
    print(f"\n  ✗ No path found. Goal {goal} is unreachable from {start}.")
    print(f"  Nodes visited: {len(visited)}")
    return {"path": [], "visited": visited, "open_order": open_order, "cost": 0, "found": False}


def _reconstruct(came_from: dict, current: tuple) -> list:
    """Trace back through came_from to build the path."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ─────────────────────────────────────────────
#  CONSOLE VISUALIZATION
# ─────────────────────────────────────────────

def visualize(maze: list, result: dict, start: tuple, goal: tuple) -> None:
    """Print a colored ASCII map of the maze and the solution."""
    path_set    = set(result["path"])
    visited_set = result["visited"]

    SYMBOLS = {
        "wall"    : "██",
        "path"    : "░░",
        "visited" : "··",
        "start"   : "S ",
        "goal"    : "G ",
        "open"    : "  ",
    }

    print(f"\n{'─'*50}")
    print("  MAZE VISUALIZATION")
    print(f"  Legend:  ██=wall  ░░=path  ··=visited  S=start  G=goal")
    print(f"{'─'*50}")

    for r, row in enumerate(maze):
        line = ""
        for c, cell in enumerate(row):
            pos = (r, c)
            if pos == start:
                line += SYMBOLS["start"]
            elif pos == goal:
                line += SYMBOLS["goal"]
            elif cell == 1:
                line += SYMBOLS["wall"]
            elif pos in path_set:
                line += SYMBOLS["path"]
            elif pos in visited_set:
                line += SYMBOLS["visited"]
            else:
                line += SYMBOLS["open"]
        print("  " + line)

    print(f"{'─'*50}")

    if result["found"]:
        print(f"  Path  : {' → '.join(str(p) for p in result['path'])}")
        print(f"  Length: {len(result['path'])} cells | Cost: {result['cost']:.1f}")
    else:
        print("  No path found — goal is unreachable!")

    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────
#  MATPLOTLIB VISUALIZATION  (optional)
# ─────────────────────────────────────────────

def plot_maze(maze: list, result: dict, start: tuple, goal: tuple,
              heuristic_name: str = "Manhattan") -> None:
    """
    Plot the maze using matplotlib with color-coded cells.
    Requires: pip install matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("  matplotlib not installed. Run: pip install matplotlib")
        return

    rows, cols = len(maze), len(maze[0])
    img = np.zeros((rows, cols, 3))

    # Colors (RGB 0-1)
    C_WALL    = [0.17, 0.17, 0.16]   # dark gray
    C_OPEN    = [0.95, 0.95, 0.95]   # white-ish
    C_VISITED = [0.71, 0.83, 0.96]   # light blue
    C_PATH    = [0.89, 0.29, 0.29]   # red
    C_START   = [0.23, 0.55, 0.83]   # blue
    C_GOAL    = [0.39, 0.60, 0.13]   # green

    path_set    = set(result["path"])
    visited_set = result["visited"]

    for r in range(rows):
        for c in range(cols):
            pos = (r, c)
            if pos == start:           img[r, c] = C_START
            elif pos == goal:          img[r, c] = C_GOAL
            elif maze[r][c] == 1:      img[r, c] = C_WALL
            elif pos in path_set:      img[r, c] = C_PATH
            elif pos in visited_set:   img[r, c] = C_VISITED
            else:                      img[r, c] = C_OPEN

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, interpolation='nearest')

    # Grid lines
    ax.set_xticks([x - 0.5 for x in range(cols + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(rows + 1)], minor=True)
    ax.grid(which='minor', color='#888', linewidth=0.3)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Labels on S and G
    ax.text(start[1], start[0], 'S', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax.text(goal[1], goal[0], 'G', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    status = (f"Path found  |  Length: {len(result['path'])} cells  |  Cost: {result['cost']:.1f}"
              if result["found"] else "No path found — goal unreachable")

    ax.set_title(f"A* Maze Solver  ·  {heuristic_name} heuristic\n{status}",
                 fontsize=13, pad=12)

    legend = [
        mpatches.Patch(color=[v/1 for v in C_START],   label='Start (S)'),
        mpatches.Patch(color=[v/1 for v in C_GOAL],    label='Goal (G)'),
        mpatches.Patch(color=[v/1 for v in C_WALL],    label='Wall'),
        mpatches.Patch(color=[v/1 for v in C_VISITED], label='Visited nodes'),
        mpatches.Patch(color=[v/1 for v in C_PATH],    label='Shortest path'),
    ]
    ax.legend(handles=legend, loc='upper right', bbox_to_anchor=(1.18, 1),
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("maze_solution.png", dpi=150, bbox_inches='tight')
    print("  Plot saved as maze_solution.png")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Run with Manhattan heuristic ──
    t0 = time.perf_counter()
    result_manhattan = astar(MAZE, START, GOAL, heuristic=manhattan)
    t1 = time.perf_counter()
    print(f"  Time: {(t1-t0)*1000:.2f} ms")

    visualize(MAZE, result_manhattan, START, GOAL)

    # ── Run again with Euclidean heuristic to compare ──
    print("\n── Euclidean heuristic comparison ──")
    t0 = time.perf_counter()
    result_euclidean = astar(MAZE, START, GOAL, heuristic=euclidean)
    t1 = time.perf_counter()
    print(f"  Time: {(t1-t0)*1000:.2f} ms")

    # ── Summary comparison ──
    print(f"\n{'='*50}")
    print("  HEURISTIC COMPARISON")
    print(f"{'='*50}")
    print(f"  {'Metric':<22} {'Manhattan':>12} {'Euclidean':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Nodes visited':<22} {len(result_manhattan['visited']):>12} {len(result_euclidean['visited']):>12}")
    print(f"  {'Path length':<22} {len(result_manhattan['path']):>12} {len(result_euclidean['path']):>12}")
    print(f"  {'Path cost':<22} {result_manhattan['cost']:>12.1f} {result_euclidean['cost']:>12.1f}")
    print(f"{'='*50}\n")

    # ── Test unreachable case ──
    print("── Unreachable goal test ──")
    blocked_maze = [row[:] for row in MAZE]
    # Surround start with walls
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = START[0]+dr, START[1]+dc
        if 0 <= nr < len(blocked_maze) and 0 <= nc < len(blocked_maze[0]):
            blocked_maze[nr][nc] = 1
    result_blocked = astar(blocked_maze, START, GOAL, heuristic=manhattan)
    visualize(blocked_maze, result_blocked, START, GOAL)

    # ── Optional matplotlib plot ──
    print("── Generating matplotlib plot ──")
    plot_maze(MAZE, result_manhattan, START, GOAL, heuristic_name="Manhattan")