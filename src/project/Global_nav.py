import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
# -----------Map class -----------------------------------------------------------


class Map:
    def __init__(self, width, height, start, end, grid, ratio_total, ratio_downscale):
        self.width = width
        self.height = height
        self.start = start
        self.end = end
        self.grid = grid
        self.path = np.empty(0)
        self.checkpoints = np.empty(0)
        self.visited = np.empty(0)
        self.ratio_total = ratio_total
        self.ratio_downscale = ratio_downscale
        self.path_solved = False

    def run_map(self, verbose=False):
        """
        Function to run the A_star algorithm on the Map.
        :param verbose: plot the Map if true
        """
        if self.start != [] or self.end != []:
            start = self.start[:2]
            self.path_solved, self.path, self.checkpoints, self.visited = run_Astar(self.grid, start, self.end,
                                                                                    self.width, self.height)
            if verbose:
                self.create_plot()
        else:
            print("No start or end to launch A_star")

    def draw_arrow(self):
        """
        Function to draw an arrow giving the angle of robot.
        Only plot it we have the angle information
        """
        if len(self.start) == 3:
            arrow_length = self.width / self.height
            plt.arrow(self.start[0], self.start[1], arrow_length * math.cos(self.start[2]),
                      arrow_length * math.sin(self.start[2]), head_width=arrow_length / 5)

    def create_plot(self):
        """
        Function to create a figure of the desired dimensions & grid
        Then draw all elements of the map (start,end,path,visited_nodes,checkpoints)
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        if self.width < self.height:
            max_val = self.height
        else:
            max_val = self.width
        # Grid and axis definition
        major_ticks = np.arange(0, max_val + 1, 5)
        minor_ticks = np.arange(0, max_val + 1, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.set_ylim([-1, self.height])
        ax.set_xlim([-1, self.width])
        ax.grid(True)
        # Start and end definition
        self.draw_arrow()
        ax.scatter(self.start[0], self.start[1], marker="o", color='green', s=200)
        ax.scatter(self.end[0], self.end[1], marker="o", color='purple', s=200)
        # Plot the best path found and the list of visited nodes
        if self.path_solved:
            ax.scatter(self.visited[0], self.visited[1], marker="o", color='orange')
            ax.plot(self.path[0], self.path[1], marker="o", color='blue')
            ax.scatter(self.checkpoints[0], self.checkpoints[1], marker="o", color='cyan', s=80)
        # Plot the Obstacles
        cmap = colors.ListedColormap(
            ['white', 'red'])  # Select the colors with which to display obstacles and free cells
        ax.imshow(self.grid.transpose(), cmap=cmap)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        plt.show()
# ---------------------------------------------------------------------------------------


# -----------_get_movements_8n-------------------------------------------------------------------

def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1)
    (up, down, left, right and the 4 diagonals).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]
# ---------------------------------------------------------------------------------------


# -----------reconstruct_path-------------------------------------------------------------------

def reconstruct_path(cameFrom, current):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node immediately
                     preceding it on the cheapest path from start to n
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    total_path = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        total_path.insert(0, cameFrom[current])
        current = cameFrom[current]
    return total_path
# ---------------------------------------------------------------------------------------


# -----------A_Star-------------------------------------------------------------------

def A_Star(start, goal, h, coords, occupancy_grid, width, height, movement_type="8N"):
    """
    A* for 2D occupancy grid. Finds a path from start to goal.
    :param start: start node (x, y)
    :param goal: goal node (x, y)
    :param h: estimates the cost to reach goal from node n.
    :param coords: matrix of the size of the Map
    :param width: width of the Map
    :param height: height of the Map
    :param occupancy_grid: the grid map
    :param movement_type: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """

    # -----------------------------------------
    # DO NOT EDIT THIS PORTION OF CODE
    # -----------------------------------------

    assert 0 <= start[0] < width and 0 <= start[1] < height, "start not contained in the map"
    assert 0 <= goal[0] < width and 0 <= goal[1] < height, "end goal not contained in the map"
    # check if start and goal nodes correspond to free spaces
    if occupancy_grid[start[0], start[1]]:
        print("Start node is not traversable")
        return [], []

    if occupancy_grid[goal[0], goal[1]]:
        print("Goal node is not traversable")
        return [], []

    # get the possible movements corresponding to the selected connectivity
    if movement_type == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')

    # --------------------------------------------------------------------------------------------
    # A* Algorithm implementation - feel free to change the structure / use another pseudo-code
    # --------------------------------------------------------------------------------------------

    # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
    # Initially, only the start node is known.
    openSet = [start]

    # The set of visited nodes that no longer need to be expanded.
    closedSet = []

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]

    # while there are still elements to investigate
    while openSet != []:

        # the node in openSet having the lowest fScore[] value
        fScore_openSet = {key: val for (key, val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet

        # If the goal is reached, reconstruct and return the obtained path
        if current == goal:
            return reconstruct_path(cameFrom, current), closedSet

        openSet.remove(current)
        closedSet.append(current)

        # for each neighbor of current:
        for dx, dy, deltacost in movements:

            neighbor = (current[0] + dx, current[1] + dy)

            # if the node is not in the map, skip
            if (neighbor[0] >= occupancy_grid.shape[0]) or (neighbor[1] >= occupancy_grid.shape[1]) or (
                    neighbor[0] < 0) or (neighbor[1] < 0):
                continue

            # if the node is occupied or has already been visited, skip
            if (occupancy_grid[neighbor[0], neighbor[1]]) or (neighbor in closedSet):
                continue

            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore = gScore[current] + deltacost

            if neighbor not in openSet:
                openSet.append(neighbor)

            if tentative_gScore < gScore[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]

    # Open set is empty but goal was never reached
    print("No path found to goal")
    return [], closedSet
# ---------------------------------------------------------------------------------------


# -----------checkpoints_path------------------------------------------------------------

def checkpoints_path(path):
    """
    Calculation of the checkpoints
    :param path: calculated path
    """
    steps = np.size(path, 1)
    checkpoints = np.reshape(path[:, 0], (2, -1))
    for i in range(steps):
        nb_i = np.reshape(path[:, i], (2, -1))
        if i == (steps - 1):
            checkpoints = np.append(checkpoints, nb_i, axis=1)
        elif i != 0:
            if (((path[0, (i + 1)] - path[0, i]) != (path[0, i] - path[0, (i - 1)])) or (
                    (path[1, (i + 1)] - path[1, i]) != (path[1, i] - path[1, (i - 1)]))):
                checkpoints = np.append(checkpoints, nb_i, axis=1)
    return checkpoints
# ---------------------------------------------------------------------------------------


# -----------run_Astar-------------------------------------------------------------------

def run_Astar(occupancy_grid, start, goal, width, height):
    """
    Initialisation and run the A*star algorithm
    :param occupancy_grid: Matrix containing the obstacles
    :param start: (X,Y) starting point
    :param goal: (X,Y) end point
    :param width: width of the map
    :param height: height of the map
    """
    # List of all coordinates in the grid
    x, y = np.mgrid[0:width:1, 0:height:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0] * x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])
    # print(coords)

    # Define the heuristic, here = distance to goal ignoring obstacles
    h = np.linalg.norm(pos - goal, axis=-1)
    h = dict(zip(coords, h))

    # Run the A* algorithm
    path, visitedNodes = A_Star(start, goal, h, coords, occupancy_grid, width, height, movement_type="8N")
    if path != []:
        path = np.array(path).reshape(-1, 2).transpose()
        visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()
        checkpoints = checkpoints_path(path)
        verbose = True
        return verbose, path, checkpoints, visitedNodes
    else:
        verbose = False
        return verbose, [], [], []
# ---------------------------------------------------------------------------------------


# -----------box_range-------------------------------------------------------------------

def box_range(coord, coord_size, Margin):
    """
    Find the smallest box around the obstacle including the Margin.
    :param coord: X or Y coordinate of the obstacle
    :param coord_size: either width or height depending of the coord analyzed
    :param Margin: Margin taken for the size of the thymio
    """
    coord_low = np.arange((coord - Margin), coord)
    coord_high = np.arange(coord, (coord + Margin + 1))
    coord_tot = np.concatenate((coord_low, coord_high), 0)
    coord_tot = coord_tot[coord_tot >= 0]
    coord_tot = coord_tot[coord_tot < coord_size]
    return coord_tot
# ---------------------------------------------------------------------------------------


# -----------Obstacles_real--------------------------------------------------------------

def Obstacles_real(size_thymio, size_pixel, grid, width, height):
    """
    Augment the size of each obstacle, to palliate the problem of point size robot
    :param size_thymio: size in pixels of the thymio
    :param size_pixel: size of a pixel
    :param grid: Matrix containing the obstacles
    :param width: width of the map
    :param height: height of the map
    """
    Margin_px = size_thymio / size_pixel
    temp_grid = grid.copy()
    Obs_coords = np.nonzero(grid > 0.5)
    steps = np.size(Obs_coords, 1)
    for i in range(steps):
        x_i = Obs_coords[0][i]
        y_i = Obs_coords[1][i]
        range_x = box_range(x_i, width, Margin_px)
        range_y = box_range(y_i, height, Margin_px)
        for j in range_x:
            for k in range_y:
                if np.linalg.norm((x_i - j, y_i - k)) < Margin_px:
                    temp_grid[int(j), int(k)] = 1
    return temp_grid

# # Size of Map and start,end
# width = 60
# height = 50
# start = (0,0)
# end = (49,49)
# size_thymio = 1.42
# size_pixel = 1
# # random obstacles
# np.random.seed(0) # To guarantee the same outcome on all computers
# data = np.random.rand(width, height) * 100 # Create a grid of width x height random values
# # Converting the random values into occupied and free cells
# limit = 95
# occupancy_grid = data.copy()
# occupancy_grid[data>limit] = 1
# occupancy_grid[data<=limit] = 0
# grid2 = Obstacles_real(size_thymio,size_pixel,occupancy_grid,width,height)
# m = Map(width,height,start,end,grid2)
# m.run_map(True)
