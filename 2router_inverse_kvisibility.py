import numpy as np
import pygame
import pandas as pd
import cv2
import time

# Load Data
data_path = "router2_rssi2.csv"
data = pd.read_csv(data_path)
x_pixel = data['x_pixel'].values
y_pixel = data['y_pixel'].values
rssi_router1_2_4 = data['router1_2.4'].values
rssi_router1_5 = data['router1_5'].values
rssi_router2_2_4 = data['router2_2.4'].values
rssi_router2_5 = data['router2_5'].values
trajectory_from_csv = list(zip(x_pixel, y_pixel, rssi_router1_2_4, rssi_router1_5, rssi_router2_2_4, rssi_router2_5))

# Constants
width, height = 900, 900
grid_size = 5
movement_speed = 5
router_points = [np.array([600, 770]), np.array([210, 350])]
#router_points = [np.array([397, 236]), np.array([830, 620])]

# Initialize Positions
robot_position = np.array([trajectory_from_csv[0][0], trajectory_from_csv[0][1]])
trajectory_grid = np.zeros((height // grid_size, width // grid_size, 3), dtype=np.uint8)

pygame.init()
window = pygame.display.set_mode((width, height))
robot_surface = pygame.Surface((10, 10))
robot_surface.fill((0, 0, 255))
router_surface = pygame.Surface((10, 10))
router_surface.fill((255, 0, 0))

# Color Map
trajectory_color_map = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)
}

occupancy_grid = 127 * np.ones((height // grid_size, width // grid_size), dtype=np.float32)
robot_trajectory = [(robot_position.tolist(), 0)]

def supercover_line(x0, y0, x1, y1):
    points = []
    dx, dy = x1 - x0, y1 - y0
    xsign, ysign = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    dx, dy = abs(int(dx)), abs(int(dy))

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        points.append((x0 + x * xx + y * yx, y0 + x * xy + y * yy))
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy

    return points


def update_grid(grid, start, end, rssi):
    updated_cells = set()
    points = supercover_line(start[0], start[1], end[0], end[1])
    robot_position_cell = (np.array(start) // grid_size).astype(int)

    k_visibility = 0 if rssi >= -41 else 1 if rssi >= -56.001 else 2
    robot_color = trajectory_color_map[k_visibility]
    trajectory_grid[robot_position_cell[0], robot_position_cell[1]] = robot_color

    k_points_indices = []

    for idx, p in enumerate(points):
        sensor_position = np.array(p)
        sensor_position_cell = (sensor_position // grid_size).astype(int)
        if 0 <= sensor_position_cell[0] < grid.shape[0] and 0 <= sensor_position_cell[1] < grid.shape[1]:
            cell_key = tuple(sensor_position_cell)
            if cell_key not in updated_cells:
                updated_cells.add(cell_key)
                if np.array_equal(trajectory_grid[sensor_position_cell[0], sensor_position_cell[1]], robot_color):
                    k_points_indices.append(idx)
                    continue

    if k_visibility == 0:
        for p in points:
            sensor_position = np.array(p)
            sensor_position_cell = (sensor_position // grid_size).astype(int)
            grid[sensor_position_cell[0], sensor_position_cell[1]] = min(255, grid[sensor_position_cell[0], sensor_position_cell[1]] + 15)
    else:
        if len(k_points_indices) <= 1:
            for p in points:
                sensor_position = np.array(p)
                sensor_position_cell = (sensor_position // grid_size).astype(int)
                if grid[sensor_position_cell[0], sensor_position_cell[1]] <= 127:
                    grid[sensor_position_cell[0], sensor_position_cell[1]] = max(0, grid[sensor_position_cell[0], sensor_position_cell[1]] - 15)
        else:
            for i in range(1, len(k_points_indices)):
                start_idx = k_points_indices[i-1]
                end_idx = k_points_indices[i]
                for j in range(start_idx + 1, end_idx):
                    sensor_position = np.array(points[j])
                    sensor_position_cell = (sensor_position // grid_size).astype(int)
                    grid[sensor_position_cell[0], sensor_position_cell[1]] = min(255, grid[sensor_position_cell[0], sensor_position_cell[1]] + 30)
    return grid

def grid_to_rgb(grid):
    grid_rgb_pre = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
    grid_rgb = cv2.resize(grid_rgb_pre, (width, height), interpolation=cv2.INTER_NEAREST)
    
    trajectory_resized = cv2.resize(trajectory_grid, (width, height), interpolation=cv2.INTER_NEAREST)
    
    mask = (trajectory_resized != [0, 0, 0]).all(axis=2)
    grid_rgb[mask] = trajectory_resized[mask]
    
    return grid_rgb

def execute_trajectory(trajectory, robot_position):
    global occupancy_grid
    computation_results = []
    for target_position in trajectory:
        rssi_values = target_position[2:]
        max_rssi = max(rssi_values)
        router_index = rssi_values.index(max_rssi) // 2
        k_visibility = 0 if max_rssi >= -41 else 1 if max_rssi >= -56.001 else 2
        direction = np.array(target_position[:2]) - robot_position
        direction_norm = np.linalg.norm(direction)
        step = direction / direction_norm * movement_speed if direction_norm != 0 else direction

        while np.linalg.norm(robot_position - np.array(target_position[:2])) > movement_speed:
            robot_position = robot_position.astype(float) + step
            robot_trajectory.append((robot_position.tolist(), k_visibility))
            robot_position_int = robot_position.astype(int)
            
            # Only store results, don't render
            occupancy_grid = update_grid(occupancy_grid, robot_position, router_points[router_index], max_rssi)
            
            # Store the occupancy grid and trajectory at each step for later rendering
            computation_results.append((occupancy_grid.copy(), robot_trajectory.copy(), robot_position_int.copy()))
    return robot_position, computation_results

start_time = time.time()  # Start the total simulation timer
robot_position, forward_results = execute_trajectory(trajectory_from_csv, robot_position)

# Reverse the trajectory
trajectory_from_csv.reverse()
robot_position, reverse_results = execute_trajectory(trajectory_from_csv, robot_position)
end_computation_time = time.time()  # End of all computations

total_computation_time = end_computation_time - start_time
print(f"Total computation time (forward + reverse): {total_computation_time:.2f} seconds")

# Combine forward and reverse results
all_results = forward_results + reverse_results

# Define desired total simulation time based on dataset size (This includes rendering)
# Example: Let’s say we want 10 seconds for 500 rows of data, and scale accordingly.
rows = len(trajectory_from_csv)
desired_total_simulation_time = max(10, rows / 50)  # Example: 10 seconds for 500 rows

# Now calculate the remaining time available for rendering after computations
remaining_time_for_rendering = desired_total_simulation_time - total_computation_time

if remaining_time_for_rendering < 0:
    remaining_time_for_rendering = 0  # If computations took longer than expected

# Calculate delay per frame so that rendering happens in the remaining time
frame_count = len(all_results)
if frame_count > 0:
    delay_per_frame = remaining_time_for_rendering / frame_count
else:
    delay_per_frame = 0  # No frames to render

# Pygame Main Loop for Rendering with Time Control
render_index = 0
running = True
while running and render_index < frame_count:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if render_index < frame_count:
        occupancy_grid, robot_trajectory, robot_position_int = all_results[render_index]
        
        rgb_grid = grid_to_rgb(occupancy_grid)
        pygame.surfarray.blit_array(window, rgb_grid)

        for i in range(1, len(robot_trajectory)):
            color = trajectory_color_map[robot_trajectory[i][1]]
            pygame.draw.line(window, color, robot_trajectory[i-1][0], robot_trajectory[i][0], 2)

        window.blit(robot_surface, tuple(robot_position_int))
        for point in router_points:
            window.blit(router_surface, tuple(point))

        pygame.display.update()

        render_index += 1

        # Adjust the rendering speed based on the remaining time
        time.sleep(delay_per_frame)  # Ensures that the total running time is distributed properly

# End of total simulation time
end_total_time = time.time()
total_running_time = end_total_time - start_time  # This is the total time from start to end

print(f"Total simulation time (including rendering): {total_running_time:.2f} seconds")

pygame.quit()