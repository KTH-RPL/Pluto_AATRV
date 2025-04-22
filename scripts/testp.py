import numpy as np
import matplotlib.pyplot as plt

class PathFollower:
    def __init__(self):
        # Straight line path along X-axis from (0,0) to (10,0)
        self.current_path = [(x, np.sin(x)) for x in np.linspace(0, 10, 50)]
        self.lookahead_distance = 1.5
        self.current_pose = [2.7, 0.5]  # (x, y)

    def find_closest_point(self, path, current_pos):
        robot_x, robot_y = current_pos
        ahead_points = [(i, point) for i, point in enumerate(path)]

        closest_idx, closest_point = min(
            ahead_points, key=lambda x: np.hypot(x[1][0]-robot_x, x[1][1]-robot_y)
        )
        return closest_idx

    def prune_passed_points(self, path, closest_idx):
        return path[closest_idx:]

    def find_lookahead_point(self, path, current_pos, closest_idx):
        lookahead_dist = self.lookahead_distance

        for i in range(closest_idx, len(path)):
            dist = np.hypot(path[i][0] - current_pos[0], path[i][1] - current_pos[1])
            if dist >= lookahead_dist:
                return path[i], i

        return path[-1], len(path) - 1

    def chkside(self, cp_idx, pose):
        x1, y1 = self.current_path[cp_idx]
        x2, y2 = self.current_path[cp_idx + 1]

        m = -(x2 - x1) / (y1 - y2) if (y1 - y2) != 0 else np.inf
        ineq = pose[1] - y1 + (pose[0] / m) - (x1 / m) if m != np.inf else pose[0] - x1

        if ineq > 0:
            return 1  # one side
        else:
            return 0  # other side

    def plot_scene(self, closest_idx, lookahead_point, lookahead_idx, chkside_result):
        path = np.array(self.current_path)
        plt.plot(path[:, 0], path[:, 1], 'b-', label='Path')
        plt.plot(self.current_pose[0], self.current_pose[1], 'ro', label='Current Pose')
        plt.plot(path[closest_idx, 0], path[closest_idx, 1], 'go', label='Closest Point')
        plt.plot(lookahead_point[0], lookahead_point[1], 'yo', label='Lookahead Point')

        plt.title(f'Closest Point: {closest_idx}, chkside: {chkside_result}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def run(self):
        closest_idx = self.find_closest_point(self.current_path, self.current_pose)
        lookahead_point, lookahead_idx = self.find_lookahead_point(self.current_path, self.current_pose, closest_idx)
        chkside_result = self.chkside(closest_idx, self.current_pose)
        self.plot_scene(closest_idx, lookahead_point, lookahead_idx, chkside_result)

# Run the simulation
follower = PathFollower()
follower.run()