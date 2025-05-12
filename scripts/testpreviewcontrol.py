import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.linalg import solve_discrete_are

def generate_path(x_points, y_points, num_samples=500, num_interpolated=200):
    spline = CubicSpline(x_points, y_points)
    x_smooth = np.linspace(min(x_points), max(x_points), num_samples)
    y_smooth = spline(x_smooth)

    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)

    s_interp = np.linspace(0, s[-1], num_interpolated)
    interp_x = interp1d(s, x_smooth, kind='linear', fill_value='extrapolate')
    interp_y = interp1d(s, y_smooth, kind='linear', fill_value='extrapolate')
    x_uniform = interp_x(s_interp)
    y_uniform = interp_y(s_interp)

    dx_interp = np.gradient(x_uniform)
    dy_interp = np.gradient(y_uniform)
    headings = np.arctan2(dy_interp, dx_interp)

    return x_smooth, y_smooth, x_uniform, y_uniform, headings

def calculate_curvature(x, y):
    curvatures = np.zeros(len(x))
    for i in range(1, len(x)-1):
        dx1 = x[i] - x[i-1]
        dy1 = y[i] - y[i-1]
        dx2 = x[i+1] - x[i]
        dy2 = y[i+1] - y[i]

        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)
        dtheta = angle2 - angle1
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        dist = np.hypot(dx1, dy1)
        if dist > 1e-6:
            curvatures[i] = dtheta / dist

    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]
    return curvatures

class PreviewController:
    def __init__(self, v=1.0, dt=0.1, preview_steps=20):
        self.v = v
        self.dt = dt
        self.preview_steps = preview_steps
        self.kd = 0.01
        self.etheta_threshold = 0.2
        self.A = np.array([[0, 1, 0, 0], [0, 0, v, 0], [0, 0, 0, 1], [0, 0, 0, -self.kd]])
        self.B = np.array([[0], [0], [0], [1]])
        self.D = np.array([[0], [-v**2], [-v], [0]])
        self.Q = np.diag([0.05,   0.001, 0.05, 1])  
        self.R = np.array([[0.05]])
        self.calc_gains()
        self.prev_ey = 0
        self.prev_etheta = 0


        

    def calc_gains(self):
        A_d = np.eye(4) + self.A * self.dt
        B_d = self.B * self.dt
        D_d = self.D * self.dt
        Q_d = self.Q * self.dt
        R_d = self.R / self.dt

        P = solve_discrete_are(A_d, B_d, Q_d, R_d)
        lambda0 = A_d.T @ np.linalg.inv((np.eye(4) + P @ B_d @ np.linalg.inv(R_d)@ B_d.T)) 
        self.Kb = np.linalg.inv(R_d + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
        self.Pc = np.zeros((4,self.preview_steps+1))
        for i in range(self.preview_steps + 1):
            Pc_column = (np.linalg.matrix_power(lambda0, i) @ P @ D_d)
            self.Pc[:, i] = Pc_column.flatten() 
        
        top = np.hstack([np.zeros((self.preview_steps, 1)), np.eye(self.preview_steps)])
    
        bottom = np.zeros((1, self.preview_steps+1))
        self.Lmatrix = np.vstack([top, bottom])
        Kf_term = P @ D_d + self.Pc @ self.Lmatrix
        self.Kf = np.linalg.inv(R_d + B_d.T @ P @ B_d) @ B_d.T @ Kf_term

    def compute_control(self, x_r, y_r, theta_r, path_x, path_y, path_theta, path_curv, idx):
        x_ref = path_x[idx]
        y_ref = path_y[idx]
        theta_ref = np.arctan2(y_ref - y_r, x_ref - x_r)

        ey =  np.sin(theta_r) * (x_r - x_ref) - np.cos(theta_r) * (y_r - y_ref)
        
        etheta = theta_r - theta_ref
        print("X_ref ",x_ref," Y_ref ",y_ref," theta ",etheta)
        print("Robot X ",x_r,"Y  ",y_r)
        print("Ey ",ey)
        eydot = (ey - self.prev_ey) / self.dt
        ethetadot = (etheta - self.prev_etheta) / self.dt

        self.prev_ey = ey
        self.prev_etheta = etheta
        x_state = np.array([ey, eydot, etheta, ethetadot])
        
        preview_curv = path_curv[idx:idx + self.preview_steps + 1]
        if len(preview_curv) < self.preview_steps + 1:
            preview_curv = np.pad(preview_curv, (0, self.preview_steps + 1 - len(preview_curv)), 'edge')

        u_fb = -self.Kb @ x_state
        u_ff = -self.Kf @ preview_curv
        omega = u_fb + u_ff

        if abs(etheta) < self.etheta_threshold:
            omega *= 0.1
        
        omega = u_fb + u_ff
        print("Omega ",omega)
        return np.clip(omega.item(), -2, 2), x_state, u_fb.item(), u_ff.item()
    
def draw_triangle(ax, x, y, heading, size=0.2, color='r'):
    triangle = np.array([[0, -size/2], [0, size/2], [size, 0]])
    R = np.array([[np.cos(heading), -np.sin(heading)], 
                  [np.sin(heading), np.cos(heading)]])
    rotated_triangle = (R @ triangle.T).T + np.array([x, y])
    ax.fill(rotated_triangle[:, 0], rotated_triangle[:, 1], color=color, alpha=0.7)

def main():
    x_pts = np.linspace(0, 10, 6)
    y_pts = np.array([0, 6, 2, 4, 2, 8])
    x_smooth, y_smooth, x_path, y_path, headings = generate_path(x_pts, y_pts)
    curvatures = calculate_curvature(x_path, y_path)

    x_r, y_r, theta_r = 1, 2, headings[0]
    controller = PreviewController(v=0.4, dt=0.1, preview_steps=10)
    total = len(x_path)
    
    x_traj, y_traj = [x_r], [y_r]
    omega_history = []
    time_history = []
    state_history = []
    u_fb_history = []
    u_ff_history = []
    steps = 500

    for t in range(steps):
        dist_to_path = np.hypot(x_path - x_r, y_path - y_r)
        closest_idx = np.argmin(dist_to_path)
        if closest_idx + 3 < total:
            closest_idx = closest_idx + 1

        omega, state, u_fb, u_ff = controller.compute_control(x_r, y_r, theta_r, x_path, y_path, headings, curvatures, closest_idx)
        x_r += controller.v * np.cos(theta_r) * controller.dt
        y_r += controller.v * np.sin(theta_r) * controller.dt
        theta_r += omega * controller.dt
        theta_r = np.arctan2(np.sin(theta_r), np.cos(theta_r))
        if t%2==0:
            x_traj.append(x_r)
            y_traj.append(y_r)
            omega_history.append(omega)
            time_history.append(t * controller.dt)
            state_history.append(state)
            u_fb_history.append(u_fb)
            u_ff_history.append(u_ff)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(x_smooth, y_smooth, 'b-', label='Reference Path')
    ax1.plot(x_traj, y_traj, 'r--', label='Tracked Path')
    ax1.scatter(x_traj[0], y_traj[0], c='y', marker='o', label='Start Position')

    for i in range(len(x_path)):
        draw_triangle(ax1, x_path[i], y_path[i], headings[i], color='b')

    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Path Following for Differential Drive Robot')
    ax1.legend()
    ax1.grid()
    ax1.axis('equal')

    ax2.plot(time_history, omega_history, 'y-', label='Omega (yaw rate)')
    ax2.plot(time_history, u_fb_history, 'b--', label='Feedback control')
    ax2.plot(time_history, u_ff_history, 'r--', label='Feedforward control')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Omega (rad/s)')
    ax2.set_title('Control Input Over Time')
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()