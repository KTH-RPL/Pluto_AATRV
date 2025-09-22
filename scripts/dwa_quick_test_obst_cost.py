import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import solve_discrete_are
import random
from scipy.ndimage import gaussian_filter1d
# *** CHANGED ***: need 2D gaussian filter
from scipy.ndimage import gaussian_filter  # *** CHANGED ***

class PreviewController:
    def __init__(self, v=0.8, dt=0.05, preview_steps=20):
        self.v = v
        self.dt = dt
        self.preview_steps = preview_steps
        self.omega = 0
        self.kd = 2
        self.etheta_threshold = 0.2
        self.A = np.array([[0, 1, 0], [0, 0, v], [0, 0, 0]])
        self.B = np.array([[0], [0], [1]])
        self.D = np.array([[0], [-v**2], [-v]])
        self.Q = np.diag([7, 5, 3])
        self.R = np.array([[0.001]])
        self.calc_gains()
        self.prev_v = 0
        self.prev_ey = 0
        self.prev_etheta = 0
        self.max_domega = 0.8
        self.prev_omega = 0
        self.max_omega = 0.8  

    def calc_gains(self):
        self.A = np.array([[0, 1, 0], [0, 0, self.v+0.001], [0, 0, 0]])
        self.B = np.array([[0], [0], [1]])
        self.D = np.array([[0], [-1*((self.v+0.001)**2)], [-(self.v+0.001)]])
        A_d = np.eye(3) + self.A * 0.01
        B_d = self.B * 0.01
        D_d = self.D * 0.01
        Q_d = self.Q * 0.01
        R_d = self.R / 0.01

        P = solve_discrete_are(A_d, B_d, Q_d, R_d)
        lambda0 = A_d.T @ np.linalg.inv((np.eye(3) + P @ B_d @ np.linalg.inv(R_d) @ B_d.T))
        self.Kb = np.linalg.inv(R_d + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
        self.Pc = np.zeros((3, self.preview_steps+1))
        for i in range(self.preview_steps + 1):
            Pc_column = (np.linalg.matrix_power(lambda0, i) @ P @ D_d)
            self.Pc[:, i] = Pc_column.flatten()
        
        top = np.hstack([np.zeros((self.preview_steps, 1)), np.eye(self.preview_steps)])
        bottom = np.zeros((1, self.preview_steps+1))
        self.Lmatrix = np.vstack([top, bottom])
        Kf_term = P @ D_d + self.Pc @ self.Lmatrix
        self.Kf = np.linalg.inv(R_d + B_d.T @ P @ B_d) @ B_d.T @ Kf_term

    def compute_control(self, x_r, y_r, theta_r, x_ref, y_ref, theta_ref, path_curv):
        theta_er = np.arctan2(y_ref - y_r, x_ref - x_r)

        ey = np.sin(theta_ref) * (x_r - x_ref) - np.cos(theta_ref) * (y_r - y_ref)

        ey = -ey
        
        etheta = - theta_ref +  theta_r
        if etheta > np.pi:
            etheta -= 2 * np.pi
        elif etheta < -np.pi:
            etheta += 2 * np.pi
        eydot =  self.v * etheta
        print(f"ey: {ey:.2f}, theta_ref: {theta_ref:.2f}, theta_r: {theta_r:.2f} ,etheta: {etheta:.2f}, eydot: {eydot:.2f}")
        self.prev_ey = ey
        self.prev_etheta = etheta
        x_state = np.array([ey, eydot, etheta])
        
        preview_curv = path_curv[0:0 + self.preview_steps + 1]
        if len(preview_curv) < self.preview_steps + 1:
            preview_curv = np.pad(preview_curv, (0, self.preview_steps + 1 - len(preview_curv)), 'edge')
        self.calc_gains()
        u_fb = -self.Kb @ x_state
        u_ff = -self.Kf @ preview_curv
        omega = (u_fb + u_ff).item()
        
        
        if omega > self.omega:
            omega = np.clip(omega, self.omega, self.omega + self.max_domega * self.dt)
        else:
            omega = np.clip(omega, self.omega - self.max_domega * self.dt, self.omega)
        
        omega = np.clip(omega, -self.max_omega, self.max_omega)
        self.prev_omega = omega
        
        return omega, x_state, u_fb.item(), u_ff.item(), ey


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

def generate_path():
    path = []
    orientations = []  
    x, y = 0, 0
    path.append((x, y))
    orientations.append(0)  
    
    max_spirals = 3  
    points_per_spiral = 50  
    max_radius = 3  
    min_radius = 1  
    
    for i in range(max_spirals * points_per_spiral):
        progress = i / (max_spirals * points_per_spiral)
        radius = min_radius + (max_radius - min_radius) * np.abs(np.sin(2 * np.pi * progress * 2))
        
        angle = 2 * np.pi * i / points_per_spiral
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        path.append((x, y))
        
        if i > 0:
            dx = x - path[-2][0]
            dy = y - path[-2][1]
            orientation = np.arctan2(dy, dx)
            orientation = (orientation + np.pi) % (2 * np.pi) - np.pi
            orientations.append(orientation)
        else:
            orientations.append(0)
    
    return path, orientations

def generate_snake_path():
    
    amplitude = 14
    wavelength = 20
    length = 40
    point_spacing = 0.3
    num_points = int(np.ceil(length / point_spacing)) + 1
    x = np.linspace(0, length, num_points)
    
    y = amplitude * np.sin(2 * np.pi * x / wavelength)
    
    path = []
    orientations = []
    
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    for i in range(len(x)):
        path.append((x[i], y[i]))
        orientation = np.arctan2(dy[i], dx[i])
        orientation = (orientation + np.pi) % (2 * np.pi) - np.pi  
        orientations.append(orientation)
    
    return path,orientations



class CarSimulation:
    def __init__(self):
        self.path, self.orientations = generate_snake_path()
        self.x_path = np.array([p[0] for p in self.path])
        self.y_path = np.array([p[1] for p in self.path])
        self.curvatures = calculate_curvature(self.x_path, self.y_path)
        
        self.ref_velocity = 1.5  
        self.controller = PreviewController(v=self.ref_velocity, dt=0.1, preview_steps=3)
        self.controller.v = self.ref_velocity
        self.controller.omega = 0
        self.x = 6
        self.y = -5
        self.theta = np.pi/1.5
        
        self.history_x = [self.x]
        self.history_y = [self.y]
        self.history_theta = [self.theta]
        self.history_time = [0]
        self.history_omega = [0]
        self.history_ey = [0]
        self.history_etheta = [0]
        self.history_eydot = [0]
        self.history_omega_fb = [0]
        
        self.target_idx = 0
        self.max_target_idx = len(self.path) - 1
        
        self.fig = plt.figure(figsize=(9, 4))
        gs = self.fig.add_gridspec(1,1)
        
        self.ax_path = self.fig.add_subplot(gs[:, 0])
        self.ax_path.set_aspect('equal')
        self.ax_path.grid(True)
        

        self.target_x = self.x_path[self.target_idx]
        self.target_y = self.y_path[self.target_idx]
        self.max_speed = 0.9
        self.min_speed = 0.05
        self.max_accel = 0.6
        self.max_domega = 1.2
        self.robot_radius = 1.2
        self.predict_time = 2
        self.dt_dwa = 0.1
        self.max_omega = 1.2
        self.obstacles = self.generate_obstacles()
        # *** CHANGED ***: costmap parameters
        self.costmap_resolution = 0.1  # meters per cell (*** CHANGED ***)
        self.costmap_margin = 5.0      # meters beyond world extents to include (*** CHANGED ***)
        # *** CHANGED ***: build costmap from obstacles with inflation
        self.build_costmap()  # *** CHANGED ***
        # *** CHANGED END ***

        self.live_plots_enabled = True
        if self.live_plots_enabled:
            self.live_fig, self.live_axes = plt.subplots(4, 1, figsize=(4, 4), sharex=True)
            

    def update(self, frame):
        if len(self.obstacles) != 0:
            self.update_obstacles()
            # *** CHANGED ***: rebuild costmap when dynamic obstacles move (minimal update)
            self.build_costmap()  # *** CHANGED ***

        while(self.distancecalc(self.x, self.y, self.x_path[self.target_idx], self.y_path[self.target_idx]) 
            and self.target_idx < self.max_target_idx):
            self.target_idx += 1
        while(self.target_idx + 1 < self.max_target_idx and 
            self.chkside(self.x, self.y, self.target_idx, self.orientations[self.target_idx])):
            self.target_idx += 1
        
        if self.target_idx >= self.max_target_idx:
            return
        
        self.target_x = self.x_path[self.target_idx]
        self.target_y = self.y_path[self.target_idx]
        target_theta = self.orientations[self.target_idx]
        preview_omega, x_state, omega_fb, omega_ff, ey = self.controller.compute_control(
            self.x, self.y, self.theta, 
            self.target_x, self.target_y, target_theta,
            self.curvatures[self.target_idx:]
            )
        

        dwa_v, dwa_omega, obsi, mindist = self.dwa_control()

        if mindist < 2*self.obstacles[obsi]['radius'] + 2*self.robot_radius:
            self.controller.omega = dwa_omega
            self.controller.v = dwa_v  
            print(f"DWA Control: v={dwa_v:.2f}, omega={dwa_omega:.2f}, mindist={mindist:.2f} (obstacle index: {obsi})")
        else:
            v_prev = self.controller.v
            v_target = self.ref_velocity
            max_accel = self.max_accel * self.controller.dt
            if v_prev < v_target:
                self.controller.v = min(v_prev + max_accel, v_target)
            else:
                self.controller.v = max(v_prev - max_accel, v_target)            
            self.controller.omega = preview_omega
            print(f"Preview Control: v={self.controller.v:.2f}, omega={self.controller.omega:.2f}, ey={ey:.2f}, etheta={x_state[2]:.2f}, mindist={mindist:.2f} (obstacle index: {obsi})")

        self.theta += self.controller.omega * self.controller.dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.x += self.controller.v * np.cos(self.theta) * self.controller.dt 
        self.y += self.controller.v * np.sin(self.theta) * self.controller.dt

        current_time = frame * self.controller.dt
        self.history_x.append(self.x)
        self.history_y.append(self.y)
        self.history_theta.append(self.theta)
        self.history_time.append(current_time)
        self.history_omega.append(self.controller.omega)
        self.history_ey.append(ey)
        self.history_etheta.append(x_state[2])  
        self.history_eydot.append(x_state[1]) 
        self.history_omega_fb.append(omega_fb)
        self.ax_path.clear()
        self.ax_path.plot(self.x_path, self.y_path, 'b-', label='Planned Path', linewidth=1)
        self.ax_path.plot(self.history_x, self.history_y, 'r-', label='Actual Path', linewidth=2)
        robot_circle = plt.Circle((self.x,self.y), self.robot_radius, color='r', fill=True)
        self.ax_path.add_patch(robot_circle)
        for obs in self.obstacles:
            circle = plt.Circle((obs['x'], obs['y']), obs['radius'], color='k', fill=True)
            self.ax_path.add_patch(circle)

        # *** CHANGED ***: optional plotting of costmap background (semi-transparent)
        try:
            # draw costmap as image (centered)
            extent = [self.costmap_origin_x, self.costmap_origin_x + self.costmap_width * self.costmap_resolution,
                      self.costmap_origin_y, self.costmap_origin_y + self.costmap_height * self.costmap_resolution]
            self.ax_path.imshow(self.costmap.T, origin='lower', extent=extent, alpha=0.4)
        except Exception:
            pass
        # *** CHANGED END ***

        arrow_len = 1.0
        dx = arrow_len * np.cos(self.theta)
        dy = arrow_len * np.sin(self.theta)
        self.ax_path.arrow(self.x, self.y, dx, dy, head_width=0.5, head_length=0.5, fc='k', ec='k')
        self.ax_path.plot(self.target_x, self.target_y, 'go', markersize=6, label='Target')
        self.ax_path.set_xlim(self.x - 10, self.x + 10)
        self.ax_path.set_ylim(self.y - 10, self.y + 10)
        self.ax_path.set_xlabel('X position (m)')
        self.ax_path.set_ylabel('Y position (m)')
        self.ax_path.set_title(f'Path Following (Time: {current_time:.1f}s)')
        self.ax_path.legend()
        self.ax_path.grid(True)

       
        plt.tight_layout()
        if self.live_plots_enabled:
            self.update_live_plots()

    def update_live_plots(self):
        t = self.history_time
        self.live_axes[0].cla()
        self.live_axes[0].plot(t, self.history_ey, label='ey')
        self.live_axes[0].set_ylabel('ey')
        self.live_axes[0].legend()
        self.live_axes[1].cla()
        self.live_axes[1].plot(t, self.history_etheta, label='etheta')
        self.live_axes[1].set_ylabel('etheta')
        self.live_axes[1].legend()
        self.live_axes[2].cla()
        self.live_axes[2].plot(t, self.history_omega, label='omega (total)')
        self.live_axes[2].set_ylabel('omega')
        self.live_axes[2].legend()
        self.live_axes[3].cla()
        self.live_axes[3].plot(t, self.history_omega_fb, label='omega_fb (LQR)')
        self.live_axes[3].set_ylabel('omega_fb')
        self.live_axes[3].set_xlabel('Time (s)')
        self.live_axes[3].legend()
        plt.tight_layout()
        plt.pause(0.00001)

    def generate_obstacles(self):
        obstacles = []
        # obstacles.append({'x': 1.33, 'y': 4.1, 'vx': -0.2, 'vy':-0.6, 'radius': 1.2})
        # obstacles.append({'x': 0.9, 'y': 5.6, 'vx': -0.3, 'vy':-0.8, 'radius': 0.5})
        obstacles.append({'x': 5, 'y': 0, 'vx': -0.3, 'vy':-0.5, 'radius': 1.2})
        obstacles.append({'x': 1, 'y': 2.5, 'vx': 0.0, 'vy':0.0, 'radius': 1.5})
        return obstacles
    
    def distancecalc(self, x1, y1, x2, y2):
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if dist < 0.8:
            return 1
        else:
            return 0    
    def chkside(self,x,y,id,orientation):
        x1 = self.x_path[id]
        y1 = self.y_path[id]
        

        m = -1/np.tan(orientation)
        ineq = y - (m*x) - (y1)+(m*x1) 

        if ineq > 0:
            t = 1 
            if orientation < 0:
                t = 0  
        else:
            t = 0
            if orientation < 0:
                t = 1
        return t
    
    def update_obstacles(self):
        for obs in self.obstacles:
            obs['x'] += obs['vx'] * self.controller.dt
            obs['y'] += obs['vy'] * self.controller.dt
            
            if obs['x'] < -10 or obs['x'] > 20:
                obs['vx'] *= -1
            if obs['y'] < -10 or obs['y'] > 10:
                obs['vy'] *= -1
    
    def motion_model(self, x, y, theta, v, omega ,dt):
        theta += omega * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        return x, y, theta
    
    def closest_point(self, x, y):
        min_dist = float('inf')
        closest_idx = 0
        for i in range(len(self.x_path)):
            dist = np.sqrt((x - self.x_path[i])**2 + (y - self.y_path[i])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return closest_idx
    
    def calc_dynamic_window(self):
        Vs = [self.min_speed, self.max_speed, 
              -self.max_omega, self.max_omega]
        
        Vd = [self.controller.v - self.max_accel * self.dt_dwa,
              self.controller.v + self.max_accel * self.dt_dwa,
              self.controller.omega - self.max_domega * self.dt_dwa,
              self.controller.omega + self.max_domega * self.dt_dwa]
        
        vr = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return vr
    
    def calc_trajectory(self, x, y, theta, v, omega):
        traj = []
        for _ in range(int(self.predict_time / self.dt_dwa)):
            x, y, theta = self.motion_model(x, y, theta, v, omega,self.dt_dwa)
            traj.append([x, y,theta])
        return np.array(traj)
    
    # *** CHANGED ***: replace calc_obstacle_cost to sample costmap along trajectory
    def calc_obstacle_cost(self, traj):
        """
        New cost: sample self.costmap along the trajectory and accumulate cost.
        Also still compute min distance to actual circular obstacles for thresholds/backwards compatibility.
        """
        if not hasattr(self, 'costmap'):
            # fallback to old behavior (should not happen)
            return 0, 0, float('inf')

        # Sample costmap at each trajectory point (bilinear interpolation)
        cost_sum = 0.0
        for pt in traj:
            x_pt, y_pt = pt[0], pt[1]
            cost_sum += self.get_cost_at(x_pt, y_pt)

        # Normalize cost a bit by number of points
        cost_avg = cost_sum / max(1, len(traj))

        # find closest obstacle index and actual min distance (retain from earlier approach)
        min_dist_actual = float('inf')
        obsi = 0
        for i, obs in enumerate(self.obstacles):
            # distance to obstacle center
            dx = traj[-1][0] - obs['x']
            dy = traj[-1][1] - obs['y']
            dist = np.hypot(dx, dy)
            if dist < min_dist_actual:
                min_dist_actual = dist
                obsi = i

        # scale cost so it can be combined with other DWA costs (tunable)
        scaled_cost = 200.0 * cost_avg  # *** CHANGED: scaling factor ***
        return scaled_cost, obsi, min_dist_actual
    # *** CHANGED END ***

    def away_from_obstacle_cost(self, traj):
        cost = 0.0
        for pt in traj:
            x, y = pt[0], pt[1]
            # costmap value
            c = self.get_cost_at(x, y)
            # Exponential penalty for high-cost areas
            cost += np.exp(5 * c)  # 5 is tunable
        return cost / len(traj)
    
    def calc_speed_ref_cost(self, v):
        return 1*np.abs((v - self.ref_velocity)) 
    
    def calc_heading_cost(self, traj):
        traj_point = traj[-1]               
        theta_error = np.abs(-traj_point[2] + self.orientations[self.target_idx] )  
        return theta_error if theta_error < np.pi else 2 * np.pi - theta_error  

    def calc_path_cost(self, traj):
        min_dist = float('inf')
        eyavg = 0
        traj_point = traj[-1]
        i = self.closest_point(traj_point[0], traj_point[1])
        target_x = self.x_path[i]
        target_y = self.y_path[i]
        theta_ref = self.orientations[i]
        ey = np.sin(theta_ref) * (self.x - target_x) - np.cos(theta_ref) * (self.y - target_y)
        
        return np.abs(ey)         
    
    def lookahead_cost(self, traj,x,y):
        traj_point = traj[-1]
        ed = (traj_point[0] - x)**2 + (traj_point[1] - y)**2
        ed = np.sqrt(ed)
        
        return 1*np.abs(ed)


    def run_simulation(self):
        total_time = 60  
        frames = int(total_time / self.controller.dt)
        
        ani = animation.FuncAnimation(
            self.fig, 
            self.update, 
            frames=frames, 
            interval=5,  
            repeat=False
        )
        plt.show()

     
    # *** CHANGED ***: new costmap utilities
    def build_costmap(self):
        """
        Build a 2D costmap raster from current self.obstacles.
        Rasterizes obstacles as circles, applies Gaussian blur for inflation,
        computes gradient field implicitly by allowing the cost to 'leak' from obstacle boundaries.
        """
        # Determine world extents from path and obstacles
        all_x = np.concatenate([self.x_path, np.array([o['x'] for o in self.obstacles])])
        all_y = np.concatenate([self.y_path, np.array([o['y'] for o in self.obstacles])])
        min_x, max_x = all_x.min() - self.costmap_margin, all_x.max() + self.costmap_margin
        min_y, max_y = all_y.min() - self.costmap_margin, all_y.max() + self.costmap_margin

        # grid size
        width = max_x - min_x
        height = max_y - min_y
        nx = int(np.ceil(width / self.costmap_resolution))
        ny = int(np.ceil(height / self.costmap_resolution))
        # Ensure at least 10x10
        nx = max(nx, 10)
        ny = max(ny, 10)

        # store origin and dims for mapping
        self.costmap_origin_x = min_x
        self.costmap_origin_y = min_y
        self.costmap_width = nx
        self.costmap_height = ny
        self.costmap_resolution = float(self.costmap_resolution)

        # create empty occupancy (0..1)
        occ = np.zeros((nx, ny), dtype=float)

        # rasterize obstacles as hard occupancy
        for obs in self.obstacles:
            # center in grid coordinates
            cxg, cyg = self.world_to_grid(obs['x'], obs['y'])
            rr = int(np.ceil(obs['radius'] / self.costmap_resolution))
            # draw filled circle
            x0 = max(0, cxg - rr)
            x1 = min(nx - 1, cxg + rr)
            y0 = max(0, cyg - rr)
            y1 = min(ny - 1, cyg + rr)
            xs = np.arange(x0, x1 + 1)
            ys = np.arange(y0, y1 + 1)
            if xs.size == 0 or ys.size == 0:
                continue
            X, Y = np.meshgrid(xs, ys, indexing='xy')
            d2 = (X - cxg)**2 + (Y - cyg)**2
            mask = d2 <= (rr + 0.5)**2
            occ[X[mask], Y[mask]] = 1.0  # set occupancy

        # apply gaussian to inflate/soften edges -> costmap
        # sigma in cells; larger sigma => larger inflation
        sigma_m = 0.6  # meters of effective inflation (tunable)
        sigma_cells = max(1.0, sigma_m / self.costmap_resolution)
        costmap = gaussian_filter(occ, sigma=sigma_cells)
        # normalize to 0..1
        if costmap.max() > 0:
            costmap = costmap / costmap.max()

        # optionally compute gradient field and add small leakage from edges:
        gx, gy = np.gradient(costmap)
        grad_mag = np.hypot(gx, gy)
        # combine base costmap and gradient (leaking from boundaries) to accent edges
        combined = costmap + 0.5 * grad_mag  # 0.5 factor controls leakage strength
        if combined.max() > 0:
            combined = combined / combined.max()

        self.costmap = combined  # store final costmap
    # *** CHANGED END ***

    def world_to_grid(self, x_w, y_w):
        """
        Convert world coordinates to integer grid indices (cxg, cyg).
        (0,0) in world corresponds to self.costmap_origin_x/y.
        """
        gx = int(np.round((x_w - self.costmap_origin_x) / self.costmap_resolution))
        gy = int(np.round((y_w - self.costmap_origin_y) / self.costmap_resolution))
        # clip
        gx = int(np.clip(gx, 0, self.costmap_width - 1))
        gy = int(np.clip(gy, 0, self.costmap_height - 1))
        return gx, gy

    def get_cost_at(self, x_w, y_w):
        """
        Bilinear interpolation of costmap at given world coordinates.
        Returns value in [0,1].
        """
        # map to fractional grid coords
        fx = (x_w - self.costmap_origin_x) / self.costmap_resolution
        fy = (y_w - self.costmap_origin_y) / self.costmap_resolution
        if fx < 0 or fy < 0 or fx >= self.costmap_width - 1 or fy >= self.costmap_height - 1:
            # outside map domain, treat as low-cost (or optionally high cost)
            return 0.0
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        dx = fx - x0
        dy = fy - y0

        # fetch four neighbors
        v00 = self.costmap[x0, y0]
        v10 = self.costmap[x0 + 1, y0]
        v01 = self.costmap[x0, y0 + 1]
        v11 = self.costmap[x0 + 1, y0 + 1]

        # bilinear interp
        v0 = v00 * (1 - dx) + v10 * dx
        v1 = v01 * (1 - dx) + v11 * dx
        v = v0 * (1 - dy) + v1 * dy
        return float(v)
    # *** CHANGED END ***

    def dwa_control(self):
        path_distance_bias = 20.0     
        goal_distance_bias = 0.5     
        occdist_scale = 20           
        speed_ref_bias = 10         
        away_bias = 50          
        vx_samples = 3                
        omega_samples = 5             

        dw = self.calc_dynamic_window()
        min_cost = float('inf')
        best_v, best_omega = self.controller.v, self.controller.omega
        best_obsi, best_mindist = 0, float('inf')

        print("----------------------------------------------------------")
        for v in np.linspace(dw[0], dw[1], vx_samples):
            for omega in np.linspace(dw[2], dw[3], omega_samples):
                traj = self.calc_trajectory(self.x, self.y, self.theta, v, omega)
                path_cost = self.calc_path_cost(traj)
                lookahead_cost = self.lookahead_cost(traj, self.x_path[self.target_idx], self.y_path[self.target_idx])
                speed_ref_cost = self.calc_speed_ref_cost(v)
                obs_cost, obsi, mindist = self.calc_obstacle_cost(traj)
                if obs_cost > 100:
                    speed_ref_cost = 0
                away_cost = self.away_from_obstacle_cost(traj)

                total_cost = (
                    # path_distance_bias * path_cost +
                    goal_distance_bias * lookahead_cost +
                    occdist_scale * obs_cost +
                    speed_ref_bias * speed_ref_cost  
                    # away_bias * away_cost
                )

                print(
                    f"total_cost={total_cost:.3f}, v={v:.2f}, omega={omega:.2f}, "
                    f"path_cost={path_cost:.2f}, lookahead_cost={lookahead_cost:.2f}, "
                    f"speed_ref_cost={speed_ref_cost:.2f}, obs_cost={obs_cost}, away_cost={away_cost:.2f}, "
                )

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_v, best_omega = v, omega
                    best_obsi, best_mindist = obsi, mindist

        # print(f"Best v={best_v:.2f}, best omega={best_omega:.2f}, min_cost={min_cost:.2f}, mindist={best_mindist:.2f} (obstacle index: {best_obsi})")
        print("----------------------------------------------------------")
        return best_v, best_omega, best_obsi, best_mindist
    
           
    
sim = CarSimulation()
sim.run_simulation()
