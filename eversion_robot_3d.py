import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class EversionRobot3D(gym.Env):
    def __init__(self, obs_use):
        super(EversionRobot3D, self).__init__()
        self.MAX_EPISODE = 100
        self.x_threshold = 5
        self.use_obstacle = obs_use

        # Initialize figure for consistent rendering
        self.fig = None
        self.ax = None

        # Updated for 3D: [x, y, z, target_x, target_y, target_z, safety_obs]
        if self.use_obstacle:
            high = np.array(
                [
                    self.x_threshold,
                    self.x_threshold,
                    self.x_threshold,
                    self.x_threshold,
                    self.x_threshold,
                    self.x_threshold,
                    2.0,
                ],
                dtype=np.float32,
            )
        else:
            high = np.array(
                [
                    self.x_threshold,
                    self.x_threshold,
                    self.x_threshold,
                    self.x_threshold,
                    self.x_threshold,
                    self.x_threshold,
                ],
                dtype=np.float32,
            )

        # Updated action space: extend, retract, bend_right, bend_left, phi_increase, phi_decrease, no_action
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.steps_left = self.MAX_EPISODE
        self.low = [-1.5, 0.5, 0]  # 3D bounds
        self.high = [1.5, 3.0, 1.0]
        self.x_target = [
            random.uniform(self.low[0], self.high[0]),
            random.uniform(self.low[1], self.high[1]),
            random.uniform(self.low[2], self.high[2]),
        ]

        if self.use_obstacle:
            self.state = [
                0,
                0,
                0,
                self.x_target[0],
                self.x_target[1],
                self.x_target[2],
                0,
            ]
        else:
            self.state = [0, 0, 0, self.x_target[0], self.x_target[1], self.x_target[2]]

        self.init_length = 0.1
        self.length = self.init_length

        # 3D bending: phi (azimuth angle) and kappa (curvature)
        self.phi = 0  # Start pointing in positive Y direction
        self.kappa = 0  # Curvature magnitude

        self.delta_length = 0.1
        self.delta_phi = 0.1
        self.delta_kappa = 0.1
        self.segment_num = 1
        self.segment_num_max = 5
        self.length_max = 1.5
        self.kappa_max = 2.0

        # Arrays to store segment parameters
        self.length_array = [self.length]
        self.phi_array = [self.phi]
        self.kappa_array = [self.kappa]

        self.T_static = np.eye(4)  # 4x4 transformation matrix for 3D
        self.safety_param = 2
        self.safety_penalty = 10

        # 3D Obstacles
        if self.use_obstacle:
            self.obs_center = []
            self.obs_center.append(np.array([0.0, 2.0, 0.0]))
            self.obs_center.append(np.array([-1.1, 1.2, 0.5]))
            self.obs_center.append(np.array([0.7, 1.0, 0.3]))
            self.radius = [0.2, 0.2, 0.2]

    def kinematic_matrix(self, phi, kappa, s):
        """
        Compute the 4x4 transformation matrix based on the provided kinematic model
        phi: bending direction angle
        kappa: curvature
        s: arc length
        """
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        cos_kappa_s = math.cos(kappa * s)
        sin_kappa_s = math.sin(kappa * s)

        if abs(kappa) < 1e-6:  # Handle straight line case
            A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, s], [0, 0, 0, 1]])
        else:
            A = np.array(
                [
                    [
                        cos_phi**2 * (cos_kappa_s - 1) + 1,
                        sin_phi * cos_phi * (cos_kappa_s - 1),
                        -cos_phi * sin_kappa_s,
                        cos_phi * (cos_kappa_s - 1) / kappa,
                    ],
                    [
                        sin_phi * cos_phi * (cos_kappa_s - 1),
                        cos_phi**2 * (1 - cos_kappa_s) + cos_kappa_s,
                        -sin_phi * sin_kappa_s,
                        sin_phi * (cos_kappa_s - 1) / kappa,
                    ],
                    [
                        cos_phi * sin_kappa_s,
                        -sin_phi * sin_kappa_s,
                        cos_kappa_s,
                        sin_kappa_s / kappa,
                    ],
                    [0, 0, 0, 1],
                ]
            )

        return A

    def static_segment_3d(self, phi_array, kappa_array, length_array, segment_index):
        """Compute cumulative transformation matrix up to segment_index"""
        T_multi = np.eye(4)

        # Corrected base rotation to make robot grow in positive Y direction
        # Rotation: 90° around X-axis to map Z to Y
        base_rotation = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],  # Map Z to -Y (will fix sign below)
                [0, 1, 0, 0],  # Map Y to Z
                [0, 0, 0, 1],
            ]
        )
        T_multi = T_multi @ base_rotation

        # Add 180° rotation around Y to flip the sign to positive Y
        flip_rotation = np.array(
            [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        T_multi = T_multi @ flip_rotation

        for i in range(segment_index):
            T_segment = self.kinematic_matrix(
                phi_array[i], kappa_array[i], length_array[i]
            )
            T_multi = T_multi @ T_segment
        return T_multi

    def pose_segment_3d(self, segment_index):
        """Get 3D coordinates along a segment"""
        T_prior_segment = self.static_segment_3d(
            self.phi_array, self.kappa_array, self.length_array, segment_index
        )

        length = self.length_array[segment_index]
        phi = self.phi_array[segment_index]
        kappa = self.kappa_array[segment_index]

        indeks_maks = max(1, math.floor(length / self.delta_length))

        x_array = []
        y_array = []
        z_array = []

        for i in range(indeks_maks + 1):
            s = (i / indeks_maks) * length if indeks_maks > 0 else 0
            T_segment = self.kinematic_matrix(phi, kappa, s)
            T_total = T_prior_segment @ T_segment

            x_array.append(T_total[0, 3])
            y_array.append(T_total[1, 3])
            z_array.append(T_total[2, 3])

        return x_array, y_array, z_array

    def check_collision_3d(self):
        """Check collision with 3D obstacles"""
        collision = False
        for seg_idx in range(len(self.length_array)):
            x_array, y_array, z_array = self.pose_segment_3d(seg_idx)
            for i, obs_pos in enumerate(self.obs_center):
                for j in range(len(x_array)):
                    point = np.array([x_array[j], y_array[j], z_array[j]])
                    distance = np.linalg.norm(point - obs_pos)
                    if distance <= self.radius[i]:
                        return True
        return collision

    def check_safety_3d(self):
        """Check safety distance from 3D obstacles"""
        danger = False
        for seg_idx in range(len(self.length_array)):
            x_array, y_array, z_array = self.pose_segment_3d(seg_idx)
            for i, obs_pos in enumerate(self.obs_center):
                for j in range(len(x_array)):
                    point = np.array([x_array[j], y_array[j], z_array[j]])
                    distance = np.linalg.norm(point - obs_pos)
                    if distance <= self.safety_param * self.radius[i]:
                        return True
        return danger

    def obstacle_vector_3d(self, dist_to_goal):
        """Calculate obstacle avoidance term for 3D"""
        limit_obs = 0.5
        gain_obs = 3.0
        min_distance = 100

        for seg_idx in range(len(self.length_array)):
            x_array, y_array, z_array = self.pose_segment_3d(seg_idx)
            for i, obs_pos in enumerate(self.obs_center):
                for j in range(len(x_array)):
                    point = np.array([x_array[j], y_array[j], z_array[j]])
                    distance = np.linalg.norm(point - obs_pos)
                    if distance < min_distance:
                        min_distance = distance

        if min_distance < limit_obs:
            distance_to_surface = min_distance - min(
                [self.radius[i] for i in range(len(self.radius))]
            )
            if distance_to_surface > 0:
                avoidance_term = (
                    1.0 / distance_to_surface - 1.0 / limit_obs
                ) * gain_obs
            else:
                avoidance_term = gain_obs * 10  # Large penalty for collision
        else:
            avoidance_term = 0

        return avoidance_term

    def step(self, action):
        self.act = action

        # Modified action mapping for 3D:
        # 0: extend, 1: retract,
        # 2: bend right (negative X), 3: bend left (positive X),
        # 4: increase phi (counter-clockwise), 5: decrease phi (clockwise),
        # 6: no action

        if action == 0:  # Extend
            self.length = min(self.length + self.delta_length, self.length_max)
        elif action == 1:  # Retract
            self.length = max(self.length - self.delta_length, 0)
        elif action == 2:  # Bend right (negative X)
            self.kappa = min(self.kappa + self.delta_kappa, self.kappa_max)
        elif action == 3:  # Bend left (positive X)
            # self.phi = math.pi / 2  # 90° or +X direction
            self.kappa = min(self.kappa - self.delta_kappa, self.kappa_max)
        elif action == 4:  # Increase phi (counter-clockwise rotation)
            self.phi = (self.phi + self.delta_phi) % (2 * math.pi)
        elif action == 5:  # Decrease phi (clockwise rotation)
            self.phi = (self.phi - self.delta_phi) % (2 * math.pi)
        # action == 6 is no action

        # Update arrays
        if 0 <= self.length < self.length_max:
            self.length_array[-1] = self.length
            self.phi_array[-1] = self.phi
            self.kappa_array[-1] = self.kappa
        elif self.length >= self.length_max:
            if len(self.length_array) < self.segment_num_max:
                # Add new segment
                self.length = 0
                self.phi = self.phi_array[-1]  # Maintain direction
                self.kappa = 0
                self.length_array.append(self.length)
                self.phi_array.append(self.phi)
                self.kappa_array.append(self.kappa)
            else:
                self.length = self.length_max
        elif self.length < 0:
            if len(self.length_array) > 1:
                # Remove last segment
                self.length_array.pop()
                self.phi_array.pop()
                self.kappa_array.pop()
                self.length = self.length_array[-1]
                self.phi = self.phi_array[-1]
                self.kappa = self.kappa_array[-1]
            else:
                self.length = 0

        # Get tip position
        self.T_static = self.static_segment_3d(
            self.phi_array, self.kappa_array, self.length_array, len(self.length_array)
        )
        x_tip, y_tip, z_tip = (
            self.T_static[0, 3],
            self.T_static[1, 3],
            self.T_static[2, 3],
        )

        # Update state
        if self.use_obstacle:
            safety_flag = self.check_safety_3d()
            safety_obs = 1 if safety_flag else 0
            self.state = [
                x_tip,
                y_tip,
                z_tip,
                self.x_target[0],
                self.x_target[1],
                self.x_target[2],
                safety_obs,
            ]
        else:
            self.state = [
                x_tip,
                y_tip,
                z_tip,
                self.x_target[0],
                self.x_target[1],
                self.x_target[2],
            ]

        # Check boundaries
        boundary = (
            abs(x_tip) > self.x_threshold
            or abs(y_tip) > self.x_threshold
            or z_tip < 0
            or z_tip > self.x_threshold
        )

        error = np.array([x_tip, y_tip, z_tip]) - np.array(self.x_target)

        # Check termination conditions
        done = bool(boundary or self.steps_left < 0 or self.length < 0)

        if self.use_obstacle:
            self.flag_collision = self.check_collision_3d()
            done = done or self.flag_collision
            reward_safety = -1 * self.obstacle_vector_3d(error)

        # Calculate reward
        if not done:
            reward = -np.linalg.norm(error) ** 2
            if self.use_obstacle:
                reward = reward + reward_safety
        else:
            reward = 0
            if boundary:
                reward += -100000
            if self.length < 0:
                reward += -100000
            elif self.use_obstacle and self.flag_collision:
                reward += -100000
            else:
                reward += 0

        if not done:
            self.steps_left -= 1

        self.cur_reward = reward
        self.cur_done = done
        return np.array([self.state]), reward, done, False, {}

    def reset(self, seed=None):
        self.x_target = [
            random.uniform(self.low[0], self.high[0]),
            random.uniform(self.low[1], self.high[1]),
            random.uniform(self.low[2], self.high[2]),
        ]

        if self.use_obstacle:
            self.state = [
                0,
                0,
                0,
                self.x_target[0],
                self.x_target[1],
                self.x_target[2],
                0,
            ]
        else:
            self.state = [0, 0, 0, self.x_target[0], self.x_target[1], self.x_target[2]]

        self.steps_left = self.MAX_EPISODE
        self.length = self.init_length
        self.phi = 0
        self.kappa = 0
        self.length_array = [self.length]
        self.phi_array = [self.phi]
        self.kappa_array = [self.kappa]
        self.T_static = np.eye(4)
        return np.array([self.state]), {}

    def draw_segment_3d(self, ax, segment_index):
        """Draw a 3D segment"""
        # Use red for odd segments (1, 3, 5...) and blue for even segments (0, 2, 4...)
        color = "red" if segment_index % 2 == 1 else "blue"

        x_array, y_array, z_array = self.pose_segment_3d(segment_index)
        ax.plot(x_array, y_array, z_array, color=color)
        ax.scatter(x_array, y_array, z_array, color=color, s=20, alpha=1.0)

    def draw_obs_3d(self, ax):
        """Draw 3D obstacles"""
        for i, center in enumerate(self.obs_center):
            # Draw sphere obstacles
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            r = self.radius[i]
            x_sphere = center[0] + r * np.outer(np.cos(u), np.sin(v))
            y_sphere = center[1] + r * np.outer(np.sin(u), np.sin(v))
            z_sphere = center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color="black")

    def render(self, mode="human"):
        # Initialize figure only once
        if self.fig is None:
            plt.ion()  # Turn on interactive mode
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.set_title("3D Eversion Robot")
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([0, 4])
            self.ax.set_zlim([0, 4])
            self.ax.view_init(elev=20, azim=-35)

            # Store references to plotted objects for updating
            self.segment_lines = []
            self.segment_points = []
            self.target_marker = None
            self.obstacle_surfaces = []
        else:

            for points in self.segment_points:
                points.remove()
            if self.target_marker:
                self.target_marker.remove()
            for surface in self.obstacle_surfaces:
                surface.remove()
            self.segment_lines = []
            self.segment_points = []
            self.obstacle_surfaces = []

        # Draw all segments with 2-color scheme
        for i in range(len(self.length_array)):
            x_array, y_array, z_array = self.pose_segment_3d(i)
            # Use red for odd segments (1, 3, 5...) and blue for even segments (0, 2, 4...)
            color = "red" if i % 2 == 1 else "blue"

            # Plot points
            points = self.ax.scatter(
                x_array, y_array, z_array, color=color, s=20, alpha=1.0
            )
            self.segment_points.append(points)

        # Draw target
        self.target_marker = self.ax.scatter(
            self.x_target[0],
            self.x_target[1],
            self.x_target[2],
            color="gold",
            s=100,
            marker="*",
        )

        # Draw obstacles if enabled
        if self.use_obstacle:
            for i, center in enumerate(self.obs_center):
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                r = self.radius[i]
                x_sphere = center[0] + r * np.outer(np.cos(u), np.sin(v))
                y_sphere = center[1] + r * np.outer(np.sin(u), np.sin(v))
                z_sphere = center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                surface = self.ax.plot_surface(
                    x_sphere, y_sphere, z_sphere, alpha=1, color="black"
                )
                self.obstacle_surfaces.append(surface)

        # Update the figure without clearing it
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Print status
        # if self.use_obstacle:
        #     print(f'State {[round(x, 3) for x in self.state]}, action: {self.act}, '
        #         f'done: {self.cur_done}, phi: {[round(x, 3) for x in self.phi_array]}, '
        #         f'kappa: {[round(x, 3) for x in self.kappa_array]}, '
        #         f'length: {[round(x, 3) for x in self.length_array]}, '
        #         f'reward: {round(self.cur_reward, 3)}, collision: {self.flag_collision}')
        # else:
        #     print(f'State {[round(x, 3) for x in self.state]}, action: {self.act}, '
        #         f'done: {self.cur_done}, phi: {[round(x, 3) for x in self.phi_array]}, '
        #         f'kappa: {[round(x, 3) for x in self.kappa_array]}, '
        #         f'length: {[round(x, 3) for x in self.length_array]}, '
        #         f'reward: {round(self.cur_reward, 3)}')

        # Small pause to allow interaction
        # plt.pause(0.001)
