from scipy.interpolate import CubicSpline
import numpy as np
from .minjerk import (
    minjerk_coefficients,
    minjerk_trajectory,
)
from ..io import import_module, log


class Waypoint(object):
    def __init__(self, positions, velocities, accelearations):
        self.positions = positions
        self.velocities = velocities
        self.accelearations = accelearations


class Path(object):
    def __init__(self, **kwargs):
        self.points = kwargs.get("points", None)
        self.velocities = kwargs.get("velocities", None)
        self.accelerations = kwargs.get("accelerations", None)
        self.times = kwargs.get("times", None)
        self.joint_names = kwargs.get("joint_names", [])
        self._path = None
        self._cspline = None

    @property
    def path(self):
        return self._path

    @property
    def size(self):
        return len(self.points)

    def __len__(self):
        return self.size

    @property
    def curve_length(self):
        s = np.linspace(0.0, 1.0, len(self.points))
        ys = self.cubic_spline(s, 1)

        length = 0

        # Euler integration
        dt = 1.0 / float(len(self.points))

        for y in ys:
            length += np.linalg.norm(y) * dt

        return length

    def reversed(self):
        self.points = np.flip(self.points, 0)
        # self.velocities = None if self.velocities is None else np.flip(self.velocities,0)
        # self.accelerations = None if self.accelerations is None else np.flip(self.accelerations,0)

        self._cspline = None

    def reverse(self):
        points = None if self.points is None else np.flip(self.points, 0)
        # velocities = None if self.velocities is None else np.flip(self.velocities,0)
        # accelerations = None if self.accelerations is None else np.flip(self.accelerations,0)

        return Path(
            points=points,
            # velocities=velocities,
            # accelerations=accelerations,
            times=self.times,
            joint_names=self.joint_names,
        )

    def cubic_spline(self, xs, derivative_order=0, extrapolate=True, use_cache=True):
        if self._cspline is None or not use_cache:
            s = np.linspace(0.0, 1.0, len(self.points))
            self._cspline = CubicSpline(s, self.points, extrapolate=extrapolate)

        return self._cspline(xs, derivative_order)

    def cubic_spline_path(
        self, xs, derivative_order=0, extrapolate=True, use_cache=True
    ):
        points = self.cubic_spline(
            xs,
            derivative_order=derivative_order,
            extrapolate=extrapolate,
            use_cache=use_cache,
        )

        return Path(points=points)

    def concat(self, other, ai=0, bi=0):
        return Path(points=np.vstack([self.points[ai:, :3], other.points[bi:, :3]]))

    def _join_array(self, arr0, arr1):
        arr = None
        if arr0 is not None and arr1 is not None:
            arr = np.vstack([arr0, arr1])

        return arr

    def join(self, other):
        points = self._join_array(self.points, other.points)
        velocities = self._join_array(self.velocities, other.velocities)
        accelerations = self._join_array(self.accelerations, other.accelerations)

        times = None
        if self.times is not None and other.times is not None:
            times = np.hstack([self.times, other.times + self.times[-1]])

        return Path(
            points=points,
            velocities=velocities,
            accelerations=accelerations,
            times=times,
            joint_names=self.joint_names,
        )

    def order(self, anchor_point):
        dists = [np.linalg.norm(p - anchor_point) for p in self.points]

        sidx = np.argsort(dists)[::-1]

        sorted_points = self.points[sidx, :]

        self._cspline = None

        self.points = sorted_points

    def time_parametrise(self, total_time=10, dt=0.2):
        npts = np.ceil(total_time / dt)
        ts = np.linspace(0.0, 1.0, int(npts))
        points = self.cubic_spline(ts, extrapolate=False, use_cache=False)
        velocities = self.cubic_spline(ts, derivative_order=1, extrapolate=False)
        accelerations = self.cubic_spline(ts, derivative_order=2, extrapolate=False)

        return Path(
            points=points,
            velocities=velocities,
            accelerations=accelerations,
            times=ts * total_time,
            joint_names=self.joint_names,
        )

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, index):
        positions = self.points[index, :]
        velocities = None if self.velocities is None else self.velocities[index, :]
        accelerations = (
            None if self.accelerations is None else self.accelerations[index, :]
        )

        return Waypoint(positions, velocities, accelerations)

    def compute_mjt(self, num_intervals=5):
        if self.points is None or self.times is None:
            return

        point_durations = [
            self.times[i + 1] - self.times[i] for i in range(len(self.times) - 1)
        ]
        # n_joints = len(self.points[0])

        # dimensions_dict = {'positions':True,
        #                     'velocities':False,
        #                     'accelerations':False}
        # self.m_matrix = compute_minjerk_coeff(n_joints,
        #                                  self.points,
        #                                  point_durations,
        #                                  dimensions_dict)

        # ts = np.linspace(0.0, 1.0, len(self.times))
        m_coeffs = minjerk_coefficients(self.points)
        points = np.array(minjerk_trajectory(m_coeffs, num_intervals, point_durations))
        times = np.linspace(0.0, self.times[-1], len(points))

        self.points = points
        self.times = times

    def as_ros_trajectory(self):
        tj = import_module("trajectory_msgs.msg")
        rclpy = import_module("rclpy")

        if rclpy is None:
            log.warning("rclpy is not available.")
            return
        if tj is None:
            log.warning("trajectory_msgs.msg is not available.")
            return

        trajectory = tj.JointTrajectory()
        trajectory.joint_names = self.joint_names

        for i in range(self.size):
            point = tj.JointTrajectoryPoint()

            point.positions = self.points[i, :].tolist()

            if self.velocities is not None:
                point.velocities = self.velocities[i, :].tolist()
            if self.accelerations is not None:
                point.accelerations = self.accelerations[i, :].tolist()

            point.time_from_start = rclpy.duration.Duration(
                seconds=self.times[i]
            ).to_msg()

            trajectory.points.append(point)

        return trajectory

    def as_ros_trajectory_action_goal(self, goal_time_tolerance=1.0):
        cm = import_module("control_msgs.action")
        rclpy = import_module("rclpy")

        trajgoal = cm.FollowJointTrajectory.Goal()

        trajgoal.trajectory = self.as_ros_trajectory()
        trajgoal.goal_time_tolerance = rclpy.duration.Duration(
            seconds=goal_time_tolerance
        ).to_msg()  # seconds

        return trajgoal

    def plot(self):
        import matplotlib.pyplot as plt

        for arr in [self.points, self.velocities, self.accelerations, self.times]:
            plt.figure()
            plt.plot(arr)

        plt.show()

    @classmethod
    def from_ros_trajectory_action_goal(cls, goal):
        points = np.asarray([np.asarray(p.positions) for p in goal.trajectory.points])
        velocities = np.asarray(
            [np.asarray(p.velocities) for p in goal.trajectory.points]
        )
        accelerations = np.asarray(
            [np.asarray(p.accelerations) for p in goal.trajectory.points]
        )

        times = np.asarray(
            [
                (p.time_from_start.sec + p.time_from_start.nanosec * 1e-9)
                for p in goal.trajectory.points
            ]
        )
        return cls(
            points=points,
            velocities=velocities,
            accelerations=accelerations,
            times=times,
            joint_names=goal.trajectory.joint_names,
        )

    @classmethod
    def from_plex_traj(cls, trajectory):
        return cls(
            points=np.array(trajectory.positions),
            velocities=np.array(trajectory.velocities),
            accelerations=np.array(trajectory.accelerations),
            times=np.array(trajectory.times),
            joint_names=trajectory.jointNames,
        )
