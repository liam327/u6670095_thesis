#!/usr/bin/env python3
from re import L
import pylie
import csv
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import yaml

import pickle
import argparse

#This animates the trajectory of an eqvio output

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def readTrajectory(fname: str, format_spec: str):
    with open(fname, 'r') as file:
        reader = csv.reader(file)
        # Skip header
        next(reader)
        times = []
        poses = []
        good_times = False
        for line in reader:
            t = float(line[0])
            pose = pylie.SE3.from_list(line[1:], format_spec)
            if t < 0:
                continue

            times.append(t)
            poses.append(pose)

    return pylie.Trajectory(poses, times)


def readEqFPoints(fname: str):
    # Read in the (bff) points from the EQVIO output file
    with open(fname, 'r') as file:
        reader = csv.reader(file)
        # Skip header
        next(reader)
        all_points = []
        for line in reader:
            t = float(line[0])
            if t < 0:
                continue

            num_points = (len(line)-1)//4

            points_dict = {int(line[1+4*i]): np.array([[float(line[1+4*i+1+j])] for j in range(3)]) for i in range(num_points)}

            all_points.append(points_dict)

    return all_points


def make_evolving_map(traj, all_points, camera_offset):
    poses = traj.get_elements()
    # Truncate the points_dict
    all_points = all_points[:len(poses)]

    evo_map = []
    cur_map = []
    current_map = {}
    for (pose, points_dict) in zip(poses, all_points):
        tf = pose * camera_offset
        cur_map.append([])
        for idx in points_dict:
            inertial_point = tf * points_dict[idx]
            current_map[idx] = inertial_point
            cur_map[-1].append(inertial_point)
        evo_map.append(np.hstack(list(current_map.values())))
        cur_map[-1] = np.hstack(cur_map[-1])

    return evo_map, cur_map


def update_animation(frame, show_points):
    if frame == 0:
        frame = 1
    if frame <= tru_pos.shape[1]:
        tru_ln.set_data_3d(tru_pos[0, :frame],
                           tru_pos[1, :frame],  tru_pos[2, :frame])
        est_ln.set_data_3d(est_pos[0, :frame],
                           est_pos[1, :frame],  est_pos[2, :frame])
        tru_pt.set_data_3d(tru_pos[0, frame-1],
                           tru_pos[1, frame-1], tru_pos[2, frame-1])
        est_pt.set_data_3d(est_pos[0, frame-1],
                           est_pos[1, frame-1], est_pos[2, frame-1])

        if show_points:
            map_plt.set_data_3d(
                evo_map[frame-1][0, :], evo_map[frame-1][1, :], evo_map[frame-1][2, :])
            cap_plt.set_data_3d(
                cur_map[frame-1][0, :], cur_map[frame-1][1, :], cur_map[frame-1][2, :])
    else:
        if show_points:
            cap_plt.set_data_3d([], [], [])
    ax.view_init(azim=0.5 * frame)
    # return [tru_ln , est_ln , tru_pt , est_pt]


def import_and_align(directory: str, config: str):
    print("Importing trajectories and points from {}.".format(directory))
    est_trajectory = readTrajectory(
        "{}/IMUState.csv".format(directory), 'xw')
    tru_trajectory = readTrajectory(
        "ground_truth.csv", 'xw')

    all_points = readEqFPoints("{}/points.csv".format(directory))

    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
        camera_offset = pylie.SE3.from_list(config_data["eqf"]["initialValue"]["cameraOffset"][1:9], 'xw')

    print("Aligning trajectories.")
    est_trajectory = pylie.analysis.align_trajectory(
        est_trajectory, tru_trajectory)
    tru_trajectory = pylie.Trajectory(
        [tru_trajectory[t] for t in est_trajectory._times], est_trajectory._times)

    print("Making maps.")
    evo_map, cur_map = make_evolving_map(
        est_trajectory, all_points, camera_offset)

    print("Caching aligned data.")
    cache_fname = "{}/aligned_animation_data.pickle".format(directory)
    with open(cache_fname, 'wb') as f:
        pickle.dump([tru_trajectory, est_trajectory, evo_map, cur_map], f)

    return tru_trajectory, est_trajectory, evo_map, cur_map


def load_cached(directory):
    print("Loading aligned trajectories and points from file.")
    cache_fname = "{}/aligned_animation_data.pickle".format(directory)
    with open(cache_fname, 'rb') as f:
        tru_trajectory, est_trajectory, evo_map, cur_map = pickle.load(f)

    return tru_trajectory, est_trajectory, evo_map, cur_map


parser = argparse.ArgumentParser("Animate the VIO output.")
parser.add_argument("directory", metavar='d',
                    help="The dataset directory from which to obtain the output.")
parser.add_argument("config", metavar='d',
                    help="The name of the config to animate")
parser.add_argument("--points", action='store_false',
                    help="Show points in the animation.")
parser.add_argument("--cached", action='store_true',
                    help="Use cached aligned data.")
parser.add_argument("--show", action='store_true',
                    help="Show the animation instead of saving to file.")
args = parser.parse_args()


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

pvg_blu = "#0854b8"
pvg_red = "#ff0c41"

if args.cached:
    tru_trajectory, est_trajectory, evo_map, cur_map = load_cached(
        args.directory)
else:
    tru_trajectory, est_trajectory, evo_map, cur_map = import_and_align(
        args.directory, args.config)


# Make the trajectory xy plots
print("Making trajectory plot")
sze_factor = 1.0
fig = plt.figure(figsize=(16.0/sze_factor, 9.0/sze_factor))
ax = fig.add_subplot(1, 1, 1, projection='3d')

est_pos = [None]
tru_pos = [None]
tru_ln = [None]
est_ln = [None]
tru_pt = [None]
est_pt = [None]
if args.points:
    map_plt = [None]
    cap_plt = [None]

est_pos = np.hstack([pose.x().as_vector()
                     for pose in est_trajectory.get_elements()])
tru_pos = np.hstack([pose.x().as_vector()
                     for pose in tru_trajectory.get_elements()])

if args.points:
    map_plt, = ax.plot(evo_map[-1][0, :], evo_map
                       [-1][1, :], evo_map[-1][2, :], 'k.', markersize=1.0)
    cap_plt, = ax.plot(cur_map[-1][0, :], cur_map
                       [-1][1, :], cur_map[-1][2, :], 'y.', markersize=2.0)
tru_ln, = ax.plot(tru_pos[0, :], tru_pos[1, :],
                  tru_pos[2, :], '-',  color=pvg_blu)
est_ln, = ax.plot(est_pos[0, :], est_pos[1, :],
                  est_pos[2, :], '--', color=pvg_red)
tru_pt, = ax.plot(tru_pos[0, 0], tru_pos[1, 0],
                  tru_pos[2, 0], '*',  color=pvg_blu)
est_pt, = ax.plot(est_pos[0, 0], est_pos[1, 0],
                  est_pos[2, 0], '*',  color=pvg_red)

set_axes_equal(ax)
ax.view_init(azim=0)

tru_ln.set_data_3d([], [], [])
est_ln.set_data_3d([], [], [])
if args.points:
    map_plt.set_data_3d([], [], [])
    cap_plt.set_data_3d([], [], [])

ax.legend([tru_ln, est_ln], ["True", "Est."])
# ax.set_title(targets[i].replace("_", "\_"), size='x-large')

# fig.suptitle(target.replace("_","\_"), size='x-large')


print("Starting Animation")
frame_limit = tru_pos.shape[1]
ani = FuncAnimation(fig, update_animation, frames=frame_limit +
                    500, fargs=(args.points,), interval=20)

if args.show:
    plt.show()
else:
    ani.save('{}/vio_animation.mp4'.format(args.directory), fps=50)

print("Done.")
