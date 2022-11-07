from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.spatial.transform import Rotation
from scipy import interpolate as interp

#This is used to compare attitude estiamte with and without gps

def read_csv(file_name,return_header=True):
    file = open(file_name)
    csvreader = csv.reader(file)
    header = []
    if not return_header:
        next(csvreader)
    else: 
        header = next(csvreader)
    rows = []
    for line in csvreader:
        rows.append(line)
    file.close()
    return np.array(rows).astype(np.float64), header


#grab the quaternion from the csv and put it into a nump array
print("importing data")
mav_imu, attitude_header = read_csv('mav_imu.csv')
quat_bad_t, attitude_header = read_csv('attitude.csv')

#generate the euler angles from the quaternions and break them into roll, pitch and yaw 
roll_good_t = mav_imu[:,8] * 180/np.pi
pitch_good_t = mav_imu[:,9] * 180/np.pi
yaw_good_t = mav_imu[:,10] * 180/np.pi
time_stamps_good_t = mav_imu[:,0]

rot = Rotation.from_quat(np.array([quat_bad_t[:,2], quat_bad_t[:,3], quat_bad_t[:,4], quat_bad_t[:,1]]).T)
rot_euler = rot.as_euler('xyz', degrees=True)
roll_bad_t = rot_euler[:,0]
pitch_bad_t = rot_euler[:,1]
yaw_bad_t = rot_euler[:,2]
time_stamps_bad_t = quat_bad_t[:,0]

roll_bad_t_interp = interp.interp1d(np.arange(roll_bad_t.size),roll_bad_t)
roll_bad_t_compress = roll_bad_t_interp(np.linspace(0,roll_bad_t.size-1,roll_good_t.size))
pitch_bad_t_interp = interp.interp1d(np.arange(pitch_bad_t.size),pitch_bad_t)
pitch_bad_t_compress = pitch_bad_t_interp(np.linspace(0,pitch_bad_t.size-1,pitch_good_t.size))
yaw_bad_t_interp = interp.interp1d(np.arange(yaw_bad_t.size),yaw_bad_t)
yaw_bad_t_compress = yaw_bad_t_interp(np.linspace(0,yaw_bad_t.size-1,yaw_good_t.size))

#generate statistics for noise 
yaw_diff = yaw_good_t-yaw_bad_t_compress
yaw_diff[abs(yaw_diff)>=30]=0

roll_diff_mean = np.mean(roll_good_t-roll_bad_t_compress)
roll_diff_std = np.std(roll_good_t-roll_bad_t_compress)
pitch_diff_mean = np.mean(pitch_good_t-pitch_bad_t_compress)
pitch_diff_std = np.std(pitch_good_t-pitch_bad_t_compress)
yaw_diff_mean = np.mean(yaw_diff)
yaw_diff_std = np.std(yaw_diff)

print("mean roll difference (deg):",roll_diff_mean)
print("mean roll difference standard devitation (deg):",roll_diff_std)
print("mean pitch difference (deg):",pitch_diff_mean)
print("mean pitch difference standard devitation (deg):",pitch_diff_std)
print("mean yaw difference (deg):",yaw_diff_mean)
print("mean yaw difference standard devitation (deg):",yaw_diff_std)

print("\ngenerating plots")
#plot the results if nessecary
fig1, (ax1, ax2, ax3) = plt.subplots(3, sharex = True)
fig1.set_size_inches(16, 9)
fig1.suptitle('Timing comparison between attitude with and without GPS')
ax1.plot(time_stamps_bad_t,roll_bad_t)
ax1.plot(time_stamps_good_t,roll_good_t)
ax1.set_ylabel("roll (deg)")
ax1.legend(["No GPS","with GPS"])
ax2.plot(time_stamps_bad_t,pitch_bad_t)
ax2.plot(time_stamps_good_t,pitch_good_t)
ax2.set_ylabel("pitch (deg)")
ax2.legend(["No GPS","with GPS"])
ax3.plot(time_stamps_bad_t,yaw_bad_t)
ax3.plot(time_stamps_good_t,yaw_good_t)
ax3.set_ylabel("yaw (deg)")
ax3.set_xlabel("time stamp (sec)")
ax3.legend(["No GPS","with GPS"])
plt.savefig("comparison.pdf")
plt.show()




