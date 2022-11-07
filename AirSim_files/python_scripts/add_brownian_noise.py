from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
import matplotlib.backends.backend_pdf
import argparse

#This file will add brownian or gaussian noise to a know good attitude file.

parser = argparse.ArgumentParser("Modify the attitude file to have brownian noise added to it")
parser.add_argument("--delta", type = float, default = 1.5, help="The magnitude of the gaussian used to generate the noise")
parser.add_argument("--NoYawNoise", type = bool, default = False, help="Set to true if you dont want any noise added to the yaw")
parser.add_argument("--NoRollPitchNoise", type = bool, default = False, help="Set to true if you dont want any noise added to the roll and pitch")
parser.add_argument("--DontSaveFigures", type = bool, default = False, help="Set to true to save a PDF with figures contain infomation about the noise added")
parser.add_argument("--GaussianNoise", type = bool, default = False, help="Use gaussian noise instead of brownian noise")
args = parser.parse_args()


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

def write_csv(file_name,data,header):
    with open(file_name, 'w') as attitude_file:
        stamp_writer = csv.writer(attitude_file)
        stamp_writer.writerow(header)
        for row in data:
            stamp_writer.writerow(row)


def generate_brownian_noise(time_stamps,delta):
    t1 = time_stamps[1:]
    t0 = time_stamps[:-1]
    dt = t1-t0
    n = len(time_stamps)
    x = np.zeros(n)

    for i in range(1,n):
        x[i] = x[i-1] + norm.rvs(scale=delta**2*dt[i-1])

    #This filter is applied to remove most of the high frequency noise. This is not present in the actual recorded data while the raw data displays it 
    xhat = savgol_filter(x, 200, 3) # window size 51, polynomial order 3

    return xhat ,x

def generate_gaussian_noise(time_stamps,delta):
    n = len(time_stamps)
    noise = np.random.normal(loc=0,scale=delta,size=n)
    return noise, np.zeros_like(time_stamps)

def keep_rotation_within_domain(data):
    for i in range(len(data)):
        if data[i] > 180:
            data[i] -= 360
        elif data[i] < -180:
            data[i] += 360
    return data

#grab the quaternion from the csv and put it into a nump array
print("importing data")
quat_array, attitude_header = read_csv('raw_attitude.csv')

#generate the euler angles from the quaternions and break them into roll, pitch and yaw 
rot = Rotation.from_quat(np.array([quat_array[:,2], quat_array[:,3], quat_array[:,4], quat_array[:,1]]).T)
rot_euler = rot.as_euler('xyz', degrees=True)
roll = rot_euler[:,0]
pitch = rot_euler[:,1]
yaw = rot_euler[:,2]
time_stamps = quat_array[:,0]

print("adding noise")
if not args.NoRollPitchNoise:
    if args.GaussianNoise:
        roll_noise_filtered, roll_noise_raw = generate_gaussian_noise(time_stamps,args.delta)
        pitch_noise_filtered, pitch_noise_raw = generate_gaussian_noise(time_stamps,args.delta)
    else: 
        roll_noise_filtered, roll_noise_raw = generate_brownian_noise(time_stamps,args.delta)
        pitch_noise_filtered, pitch_noise_raw = generate_brownian_noise(time_stamps,args.delta)
else: 
    pitch_noise_filtered = np.zeros_like(pitch)
    roll_noise_filtered = np.zeros_like(roll)
    pitch_noise_raw = np.zeros_like(pitch)
    roll_noise_raw = np.zeros_like(roll)
if not args.NoYawNoise:
    if args.GaussianNoise:
        yaw_noise_filtered, yaw_noise_raw = generate_gaussian_noise(time_stamps,args.delta*2)
    else: 
        yaw_noise_filtered, yaw_noise_raw = generate_brownian_noise(time_stamps,args.delta*2)
else: 
    yaw_noise_filtered = np.zeros_like(yaw)
    yaw_noise_raw = np.zeros_like(yaw)

#add in noise. Ensure that yaw stays between -180 and 180 degrees when we add in noise. 
roll_noisy = roll+roll_noise_filtered
pitch_noisy = pitch+pitch_noise_filtered
yaw_noisy = keep_rotation_within_domain(yaw+yaw_noise_filtered)

print("saving noisy attitude data")
#create our new quaternions
new_quat_array = np.zeros_like(quat_array)
new_quat_array[:,0] = time_stamps
rot = Rotation.from_euler('xyz', np.array([roll_noisy,pitch_noisy,yaw_noisy]).T, degrees=True)
quat_xyzw = rot.as_quat()
#rotate the matrix to have the correct order
quat_wxyz = np.roll(quat_xyzw,shift = 1,axis = 1)

#This code here will make the quaternion look nice but is not actually nessecary for operation
reverse_flag = False
for i in range(1,len(yaw_noisy)):
    if (yaw_noisy[i-1] >= 100) and (yaw_noisy[i]<-100):
        if reverse_flag:
            reverse_flag = False
        else:
            reverse_flag = True
    elif (yaw_noisy[i-1] <= -100) and (yaw_noisy[i] > 100):
        if reverse_flag:
            reverse_flag = False
        else:
            reverse_flag = True
    if reverse_flag:
        quat_wxyz[i] = quat_wxyz[i]*-1
    
new_quat_array[:,1:] = quat_wxyz
#now save the new csv
write_csv("attitude.csv",new_quat_array,attitude_header)

if not args.DontSaveFigures:
    print("generating plots")
    #plot the results if nessecary
    fig1, (ax1, ax2, ax3) = plt.subplots(3, dpi = 2000, sharex = True)
    fig1.set_size_inches(12, 6.75)
    fig1.suptitle('Attitude as euler angle with and without noise')
    ax1.plot(time_stamps,roll)
    ax1.plot(time_stamps,roll_noisy)
    ax1.set_ylabel("roll noise (deg)")
    ax1.legend(["roll","roll + smoothed noise"])
    ax2.plot(time_stamps,pitch)
    ax2.plot(time_stamps,pitch_noisy)
    ax2.set_ylabel("pitch noise (deg)")
    ax2.legend(["pitch","pitch + smoothed noise"])
    ax3.plot(time_stamps,yaw)
    ax3.plot(time_stamps,yaw_noisy)
    ax3.set_ylabel("yaw noise (deg)")
    ax3.set_xlabel("time stamp (sec)")
    ax3.legend(["yaw","yaw + smoothed noise"])

    fig2, (ax1, ax2, ax3) = plt.subplots(3, dpi = 2000, sharex = True)
    fig2.set_size_inches(12, 6.75)
    fig2.suptitle('Comparison of smooth noise vs raw noise added to system')
    ax1.plot(time_stamps,roll_noise_raw)
    ax1.plot(time_stamps,roll_noise_filtered)
    ax1.set_ylabel("roll noise (deg)")
    ax1.legend(["raw roll noise","smoothed roll noise"])
    ax2.plot(time_stamps,pitch_noise_raw)
    ax2.plot(time_stamps,pitch_noise_filtered)
    ax2.set_ylabel("pitch noise (deg)")
    ax2.legend(["raw pitch noise","smoothed pitch noise"])
    ax3.plot(time_stamps,yaw_noise_raw)
    ax3.plot(time_stamps,yaw_noise_filtered)
    ax3.set_ylabel("yaw noise (deg)")
    ax3.set_xlabel("time stamp (sec)")
    ax3.legend(["raw yaw noise","smoothed yaw noise"])

    fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, dpi = 2000, sharex = True)
    fig3.set_size_inches(12, 6.75)
    fig3.suptitle('Comparison of attitude as quaternion with and without noise')
    ax1.plot(time_stamps,quat_array[:,1])
    ax1.plot(time_stamps,new_quat_array[:,1])
    ax1.set_ylabel("W quat")
    ax1.legend(["W","W with noise"])
    ax2.plot(time_stamps,quat_array[:,2])
    ax2.plot(time_stamps,new_quat_array[:,2])
    ax2.set_ylabel("X quat")
    ax2.legend(["X","X with noise"])
    ax3.plot(time_stamps,quat_array[:,3])
    ax3.plot(time_stamps,new_quat_array[:,3])
    ax3.set_ylabel("Y quat")
    ax3.legend(["Y","Y with noise"])
    ax4.plot(time_stamps,quat_array[:,4])
    ax4.plot(time_stamps,new_quat_array[:,4])
    ax4.set_ylabel("Z quat")
    ax4.set_xlabel("time stamp (sec)")
    ax4.legend(["Z","Z with noise"])

    pdf = matplotlib.backends.backend_pdf.PdfPages("noise_comparison_data.pdf")
    for fig in [fig1,fig2,fig3]: ## will open an empty extra figure :(
        pdf.savefig( fig )
    pdf.close()
