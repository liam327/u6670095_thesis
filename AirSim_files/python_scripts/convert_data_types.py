import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from pylie import SE3
from scipy.spatial.transform import Rotation
import glob

###################################################from mav2poses
def lat2radius(lat):
    equatorial_radius = 6378.1370e3 # metres
    polar_radius = 6356.7523e3 # metres

    c = np.cos(lat * np.pi / 180.0)
    s = np.sin(lat * np.pi / 180.0)

    radius = (equatorial_radius**2 * c)**2 + (polar_radius**2 * s)**2
    radius = radius / ((equatorial_radius * c)**2 + (polar_radius * s)**2)
    radius = np.sqrt(radius)
    return float(radius)

def gps2xyz(gps_origin, gps):
    # The origin defines the frame NED using a local approximation
    
    local_radius = lat2radius(gps_origin[0])

    nor = float(np.sin((gps[0]- gps_origin[0]) * np.pi / 180.0) * local_radius)
    eas = float(np.sin((gps[1]- gps_origin[1]) * np.pi / 180.0) * local_radius)
    down = gps_origin[2] - gps[2]

    return [nor, eas, down]
    

print("Creating ground truth file")
with open('mav_imu.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip header

    times = []
    att_data = []
    gps_data = []

    for row in reader:
        t = float(row[0])
        att = [float(row[i]) for i in range(8,11)]
        gps = [float(row[i]) for i in range(11,14)] #Use the ArduPilot estimate 
        #gps = [float(row[i]) for i in range(14,17)]  # Use the RAW GPS datas

        # Avoid repeat entries
        if len(times) > 0 and all(att_data[-1][i] == att[i] for i in range(3)) and all(gps_data[-1][i] == gps[i] for i in range(3)):
            continue
        
        if len(times) > 0 and all(gps_data[-1][i] == gps[i] for i in range(3)):
            continue
        

        times.append(t)
        att_data.append(att)
        gps_data.append(gps)

# Convert into wx pose format and write
output_fname = "ground_truth.csv"
with open(output_fname, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Time (s)", "tx", "ty", "tz", "qw", "qx", "qy", "qz"])
    N = len(times)
    for i in range(N):

        position = gps2xyz(gps_data[0], gps_data[i])
        attitude = Rotation.from_euler('xyz', att_data[i])

        pose = SE3(attitude, position)
        writer.writerow([times[i]] + pose.to_list('xw'))

###################################################from convert_time
t0 = np.genfromtxt('timestamp.txt',dtype=float)
#offset to account for differences in time between desktop and laptop if nessecary 
tDiff = 100 #micro

img_entrys = np.genfromtxt('airsim_rec.txt',delimiter=',',skip_header=1,dtype=str)

for i in range(len(img_entrys)):
    img_entrys[i][0] = (float(img_entrys[i][0])-t0)

np.savetxt('cam.csv',img_entrys,delimiter=',',fmt='%1.13s')


############################################from convert_image_type
print("changing type of images and renumbering them")
cam_csv = np.genfromtxt('cam.csv',skip_header = 1, delimiter=',',dtype=[float,int])
number_files = len(cam_csv)

path = 'frames/'

files = os.listdir(path)

count = 0
for i in range(number_files+1):
    if os.path.isfile(os.path.join(path,'frame_'+str(i)+'.png')):
        image = cv2.imread(os.path.join(path,'frame_'+str(i)+'.png'))
    	# Save .jpg image
        cv2.imwrite(os.path.join(path,'frame_'+str(count)+'.jpg'), image, [int(cv2.IMWRITE_JPEG_QUALITY),100])
        os.remove(os.path.join(path,'frame_'+str(i)+'.png'))
        count += 1


if (number_files<1000):
    #create video
    print("creating video of images")
    img_array = []
    for filename in sorted(glob.glob('frames/*.jpg'), key=os.path.getmtime):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('sim_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
else:
    print("An external command is required to generate a video of the flight from the images as to many images are present. The command required is \n ffmpeg -start_number 0 -i frame_%d.jpg -vcodec copy -q:v 1 sim_video.avi \n where this command should be run in the frames folder")
