
#!/usr/bin/env python3
from email import message
from re import A
import time, datetime
import argparse

#This is the main script for recording AirSim data

parser = argparse.ArgumentParser("Record camera, IMU, GPS, attitude.")
parser.add_argument("-i", action='store_true', help="Use initial exposure estimate for indoor lighting.")
parser.add_argument("--no-video", action='store_true', help="Disable video recording.")
parser.add_argument("--no-imu", action='store_true', help="Disable IMU recording.")
parser.add_argument("--codec", type=str, default="H264", help="The desired video writer codec. Default: 'H264'")
parser.add_argument("--videodev", type=int, default=0, help="video device number")
parser.add_argument("--serialdev", default="/dev/serial0", help="mavlink serial device")
parser.add_argument("--verbose", action='store_true', help="show verbode debug")
parser.add_argument("--delay", type=float, default=0.0, help="delay on camera read")
parser.add_argument("--record-video", action='store_true', help="record a video instead of frames")
parser.add_argument("--noGPS", type=bool, default=False, help="If you are recording attitude without GPS")
parser.add_argument("--noAttitude", type=bool, default=False, help="If you do not want to record attitutde")
args = parser.parse_args()

from pymavlink import mavutil
from pymavlink import quaternion
from pymavlink import rotmat
import csv
import sys, os
import threading
import cv2
import numpy as np
import queue
from dataclasses import dataclass

@dataclass
class FrameData:
    csv_row: list
    image: np.ndarray

@dataclass
class MavData:
    csv_row: list

@dataclass
class AttitudeData:
    csv_row: list


def write_from_buffer(t0, vio_cap_dir):
    global write_buffer

    mav_fname = vio_cap_dir+"/mav_imu.csv"
    stamp_fname = vio_cap_dir+"/cam.csv"
    attitude_fname = vio_cap_dir+"/attitude.csv"
    
    if args.record_video:
        video_fname = vio_cap_dir+"/cam.mkv"
    else:
        os.mkdir(vio_cap_dir+"/frames")

    video_writer = None
    with open(stamp_fname, 'w') as stamp_file, open(mav_fname, 'w') as mav_file, open(attitude_fname, 'w') as attitude_file:
        
        # Set up video data writer
        stamp_writer = csv.writer(stamp_file)
        stamp_writer.writerow([
            "Timestamp (s)",
            "Frame #",
            "Exposure"
        ])

        # Set up IMU data writer
        mav_writer = csv.writer(mav_file)
        mav_writer.writerow([
            "Timestamp (s)",
            "xgyro (rad/s)",
            "ygyro (rad/s)",
            "zgyro (rad/s)",
            "xacc (m/s/s)",
            "yacc (m/s/s)",
            "zacc (m/s/s)",
            "Mav Time (s)",
            "Roll (rad)",
            "Pitch (rad)",
            "Yaw (rad)",
            "Lat (deg)",
            "Lon (deg)",
            "Alt (m)",
            "GLat (deg)",
            "GLon (deg)",
            "GAlt (m)"
        ])

        # Set up attitude data writer
        attitude_writer = csv.writer(attitude_file)
        attitude_writer.writerow([
            "Timestamp (s)",
            "Quat w", 
            "Quat x", 
            "Quat y", 
            "Quat z"
        ])

        print("Writing buffer is started.")

        flush_counter = 0
        max_flush_count = 200
        item_count = 0
        while True:
            data_item = write_buffer.get(block=True)
            # print("got frame %u" % (item_count))
            item_count += 1
             
            if isinstance(data_item, MavData):
                mav_writer.writerow(data_item.csv_row)
            elif isinstance(data_item,AttitudeData):
                attitude_writer.writerow(data_item.csv_row)
            elif isinstance(data_item, FrameData):
                if args.record_video:
                    if video_writer is None:
                        # Initialise the video writer
                        frame = data_item.image
                        video_writer = cv2.VideoWriter(video_fname, cv2.VideoWriter_fourcc(*args.codec), 20.0, (frame.shape[1],frame.shape[0]))
                    video_writer.write(data_item.image)
                else:
                    cv2.imwrite(vio_cap_dir+"/frames/frame_{}.jpg".format(data_item.csv_row[1]), data_item.image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                stamp_writer.writerow(data_item.csv_row)
            
            flush_counter += 1
            if flush_counter % max_flush_count == 0:
                mav_file.flush()
                stamp_file.flush()
                attitude_file.flush()


def record_cam(t0, indoor_lighting = True):
    global write_buffer

    cap = cv2.VideoCapture(args.videodev)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) #means manual
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.7) #means manual

    cap.set(cv2.CAP_PROP_FPS, 15)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #full res is 1280
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #full res is 720
    if indoor_lighting:
        exposure = 0.3 # Indoor initial value
    else:
        exposure = 0.001 # Outdoor initial value
    gain = 1e-4
    
    ret, frame = cap.read()

    frame_count = 0
    last_frame_count = 0
    last_t = msg.time_usec*1.0e-6
    if args.record_video:
        print("Video recording is started.")
    else:
        print("Frame recording is started.")
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Video frame did not return correctly!")
            continue
        t = msg.time_usec*1.0e-6
        row = [ t - t0, frame_count, exposure ]

        write_buffer.put(FrameData(row, frame))
        
        frame_count += 1

        # Control camera exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        img_mean = np.mean(frame)

        if img_mean > 128-32 and img_mean < 128+32:
            continue

        exposure += gain * (128 - img_mean) * exposure
        if exposure > 0.7:
            exposure = 0.7
        elif exposure <= 0.0:
            exposure = 1e-6

        if args.delay > 0:
            time.sleep(args.delay)

        if args.verbose:
            t = time.time()
            if t - last_t > 1.0:
                fps = (frame_count-last_frame_count)/(t-last_t)
                print("%.2f FPS qsize=%u" % (fps, write_buffer.qsize()))
                last_t = t
                last_frame_count = frame_count

        if write_buffer.qsize() > 2000:
            print("Draining queue")
            while (write_buffer.qsize() > 2000):
                write_buffer.get()

def record_imu(t0):
    global write_buffer

    gyro_factor = 1e-3
    acc_factor = 9.81 * 1e-3

    ATTITUDE = None
    GLOBAL_POSITION_INT = None
    GPS_RAW_INT = None
    ATTITUDE_QUATERNION = None
    t_old = 0
    scale_factor = 1

    if args.noGPS:
        message_request = ["RAW_IMU",'ATTITUDE','GLOBAL_POSITION_INT','GPS_RAW_INT','AHRS2']
    else: 
        message_request = ["RAW_IMU",'ATTITUDE','GLOBAL_POSITION_INT','GPS_RAW_INT','ATTITUDE_QUATERNION']

    print("MAV/IMU recording is started.")
    while(True):
        msg = mav_connection.recv_match(type=message_request, blocking=True)
        mtype = msg.get_type()
        #if we have an attitude quartonian message then add that to the buffer. Either get attitude data directly from quartonian or from AHRS2 depending if GPS is being turned off 
        if ((mtype == 'ATTITUDE_QUATERNION') and (args.noAttitude==False)):
            ATTITUDE_QUATERNION = msg
            t = ATTITUDE_QUATERNION.time_boot_ms*1.0e-3 - t0
            row = [t, ATTITUDE_QUATERNION.q1, ATTITUDE_QUATERNION.q2, ATTITUDE_QUATERNION.q3, ATTITUDE_QUATERNION.q4]
            write_buffer.put(AttitudeData(row))
            continue
        if ((mtype == 'AHRS2') and (args.noAttitude==False)):
            ATTITUDE_EULER = msg
            t = (time.time() - t_noGPS)*scale_factor
            m = rotmat.Matrix3()
            m.from_euler(ATTITUDE_EULER.roll, ATTITUDE_EULER.pitch, ATTITUDE_EULER.yaw)
            q = quaternion.Quaternion(m)
            row = [t, q[0], q[1], q[2], q[3]]
            #print(row)
            write_buffer.put(AttitudeData(row))
            continue
        #otherwise record mav_imu as normal 
        if mtype == 'ATTITUDE':
            ATTITUDE = msg
            continue
        if mtype == 'GLOBAL_POSITION_INT':
            GLOBAL_POSITION_INT = msg
            continue
        if mtype == 'GPS_RAW_INT':
            GPS_RAW_INT = msg
            continue
        if mtype != 'RAW_IMU':
            continue
        if ATTITUDE is None or GLOBAL_POSITION_INT is None or GPS_RAW_INT is None:
            continue
        t = msg.time_usec*1.0e-6 - t0
        #This helps to stop repeat time stamps of the exact same time
        if t_old == t: 
            continue
        
        if (args.noAttitude==False and args.noGPS):
            t_wall = time.time() - t_noGPS
            scale_factor = t/t_wall

        assert msg is not None
        row = [
            t,
            msg.xgyro * gyro_factor,
            msg.ygyro * gyro_factor,
            msg.zgyro * gyro_factor,
            msg.xacc * acc_factor,
            msg.yacc * acc_factor,
            msg.zacc * acc_factor,
            msg.time_usec*1.0e-6,
            ATTITUDE.roll,
            ATTITUDE.pitch,
            ATTITUDE.yaw,
            GLOBAL_POSITION_INT.lat*1.0e-7,
            GLOBAL_POSITION_INT.lon*1.0e-7,
            GLOBAL_POSITION_INT.alt*0.001,
            GPS_RAW_INT.lat*1.0e-7,
            GPS_RAW_INT.lon*1.0e-7,
            GPS_RAW_INT.alt*0.001
        ]
        t_old = t
        write_buffer.put(MavData(row))


def request_message_interval(master : mavutil.mavudp, message_id: int, frequency_hz: float):
    """
    Request MAVLink message in a desired frequency,
    documentation for SET_MESSAGE_INTERVAL:
        https://mavlink.io/en/messages/common.html#MAV_CMD_SET_MESSAGE_INTERVAL

    Args:
        message_id (int): MAVLink message ID
        frequency_hz (float): Desired frequency in Hz
    """
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        message_id, # The MAVLink message ID
        1e6 / frequency_hz, # The interval between two messages in microseconds. Set to -1 to disable and 0 to request default rate.
        0, # Target address of message stream (if message has target address fields). 0: Flight-stack default (recommended), 1: address of requestor, 2: broadcast.
        0, 0, 0, 0)


print("Requesting Connection to MAV...")
# mav_connection = mavutil.mavlink_connection('udpout:raspberrypi.local:14550', )

if not args.no_imu:
    mav_connection = mavutil.mavlink_connection('udpin:127.0.0.1:14550')
    mav_connection.wait_heartbeat()
    print("Heartbeat from system (system %u component %u)" % (mav_connection.target_system, mav_connection.target_system))

    print("Requesting 200 Hz IMU")

    # RAW_IMU at 200Hz
    request_message_interval(mav_connection, mavutil.mavlink.MAVLINK_MSG_ID_RAW_IMU, 200.0)

    # ATTITUDE and GLOBAL_POSITION_INT at 20Hz
    request_message_interval(mav_connection, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 20.0)
    request_message_interval(mav_connection, mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 20.0)
    request_message_interval(mav_connection, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 20.0)
    #QUATERNION attitude at 200Hz
    request_message_interval(mav_connection, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 100.0)
    request_message_interval(mav_connection, mavutil.mavlink.MAVLINK_MSG_ID_AHRS2, 100.0)


now = datetime.datetime.now()
dt = now.strftime("%Y-%m-%d_%H-%M-%S")
vio_cap_dir = "VIO_Capture_{}".format(dt)
os.mkdir(vio_cap_dir)

# Start the IMU and camera threads
write_buffer = queue.Queue()

#This is used to help adjust the starting recording time 
while True:
    msg = mav_connection.recv_match(type=["SYSTEM_TIME"], blocking=True)
    mtype = msg.get_type()
    if mtype != 'SYSTEM_TIME':
        continue
    else:
        t0 = msg.time_boot_ms*1.0e-3
        t_noGPS = time.time()
        break



#possibly start IMU reccording thread 
if not args.no_imu:
    imu_thread = threading.Thread(target=record_imu, args=(t0,))
    imu_thread.start()

#possibly start image recording thread
if not args.no_video:
    print("Video recording is not active!")

else:
    cam_thread = threading.Thread(target=record_cam, args=(t0, args.i))
    cam_thread.start()

#start thread that writes data out of the buffer 
write_thread = threading.Thread(target=write_from_buffer, args=(t0, vio_cap_dir))
write_thread.start()

print("Data is saved to {}/".format(vio_cap_dir))
np.savetxt(vio_cap_dir+'/timestamp.txt',[t0],fmt='%3.7f')

if args.noGPS:
    print("attitude data is being recorded assuming that you have no GPS input")




