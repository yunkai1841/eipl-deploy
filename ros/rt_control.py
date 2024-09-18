#!/usr/bin/env python3
"""
Copyright (c) Since 2023 Ogata Laboratory, Waseda University
Released under the AGPL license.
see https://www.gnu.org/licenses/agpl-3.0.txt
"""

import os
import time
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError   # for conversion between ros data and cv2
import torch
from dynamixel_sdk import *
from dynamixel_driver import *
# ROS
import rospy
from sensor_msgs.msg import JointState, Image

from eipl.utils import restore_args
from eipl.utils import resize_img, normalization, tensor2numpy, deprocess_img
from eipl.model import SARNN

from trt_model import SARNNTRT

joint_arr = None
img_arr = None

def followerCallback(msg):
    global joint_arr
    joint_arr = msg.position
    

def cameraCallBack(msg):
    global img_arr
    np_img = np.frombuffer(msg.data, np.uint8)
    np_img = np_img.reshape((msg.height, msg.width, 3))
    # np_img = np_img[:-120,400:1000]
    img_arr = np.array(cv2.resize(np_img, (64, 64)))

def main(freq, exp_time, motor_list, model_path, input_param):
    rospy.loginfo_once(model_path)
    motor_names = ['motor_{}'.format(m) for m in motor_list]
    
    # publisher
    rospy.Subscriber("/follower/joint_states", JointState, followerCallback)
    rospy.Subscriber("/camera/color/image_raw", Image, cameraCallBack)
    joint_pub = rospy.Publisher('/leader/joint_states', JointState, queue_size=1)
    img_pub   = rospy.Publisher("/camera/concat/image", Image, queue_size=10)

    # setup the message
    joint_msg = JointState()
    joint_msg.name = motor_names

    # restore parameters
    model_dir_name = os.path.split(model_path)[0]
    params = restore_args(os.path.join(model_dir_name, "args.json"))
    
    # load dataset
    minmax = [params["vmin"], params["vmax"]]
    joint_bounds = np.load("/home/zhu/catkin_ws/src/om_teleop2/SARNN/data/joint_bounds.npy")
    # joint_msg.header.stamp = rospy.Time.now()
    # joint_msg.position = initial_joint
    # joint_pub.publish(joint_msg)
    # print("initial", initial_joint)


    # model = SARNN(
    #     rec_dim=params["rec_dim"],
    #     joint_dim=5,
    #     k_dim=params["k_dim"],
    #     heatmap_size=params["heatmap_size"],
    #     temperature=params["temperature"],
    #     im_size=[64,64]
    # )
    # engine_path = model_path.replace(".pth", ".engine")
    engine_path = "/home/zhu/Documents/eipl-deploy/models/sarnn-om/sarnn.engine"
    model_trt = SARNNTRT(engine_path=engine_path)
    
    # ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    # model.load_state_dict(ckpt["model_state_dict"])
    # model = model.cuda()
    # model.eval()
    
    state = None
    y_img, y_joint = 0.0, 0.0
    img_size = 64                   # image size for prediction
    nloop = int(freq * exp_time)    # loops for prediction
    rate = rospy.Rate(freq)         # opelation cycle
    # rate = rospy.Rate(1)         # opelation cycle
    stay_loop = int(freq * 2.0)     # loops to stabilize the initial pause
    
    # t_img, t_joint : test image, test joint
    # y_img, y_joint : predicted image, predicted joint
    # rt_img : real time image

    initial_joint = np.load("/home/zhu/catkin_ws/src/om_teleop2/SARNN/data/train/joints.npy")[0, 0]
    joint_msg.header.stamp = rospy.Time.now()
    joint_msg.position = initial_joint
    joint_pub.publish(joint_msg)
    print("initial", initial_joint)
    rate.sleep()

    rospy.logwarn("Playback: Starting execution")
    for loop_cnt in range(nloop):
        if (img_arr is not None) and (joint_arr is not None):

            rt_img = img_arr
            # t_img = np.expand_dims(rt_img, 0)
            t_img = normalization(rt_img, (0,255), minmax )
            t_img = np.transpose(t_img.astype(np.float32), (2, 0, 1))
            # t_img = torch.Tensor(t_img).cuda()
            # normalize joint
            t_joint = np.array(joint_arr, dtype=np.float32)
            # t_joint = np.expand_dims(t_joint, 0)
            t_joint = normalization(t_joint, joint_bounds, minmax)
            # t_joint = torch.Tensor(t_joint).cuda()

            # predict image and joint
            # with torch.inference_mode():
            #     y_img, y_joint, ect_pts, dec_pts, state = model(t_img, t_joint, state)

            y_img, y_joint, ect_pts, dec_pts = model_trt(t_img, t_joint)
            # print(y_joint)
            rospy.loginfo("next")

            # denormalization
            # pred_image = tensor2numpy(y_img[0])
            pred_image = deprocess_img(y_img, params["vmin"], params["vmax"])
            pred_image = pred_image.reshape([3, 64, 64])
            pred_image = pred_image.transpose(1, 2, 0)
            # pred_joint = tensor2numpy(y_joint[0])
            pred_joint = normalization(y_joint, minmax, joint_bounds)
            
            # set message
            joint_msg.header.stamp = rospy.Time.now()
            joint_msg.position = pred_joint
            # rospy.loginfo(str(pred_joint))
            rospy.loginfo(str(joint_arr))

            # ? ignore the first several loops
            if loop_cnt > 10:
                joint_pub.publish(joint_msg)

            # converting the position of attention points
            # ect_pts = tensor2numpy(ect_pts)
            # dec_pts = tensor2numpy(dec_pts)
            ect_pts = ect_pts.reshape(params["k_dim"], 2) * img_size
            dec_pts = dec_pts.reshape(params["k_dim"], 2) * img_size
            ect_pts = np.clip(ect_pts, 0, img_size).astype(np.int8)
            dec_pts = np.clip(dec_pts, 0, img_size).astype(np.int8)

            # plot attention points on the predicted image
            rt_img = rt_img[:,:,::-1].copy()
            pred_image = pred_image[:,:,::-1].copy()
            for i in range(params["k_dim"]):
                cv2.circle(pred_image, tuple(ect_pts[i]), 1, (0,0,255), thickness=-1)
                cv2.circle(pred_image, tuple(dec_pts[i]), 1, (255,255,255), thickness=-1)
            
            # risize for display, concatenate real time image and predicted image
            # put text on the concatenated image
            out_img = np.concatenate((rt_img, pred_image), axis=1)
            
            # publish the concatenated image displayed at image view
            bridge = CvBridge()
            out_img_msg = bridge.cv2_to_imgmsg(out_img[:,:,::-1], encoding="rgb8")
            img_pub.publish(out_img_msg)

        rate.sleep()
        # stay for several loop to stabilize the initial pause
        if loop_cnt == 1:
            for stay_loop_cnt in range(stay_loop):
                rate.sleep()
    
    rospy.logwarn("Playback: Finished execution")
    




if __name__ == '__main__':
    joint_arr = None
    img_arr = None
    try:
        rospy.init_node('virtual_leader_node', anonymous=True)
        sleep_time = rospy.get_param('virtual_leader_node/sleep_time')      # delay time for starting this node
        freq = rospy.get_param('virtual_leader_node/freq')
        exp_time = rospy.get_param('virtual_leader_node/exp_time')
        motor_list = rospy.get_param('virtual_leader_node/motor_list')
        motor_list = [ eval(m) for m in motor_list.split(',') ]
        model_path = rospy.get_param('virtual_leader_node/model_path')
        input_param = rospy.get_param('virtual_leader_node/input_param')    # percentage of measurements used in predictive model
        
        time.sleep(sleep_time)
        main(freq, exp_time, motor_list, model_path, input_param)
    except rospy.ROSInterruptException:
        pass

