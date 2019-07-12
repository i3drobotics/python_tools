#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_bag_name = "i3dr_ROS_room_scan_cam_workshop_light+laser"
    default_bag_folder = "/media/i3dr/Seagate Backup Plus Drive/SBRI_room_scans/workshop_scan"
    default_camera_name = "phobos_nuclear"

    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("-i","--bag_file", help="Input ROS bag.", default=default_bag_folder+"/"+default_bag_name+".bag")
    parser.add_argument("-o","--output_dir", help="Output directory.", default=script_dir+"/output/"+default_bag_name+"/")
    parser.add_argument("-c","--camera_name", help="Camera name (used for naming of image files)", default=default_camera_name)
    parser.add_argument("-r","--right_image_topic", help="Right image topic.", default="/"+default_camera_name+"/right/image_raw")
    parser.add_argument("-l","--left_image_topic", help="Left image topic.", default="/"+default_camera_name+"/left/image_raw")

    args = parser.parse_args()

    print "Extracting images from %s on topics (%s, %s) into %s" % (args.bag_file,
                                                          args.left_image_topic, args.right_image_topic, args.output_dir)

    bag = rosbag.Bag(args.bag_file, "r")
    print "Bag file loaded."
    bridge = CvBridge()
    count = 0
    bag_maxCount = bag.get_message_count(args.left_image_topic) + bag.get_message_count(args.right_image_topic)
    for topic, msg, t in bag.read_messages(topics=[args.left_image_topic, args.right_image_topic]):
        if (topic == args.left_image_topic or args.right_image_topic):
            rightLeft = ""
            if (topic == args.left_image_topic):
                rightLeft = "left"
            elif (topic == args.right_image_topic):
                rightLeft = "right"

            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            path = os.path.join(args.output_dir, args.camera_name + ("_{}_".format(t.secs)) + rightLeft + ".png")

            cv2.imwrite(path, cv_img)
        else:
            print "Ignoring unknown topic"

        count += 1
        perc = float(float(count)/float(bag_maxCount)) * 100

        print "Saving data... {:0.2f}%".format(perc)

    bag.close()
    print "Done."

    return

if __name__ == '__main__':
    main()