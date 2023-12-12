#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from active_slam.msg import MeasurementPacket, SegmentMeasurement
from cv_bridge import CvBridge, CvBridgeError
import cv2
from message_filters import TimeSynchronizer, Subscriber
from std_msgs.msg import Header
import numpy as np
from collections import deque

class SegmentViz:
    def __init__(self):
        rospy.init_node('segment_viz', anonymous=True)

        # Define parameters
        self.image_topic = '/airsim_node/Multirotor/front_center_custom/Scene'
        self.measurement_topic = '/measurement_packet'
        self.publish_topic = '/annotated_image'
        self.fifo_size = 10

        # Initialize FIFO and image dictionary
        self.image_fifo = deque(maxlen=self.fifo_size)
        self.segment_dict = {}

        # Initialize ROS publishers and subscribers
        self.image_sub = Subscriber(self.image_topic, Image)
        self.measurement_sub = Subscriber(self.measurement_topic, MeasurementPacket)
        self.ts = TimeSynchronizer([self.image_sub, self.measurement_sub], 100)
        self.ts.registerCallback(self.callback)

        self.image_pub = rospy.Publisher(self.publish_topic, Image, queue_size=10)

        # Initialize CvBridge
        self.bridge = CvBridge()

    def callback(self, image_msg, measurement_msg):
        print(f"processing sequence number: {measurement_msg.sequence}")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Process segments
        # print(len(measurement_msg.segments))
        print([seg.sequence for seg in measurement_msg.segments])
        for segment in measurement_msg.segments:
            if segment.sequence not in self.segment_dict:
                self.segment_dict[segment.sequence] = [segment]
            else:
                self.segment_dict[segment.sequence].append(segment)
        if measurement_msg.sequence in self.segment_dict:
            print(f"this frame had this many: {len(self.segment_dict[measurement_msg.sequence])}")

        # Process images
        self.image_fifo.append((measurement_msg.sequence, cv_image))
        self.process_images()
        print("finished cb")

    def process_images(self):
        if len(self.image_fifo) == self.fifo_size:
            seq, img = self.image_fifo.popleft()
            segments = self.segment_dict.get(seq, [])
            # segments = self.segment_dict[seq]

            # Draw segments on the image
            for segment in segments:
                center = (int(segment.center.x), int(segment.center.y))
                cv2.circle(img, center, 5, (0, 255, 0), -1)
                cv2.putText(img, str(segment.id), (center[0] + 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Publish annotated image
            annotated_img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            annotated_img_msg.header = Header(stamp=rospy.Time.now())
            self.image_pub.publish(annotated_img_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = SegmentViz()
        node.run()
    except rospy.ROSInterruptException:
        pass


# #!/usr/bin/env python

# import rospy
# import cv2
# from sensor_msgs.msg import Image
# from std_msgs.msg import Header
# from geometry_msgs.msg import PoseWithCovariance, Pose2D, Pose
# from active_slam.msg import SegmentMeasurement
# from cv_bridge import CvBridge
# from message_filters import TimeSynchronizer, Subscriber

# class SegmentViz:
#     def __init__(self):
#         rospy.init_node('segment_viz')

#         # Parameters
#         self.image_topic = '/airsim_node/Multirotor/front_center_custom/Scene'
#         self.measurement_topic = '/measurement_packet'

#         # Image and segment FIFO
#         self.image_fifo = []
#         self.segment_fifo = []

#         # Initialize ROS publishers and subscribers
#         self.image_sub = Subscriber(self.image_topic, Image)
#         self.measurement_sub = Subscriber(self.measurement_topic, SegmentMeasurement)
#         self.ts = TimeSynchronizer([self.image_sub, self.measurement_sub], 10)
#         self.ts.registerCallback(self.callback)

#         self.annotated_image_pub = rospy.Publisher('/annotated_image', Image, queue_size=10)

#         self.bridge = CvBridge()

#         rospy.spin()

#     def callback(self, image_msg, measurement_msg):
#         # Convert image to OpenCV format
#         cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")

#         # Get current and old segments
#         for seg in ea

#         # Store image and measurement in FIFO
#         self.image_fifo.append((measurement_msg.sequence, cv_image))
#         self.segment_fifo.append((measurement_msg.sequence, measurement_msg.segments))

#         # Process FIFO if it is full
#         if len(self.image_fifo) > 20:
#             self.process_fifo()

#     def process_fifo(self):
#         # Get the oldest image and segments
#         oldest_image_seq, oldest_image = self.image_fifo.pop(0)
#         oldest_measurement_seq, oldest_measurement = self.segment_fifo.pop(0)

#         # Process image and segments
#         annotated_image = self.process_image(oldest_image, oldest_measurement)

#         # Publish annotated image
#         self.publish_annotated_image(annotated_image, oldest_image_seq)

#     def process_image(self, image, segments):
#         # Process the image and draw segments
#         for segment in segments:
#             center_point = (int(segment.center.x), int(segment.center.y))
#             cv2.circle(image, center_point, 5, (0, 255, 0), -1)  # Draw a green circle at the center
#             cv2.putText(image, str(segment.id), center_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Write ID next to the center

#         return image

#     def publish_annotated_image(self, annotated_image, seq):
#         # Convert annotated image to ROS format
#         annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="passthrough")

#         # Set header information
#         annotated_image_msg.header = Header()
#         annotated_image_msg.header.stamp = rospy.Time.now()
#         annotated_image_msg.header.seq = seq

#         # Publish the annotated image
#         self.annotated_image_pub.publish(annotated_image_msg)

# if __name__ == '__main__':
#     try:
#         SegmentViz()
#     except rospy.ROSInterruptException:
#         pass
