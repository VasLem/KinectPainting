'''
Module containing the bases of a ROS thread to be inherited
'''
# pylint: disable=R0903
import time
import threading
from ast import literal_eval
import numpy as np
# pylint: disable=unused-import
from sensor_msgs.msg import Image
from sensor_msgs.msg import TimeReference
# pylint: enable=unused-import
import rospy
import message_filters as mfilters
from cv_bridge import CvBridge, CvBridgeError
import class_objects as co


class Time:
    '''
    Class to keep time, necessary for rouse TimeReference message
    '''

    def __init__(self):
        self.secs = 0
        self.nsecs = 0

class ROSPipelineElement(threading.Thread):
    '''
    Class containing the basis of a ROS thread to be inherited
    '''

    def __init__(self, name, loglevel='INFO'):
        '''
        `name`:: key given in ROSNodes in config.yaml
        `loglevel`:: the logger level, defaults to `INFO`
        '''
        threading.Thread.__init__(self)
        name = co.CONST['ROSNodes'][name]['name']
        subscribe_channels = co.CONST['ROSNodes'][name]['subscribing_to']
        publish_channels = co.CONST['ROSNodes'][name]['publishing_to']
        self._name = name
        self._loglevel = loglevel
        (subscribe_channels,
         subscribe_types) = self._check_channels(subscribe_channels)

        (publish_channels,
         publish_types) = self._check_channels(publish_channels)
        if subscribe_channels is not None:
            self.image_ts = mfilters.TimeSynchronizer(
                [mfilters.Subscriber(channel,
                                     channel_type) for channel, channel_type in
                 zip(subscribe_channels, subscribe_types)], 30)
            self.image_ts.registerCallback(self.get)
        if publish_channels is not None:
            self._publishers = {channel: rospy.Publisher(channel, channel_type)
                                for channel, channel_type
                                in zip(publish_channels, publish_types)}
        else:
            self._publishers = None
        self._stop_event = threading.Event()
        self.bridge = CvBridge()

    @staticmethod
    def _check_channels(channels):
        '''
        Check input channels integrity
        '''
        if channels is None:
            return channels, None
        assert isinstance(channels, (str, list)),\
            'Incorrect argument type for channels'
        if isinstance(channels, str):
            channels = [channels]
        channels_types = [literal_eval(co.CONST['ROSChannelsTypes'][channel])
                          for channel in channels]
        return channels, channels_types

    def get_publisher(self, channel):
        '''
        Get requested publisher object
        '''
        try:
            return self._publishers[channel]
        except KeyError:
            raise BaseException(
                'Invalid provided channel, publisher does not exist')
        except TypeError:
            if self._publishers is None:
                raise TypeError("No publishers exist"
                                ", wrong class initialization")

    def run(self):
        rospy.init_node(self._name, anonymous=True, disable_signals=True)
        rospy.spin()

    def _callback(self, *channels_data):
        self.get(*channels_data)
        self.post()

    def get(self, *channels_data):
        '''
        Callback to handle channels subscribers
        '''
        pass

    def post(self):
        '''
        Function to handle channels publishers
        '''
        pass

    def stop(self):
        '''
        Stop ROS Thread
        '''
        self._stop_event.set()

    def stopped(self):
        '''
        Check if ROS Thread is stopped
        '''
        return self._stop_event.is_set()

    def post_array(self, channel, array):
        '''
        Publish a numpy array to the given channel
        '''
        self.get_publisher(channel).publish(self.bridge.cv2_to_imgmsg(
            np.atleast_2d(array)))

    def get_array(self, ros_message):
        '''
        Get a numpy array from ros_message coming from a publisher,
        who used post_array(). If error shows up, ignore it.
        '''
        try:
            return self.bridge.imgmsg_to_cv2(ros_message,
                                             desired_encoding="passthrough")
        except CvBridgeError as err:
            print(err)
        return None

    def post_time_reference(self, channel, data):
        '''
        Publish a data as time reference source to the given channel
        '''
        msg = TimeReference()
        msg.source = data
        msg.time_ref = Time()
        time_struct = rospy.Time.from_sec(time.time())
        seconds = time_struct.to_sec()  # floating point
        nanoseconds = time_struct.to_nsec()
        msg.time_ref.secs = seconds
        msg.time_ref.nsecs = nanoseconds
        self.get_publisher(channel).publish(msg)

    @staticmethod
    def get_time_reference(message):
        '''
        Get from a Time Reference message created using
        post_time_reference the needed data
        '''
        return message.source
