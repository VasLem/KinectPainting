
import signal
import sys
from multiprocessing import Process, Semaphore, Queue, Event, Lock
import poses_to_actions
import time



class Generator(Process):
    '''
    <term_queue>: Queue to write termination events, must be same for all
                processes spawned
    <function>: function to call. None value means that the current class will
        be used as a template for another class, with <function> being defined
        there
    <input_queues> : Queue or list of Queue objects , which refer to the input
        to <function>.
    <output_queues> : Queue or list of Queue objects , which are used to pass
        output
    <sema_to_acquire> : Semaphore or list of Semaphore objects, which are
        blocking function execution
    <sema_to_release> : Semaphore or list of Semaphore objects, which will be
        released after <function> is called
    '''

    def __init__(self, term_queue,
                 function=None, input_queues=None, output_queues=None, sema_to_acquire=None,
                 sema_to_release=None,name=None):
        Process.__init__(self)
        self.name = name
        self.term_queue = term_queue
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.sema_to_acquire = sema_to_acquire
        self.sema_to_release = sema_to_release
        if function is not None:
            self.function = function

    def run(self):
        if self.sema_to_release is not None:
            try:
                self.sema_to_release.release()
            except AttributeError:
                deb = [sema.release() for sema in self.sema_to_release]
        while True:
            if not self.term_queue.empty():
                self.term_queue.put(('ka', 0))
                break
            try:
                if self.sema_to_acquire is not None:
                    try:
                        self.sema_to_acquire.acquire()
                    except AttributeError:
                        deb = [sema.acquire() for sema in self.sema_to_acquire]

                if self.input_queues is not None:
                    try:
                        data = self.input_queues.get()
                    except AttributeError:
                        data = tuple([queue.get()
                                      for queue in self.input_queues])
                    time1=time.time()
                    res = self.function(data)
                    print self.name,time.time()-time1
                else:
                    res = self.function()
                if self.output_queues is not None:
                    try:
                        if self.output_queues.full():
                            self.output_queues.get(res)
                        self.output_queues.put(res)
                    except AttributeError:
                        deb = [queue.put(res) for queue in self.output_queues]
                if self.sema_to_release is not None:
                    if self.sema_to_release is not None:
                        try:
                            self.sema_to_release.release()
                        except AttributeError:
                            deb = [sema.release() for sema in self.sema_to_release]
            except Exception as exc:
                self.term_queue.put(('ba', exc))
                break


import classifiers as cs
import rospy
import roslaunch
import message_filters as mfilters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, TimeReference
import subprocess
import extract_and_process_rosbag as epr
import class_objects as co
import poses_to_actions as p2a


class KinectStreamer(Process):

    def __init__(self, term_queue, output_queue):
        Process.__init__(self)
        self.term_queue = term_queue
        self.output_queue = output_queue
        node = roslaunch.core.Node("kinect2_bridge", "kinect2_bridge")
        rospy.set_param('fps_limit', 10)
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        self.kinect_process = launch.launch(node)

    def run(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/kinect2/sd/image_depth", Image, self.callback, queue_size=10)
        rospy.init_node('streamer', anonymous=True, disable_signals=True)
        rospy.spin()

    def callback(self, data):
        '''
        Callback function, <data> is the depth image
        '''
        time1=time.time()
        if not self.term_queue.empty():
            self.kinect_process.stop()
            #rospy.signal_shutdown('Exiting')
            self.term_queue.put((self.name, 0))
        try:
            self.output_queue.put(self.bridge.imgmsg_to_cv2(data,
                                                            desired_encoding="passthrough"))
            print 'kinect',time.time()-time1
        except CvBridgeError as err:
            print err
            return


class Preprocessing(Generator):

    def __init__(self, *args, **kwargs):
        Generator.__init__(self, *args, **kwargs)
        self.function = self.preprocess
        self.prepare_frame = epr.DataProcess(save=False)

    def preprocess(self, frame):
        data = self.prepare_frame.process(frame, low_ram=False, derotate=False)
        try:
            processed = data['hand'].frames[0]
            [angle, center] = data['hand'].info[0]
            self.prepare_frame.data = {}
            return processed, angle, center
        except (KeyError, IndexError):
            self.prepare_frame.data = {}
            return None, None, None


class Classifier(Generator):

    def __init__(self, term_queue, classifier, *args, **kwargs):
        Generator.__init__(self, term_queue, *args, **kwargs)
        self.classifier = classifier
        if 'function' in kwargs:
            self.test = kwargs['function']
        else:
            self.test = self.classifier.run_testing
        self.function = self.classify
        self.test_params = {'online': True,
                            'against_training': False,
                            'scores_filter_shape': 5,
                            'std_small_filter_shape': co.CONST['STD_small_filt_window'],
                            'std_big_filter_shape': co.CONST['STD_big_filt_window'],
                            'ground_truth_type': co.CONST['test_actions_ground_truth'],
                            'img_count': None, 'save': True, 'scores_savepath': None,
                            'load': False, 'testname': None, 'display_scores': True,
                            'derot_angle': None, 'derot_center': None,
                            'construct_gt': False, 'just_scores': True}

    def reset(self, online=True):
        self.classifier.init_testing(self.test_params)
        if not self.test_params['online']:
            self.classifier.reset_offline_test()
        else:
            self.classifier.reset_online_test()

    def setparam(self, **kwargs):
        for key in kwargs:
            self.test_params[key] = kwargs[key]

    def classify(self, *args):
        _, scores = self.test(
            *args, **self.test_params)
        if not self.test_params['just_scores']:
            if isinstance(self.classifier.recognized_classes[-1],
                          cs.ClassObject):
                return scores, self.classifier.recognized_classes[-1]
            else:
                return scores, self.classifier.recognized_classes[-1]
        else:
            return scores


def main():
    run_svm = Semaphore()
    run_rf = Semaphore()
    inp_rf = Queue()
    inp_svm = Queue()
    out_rf = Queue()
    out_svm = Queue()
    kin_stream = Queue()
    res_mixed = Queue()
    term_queue = Queue()
    processes = {}
    processes['preproc'] = Preprocessing(term_queue, input_queues=kin_stream,
                                         output_queues=[inp_rf,
                                                        inp_svm],
                                         name='preproc')
    processes['svm_class'] = Classifier(term_queue, cs.ACTIONS_CLASSIFIER_SIMPLE,
                                        input_queues=inp_svm,
                                        output_queues=out_svm,
                                        sema_to_acquire=run_svm,
                                        sema_to_release=run_rf,
                                        name='svm_class')
    processes['rf_class'] = Classifier(term_queue, cs.POSES_CLASSIFIER,
                                       input_queues=inp_rf,
                                       output_queues=out_rf,
                                       sema_to_acquire=run_rf,
                                       sema_to_release=run_svm,
                                       name='rf_class')
    mixedclassifier_simple = p2a.MixedClassifier(cs.ACTIONS_CLASSIFIER_SIMPLE,
                                                 cs.POSES_CLASSIFIER,
                                                 add_info='without sparse coding')
    mixedclassifier_simple.run_training()
    processes['mixed_class'] = Classifier(term_queue, mixedclassifier_simple,
                                          function=mixedclassifier_simple.run_mixer,
                                          input_queues=[out_rf, out_svm],
                                          output_queues=res_mixed,
                                          name='mixed_class')
    processes['mixed_class'].setparam(just_scores=False)
    processes['stream_proc'] = KinectStreamer(term_queue, kin_stream)
    signal.signal(signal.SIGINT, lambda sig,frame: signal_handler(sig,frame,
                                                                  term_queue,processes))
    [processes[key].start() for key in processes]
    while True:
        time1 = time.time()
        if not term_queue.empty():
            [processes[key].join() for key in processes]
            break
        res = res_mixed.get()
        print time.time()-time1


def signal_handler(sig, frame, term_queue, processes):
    term_queue.put((__name__, 'SIGINT'))
    try:
        [processes[key].join() for key in processes]
        while not term_queue.empty():
            print term_queue.get()
    except AssertionError:
        pass
    sys.exit(0)

if __name__ == '__main__':
    main()
