Hi. I am Vassilis Lemonidis, student of ECE,NTUA. Currently focusing on computer vision and robotics.
# ProjectionPainting
Paint on any surface using Kinect and Projecting hardware.

# Current Progress
Till now I am able to read either a stream from windows kinect sdk, by using KinectStreamer, or a stream from the Kinect itself, followed by background modelling and subtraction (the background is supposed to be static and not reflecting, so that IR sensors identify depth accurately), moving object detection (the object is assumed initially to be an arm) and hand detection, by approximating forearm shape. Currently I am between the improvement of previous results by infusing color images to the process, something quite frustrating and time consuming (probably will leave it out), and the implementation of a 3d CNN that is capable of classifying 3 actions and 4 static gestures. I am using the action gestures training dataset from Cambridge University as starting point (http://www.iis.ee.ic.ac.uk/icvl/ges_db.htm). After this stage, I will match each gesture to an equivalent module behaviour ,furnish code in order to make the project friendlier to a normal user ,construct the GUI and join every part together. I am a bit off in terms of realtime application (~80 ms procedure for each frame till now)

# Update (10/10/2016)

Through experimentation, I ended up changing totally the background substraction algorithm. The reason for this change is that the assumption that the background is static is wrong, mainly because the moving object tends to reflect IR radiation not only to the sensors that 'are looking towards" it, but also to the sensors that are used for reference of the background. As a result, I created a "distortion-field" algorithm, as follows:

1. Segmentation of the initial background into small tiles, without altering the initial edges, thus making a grid of objects.
2. Calculation of the initial center of mass for each of the objects, by using the intensity image ,without the moving object.

For each frame after this:

3. Calculation of the new center of mass and the dislocation observed.
4. Matching of each dislocation to the appropriate line of movement and checking the intersection points of those lines.


The main concept is that the intersection points will belong to the moving object. However, as for now, I have not been able to produce admirable results, therefore I am currently in the process of debugging/tuning. Due to this situation, it was impossible to move further with the rest of the project.

# Update (17/10/2016)

I ended up abbandoning the idea of using the intersection points to identify the location of the object. I now use the mean intensity in each uniquely found object tile area and I assume that this corresponds to the object intensity. Yet, it seems like a real challenge to create a completely invariant method to changes of intensity and Kinect sensors aberrations. I am reaching the completion of the suggested algorithm, by addressing to small bugs, that prevent me from making any more progress. When I have solved them all, while I believe that after that the speed of the computations will have drastically increased, I might add a scipy library to accelerate the 'find the center of mass' part. Apart from the above, I implemented a method to read from recorded data, aka rosbag files. This will enable me to assess the quality of my method on known video data, before proceeding to the real-time implementation. 

#Update (22/10/2016)

Almost after a week of no particular development and a lot of brainstorming, I found out some beautiful ideas to implement, based on the center of masses concept. The first one is that the dislocation of the center of mass can approximate the vertical vector to the contour of the object, if linearity is assumed, captured inside a cell from the active ones. Sadly, this idea doesn't always work, because the assumption of contour linearity does not stand true in every cell and there is high susceptibility to noisy contours. The second one is less complex and seems it will work. Based on the idea presented on the previous update, I instead firstly normalise the intensity in each cell and compare that with the initial normalised one. By setting a threshold (which, I believe, can be found to be dependent of the total intensity variance in a cell neighborhood), I receive a mask, with the object inside. There are several bugs that don't allow this to happen, however it is a matter of little time and great patience that this piece of work is finally coming to an end. 

#Update (27/10/2016)

At last the object detection is over for now. This is because I have set high depth threshold, which in realtime applications will be probably useless, considering as normal action the touching of the surface on which someone wants to draw. At least now I have an algorithm highly invariant to Kinect noise, highly realtime (~100fps with CPU only) thus it is time to focus on the next big thing: How can I perform hand pose recognition. To this end, I found very interesting and highly promising the approach of a 2015 paper, in which random forests mixed with hand modeling are used, to find hand pose, with only input the depth image. They use many metrics to give outstanding results in above realtime processing with CPU utilisation. The paper is called 'Fast and Robust Hand Tracking Using Detection-Guided Optimization' and its respective page is http://handtracker.mpi-inf.mpg.de/projects/FastHandTracker/ 

#Update (30/10/2016)

Created hand model (create_hand_model.py) , by inspecting (and slightly correcting) the DH parameters presented in the PHD thesis "Kinematic Model of the Hand using Computer Vision" of Edgar Simo Serra. I now need to give some volume to this model, by implementing the 3D gaussians, and also speed it up a little after this, as I think it might slow down a little the whole procedure.


#Update (16/11/2016)

I ended up abbandoning the whole idea of the paper  as it needed different
dataset from the one provide and too much effort and I had no time to 
accomplish such a target. Instead I created a fast hand segmentation 
algorithm, which makes use of polar coordinates to identify joints
and links that form the arm and by scanning for anomalies it ends up
identifying hand and fingers area. The algorithm is robust to noise and
low missing data rates, but needs tuning and, while it was pretty efficient in
identifying the example given (run 'python hand_segmentation_alg.py' to view
it), does not provide good results while it is matched with the 'distortion
field' algorithm. Apart from the tuning, there is not taken into account the
possibility that more than one hands can enter the detection field, although
this can be fixed with simple operations. SO, when I finish tuning, I hope to
start organizing the long-awaited action recognition algorithm.




