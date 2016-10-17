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

I ended up abbandoning the idea of using the intersection points to identify the location of the object. I now use the mean intensity in each uniquely found object tile area and I assume that this corresponds to the object intensity. Yet, it seems like a real challenge to create a completely invariant method to changes of intensity and Kinect sensors aberrations. I am reaching the completion of the suggested algorithm, by addressing to small bugs, that prevent me from making any more progress. When I have solved them all, while I believe that after that the speed of the computations will have drastically increased, I might add a scipy library to accelerate the 'find the center of mass' pass. Apart from the above, I implemented a method to read from recorded data, aka rosbag files. This will enable me to assess the quality of my method on known video data, before proceeding to the real-time implementation. 



