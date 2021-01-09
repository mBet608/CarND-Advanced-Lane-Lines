## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./reportImages/chessboardCorners_calibration12.jpg "Chessboard Points"
[image2]: ./reportImages/chessBoard_beforeAfter_distortion.png "Chessboard undistorted"
[image3]: ./reportImages/straightLine_beforeAfter_distortion.png "Straight line undistorted"
[image4]: ./reportImages/binaryImage.jpg "Binary Image"
[image5]: ./reportImages/trapezoidFinder.png "Trapezoid Finder"
[image6]: ./reportImages/slidingWindow.png "Sliding Windows"
[image7]: ./reportImages/trackedPolyfit.png "Tracked polyfit window"
[image8]: ./reportImages/finalImage.jpg "Final Image"


[video1]: test_videos_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration
Using the provided camera calibration pictures (chessboard pictures) I determined the intrinsics and extrinsics of the camera used. To allow easy use I create the class `CameraCalibrator` which can be found in `cameraCalibraton.py` and used in `main.py`in the lines `#101-103`. In the next images we see how the edges points of the chessboard are successfully detected.
![alt text][image1]
A distorted vs. an undistorted chessboard image can be observed in the next image.
![alt text][image2]

### Pipeline (single images)

#### 1. Undistortion of road image
The first step in the pipline, after the camera calibration, has to be to use the obtained camera parameters to undistort the image of the road. The difference is not obvious at the first glance but gets quite prominent in the corners. The next images shows an example.
![alt text][image3]
The opertion is performed in line `#28`in `main.py`. 
#### 2. Creation of binary image
A set of filters are being used for the binary images generation:
* Gradient filter in X direction (min: 15 | max: 200 | kernel: 5 )
* Magnitude filter (min: 38 | max: 124 | kernel: 5 )
* S-channel filter (min: 103 | max: 200 )
* Gaussian blur filter (kernel: 5)

The Gaussian blur filter is applied at first to be less sensitive to small clutter. The gradient and magnitude filters are combined via and logical AND and the result then combined with the s-channel filter via a logical OR. The code for this can be found in the lines `#8-25` in `main.py` using helper functions defined in `colorAndGradientThresholding.py`. with an example picture shown next:

The s-channel filter has a dominant influence due to the relative low min and relative high max value chosen. This was done to obtain a maximum detection especially of yellow lines in difficult light conditions. The obvious tradeoff are more unwanted gradients like the shadow patch in the ego lane at height of the neighbor lane vehicle, as depicted in the next image:
![alt text][image4]
#### 3. Perspective transformation
The actual transforming or "warping" of the picture can be done relatively straight forward using the openCV provided function `cv2.getPerspectiveTransform(src, dst)`. The step itself is done in line `#36` with the aid of `perspectiveTransformation.py`. The more tricky part is to come up with some good source image points. To do this, a openCV based, simple GUI app "TrapezoidFinder" was developed. With the aid of an image where the lane is assumed straight, the parameters are tweaked so long until a satisfactory projection is obtained in the birdseye window. An example can be seen here:
![alt text][image5]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 720       | 100, 720       | 
| 1280, 720    | 1180, 720      |
| 716, 456     | 1180, 0        |
| 562, 456     | 100, 0         |

#### 4. Lane pixel and boundary detection
Up to that point we did the pre-processing for the actual lane boundary detection. The task on hand is to detect those pixels that describe the lane boundaries. This happens in the warped birdseye view. To get an first idea where to start, knowing where the rough lateral position of the left and right ego lane boundary is, a histogram is used. The histogram is split in the middle and highest peaks are taken. Next step is to start from there using the so called "sliding-window" method to "capture" those pixel that are in the ROI (region of interest). The sliding window is thereby propagated from the bottom of the picture going upwards in a predefined step size. Each window is centered laterally using the average x coordinate of the "captured" pixels of in the window. The code where is step is performed can be found in line `#47` in `main.py` with the aid of `laneDetectionHelper.py`.
The following image gives an idea how this looks:
![alt text][image6]

To obtain a smooth lane boundary 2nd order polynomials are fitted through the masked pixels from the the sliding window detection (red and blue pixels) both for left and right side. The main drawback of the sliding window method is, it does not perform well if line markings are not visible well or not in sufficient number. To overcome this, the last polyfit is being used also in the next cycle under the assumption the situation has not changed much from cycle to cycle (or frame to frame). The next image illustrate this step. The code where is step is performed can be found in line `#42` in `main.py` with the aid of `laneDetectionHelper.py`.
![alt text][image7]
To even more stabilize the estimated polyfit, the history is kept and a average is of up to 5 cycles is used for the final fit. Also there is a minium of 5000 pixels that must be detected by the tracked polyfit window method for each side at all times. If this condition is violated, the tracked history is reset and the sliding window is used to search again from scratch. 
The code where these steps are performed can be found in the lines `#40-56` in `main.py` with the aid of `laneDetectionHelper.py`. 

#### 5. Calculation of lane meta data (curvature radius, lane width, lane middle offset)
Now that we have a polyfit for left and right lane boundary, we can obtain more meta data of the found ego lane. First a transformation of pixels to real world units needs to be performed. The transformation in x direction per pixel as been defined as `xm_per_pix = 3.7 / 700` and in y direction as `ym_per_pix = 30 / 720`. The curvature radius can be obtained via a simple formula:

 ```curverad = ((1 + (2 * polyFit[0] * y_eval + polyFit[1]) ** 2) ** 1.5) / np.absolute(2 * polyFit[0])```
 
With `y_eval = imgHeight * ym_per_pix` and the `polyFit` with transformed filtered pixels into meters using `ym_per_pix` and `xm_per_pix`.
The lane width is obtained by determining the x value at the bottom of the image using the redone `polyFit` in meters for left and right boundary. The offset is determined by calculating the delta between ideal middle (mid of image) and the actual mid between the found lane boundaries right at the bottom of the image.
The code where these steps are performed can be found in line `#60` in `main.py` with the aid of `laneDetectionHelper.py`. 

#### 6. Final image
The final result looks like this:
![alt text][image8]

---

### Pipeline (video)

#### 1. Applied on the "project_video.mp4""
[See the video here](test_videos_output/project_video.mp4)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are few issues still reaming. The found lane boundaries are getting wobbly in some situations for example for multiple reasons. First, due to the relaxed s-channel filter, a lot of noise was introduced. With more targeted filters as well as special color filters a better result could be obtained. But also the polyfit method can be improved using algorithms like RANSAC. Also range of the polynomial parameters can be more restricted to only allow for so much dynamic/flexibility. State of the art is to use clothoids as mathematical representation as they are also used when the road is designed by the infrastructure architects.
Another improvement point would be higher sophisticated sanity checks for the estimation (currently only number of points is used as feature). Also instead of taking a simple low pass (average) filter, one could apply a Kalman filter to allow for more accurate tracking of the lane boundaries.
