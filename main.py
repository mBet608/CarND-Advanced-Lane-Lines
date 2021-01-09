import cameraCalibration as cal
from laneDetectionHelper import *
from perspectiveTransformation import transform
import misc
import cv2
import numpy as np

def colorAndGradientFilter(img):
    # Setup hyper parameters
    blurKernelSize = 5
    sobelKernelSize = 5
    absSobelXThresh = (15, 200)
    magSobelThresh = (38, 124)
    sChannelThresh = (103, 200)
    # Blur image
    blurImg = colGrad.blurFilter(img, blurKernelSize)
    # Get gradient image via combination of x-dir abs. sobel and magnitude filter
    xGradImg = colGrad.absSobelThreshFilter(blurImg, 'x', sobelKernelSize, absSobelXThresh)
    magGradImg = colGrad.magSobelThreshFilter(blurImg, sobelKernelSize, magSobelThresh)
    gradImg = colGrad.mergeWithAnd(xGradImg, magGradImg)
    # Filter via s-channel filter esp. for yellow this is important
    sChannelImg = colGrad.sChannelThreshFilter(blurImg, sChannelThresh)
    # Merge gradient and s-channel images via or to get best out of both worlds
    finalImg = colGrad.mergeWithOr(gradImg, sChannelImg)
    return finalImg

def detectLaneBoundaries(image, trackingState):
    image = calibrator.undistortImage(image)
    imgHeight = image.shape[0]
    imgWidth = image.shape[1]
    # Apply color and gradient threshold filter
    binaryImage = colorAndGradientFilter(image)
    #####################################################################################
    # Change perspective to birdseye
    src_pts = [np.array([0, 720]),np.array([1280, 720]),np.array([716, 456]),np.array([562, 456])]
    binaryBirdseye = transform(binaryImage,src_pts, 'bird')
    #####################################################################################
    # Find left and right pixels around tracked left and right fit. Don't do it in the first cycle
    # or if quality of fit were not good enough (-> use sliding window instead)
    if not trackingState.useSlidingWindow or trackingState.cycleCounter != 0:
        trackedLeftPoly, trackedRightPoly = trackingState.calcLeftAndRightFit()
        leftX, leftY, rightX, rightY = search_around_poly(trackedLeftPoly,trackedRightPoly, binaryBirdseye)
        if len(leftX) < 5000 or len(rightX) < 5000:
            trackingState.useSlidingWindow = True
    # Use sliding window in first cycle and whenver prev. fit wasn't good enough. Make sure to reset tracking state
    if trackingState.useSlidingWindow:
        leftX, leftY, rightX, rightY, debugImg = findLanePixelsWithSlidingWindow(binaryBirdseye)
        trackingState.useSlidingWindow = False
        trackingState.resetLeftAndRightFit()

    # Get fit for selected pixel and update tracking state
    leftFit, rightFit = fitPolynomial(leftX, leftY, rightX, rightY)
    trackingState.addLeftAndRightFit(leftFit,rightFit)
    trackingState.cycleCounter = trackingState.cycleCounter + 1
    # The fits are averaged with old fits to stabilize fits
    leftFit, rightFit = trackingState.calcLeftAndRightFit()

    #####################################################################################
    # Find width, offset to middle and curvature of lane
    laneWidth, laneMiddleOffsetReal, avgCurvature = determineLaneMetaData(
        leftX, leftY, rightX, rightY, imgWidth, imgHeight)

    #####################################################################################
    # Prepare the drawings
    # Get overlay
    overlay = getOverlay(leftFit, rightFit, image.shape)
    # Combine overlay with binary birds eye
    overlayBirdsEye = weighted_img(overlay, cv2.cvtColor(binaryBirdseye*255,cv2.COLOR_BGR2RGB))
    overlayUnwarped = transform(overlay,src_pts, 'normal')
    # Also draw overlay into original image
    overlayNormal = weighted_img(overlayUnwarped, image)
    # Resize image to prepare for in-picture effect
    overlayBirdsEyeSmall = cv2.resize(overlayBirdsEye,
                                      (int(imgWidth * 0.25), int(imgHeight * 0.25)),
                                      interpolation=cv2.INTER_AREA)
    # Draw small version of birds eye with overlay, directly into normal perspective image
    overlayNormal[0:overlayBirdsEyeSmall.shape[0], 0:overlayBirdsEyeSmall.shape[1]] = overlayBirdsEyeSmall
    # Get a window around the text information for better readability
    pos = (855, 50)
    # Set up box dimensions
    x, y, w, h = pos[0], 0, imgWidth-pos[0], 150
    # Copy section of image where we want to draw in
    subImg = overlayNormal[y:y + h, x:x + w]
    # Get rect in gray'ish color
    white_rect = np.ones(subImg.shape, dtype=np.uint8) * 120
    # Combine rect with extracted sub image
    res = cv2.addWeighted(subImg, 0.5, white_rect, 0.5, 1.0)
    # Overwrite image with our composition
    overlayNormal[y:y + h, x:x + w] = res
    # Place lane meta data
    misc.putTextInImage(overlayNormal,'Lane offset: {:0.2}m'.format(laneMiddleOffsetReal), pos)
    pos = (pos[0], pos[1]+35)
    misc.putTextInImage(overlayNormal,'Lane Width: {:.4}m'.format(laneWidth), pos)
    pos = (pos[0], pos[1]+35)
    misc.putTextInImage(overlayNormal,'Avg.Curvature: {:.4}m'.format(avgCurvature), pos)

    return overlayNormal

#####################################################################################
# Calibrate camera
calibrator = cal.CameraCalibrator()
calibImageCatalog = misc.getFileRelativeFilepathsInDirectory('camera_cal')
calibrator.calibCamera(calibImageCatalog, 9, 6)

def processTestImages():
    testImages = misc.getFileRelativeFilepathsInDirectory('test_images')
    for imageName in testImages:
        image = cv2.imread(imageName)
        trackingState = LaneBoundaryTrackingState()
        overlayNormal = detectLaneBoundaries(image, trackingState)
        cv2.imwrite('output_images/' + misc.getFilenameFromPath(imageName), overlayNormal)

def processVideo(videoName):
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    output = 'test_videos_output/' + videoName

    #Init tracking state
    trackingState = LaneBoundaryTrackingState()

    clip1 = VideoFileClip(videoName)
    # Dp the lambda trick to give additional arguments
    white_clip = clip1.fl_image(lambda image: detectLaneBoundaries(image, trackingState))
    white_clip.write_videofile(output, audio=False)

#processTestImages()
processVideo('project_video.mp4')
