import colorAndGradientThresholding as colGrad
import cv2
import numpy as np


def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car

    bottom_half = None

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

    return histogram

def findLanePixelsWithSlidingWindow(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) &
                          (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) &
                           (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img

def fitPolynomial(leftx, lefty, rightx, righty):
    # Find our lane pixels first
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def measureCurvature(leftPolynomial, rightPolynomial, yCoordWhereToMeasureCurveRadius):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = yCoordWhereToMeasureCurveRadius * ym_per_pix

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * leftPolynomial[0] * y_eval + leftPolynomial[1]) ** 2) ** 1.5) / np.absolute(
        2 * leftPolynomial[0])
    right_curverad = ((1 + (2 * rightPolynomial[0] * y_eval + rightPolynomial[1]) ** 2) ** 1.5) / np.absolute(
        2 * rightPolynomial[0])

    return left_curverad, right_curverad

def getPointsForEgoBoundaryParabola(parabola, start, end):
    y = np.arange(start, end)
    # x = Ay2 + By + C
    x = parabola[0]*(y**2) + parabola[1]*y + parabola[2]

    xy = np.vstack((x, y)).T
    return xy

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def calcEgoLanePosAndWidth(leftFitReal, rightFitReal, idealMiddleReal, birdsEyeHeightReal):
    # Determine intersection with bottom of image and poly-fits
    leftLanePosReal = leftFitReal[0]*birdsEyeHeightReal**2 + leftFitReal[1]*birdsEyeHeightReal + leftFitReal[2]
    rightLanePosReal = rightFitReal[0] * birdsEyeHeightReal ** 2 + rightFitReal[1] * birdsEyeHeightReal + rightFitReal[2]
    # Lane width is the distance between left and right lane boundary
    laneWidth = np.abs(rightLanePosReal - leftLanePosReal)
    # Actual middle is the mid-point between left and right lane boundary
    actualMiddleReal = leftLanePosReal + laneWidth/2.0
    # Lane middle offset is described as the difference between the ideal (middle of image) and actual middle
    laneMiddleOffsetReal = np.abs(actualMiddleReal - idealMiddleReal)
    return laneWidth, actualMiddleReal, laneMiddleOffsetReal

def getOverlay(leftFit, rightFit, shape):
    # Get sample points with 1px resolution in y
    pointsLeft = getPointsForEgoBoundaryParabola(leftFit, 0, shape[0])
    pointsRight = getPointsForEgoBoundaryParabola(rightFit, 0, shape[0])
    # Combine left and right side to polygon -> flip on side to produce continuous shape
    points = np.vstack((pointsLeft, np.flipud(pointsRight)))
    points = points.reshape((1, -1, 2)).astype(int)
    # Create actual overlay
    overlay = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(overlay, points, (100, 100, 50))
    return overlay

def search_around_poly(left_fit,right_fit, binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = (nonzerox > (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy +
                                  left_fit[2] - margin)) & (nonzerox < (left_fit[0] * nonzeroy ** 2 +
                                                                        left_fit[1] * nonzeroy + left_fit[2] + margin))
    right_lane_inds = (nonzerox > (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy +
                                   right_fit[2] - margin)) & (nonzerox < (right_fit[0] * nonzeroy ** 2 +
                                                                          right_fit[1] * nonzeroy + right_fit[
                                                                              2] + margin))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty,

def determineLaneMetaData(leftX, leftY, rightX, rightY, imgWidth, imgHeight):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    # Get polynomial based on meter

    leftFitReal, rightFitReal = fitPolynomial(leftX * xm_per_pix, leftY * ym_per_pix, rightX * xm_per_pix,
                                              rightY * ym_per_pix)
    leftCurvature, rightCurvature = measureCurvature(leftFitReal, rightFitReal, imgHeight)
    # Get extra information like lane width, actual in lane position and lane center offset in meter vs.
    laneWidthReal, actualMiddleReal, laneMiddleOffsetReal = calcEgoLanePosAndWidth(leftFitReal, rightFitReal,
                                                                               0.5 * imgWidth * xm_per_pix,
                                                                               imgHeight * ym_per_pix)
    return laneWidthReal, laneMiddleOffsetReal, np.mean([leftCurvature,rightCurvature])


class LaneBoundaryTrackingState:
    def __init__(self):
        self.useSlidingWindow = True
        self.cycleCounter = 0
        self.leftFits = []
        self.rightFits = []
        self.bufferLen = 5

    def addLeftAndRightFit(self, leftFit, rightFit):
        # Lets implement a ring buffer
        if len(self.leftFits) == self.bufferLen:
            self.leftFits.pop(0)
            self.rightFits.pop(0)

        self.leftFits.append(leftFit)
        self.rightFits.append(rightFit)

    def calcLeftAndRightFit(self):
        # First cycle, this should not happen
        if self.cycleCounter == 0:
           return np.zeros(3,np.float), np.zeros(3,np.float)

        leftFit = np.mean(self.leftFits, axis=0)
        rightFit = np.mean(self.rightFits, axis=0)

        return leftFit, rightFit

    def resetLeftAndRightFit(self):
        self.leftFits.clear()
        self.rightFits.clear()