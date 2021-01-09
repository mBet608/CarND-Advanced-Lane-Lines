import numpy as np
import cv2
import cameraCalibration as cal
import misc
class TrapezoidFinder:
    def __init__(self, image):
        self._image = image
        self._lowLeftX = 0
        self._lowLeftY = 720
        self._lowRightX = 1280
        self._lowRightY = 720
        self._highRightX = 716
        self._highRightY = 456
        self._highLeftX = 562
        self._highLeftY = 456

        def onChangeLowLeftX(pos):
            self._lowLeftX = pos
            self.render()

        def onChangeLowLeftY(pos):
            self._lowLeftY = pos
            self.render()

        def onChangeLowRightX(pos):
            self._lowRightX = pos
            self.render()

        def onChangeLowRightY( pos):
            self._lowRightY = pos
            self.render()

        def onChangeHighRightX(pos):
            self._highRightX = pos
            self.render()

        def onChangeHighRightY(pos):
            self._highRightY = pos
            self.render()

        def onChangeHighLeftX(pos):
            self._highLeftX = pos
            self.render()

        def onChangeHighLeftY(pos):
            self._highLeftY = pos
            self.render()

        # Get window
        cv2.namedWindow('Trapezoid Finder')
        cv2.namedWindow('Birdseye')
        # Add trackbars for all values
        cv2.createTrackbar('Low Left X', 'Trapezoid Finder', self._lowLeftX, image.shape[1] , onChangeLowLeftX)
        cv2.createTrackbar('Low Left Y', 'Trapezoid Finder', self._lowLeftY, image.shape[0], onChangeLowLeftY)
        cv2.createTrackbar('Low Right X', 'Trapezoid Finder', self._lowRightX, image.shape[1] , onChangeLowRightX)
        cv2.createTrackbar('Low Right Y', 'Trapezoid Finder', self._lowRightY, image.shape[0], onChangeLowRightY)
        cv2.createTrackbar('High Right X', 'Trapezoid Finder', self._highRightX, image.shape[1] , onChangeHighRightX)
        cv2.createTrackbar('High Right Y', 'Trapezoid Finder', self._highRightY, image.shape[0], onChangeHighRightY)
        cv2.createTrackbar('High Left X', 'Trapezoid Finder', self._highLeftX, image.shape[1] , onChangeHighLeftX)
        cv2.createTrackbar('High Left Y', 'Trapezoid Finder', self._highLeftY, image.shape[0], onChangeHighLeftY)

        # Lets draw!
        self.render()
        # Close if any key was pressed
        cv2.waitKey(0)

    def render(self):

        pts = np.array([[self._lowLeftX, self._lowLeftY],
                        [self._lowRightX, self._lowRightY],
                        [self._highRightX, self._highRightY],
                        [self._highLeftX, self._highLeftY]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        trapImage = np.copy(self._image)
        cv2.polylines(trapImage, [pts], True, (0, 255, 255))
        cv2.imshow('Trapezoid Finder', trapImage)

        src_pts = [[self._lowLeftX, self._lowLeftY],
                                 [self._lowRightX, self._lowRightY],
                                 [self._highRightX, self._highRightY],
                                 [self._highLeftX, self._highLeftY]]
        warped = transform(self._image,src_pts, 'bird')
        cv2.imshow('Birdseye', warped)

        return

def transform( img,src_pts, direction='bird'):
    height = img.shape[0]
    width = img.shape[1]
    margin = 100
    # Set source points
    src_leftLow, src_rightLow, src_rightHigh, src_leftHigh = src_pts

    src = np.float32([src_leftHigh, src_rightHigh, src_leftLow, src_rightLow])
    # Set destination points
    dst_leftLow = np.array([margin, height])
    dst_rightLow = np.array([width - margin, height])
    dst_rightHigh = np.array([width - margin, 0])
    dst_leftHigh = np.array([margin, 0])
    dst = np.float32([dst_leftHigh, dst_rightHigh, dst_leftLow, dst_rightLow])
    # With the src and dst point we can get the perspective transformation matrix
    if direction == 'bird':
        M = cv2.getPerspectiveTransform(src, dst)
    if direction == 'normal':
        M = cv2.getPerspectiveTransform(dst, src)

    # Now all ingredients are there, we can warp
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped


if __name__ == '__main__':
    # Calibrate camera, we need to undistort before perspective transformation
    calibrator = cal.CameraCalibrator()
    calibImageCatalog = misc.getFileRelativeFilepathsInDirectory('camera_cal')
    calibrator.calibCamera(calibImageCatalog, 9, 6)
    # Read in and right away undistort image with calibrator
    imgStraight = calibrator.undistortImage(cv2.imread('test_images/straight_lines1.jpg'))

    img_size = (imgStraight.shape[1], imgStraight.shape[0])

    traFinder = TrapezoidFinder(imgStraight)
    cv2.destroyAllWindows()

    height = imgStraight.shape[0]
    width = imgStraight.shape[1]
    margin = 100
    # Set source points
    src_leftLow = np.array([traFinder._lowLeftX,traFinder._lowLeftY])
    src_rightLow = np.array([traFinder._lowRightX,traFinder._lowRightY])
    src_rightHigh = np.array([traFinder._highRightX,traFinder._highRightY])
    src_leftHigh = np.array([traFinder._highLeftX,traFinder._highLeftY])
    src = np.float32([src_leftHigh, src_rightHigh, src_leftLow, src_rightLow])
    # Set destination points
    dst_leftLow = np.array([margin,height])
    dst_rightLow = np.array([width-margin,height])
    dst_rightHigh = np.array([width-margin,0])
    dst_leftHigh = np.array([margin,0])
    dst = np.float32([dst_leftHigh, dst_rightHigh, dst_leftLow, dst_rightLow])
    # With the src and dst point we can get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Now all ingredients are there, we can warp
    warped = cv2.warpPerspective(imgStraight, M, (width, height))

    s2 = [width // 2 + 76, height * 0.625]
    s1 = [width // 2 - 76, height * 0.625]
    s3 = [-100, height]
    s4 = [width + 100, height]

    import misc
    misc.showBeforeAfter(cv2.cvtColor(imgStraight, cv2.COLOR_BGR2RGB), 'Undistorted', cv2.cvtColor(warped,cv2.COLOR_BGR2RGB), 'warped')
    misc.showFigureUntilKeyHit()




