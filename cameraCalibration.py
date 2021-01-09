import numpy as np
import misc

import os
import cv2

debugFolder = 'debugOutput/'

class CameraCalibrator:
    def __init__(self):
        self._mtx = None
        self._dist = None
        self._rvecs = None
        self._tvecs = None
        self._isCalibrated = False

    def reset(self):
        self._mtx = None
        self._dist = None
        self._rvecs = None
        self._tvecs = None
        self._isCalibrated = False

    def calibCamera(self, calibrationCatalog: [str], nx: int, ny: int, verbose = False) -> bool:
        '''
        Calibrates a camera with a given set of chessboard pictures.
        :param calibrationCatalog: List of calibration image filenames (absolute or relative)
        :param nx: Size of chessboard in X direction
        :param ny: Size of chessboard in Y direction
        :returns: success (bool): Returns True if successful and False if calibration failed
        '''
        # Reset existing calibration
        self.reset()
        # Prepare obj and img point arrays

        objPts = []  # 3d points in real world
        imgPts = []  # 2d points in image space
        # Prepare object ref. points -> they don't change as calibration pictures are all images form the
        # same object in the real world
        objP = np.zeros((nx * ny, 3), np.float32)
        objP[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        cameraShape = None
        # Loop through calibration image catalog
        for imageFilename in calibrationCatalog:
            # Read in image
            image = cv2.imread(imageFilename)
            # Store the shape of an image if not done already
            if cameraShape == None:
                cameraShape = image.shape
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            # If found add them to the list
            if ret == True:
                # Append found image points plus standard object points
                imgPts.append(corners)
                objPts.append(objP)
                # If verbose is ON, let's draw the images and store them
                if verbose:
                    cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
                    filename = debugFolder + 'chessboardCorners_' + misc.getFilenameFromPath(imageFilename)
                    cv2.imwrite(filename, image)
        # Determine calibration parameters and store them
        self._isCalibrated, self._mtx, self._dist, self._rvecs, self._tvecs = cv2.calibrateCamera(objPts, imgPts, cameraShape[1::-1], None, None)
        return self._isCalibrated

    def undistortImage(self, img: np.ndarray) -> np.ndarray:
        '''
        Takes in a distorted image img, distorts it, and returns the undistorted image based on calibration.
        If camera was not calibrated, it returns the original image.
        :param img: An distorted image
        :returns dst: An undistorted image
        '''
        dst = img.copy()
        if self._isCalibrated:
            dst = cv2.undistort(img, self._mtx, self._dist, None, self._mtx)
        else:
            print("Could not start undistortion! Camera has not been calibrated yet!")
        return dst



if __name__ == "__main__":
    # Load to be undistort image
    distortedImage = cv2.cvtColor(cv2.imread("test_images/straight_lines1.jpg"), cv2.COLOR_BGR2RGB)
    # Make a list of calibration images
    import misc
    calibrationCatalog = misc.getFileRelativeFilepathsInDirectory("camera_cal/")
    # Chessboard size
    nx = 9
    ny = 6
    # Init camera calibrator up and running
    camCalib = CameraCalibrator()
    # Try camera calibration
    if camCalib.calibCamera(calibrationCatalog,nx,ny, True):
        print("Camera Calibration successful")
        # Use calibrator to undistort image
        undistorted = camCalib.undistortImage(distortedImage)
        # Show distorted and undistorted next to each other
        f = misc.showBeforeAfter(distortedImage, "Original", undistorted, "Undistorted")
        f.savefig(debugFolder + 'beforeAfter_distortion.png')
        # Save it to debugFolder
    else:
        print("Failed")
