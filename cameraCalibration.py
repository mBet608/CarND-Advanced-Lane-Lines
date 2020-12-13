import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

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

    def calibCamera(self, calibrationCatalog, nx, ny):
        '''
        Calibrates a camera with a given set of chessboard pictures.

                Parameters:
                        calibrationCatalog ([str]): List of calibration image filenames (absolute or relative)
                        nx (int): Size of chessboard in X direction
                        ny (int): Size of chessboard in Y direction

                Returns:
                        success (bool): Returns True if successful and False if calibration failed
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
            # If found, draw corners
            if ret == True:
                # Append found image points plus standard object points
                imgPts.append(corners)
                objPts.append(objP)
        # Determine calibration parameters and store them
        self._isCalibrated, self._mtx, self._dist, self._rvecs, self._tvecs = cv2.calibrateCamera(objPts, imgPts, cameraShape[1::-1], None, None)
        return self._isCalibrated

    def undistortImage(self, img):
        '''
        Takes in a distorted image img, distorts it, and returns the undistorted image based on calibration.
        If camera was not calibrated, it returns the original image.

                Parameters:
                        img: An distorted image

                Returns:
                        dst : An undistorted image
        '''
        dst = img.copy()
        if self._isCalibrated:
            dst = cv2.undistort(img, self._mtx, self._dist, None, self._mtx)
        else:
            print("Could not start undistortion! Camera has not been calibrated yet!")

        return dst



if __name__ == "__main__":

    # Make a list of calibration images
    distortedImage = cv2.cvtColor(cv2.imread("test_images/test1.jpg"), cv2.COLOR_BGR2RGB)
    calibrationCatalog = []
    for root, dirs, files in os.walk("camera_cal/"):
        for f in files:
            calibrationCatalog.append(os.path.relpath(os.path.join(root, f), "."))

    # Chessboard size
    nx = 9
    ny = 6

    camCalib = CameraCalibrator()
    if camCalib.calibCamera(calibrationCatalog,nx,ny):
        print("Camera Calibration successful")

        undistorted = camCalib.undistortImage(distortedImage)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(distortedImage)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undistorted)
        ax2.set_title('Undistorted Image', fontsize=50)
    else:
        print("Failed")

    debuggerBreakPoint =5
