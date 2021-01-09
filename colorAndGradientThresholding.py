import cv2
import numpy as np


def absSobelThreshFilter(image, orient='x', sobelKernel=6, thresh=(0, 255)):
    # Use lightness channel of HLS color space for better performance over grayscale
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobelKernel)
    else:
        sobel = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)
    # Get absolute value, direction is of no importance
    absSobel = np.absolute(sobel)
    # Scale to max value and apply threshold filter
    binaryImg = scaleAndApplyThresholdsOnAbsoluteSobel(absSobel, thresh)
    return binaryImg

def magSobelThreshFilter(image, sobelKernel=6, magThresh=(0, 255)):
    # Use lightness channel of HLS color space for better performance over grayscale
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lChannel = hls[:, :, 1]
    # Apply Sobel in both directions
    sobelX = cv2.Sobel(lChannel, cv2.CV_64F, 1, 0, ksize=sobelKernel)
    sobelY = cv2.Sobel(lChannel, cv2.CV_64F, 0, 1, ksize=sobelKernel)
    # Get absolute sum out of both, we are targeting for the magnitude,
    # no filtering for either direction
    absSobel = np.sqrt(np.square(sobelX) + np.square(sobelY))
    # Scale to max value and apply threshold filter
    binaryImg = scaleAndApplyThresholdsOnAbsoluteSobel(absSobel, magThresh)

    return binaryImg

def scaleAndApplyThresholdsOnAbsoluteSobel(absoluteSobel, thresh=(0, 255)):
    # Scale to max found value and convert to resolution 8bit
    scaledSobel = np.uint8(255 * absoluteSobel / np.max(absoluteSobel))
    # Convert to binary image
    binaryImg = convertToBinaryFilter(scaledSobel, thresh)

    return binaryImg

def convertToBinaryFilter(img, thresh=(0, 255)):
    # Prepare empty binary image
    binaryImg = np.zeros_like(img)
    # Only keep those pixels that are between the threshold
    binaryImg[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binaryImg

def sChannelThreshFilter(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    sChannel = hls[:, :, 2]
    # Convert to binary image
    binaryImg = convertToBinaryFilter(sChannel, thresh)
    return binaryImg

def mergeWithAnd(binImg1 , binImg2):
    mergedBinImg = np.zeros_like(binImg1)
    mergedBinImg[((binImg1 == 1)) & ((binImg2 == 1))] = 1
    return mergedBinImg


def mergeWithOr(binImg1, binImg2):
    mergedBinImg = np.zeros_like(binImg1)
    mergedBinImg[((binImg1 == 1)) | ((binImg2 == 1))] = 1
    return mergedBinImg

def blurFilter(img, blurKernel=5):
    return cv2.blur(img, (blurKernel,blurKernel))

if __name__ == '__main__':
    import misc

    testImages = misc.getFileRelativeFilepathsInDirectory('test_images')
    for imageName in testImages:
        img = cv2.imread(imageName)
        # Setup hyper parameters
        blurKernelSize = 5
        sobelKernelSize = 5
        absSobelXThresh = (15, 200)
        magSobelThresh = (38, 124)
        sChannelThresh = (103, 200)
        # Blur image
        blurImg = blurFilter(img, blurKernelSize)
        # Get gradient image via combination of x-dir abs. sobel and magnitude filter
        xGradImg = absSobelThreshFilter(blurImg,'x',sobelKernelSize, absSobelXThresh)
        magGradImg = magSobelThreshFilter(blurImg, sobelKernelSize, magSobelThresh)
        gradImg = mergeWithAnd(xGradImg, magGradImg)
        # Filter via s-channel filter esp. for yellow this is important
        sChannelImg = sChannelThreshFilter(blurImg, sChannelThresh)
        # Merge gradient and s-channel images via or to get best out of both worlds
        finalImg = mergeWithOr(gradImg, sChannelImg)
        # Show before after
        #misc.showBeforeAfter(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original", cv2.cvtColor(finalImg*255,cv2.COLOR_BGR2RGB), "Gradient and color threshold filter")
        #misc.showImageInNewFigure(cv2.cvtColor(finalImg*255,cv2.COLOR_BGR2RGB), "Gradient and color threshold filter")
        cv2.imwrite('debugOutput/' + misc.getFilenameFromPath(imageName), cv2.cvtColor(finalImg*255,cv2.COLOR_BGR2RGB))


