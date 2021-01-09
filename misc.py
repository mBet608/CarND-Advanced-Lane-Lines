import os
import matplotlib.pyplot as plt
import cv2

def getFileRelativeFilepathsInDirectory(dirName: str) -> [str]:
    """
    Gets relative file paths for given directory. Is NOT recursive amd ONLY
    searches for files, nothing else. Will return empty list if nothing is found or
    dirName does not exist.
    :param dirName: The root dir of where the file search should happen
    :return: Returns list with relative file paths as strings
    """
    relativeFilepath = []
    for root, dirs, files in os.walk(dirName):
        for f in files:
            relativeFilepath.append(os.path.relpath(os.path.join(root, f), "."))
    return relativeFilepath

def showBeforeAfter(beforeImg, beforeTitle: str, afterImg, afterTitle: str):
    """
    Show two images next to each other
    :param beforeImg: First image
    :param beforeTitle: First image's title
    :param afterImg: Second image
    :param afterTitle: Second image's title
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(beforeImg)
    ax1.set_title(beforeTitle, fontsize=50)
    ax2.imshow(afterImg)
    ax2.set_title(afterTitle, fontsize=50)
    return f

def showImageInNewFigure(image, title):
    plt.figure(title)
    plt.tight_layout()
    plt.imshow(image)
    plt.title(title)

def getFilenameFromPath(path):
    return os.path.basename(path)

def showFigureUntilKeyHit():
    plt.waitforbuttonpress()

def putTextInImage(image, text, pos):
    cv2.putText(
        image,
        text,
        pos,
        cv2.FONT_HERSHEY_DUPLEX,
        1,  # font size
        (0, 0, 0, 255),  # font color
        2)  # font stroke
