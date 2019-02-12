import cv2

#sanity check
print('Your using OpenCV, ver. {}'.format(cv2.__version__))

#where the video is
mypath = '/home/grzeszo/myprojects/brazilian_or_not/vids/'

# Function to extract frames
def FrameCapture(path):
'''Extract frames from video as pictures. 
The larger the count, the less frames you'll extract'''

    # Path to video file
    vidObj = cv2.VideoCapture(path + 'input/brazilian4.mp4')

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:

        # vidObj object calls read
        # function extract frames
        vidObj.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        success, image = vidObj.read()
        print ('Read a new frame: ', success)
        # Saves the frames with frame-count
        cv2.imwrite("output/vid4/frame%d.jpg" % count, image)

        count += 1

# Driver Code
#if __name__ == '__main__':

# Calling the function
FrameCapture(path = mypath)
