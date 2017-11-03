
import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")

def detectFace(grayScaleImage, originalColorImage):
    faceImageScaleFactor = 1.3
    faceMinNumOfAcceptedNeighbors = 5
    faces = faceCascade.detectMultiScale(grayScaleImage, faceImageScaleFactor, 
                                         faceMinNumOfAcceptedNeighbors)
    
    eyeImageScaleFactor = 1.1
    eyeMinNumOfAcceptedNeighbors = 15
    
    smileImageScaleFactor = 1.7
    smileMinNumOfAcceptedNeighbors = 22
    
    for (xCoordinate, yCoordinate, width, height) in faces:
        upperLeftCorner = (xCoordinate, yCoordinate)
        lowerRightCorner = (xCoordinate + width, yCoordinate + height)
        faceRectangleColour = (255, 0, 0)
        rectangleThickness = 2
        
        cv2.rectangle(originalColorImage, upperLeftCorner, lowerRightCorner, 
                      faceRectangleColour, rectangleThickness)
        
        grayRegionOfInterest = grayScaleImage[upperLeftCorner[1]:lowerRightCorner[1], 
                                                upperLeftCorner[0]:lowerRightCorner[0]]
        colorRegionOfInterest = originalColorImage[
                                        upperLeftCorner[1]:lowerRightCorner[1],
                                        upperLeftCorner[0]:lowerRightCorner[0]]
        
        eyes = eyeCascade.detectMultiScale(grayRegionOfInterest, 
                                           eyeImageScaleFactor, 
                                           eyeMinNumOfAcceptedNeighbors)
        for (eyeXCoordinate, eyeYCoordinate, eyeWidth, eyeHeight) in eyes:
            eyeUpperLeftCorner = (eyeXCoordinate, eyeYCoordinate)
            eyeLowerRightCorner = (eyeXCoordinate + eyeWidth, 
                                eyeYCoordinate + eyeHeight)
            eyeRectangleColour = (0, 255, 0)
            
            cv2.rectangle(colorRegionOfInterest, eyeUpperLeftCorner, 
                          eyeLowerRightCorner, eyeRectangleColour, 
                          rectangleThickness)
        smile = smileCascade.detectMultiScale(grayRegionOfInterest,
                                              smileImageScaleFactor,
                                              smileMinNumOfAcceptedNeighbors)
        for (smileXCoordinate, smileYCoordinate, smileWidth, smileHeight) in smile:
            smileUpperLeftCorner = (smileXCoordinate, smileYCoordinate)
            smileLowerRightCorner = (smileXCoordinate + smileWidth, 
                                smileYCoordinate + smileHeight)
            smileRectangleColour = (0, 0, 255)
            
            cv2.rectangle(colorRegionOfInterest, smileUpperLeftCorner, 
                          smileLowerRightCorner, smileRectangleColour, 
                          rectangleThickness)
    return originalColorImage

if __name__ == "__main__":
    videoCapture = cv2.VideoCapture(0)
    while (cv2.waitKey(1) & 0xFF != ord('q')):
        _, originalColorImage = videoCapture.read()
        grayScaleImage = cv2.cvtColor(originalColorImage, cv2.COLOR_BGR2GRAY)
        canvas = detectFace(grayScaleImage, originalColorImage)
        cv2.imshow("Video", canvas)
    videoCapture.release()
    cv2.destroyAllWindows()
