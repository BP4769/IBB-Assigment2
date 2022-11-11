import cv2

def calculate_pixel_coordinates(xCenter, yCenter, boxHeight, boxWidth, imgHeight, imgWidth):
    xmin = (xCenter - (boxWidth/2))*imgWidth
    xmax = (xCenter + (boxWidth/2))*imgWidth
    ymin = (yCenter - (boxHeight/2))*imgHeight
    ymax = (yCenter + (boxHeight/2))*imgHeight
    
    return [xmin, ymin, xmax, ymax]



def calculate_coordinates(x,y,w,h):
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    
    return [xmin, ymin, xmax, ymax]



def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # Intersection
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # Union
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union = float(boxAArea + boxBArea - intersection)
    iou = intersection / union
    
    return float(iou)



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized