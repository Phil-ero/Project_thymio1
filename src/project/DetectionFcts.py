import cv2
import numpy as np
import math
import Global_nav as GN
from Global_nav import Map


# import matplotlib.pyplot as plt
# from matplotlib import colors


# -----------FindOuterContour-----------------------------------------------------------

def FindOuterContour(img, areaMin, ShowCanny=False, ShowImg=False):
    """
    Finds the outer most border of the image. Uses Canny filter. finds the one with the
    biggest area and sets it to the outer border
    :output maxContour: gives the number of contours we detect,the area, the perimeter,approx: corner values of the contour
    :param img: image from the camera
    :param areaMin: threshold for minimum size of map
    :param ShowCanny: print canny image result
    :param ShowImg: print result in cv2
    """
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    # cv2.imshow('blured',blur)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgCanny = cv2.Canny(blur, 90, 100)
    if ShowCanny: cv2.imshow('Çanny', imgCanny)

    # contours detection
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # initialisation
    area_max = 0
    maxContour = []

    # searching for biggest contour
    for i in contours:
        if cv2.contourArea(i) > area_max and cv2.contourArea(i) > areaMin:
            # if len(i)=3:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # gives us corner points
            if len(approx) == 4:
                maxContour = []
                area_max = cv2.contourArea(i)
                bbox = cv2.boundingRect(approx)
                maxContour.append(area_max)
                maxContour.append(peri)
                maxContour.append(approx)
                maxContour.append(bbox)
                cv2.drawContours(img, contours, -1, (0, 255, 255), 3)
                cv2.drawContours(img, maxContour[2], -1, (0, 255, 255), 3)
    if len(maxContour) > 0:
        for points in approx:
            x, y = points[0]
            cv2.circle(img, (x, y), 3, (255, 0, 255), -1)
    if ShowImg:
        cv2.imshow('original2', img)
    # Plotting the original photo with map borders detected
    # plt.figure(figsize=(7, 7))
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(rgb_img)
    # plt.show()
    return maxContour, area_max


# ---------------------------------------------------------------------------------------


# -----------CropImage-------------------------------------------------------------------

def CropImage(img, contours, pad=15):
    """
    Crops the image around the outer border. uses the corner points, creates a new rectangle
    and wraps the points to the right place
    :output cropped: image cropped to map borders
    :param img: image from the outer contours function
    :param contours: data from the contours
    :param pad: padding
    """
    np_cont = np.array(contours[2])  # contours[2]: corner points
    rect = cv2.minAreaRect(np_cont)
    box = cv2.boxPoints(rect)  # bottom point is first then clockwise
    box = np.int0(box)
    width = int(rect[1][1])
    height = int(rect[1][0])

    if width < height:
        temp = height
        height = width
        width = temp

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    Cropped = warped[pad:-pad, pad:-pad]

    return Cropped, width, height


# ---------------------------------------------------------------------------------------


# -----------FindOuterContour-----------------------------------------------------------

def FindContour(img, color, areaMin, areaMax, show_img=False):
    """
    Finds the borders of the elements in the outer border. similar to FindOuterContour
    then is then used to draw rectangles representing the obstacles on the map
    :output contours: number of contours inside, area of contours, perimeter of contours, approx: corner points
    bbox: corner points, number of points per contour
    :param img: image from the crop function
    :param color: color mask to be applied
    :param areaMin: minimum threshold for detection
    :param areaMax: maximum threshold for detection
    :param show_img: plot result in cv2
    """
    # lines 151 -161 are used to detect the outermost points. this is then used to draw the
    # rectangle representing the obstacles on the map

    if color == 'red':
        new_img = red_mask(img)
    elif color == 'blue':
        new_img = blue_mask(img)
    elif color == 'green':
        new_img = green_mask(img)
    else:
        new_img = img

    blur = cv2.bilateralFilter(new_img, 9, 75, 75)
    imgCanny = cv2.Canny(blur, 100, 100)
    cont, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # initialisation
    area_max = 0
    Contours = []
    Contours.append(len(cont))
    OuterPnts = []
    approx = 0
    # if contours detected
    if (len(cont)) > 0:
        indexs = []
        for i in cont:
            if areaMin < cv2.contourArea(i) < areaMax:
                area = cv2.contourArea(i)
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # gives us corner points
                numbPnts = len(approx)

                # finding outerpoints
                X = []
                Y = []
                Pnts = []
                for points in approx:
                    x, y = points[0]
                    X.append(x)
                    Y.append(y)
                    cv2.circle(img, (x, y), 3, (255, 0, 255), -1)  # draw points of countours
                    Pnts.append((x, y))
                indexs = IndexCalc(X, Y)
                if len(indexs) > 3:
                    for m in range(len(indexs)):
                        idx = indexs[m]
                        OuterPnts.append(Pnts[idx])
                bbox = cv2.boundingRect(approx)
                Contours.append(area)
                Contours.append(peri)
                Contours.append(approx)
                Contours.append(bbox)
                Contours.append(numbPnts)
                cv2.drawContours(img, cont, -1, (0, 255, 255), 3)
    if show_img:
        cv2.imshow('borders_{}'.format(color), img)

    return Contours, imgCanny, OuterPnts


# ---------------------------------------------------------------------------------------


# -----------CreateMatrix----------------------------------------------------------------

def CreateMatrix(mat, pnt):
    """
    fills the matrix with the obstacles.
    takes the points (4 outermost corner points) and fills the matrix
    :output mat: matrix filled with obstacles
    :param mat: empty matrix to be filled
    :param pnt: points that are obstacles
    """
    length = len(pnt)
    for i in range(0, length, 4):
        min_X_pnt = pnt[i]
        min_Y_pnt = pnt[i + 1]
        max_X_pnt = pnt[i + 2]
        max_Y_pnt = pnt[i + 3]
        minX = min_X_pnt[0]
        minY = min_Y_pnt[1]
        maxX = max_X_pnt[0]
        maxY = max_Y_pnt[1]
        for x in range(minX, maxX, 1):
            for y in range(minY, maxY, 1):
                # print("x:"+str(x-1))
                # print("y:"+str(y-1))
                mat[y - 1][x - 1] = 1  # occupied!
    return mat


# ---------------------------------------------------------------------------------------


# -----------Mat2img---------------------------------------------------------------------

def mat2img(mat):
    """
    Transform Obstacles matrix into a grayscale image allows for visualisation of the matrix
    :output grayImage: Black and white image of the obstacles
    :param mat: matrix of obstacles
    """
    uint_img = ((np.array(mat)) * 255).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    return grayImage


# ---------------------------------------------------------------------------------------


# -----------IndexCalc-------------------------------------------------------------------

def IndexCalc(X, Y):
    """
    Takes 2 arrays : one for X and one for Y. from this array determines the outermost points indexes
    :output grayImage: Black and white image of the obstacles
    :param X: index X
    :param Y: index Y
    """
    idx = []
    min_X = min(X)
    min_idx_X = X.index(min_X)
    min_Y = min(Y)
    min_idx_Y = Y.index(min_Y)
    max_X = max(X)
    max_idx_X = X.index(max_X)
    max_Y = max(Y)
    max_idx_Y = Y.index(max_Y)

    idx.append(min_idx_X)
    idx.append(min_idx_Y)
    idx.append(max_idx_X)
    idx.append(max_idx_Y)

    return idx


# ---------------------------------------------------------------------------------------


# -----------thymio_detect---------------------------------------------------------------

def thymio_detect(img, area_max, show_img=False):
    """
    Uses FindContour function to find the Thymio (green mask). finds the 2 boundaries representing the Thymio.
    Finds the center point of each and sets the biggest area as the front point of the Thymio.
    Uses angle_thymio to determine the angle. Returns ((0,0),0) if Thymio not detected
    :param img: cropped image
    :param area_max: max Threshold for detection
    :param show_img: plot result in cv2
    """
    position = []
    contours, _, obst_map = FindContour(img, color='green', areaMin=5, areaMax=area_max)

    # at least two green elements detected representing thymio's position
    if len(contours) > 6:
        for i in range(0, len(contours) - 4, 5):
            s = 0
            sumx = 0
            sumy = 0
            numbPnts = contours[i + 5]
            # searching for center
            for pnts in contours[3 + i]:
                x, y = pnts[0]
                sumx = sumx + x
                sumy = sumy + y
                s = s + 1
                if s == numbPnts and numbPnts != 0:
                    Xpos = int(sumx / numbPnts)
                    Ypos = int(sumy / numbPnts)
                    position.append((int(Xpos), int(Ypos)))
                    position.append(contours[i + 1])
                    cv2.circle(img, (Xpos, Ypos), 3, (255, 0, 255), -1)
                    if show_img:
                        cv2.imshow('centers', img)

        area_max = 0
        thymio_back = []
        thymio_front = []

        for i in range(0, len(position), 2):
            thymio_back.append(position[i])  # set all points to back
            if position[i + 1] > area_max:
                area_max = position[i + 1]
                thymio_front = position[i]
                idx_max = i
        thymio_back.remove(position[idx_max])

        thymio_front = list(thymio_front)
        thymio_back = list(thymio_back[0])

        # center if the 2 blocks
        center = (int(thymio_back[0])), int(thymio_back[1])
        # center = (int((thymio_front[0] + thymio_back[0]) / 2), int((thymio_front[1] + thymio_back[1]) / 2))
        cv2.circle(img, center, 3, (255, 0, 255), -1)

        angle = angle_thymio(thymio_front, thymio_back)
        final_pos = (center, angle)

    else:
        final_pos = ((0, 0), 0)
        print("Thymio not found")

    return final_pos


# ---------------------------------------------------------------------------------------


# -----------finish_detect---------------------------------------------------------------
def finish_detect(img, area_max, show_img=False):
    """
    Finds the finish point by applying a blue mask and finding it's contour.
    returns (0,0) if no finish point found.
    :param img: Cropped image
    :param area_max: max Threshold for detection
    :param show_img: show result in cv2
    """
    position = []
    contours, _, obst_map = FindContour(img, color='blue', areaMin=200, areaMax=area_max)

    if (len(contours) > 5):  # if at least one contour found
        for i in range(0, len(contours) - 4, 5):
            s = 0
            sumx = 0
            sumy = 0
            numbPnts = contours[i + 5]
            for pnts in contours[3 + i]:
                x, y = pnts[0]
                sumx = sumx + x
                sumy = sumy + y
                s = s + 1
                if (s == numbPnts and numbPnts != 0):
                    Xpos = int(sumx / numbPnts)
                    Ypos = int(sumy / numbPnts)
                    position = int(Xpos), int(Ypos)
                    cv2.circle(img, (Xpos, Ypos), 3, (255, 0, 255), -1)
                if show_img:
                    cv2.imshow('final position center', img)

    else:
        print("End position not found")
        return (0, 0)

    return position


# ---------------------------------------------------------------------------------------


# -----------angle_thymio----------------------------------------------------------------

def angle_thymio(Thymio_front, Thymio_back):
    """
    Returns the angle of the thymio depending on two points: the front and the back one
    :output res: Thymio angle between [-pi,pi] =[rad]
    :param Thymio_front: (X,Y) of Thymio front
    :param Thymio_back: (X,Y) of Thymio back
    """
    dy = Thymio_front[1] - Thymio_back[1]
    dx = Thymio_front[0] - Thymio_back[0]
    res = math.atan2(dy, dx)
    return res


# ---------------------------------------------------------------------------------------

# -----------red_mask-------------------------------------------------------------------

def red_mask(Img_rescaled):
    """
    Extracts the red bits of the image. uses HSV color scheme
    :param Img_rescaled: Image to apply color mask
    """
    hsv_rescaled = cv2.cvtColor(Img_rescaled, cv2.COLOR_RGB2HSV)
    # Picking out a range
    lower_red = np.array([200 / 2, 4 / 10 * 255, 25 / 100 * 255])
    upper_red = np.array([280 / 2, 255, 255])
    # Binary mask ('1' indicate values within the range, and '0' values indicate values outside)
    mask = cv2.inRange(hsv_rescaled, lower_red, upper_red)
    return mask


# ---------------------------------------------------------------------------------------

# -----------green_mask------------------------------------------------------------------

def green_mask(Img_rescaled):
    """
    Extracts the green bits of the image. uses HSV color scheme
    :param Img_rescaled: Image to apply color mask
    """
    hsv_rescaled = cv2.cvtColor(Img_rescaled, cv2.COLOR_RGB2HSV)
    # Picking out a range
    lower_green = np.array([80 / 2, 4 / 10 * 255, 25 / 100 * 255])
    upper_green = np.array([160 / 2, 255, 255])
    # Binary mask ('1' indicate values within the range, and '0' values indicate values outside)
    mask = cv2.inRange(hsv_rescaled, lower_green, upper_green)
    return mask


# ---------------------------------------------------------------------------------------

# -----------blue_mask-------------------------------------------------------------------
def blue_mask(Img_rescaled):
    """
    Extracts the blue bits of the image. uses HSV color scheme
    :param Img_rescaled: Image to apply color mask
    """
    hsv_rescaled = cv2.cvtColor(Img_rescaled, cv2.COLOR_RGB2HSV)
    # Picking out a range
    lower_blue = np.array([0, 4 / 10 * 255, 25 / 100 * 255])
    upper_blue = np.array([40 / 2, 255, 255])
    # Binary mask ('1' indicate values within the range, and '0' values indicate values outside)
    mask1 = cv2.inRange(hsv_rescaled, lower_blue, upper_blue)
    lower_blue2 = np.array([318 / 2, 4 / 10 * 255, 25 / 100 * 255])
    upper_blue2 = np.array([358 / 2, 255, 255])
    mask2 = cv2.inRange(hsv_rescaled, lower_blue2, upper_blue2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask


# ---------------------------------------------------------------------------------------

# -----------downscale_img-------------------------------------------------------------------
def downscale_img(width, height, ratio_downscale, start, end, img):
    """
    Downscale everything from the photo to get a fast A_star algorithm
    Uses image interpolation, followed by thresholding
    Downscale start,end and obstacles
    :param width: width of initial image
    :param height: height of initial image
    :param ratio_downscale: ratio between the image and the one used in A_star algorithm
    :param start: start of the initial image
    :param end: end of the initial image
    :param img: img containing obstacles map to be downscaled
    """
    w_down = int(round(ratio_downscale * width))
    h_down = int(round(ratio_downscale * height))
    s_x = int(round(ratio_downscale * start[0]))
    s_y = int(round(ratio_downscale * start[1]))
    e_x = int(round(ratio_downscale * end[0]))
    e_y = int(round(ratio_downscale * end[1]))
    if len(start) == 3:
        start_d = (s_x, s_y, start[2])
    else:
        start_d = (s_x, s_y)
    end_d = (e_x, e_y)
    img_interpolate = cv2.resize(img, (w_down, h_down), interpolation=cv2.INTER_AREA)
    mat_grid = np.asarray(img_interpolate)
    mat_grid = mat_grid[:, :, 0]
    # boolean map
    Threshold = 1
    bin_grid = mat_grid.copy()
    bin_grid[mat_grid < Threshold] = 0
    bin_grid[mat_grid >= Threshold] = 1
    bin_grid = bin_grid.transpose()
    return w_down, h_down, start_d, end_d, bin_grid


# ---------------------------------------------------------------------------------------

# -----------Analyse---------------------------------------------------------------------
def analyse(img, show_img=False):
    """
    Perform a full analyse of an image provided by the camera
    :param img: image from the camera
    :param show_img: print result
    """
    # Find the outer boundary
    contours, area_max = FindOuterContour(img, areaMin=280, ShowCanny=False)
    # if outer boundary found
    if len(contours) > 0:
        # resize the image with outer contours
        croppedImg, width, height = CropImage(img, contours)
        if show_img:
            cv2.imshow('cropped', croppedImg)
        # Memory for the rest of the analysis
        # copy image because drawing on them modifies them
        Matrix = [[0 for x in range(width)] for y in range(height)]
        croppedImgTh = croppedImg.copy()
        croppedImg_finish = croppedImg.copy()

        # Find the obstacles(in red), the Thymio (in green), the end point (in blue)
        _, innerImg, obstacle_map = FindContour(croppedImg, color='red', areaMin=10, areaMax=area_max)
        Thymio_pos = thymio_detect(croppedImgTh, area_max)
        finish_pos = finish_detect(croppedImg_finish, area_max)

        # Prepare data for A star algorithm
        Matrix_filled = CreateMatrix(Matrix, obstacle_map)
        matrixImg = mat2img(Matrix_filled)

        # data defined by user: real measurement for real world
        ratio_downscale = 0.1
        real_width = 554  # Map size in mm
        real_height = 380  # Map size in mm
        ratio_camera = (real_width / width + real_height / height) / 2  # Ratio [mm/px]
        real_thymio = 75  # 1/2 Thymio size = 60mm
        ratio_total = ratio_camera / ratio_downscale  # mm/px in A_star

        # Parameters for A_star
        size_thymio = real_thymio / ratio_total
        size_pixel = 1
        Thymio_coord = Thymio_pos[0]
        start = (Thymio_coord[0], Thymio_coord[1], Thymio_pos[1])
        end = finish_pos

        if end != [] and start != []:
            # Downscale the image
            w_down, h_down, start_d, end_d, occupancy_grid = downscale_img(width, height, ratio_downscale, start,
                                                                           end, matrixImg)
            # Augment size of obstacles taking account of robot size
            grid2 = GN.Obstacles_real(size_thymio, size_pixel, occupancy_grid, w_down, h_down)
            # instantiate the map and running it
            m = Map(w_down, h_down, start_d, end_d, grid2, ratio_total, ratio_downscale)
            m.run_map(True)
            if m.path != []:
                xMap = m.path[0] / ratio_downscale
                xMap = xMap.astype(int)
                yMap = m.path[1] / ratio_downscale
                yMap = yMap.astype(int)
                for i in range(0, len(xMap) - 1):
                    x = xMap[i]
                    y = yMap[i]
                    cv2.circle(croppedImg, (x, y), 3, (0, 255, 255), -1)
            return m, contours, area_max
        if show_img:
            cv2.imshow('path', croppedImg)
    else:
        print("did not find outer contours")
        return []


# ---------------------------------------------------------------------------------------


# -----------analyse_thymio--------------------------------------------------------------
def analyse_thymio(img, contours, area_max, ratio_downscale):
    """
    Perform detection of thymio via image provided by the camera
    :param img: image from the camera
    :param contours: contours of outer contour
    :param area_max: threshold for max detection
    :param ratio_downscale: ratio for real size of the world
    """
    # Find the outer boundary
    # contours, area_max = FindOuterContour(img, areaMin=280, ShowCanny=False, ShowImg= False)
    # if outer boundary found
    if len(contours) > 0:  # contours ! = []:
        # resize the image with outer contours
        croppedImg, _, _ = CropImage(img, contours)
        Thymio_pos = thymio_detect(croppedImg, area_max)
        if Thymio_pos != []:
            Thymio_xy = Thymio_pos[0]
            Thymio_coord = (Thymio_xy[0], Thymio_xy[1], Thymio_pos[1])
            Thymio_real_x = int(round(ratio_downscale * Thymio_coord[0]))
            Thymio_real_y = int(round(ratio_downscale * Thymio_coord[1]))
            if len(Thymio_coord) == 3:
                real_thymio = (Thymio_real_x, Thymio_real_y, Thymio_coord[2])
            else:
                real_thymio = (Thymio_real_x, Thymio_real_y)
            return real_thymio, True
        else:
            return [], False
    else:
        return [], False
