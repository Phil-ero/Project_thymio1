import cv2
import numpy as np
import DetectionFcts as DF
import Global_nav as GN
from Global_nav import Map

webcam = True
path = 'edgeDetect3.jpg'

# Open the camera with the name cap
# Global variable stays active during entire program
cap = cv2.VideoCapture(1)
cap.open(1, cv2.CAP_DSHOW)

while True:
    if webcam:
        # At each begin of while take a new image
        success, img = cap.read()
        if not success:
            break
    else:
        img = cv2.imread(path)
    # Show the image which will be analyzed
    cv2.imshow('original', img)

    # waitKey: window of 42 ms to let the user decide
    # Press "a" to begin the analyse and "esc" to quit
    k = cv2.waitKey(42)  # 42 ms between two images: correspond to 24 frames/s
    if k == 27:  # esc
        cv2.destroyAllWindows()  # Closing camera
        break

    elif k == 97:  # a-analysis
        # Find the outer boundary
        contours, area_max = DF.FindOuterContour(img, areaMin=280, ShowCanny=False)
        # if outer boundary found
        if (contours[0]) > 0:
            # resize the image with outer contours
            croppedImg, width, height = DF.CropImage(img, contours)
            cv2.imshow('cropped', croppedImg)

            # Memory for the rest of the analysis
            # copy image because drawing on them modifies them
            Matrix = [[0 for x in range(width)] for y in range(height)]
            croppedImgTh = croppedImg.copy()
            croppedImg_finish = croppedImg.copy()

            # Find the obstacles(in red), the Thymio (in green), the end point (in blue)
            _, innerImg, obstacle_map = DF.FindContour(croppedImg, color='red', areaMin=10, areaMax=area_max)
            Thymio_pos = DF.thymio_detect(croppedImgTh, area_max)
            finish_pos = DF.finish_detect(croppedImg_finish, area_max)

            # Prepare data for A star algorithm
            Matrix_filled = DF.CreateMatrix(Matrix, obstacle_map)
            matrixImg = DF.mat2img(Matrix_filled)

            # data defined by user: real measurement for real world
            ratio_downscale = 0.1
            real_width = 554  # Map size in mm
            real_height = 380  # Map size in mm
            ratio_camera = (real_width / width + real_height / height) / 2  # Ratio [mm/px]
            real_thymio = 60  # 1/2 Thymio size = 60mm
            ratio_total = ratio_camera / ratio_downscale  # mm/px in A_star

            # Parameters for A_star
            size_thymio = real_thymio / ratio_total
            size_pixel = 1
            Thymio_coord = Thymio_pos[0]
            start = (Thymio_coord[0], Thymio_coord[1], Thymio_pos[1])
            end = finish_pos

            if end != [] and start != []:
                # Downscale the image
                w_down, h_down, start_d, end_d, occupancy_grid = DF.downscale_img(width, height, ratio_downscale, start,
                                                                                  end, matrixImg)
                # Augment size of obstacles taking account of robot size
                grid2 = GN.Obstacles_real(size_thymio, size_pixel, occupancy_grid, w_down, h_down)
                # instantiate the map and running it
                m = Map(w_down, h_down, start_d, end_d, grid2,ratio_total, ratio_downscale)
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
                   
        else:
            print("did not find outer contours")
