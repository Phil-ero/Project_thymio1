import cv2
import numpy as np
import math


# -----------FindOuterContour-----------------------------------------------------------

def FindOuterContour(img, areaMin, ShowCanny=False):
	# Finds the outer most border of the image. Uses Canny filter. finds the one with the
	# biggest area and sets it to the outer border
	# maxcontours takes several values:
	# 	- the number of contour we detect
	#	- the area
	#	- the perimeter
	#	- approx: gives the corner values of the contour. for example 4 corner points of the square

	blur = cv2.bilateralFilter(img, 9, 75, 75)
	# cv2.imshow('blured',blur)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	imgCanny = cv2.Canny(blur, 90, 100)
	if ShowCanny: cv2.imshow('Çanny', imgCanny)

	contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	area_max = 0
	maxContour = []

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
	if (len(maxContour) > 0):
		for points in approx:
			x, y = points[0]
			cv2.circle(img, (x, y), 3, (255, 0, 255), -1)
	cv2.imshow('original2', img)

	return maxContour, area_max


# ---------------------------------------------------------------------------------------


# -----------CropImage-------------------------------------------------------------------

def CropImage(img, contours, pad=15):
	# crops the image around the outer border. uses the corner points, creates a new rectangle
	# and wraps the points to the right place

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
	Croped = warped[pad:-pad, pad:-pad]

	# lines used to make a smaller map

	# Croped=cv2.resize(Croped,(57,42),interpolation=cv2.INTER_AREA)
	# width=57
	# height=42

	return Croped, width, height


# ---------------------------------------------------------------------------------------


# -----------FindOuterContour-----------------------------------------------------------

def FindContour(img, color, areaMin, areaMax):
	# finds the borders of the elements in the outer border. similar to FindOuterContour
	# The output Countours take several values:
	#	- the number of contours inside
	#	- the area
	#	- the perimeter
	#	- approx : corner points
	#	- bbox : bounding box
	#	- number of points per contours

	# lines 140 -151 are used to detect the outermost points. this is then used to draw the
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
	area_max = 0
	Contours = []
	Contours.append(len(cont))
	OuterPnts = []
	approx = 0
	if (len(cont)) > 0:
		s = 0
		indexs = []
		for i in cont:
			if cv2.contourArea(i) > areaMin and cv2.contourArea(i) < areaMax:
				s = s + 1
				area = cv2.contourArea(i)
				peri = cv2.arcLength(i, True)
				approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # gives us corner points
				numbPnts = len(approx)
				X = []
				Y = []
				Pnts = []
				StartPnts = []
				Width_arr = []
				Height_arr = []
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
				# print(bbox)
				Contours.append(numbPnts)
				cv2.drawContours(img, cont, -1, (0, 255, 255), 3)
	cv2.imshow('borders_{}'.format(color), img)

	return Contours, imgCanny, OuterPnts


# ---------------------------------------------------------------------------------------


# -----------CreateMatrix----------------------------------------------------------------

def CreateMatrix(mat, pnt, width, height):
	# fills the matrix with the obstacles.
	# takes the points (4 outermost corner points) and fills the matrix

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
	# takes a matrix and transforms into a grayscale image
	# allows for visualisation of the matrix

	uint_img = ((np.array(mat)) * 255).astype('uint8')
	grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)

	return grayImage


# ---------------------------------------------------------------------------------------


# -----------IndexCalc-------------------------------------------------------------------

def IndexCalc(X, Y):
	# takes 2 arrays : one for X and one for Y. from this array
	# determines the outermost points indexes

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

def thymio_detect(img, area_max):
	# uses FindContour fucntion to find the Thymio (green mask). finds the 2 buondaries
	# representing the Thymio. Finds the center point of each and sets the biggest area as the
	# front point of the thymio. Uses angle_thymio to determine the angle. Returns ((0,0),0)
	# if thymio not detected

	# check if angle corect!!

	position = []
	contours, _, obst_map = FindContour(img, color='green', areaMin=5, areaMax=area_max)

	if (contours[0] > 1):  # at least two green elements detected representing thymio's position
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
					position.append((int(Xpos), int(Ypos)))
					position.append(contours[i + 1])
					cv2.circle(img, (Xpos, Ypos), 3, (255, 0, 255), -1)
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
		center = (int((thymio_front[0] + thymio_back[0]) / 2), int((thymio_front[1] + thymio_back[1]) / 2))
		cv2.circle(img, center, 3, (255, 0, 255), -1)

		angle = angle_thymio(thymio_front, thymio_back)
		final_pos = (center, angle)

	else:
		print("Could not find Thymio")
		final_pos = ((0, 0), 0)

	return final_pos


# ---------------------------------------------------------------------------------------


# -----------finish_detect---------------------------------------------------------------
def finish_detect(img, area_max):
	# finds the finish point by applying a blue mask and finding it's contour.
	# returns (0,0) if no finish point found.

	position = []
	contours, _, obst_map = FindContour(img, color='blue', areaMin=50, areaMax=area_max)
	# if(len(contours)>0):
	if (contours[0]) > 0:  # if at least one contour found
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
					cv2.imshow('final position center', img)
	# print(position)
	else:
		return (0, 0)

	return position


# ---------------------------------------------------------------------------------------


# -----------angle_thymio----------------------------------------------------------------

def angle_thymio(Thymio_front, Thymio_back):
	# returns the angle of the thymio depending on two points: the front and the back one
	dy = Thymio_front[1] -Thymio_back[1]
	dx = Thymio_front[0] - Thymio_back[0]
	res = math.atan2(dy,dx)
	return res


# ---------------------------------------------------------------------------------------

# -----------blue_mask-------------------------------------------------------------------

def red_mask(Img_rescaled):
	# extracts the red bits of the image. uses HSV color scheme

	hsv_rescaled = cv2.cvtColor(Img_rescaled, cv2.COLOR_RGB2HSV)
	# Picking out a range
	lower_red = np.array([200 / 2, 4 / 10 * 255, 25 / 100 * 255])
	upper_red = np.array([280 / 2, 255, 255])
	# Binary mask ('1s' indicate values within the range, and '0s' values indicate values outside)
	mask = cv2.inRange(hsv_rescaled, lower_red, upper_red)
	return mask


# ---------------------------------------------------------------------------------------

# -----------green_mask------------------------------------------------------------------

def green_mask(Img_rescaled):
	# extracts the green bits of the image. uses HSV color scheme

	hsv_rescaled = cv2.cvtColor(Img_rescaled, cv2.COLOR_RGB2HSV)
	# Picking out a range
	lower_green = np.array([80 / 2, 4 / 10 * 255, 25 / 100 * 255])
	upper_green = np.array([160 / 2, 255, 255])
	# Binary mask ('1s' indicate values within the range, and '0s' values indicate values outside)
	mask = cv2.inRange(hsv_rescaled, lower_green, upper_green)
	return mask


# ---------------------------------------------------------------------------------------

# -----------blue_mask-------------------------------------------------------------------
def blue_mask(Img_rescaled):
	# extracts the blue bits of the image. uses HSV color scheme

	hsv_rescaled = cv2.cvtColor(Img_rescaled, cv2.COLOR_RGB2HSV)
	# Picking out a range
	lower_blue = np.array([0, 4 / 10 * 255, 25 / 100 * 255])
	upper_blue = np.array([40 / 2, 255, 255])
	# Binary mask ('1s' indicate values within the range, and '0s' values indicate values outside)
	mask1 = cv2.inRange(hsv_rescaled, lower_blue, upper_blue)
	lower_blue2 = np.array([318 / 2, 4 / 10 * 255, 25 / 100 * 255])
	upper_blue2 = np.array([358 / 2, 255, 255])
	mask2 = cv2.inRange(hsv_rescaled, lower_blue2, upper_blue2)
	mask = cv2.bitwise_or(mask1, mask2)
	return mask


# ---------------------------------------------------------------------------------------

#-----------downscale_img-------------------------------------------------------------------
def downscale_img(width, height, ratio_downscale, start, end, img):

	# We have to downscale everything from the photo to get a fast Astar
	# downscale start,end,obstacles

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
	bin_grid [mat_grid < Threshold] = 0
	bin_grid [mat_grid >= Threshold] = 1
	return w_down,h_down,start_d, end_d, bin_grid
#---------------------------------------------------------------------------------------