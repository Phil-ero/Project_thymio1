import cv2
import numpy as np
import DetectionFcts as DF
import Global_nav as GN
from Global_nav import Map

webcam= True
path= 'edgeDetect3.jpg'

cap= cv2.VideoCapture(1)
cap.open(1, cv2.CAP_DSHOW)

while True:

	if webcam:
		success,img = cap.read()
		if success==False:
			break
	else: img=cv2.imread(path)
	cv2.imshow('original', img)
	
#waitKet pour permettre d'afficher les images et lire les entrées du clavier
#il faut appuyer sur "a" pour commencer l'analyse et "esc" pour quitter

	k = cv2.waitKey(42) #### 42 ms between two images: correspond to 24 frames/s
	if k==27: #esc
		cv2.destroyAllWindows()
		break
	
	elif k==97: #a-analysis

		contours,area_max=DF.FindOuterContour(img,areaMin=280,ShowCanny=False)

		if (contours[0])>0:
			cropedImg,width,height=DF.CropImage(img, contours)

			cv2.imshow('croped',cropedImg)

			Matrix = [[0 for x in range(width)] for y in range(height)]
			
#ca c'est pour copier les photos vu que je dessine dessus et du coup ca fait des pbms pour la detection sinon
#je vais essayer de voir si on peut changer ca mais c'est pas le plus important pr le moment
			cropedImgTh=cropedImg.copy()
			cropedImgfinish=cropedImg.copy()

			_,innerImg,obstacle_map= DF.FindContour(cropedImg,color='red',areaMin=10,areaMax=area_max)
			Thymio_pos=DF.thymio_detect(cropedImgTh,area_max)
			finish_pos=DF.finish_detect(cropedImgfinish,area_max)
			print("thymio position et angle"+str(Thymio_pos))
			print("finish point position"+str(finish_pos))
			Matrix_filled=DF.CreateMatrix(Matrix,obstacle_map,width,height)

			### Ne pas enlever sinon pas de downscale###########
			matrixImg=DF.mat2img(Matrix_filled)
			ratio_downscale = 0.1
			###########################################

			real_width = 554 #Map size in mm
			real_height = 380 # Map size in mm
			ratio_camera = (real_width/width + real_height/height)/2 # Ratio [mm/px]
			real_thymio = 60 # 1/2 Thymio size = 60mm
			ratio_total = ratio_camera/ratio_downscale # mm/px in Astar

			cv2.imshow('matrix', matrixImg)
			size_thymio = real_thymio/ratio_total
			size_pixel = 1
			Thymio_coord = Thymio_pos[0]
			start=(Thymio_coord[0],Thymio_coord[1],Thymio_pos[1])
			print("start"+str(start))
			print("width"+str(width))
			print("height"+str(height))
			end= finish_pos

#pour utilsier ta fonction
			w_down,h_down,start_d,end_d,occupancy_grid = DF.downscale_img(width,height,ratio_downscale,start,end,matrixImg)
			occupancy_grid = occupancy_grid.transpose()
			grid2 = GN.Obstacles_real(size_thymio, size_pixel, occupancy_grid, w_down, h_down)
			m = Map(w_down,h_down,start_d,end_d, grid2)
			#m.create_plot()
			m.run_map(True)
			#print("checkpoints"+str(m.checkpoints))
			#print("path"+str(m.path))


#ca ca permet de dessiner directemement sur l'image, c'était avant quand j'arrivais pas a afficher ta fonction
#mais je laisse, ca pourra être mieux pour visualiser
			if(len(m.path != 0)):
				xMap= m.path[0]/ratio_downscale
				xMap = xMap.astype(int)
				yMap=m.path[1]/ratio_downscale
				yMap = yMap.astype(int)
				for i in range(0,len(xMap)-1):
					x=xMap[i]
					y=yMap[i]
					cv2.circle(cropedImg,(x,y),3,(0,255,255), -1)
				cv2.imshow('path',cropedImg)
	elif k == 115: #s-analysis
		#Press s after one good initialisation with a
		#



# tout ca tu peux ignorer 
#c'est juste pour me rappeller de comment j'avais fait qqch

# #yesNo=0
# 		#while (yesNo!=1):
# 		contours,area_max=DF.FindOuterContour(img,areaMin=280,ShowCanny=False)
# 		#	print("Is the contour the right one? y/n")
# 		#	yesNo = cv2.waitKey(1)
# 			# inp=input('Is the contour the right one? y/n')
# 			# if inp=='y':
# 			# 	yesNo=1
# 			# else:
# 			# 	if webcam:
# 			# 		success,img = cap.read()
# 			# 		cv2.imshow('original', img)
# 			# 	else: img=cv2.imread(path)