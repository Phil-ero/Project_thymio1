# Basics of Mobile robotics - Project report
### Students: Sushen Jilla Venkatesa 284402, Forero Philippe 284213, Mael Mouhoub 288287, Selina Bothner 284028
## 1.Introduction
We have realized this project in the framework of the course Basics of mobile robotics at EPFL in 2021. The goal of this project is to offer students the possibility to learn the basics of robotics, building a complex project within 4 weeks. We used a Thymio robot [1] and a webcam for this project. Our main constraints for this project were to integrate:
- Computer vision: Take a picture and do image processing to detect all elements of the map.
-	Motion control:  A Finite State Machine (FSM) with different modes, controlling the robot motors
-	Global and Local navigation: Find the shortest path to the goal, and avoids obstacles placed randomly while moving towards the goal 
-	Filtering: diverse filters in Computer vision and Kalman filter in Motion control
 We will combine these elements into a global project that will allow us to maneuver the Thymio robot in a defined map around obstacles. For more details, we will discuss the challenges linked to each part in the following sub-sections.
Link to video :https://github.com/Phil-ero/Project_thymio1/blob/main/ThymioFast.mp4
