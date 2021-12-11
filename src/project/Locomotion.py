from threading import Timer
import numpy as np
import math


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False
# ---------------------------------------------------------------------------------------


# -----------kalman_filter---------------------------------------------------------------

def kalman_filter(speed, cam_available, cam_data, x_est_prev, P_est_prev, Q, R, Ts):
    """
    Estimates the current state using camera measurement data, speed data and the previous state
    
    param speed: measured speed (Thymio units) of right and left wheels
    
    param cam_available: boolean letting us know a camera measurement is available
    param cam_data: position and orientation (state) measurement from the camera

    param x_est_prev: previous state a posteriori estimation
    param P_est_prev: previous state a posteriori covariance
    
    return x_est: new a posteriori state estimation
    return P_est: new a posteriori state covariance
    """

    # Constants
    distance_between_wheels = 60  # [mm], in reality distance is only 45mm

    # Variables
    [vl, vr] = speed  # speed contains non metric values
    # Transform vl and vr to metric [mm/s]
    vl = 0.43478260869565216 * vl
    vr = 0.43478260869565216 * vr
    theta_est = x_est_prev[2]  # changement car pas initialisé je met x_est_prev à la place de x_est

    # Prediction through the a priori estimate

    # Estimated mean of the state
    # xk+1 = I + Ts*f(xk,uk) = fd(xk,uk)
    x_est_a_priori = x_est_prev + Ts * np.array([0.5 * (vr + vl) * math.cos(theta_est),
                                                 0.5 * (vr + vl) * math.sin(theta_est),
                                                 (0.5 * (
                                                             vr - vl)) / distance_between_wheels])
    # d(fd)/dx evaluated at xk_est and uk                                                    
    A = np.identity(3) + Ts * np.array([[0, 0, -0.5 * (vr + vl) * math.sin(theta_est)],
                                        [0, 0, 0.5 * (vr + vl) * math.cos(theta_est)],
                                        [0, 0, 0]])  # j'ai rajouté les crochets

    # Estimated covariance of the state
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T))
    P_est_a_priori = P_est_a_priori + Q if type(Q) != type(None) else P_est_a_priori

    # Update/Correction

    # y, C, and R for a posteriori estimate, depending on available camera measurement
    if cam_available:  # cam_measure_available boolean
        y = cam_data
        H = np.identity(3)

        # innovation / measurement residual
        i = y - np.dot(H, x_est_a_priori)

        # measurement prediction covariance
        S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R

        # Kalman gain (tells how much the predictions should be corrected based on the measurements)
        K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))

        # a posteriori estimate
        x_est = x_est_a_priori + np.dot(K, i)
        P_est = P_est_a_priori - np.dot(K, np.dot(H, P_est_a_priori))

    else:
        # no measurement so we just use x a posteriori(k+1) to be equal to the a priori(k+1)
        # since no measurement to correct
        # a posteriori estimate
        x_est = x_est_a_priori
        P_est = P_est_a_priori

    return x_est, P_est
# ---------------------------------------------------------------------------------------


# -----------control_law-----------------------------------------------------------------

def control_law(sensor_data, checkpoints_data, current_checkpoint_index, fsm_state, state_kalman, Thymio_found):
    global reached_end
    """
    Finite State Machine to decide the wheel velocities for the next step
    
    param sensor_data: array containing the values of the 5 front horizontal proximity sensors
    param checkpoints_data: set/array containing all the checkpoints the thymio has to go through !! each row is a checkpoint
    param current_checkpoint_index: indicates what is the current checkpoint we want to go to
    param fsm_state: indicates what state of the fsm we are in ("starting", "moving", "rotating", "obstacle_avoidance", "stop")
    param state_kalman: state estimate from kalman

    return left_velocity: velocity of left wheel as decided by control law in metric [mm/s]
    return right_velocity: velocity of left wheel as decided by control law in metric [mm/s]
    return new_fsm_state: updated fsm_state
    return new_checkpoint_index: updated checkpoint index
    """

    # Constant parameters
    distance_threshold = 15  # [mm]
    angle_threshold_high = 5  # [deg]
    angle_threshold_low = 1  # [deg]
    base_translational_velocity = 100  # thymio speed
    # to get speed in metric, just multiply by constant, example : 50*0.43478260869565216 = 21.73913043478261 [mm/s]

    # Parameters related to sensors
    prox_horizontal = sensor_data
    threshold_sensor = 900
    obstacle_speed_gain = 5
    scale_positioning = 100

    # Parameters related to estimated postion and current checkpoint

    [total_rows, total_columns] = checkpoints_data.shape  # we want to know the amount of checkpoints we have, that is give by the amount of rows
    first_checkpoint_bool = (current_checkpoint_index == 0)  # boolean to check whether our current goal is the last checkpoint
    last_checkpoint_bool = (current_checkpoint_index == (total_rows - 1))  # boolean to check whether our current goal is the last checkpoint
    # print('Checkpoint number : '+str(current_checkpoint_index) + '\n')

    current_checkpoint = checkpoints_data[current_checkpoint_index]  # position of the checkpoint we aim for
    current_checkpoint_x = current_checkpoint[0]  # position x of the checkpoint we aim for
    current_checkpoint_y = current_checkpoint[1]  # position y of the checkpoint we aim for

    est_pos_x = state_kalman[0]
    est_pos_y = state_kalman[1]
    est_orientation = state_kalman[2]
    est_orientation_deg = (est_orientation * 180) / math.pi  # [deg]

    delta_x = current_checkpoint_x - est_pos_x
    delta_y = current_checkpoint_y - est_pos_y

    angle = math.atan2(delta_y, delta_x)  # [rad]
    angle_deg = (angle * 180) / math.pi  # [deg]
    delta_angle = angle_deg - est_orientation_deg  # [deg]
    euler_distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

    # print ('Position x :' + str(est_pos_x) + ', Position y :' + str(est_pos_y) +'\n')
    # print ('Estimated orientation : ' + str(est_orientation_deg) + '\n')

    # Beginning of Finite State Machine
    if not (fsm_state == "stop"):
        if max(prox_horizontal) > threshold_sensor:
            fsm_state = "obstacle_avoidance"

    if fsm_state == "starting":
        if Thymio_found :
            new_fsm_state = "rotating"
        else:
            new_fsm_state = "starting"

        left_velocity = 0
        right_velocity = 0
        # new_fsm_state = "rotating"  # ! line to be removed once camera is integrated
        new_checkpoint_index = current_checkpoint_index

        return left_velocity, right_velocity, new_fsm_state, new_checkpoint_index

    if fsm_state == "moving":

        if euler_distance <= distance_threshold:

            if not last_checkpoint_bool:
                # We need to reorient ourselves until our orientation is good to go to the next checkpoint
                left_velocity = 0
                right_velocity = 0
                new_fsm_state = "rotating"
                new_checkpoint_index = current_checkpoint_index + 1

                return left_velocity, right_velocity, new_fsm_state, new_checkpoint_index

            if last_checkpoint_bool:
                # We have reached the end point of the trajectory
                left_velocity = 0
                right_velocity = 0
                new_fsm_state = "stop"
                new_checkpoint_index = current_checkpoint_index

                return left_velocity, right_velocity, new_fsm_state, new_checkpoint_index

        elif abs(delta_angle) > angle_threshold_high:
            # will be used when camera adjusts angle estimate
            # and we realize we are not going in the proper direction angle-wise
            left_velocity = 0
            right_velocity = 0
            new_fsm_state = "rotating"
            new_checkpoint_index = current_checkpoint_index

            return left_velocity, right_velocity, new_fsm_state, new_checkpoint_index

        else:
            # We can continue moving freely
            left_velocity = base_translational_velocity
            right_velocity = base_translational_velocity
            new_fsm_state = "moving"
            new_checkpoint_index = current_checkpoint_index
            # print('Distance to checkpoint : '+str(euler_distance) + '\n')

            return left_velocity, right_velocity, new_fsm_state, new_checkpoint_index

    elif fsm_state == "rotating":

        # print('Current orientation estimate : '+str(est_orientation_deg) + ' [degrees]\n')
        # print('delta_angle : '+str(delta_angle) + ' [degrees]\n')

        if delta_angle > angle_threshold_low:
            left_velocity = -base_translational_velocity // 3
            right_velocity = base_translational_velocity // 3

            # slower speed to be more precise
            if delta_angle > angle_threshold_high:
                left_velocity = -base_translational_velocity
                right_velocity = base_translational_velocity

            new_fsm_state = "rotating"
            new_checkpoint_index = current_checkpoint_index

            return left_velocity, right_velocity, new_fsm_state, new_checkpoint_index

        elif delta_angle < -angle_threshold_low:
            left_velocity = base_translational_velocity // 3
            right_velocity = -base_translational_velocity // 3

            # slower speed to be more precise
            if delta_angle < -angle_threshold_high:
                left_velocity = base_translational_velocity
                right_velocity = -base_translational_velocity

            new_fsm_state = "rotating"
            new_checkpoint_index = current_checkpoint_index
            return left_velocity, right_velocity, new_fsm_state, new_checkpoint_index

        elif abs(delta_angle) < angle_threshold_low:
            left_velocity = 0
            right_velocity = 0
            new_fsm_state = "moving"
            new_checkpoint_index = current_checkpoint_index
            return left_velocity, right_velocity, new_fsm_state, new_checkpoint_index

    elif fsm_state == "obstacle_avoidance":

        # positioning with respect to obstacle when needed
        left_velocity = base_translational_velocity
        right_velocity = base_translational_velocity
        new_checkpoint_index = current_checkpoint_index

        if max(prox_horizontal[0:5]) > threshold_sensor:
            for i in range(5):
                left_velocity += (obstacle_speed_gain * prox_horizontal[i]) // scale_positioning
                right_velocity -= (obstacle_speed_gain * prox_horizontal[i]) // scale_positioning
            new_fsm_state = "obstacle_avoidance"

        elif max(prox_horizontal[0:5]) < threshold_sensor:
            left_velocity = base_translational_velocity
            right_velocity = base_translational_velocity
            new_fsm_state = "rotating"

        return left_velocity, right_velocity, new_fsm_state, new_checkpoint_index

    elif fsm_state == "stop":
        # Once we enter the stop mode we shouldn't leave it anymore since we have reached our final destination
        reached_end = True
        left_velocity = 0
        right_velocity = 0
        new_fsm_state = "stop"
        new_checkpoint_index = current_checkpoint_index
        return left_velocity, right_velocity, new_fsm_state, new_checkpoint_index
# ---------------------------------------------------------------------------------------


# -----------motors----------------------------------------------------------------------

def motors(left_velocity, right_velocity):
    """
    Function to set the wheel velocities according to the control law, they'll be sent to the thymio in the main though
    
    param left_velocity: velocity of left wheel as decided by control law in metric [mm/s]
    param right_velocity: velocity of right wheel as decided by control law in metric [mm/s]
    
    return : motor variables set to the input velocities
    """

    return {
        "motor.left.target": [left_velocity],
        "motor.right.target": [right_velocity],
    }
