#!/usr/bin/env python

# Ros libraries
import rospy
import tf
from tf.transformations import euler_from_quaternion

# The usual libraries
import random
import math
import time
import numpy as np
import quaternion
import matplotlib.pyplot as plt
import scipy

from math import sin,cos,atan2,sqrt,fabs,pi
from mpl_toolkits.mplot3d import Axes3D

# Messages and Services modules:
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import *
from geometry_msgs.msg import Point, WrenchStamped, Vector3
from sensor_msgs.msg import JointState

# Personal Modules:
from robot3 import *
from simulation import*


class panda_simulation:
    def __init__(self, ros_state):
        self.state = ros_state
        self.listener = tf.TransformListener() #This will listen to the tf data later
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.hz = 100.0
        self.duration = 10.0

    def ros_output(self, rb):
        gazebo_link_states()
        Xf = np.array([end_effector_position[0], end_effector_position[1], end_effector_position[2], link4quat[1], link4quat[2], link4quat[3]])
        quatf = quaternion.from_float_array(link4quat)

        A = rb.quat2Ja(quatf)
        Xquatd_return = A.dot(np.array([link4ang[0], link4ang[1], link4ang[2]]))
        quatf_d = quaternion.from_float_array(Xquatd_return)
        Xfd = np.array([end_effector_vel[0], end_effector_vel[1], end_effector_vel[2], Xquatd_return[1], Xquatd_return[2], Xquatd_return[3]])

        self.state.x = Xf
        self.state.xd = Xfd
        self.state.quat = quatf
        self.state.quat_d = quatf_d

    def plot_data(self, sim):
        self.pause()
        fig, Fx1 = plt.subplots()

        Fx1.plot(sim.data.x[:3,:].T)
        # Fx1.plot(xincollect[0,:].T,xincollect[1,:].T)
        # Fx1.grid()
        Fx1.set_title('Trajectory')
        Fx1.set_xlabel('X-axis $[m]$')
        Fx1.set_ylabel('Y-axis $[m]$')
        Fx1.legend(['$x$', '$y$', '$z$'],loc="upper right")
        plt.show()

        fig, Fx3 = plt.subplots()

        Fx3.plot(sim.data.x[3:,:].T)
        # Fx1.plot(xincollect[0,:].T,xincollect[1,:].T)
        # Fx3.grid()
        Fx3.set_title('Unit Quat')
        # Fx1.set_xlabel('X-axis $[m]$')
        # Fx1.set_ylabel('Y-axis $[m]$')
        Fx3.legend(['$x$', '$y$', '$z$'],loc="upper right")
        plt.show()

        fig, Fx2 = plt.subplots()

        Fx2.plot(sim.data.tau.T)
        # Fx1.plot(xincollect[0,:].T,xincollect[1,:].T)
        # Fx2.grid()
        Fx2.set_title('Torque')
        # Fx1.set_xlabel('X-axis $[m]$')
        # Fx1.set_ylabel('Y-axis $[m]$')
        Fx2.legend(['$q_1$', '$q_2$', '$q_3$','$q_4$', '$q_5$', '$q_6$', '$q_7$'],loc="upper right")
        plt.show()

    def init_gazebo_services(self):
        model_name = 'panda'
        joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
        joint_positions = [0.0, 0.0, 0.0, -1.5707, 0.0, 1.5707, 1.5707]

        # rospy.wait_for_service('gazebo/set_model_configuration')
        # try:
        #     set_model_configuration = rospy.ServiceProxy('gazebo/set_model_configuration', SetModelConfiguration)
        #     resp = set_model_configuration(model_name, "", joint_names, joint_positions)
        #     print "set model configuration status: ", resp.status_message
        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e

        # Retrieving default physics parameters
        rospy.wait_for_service('gazebo/get_physics_properties')
        try:
            get_physics_properties = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)
            resp2 = get_physics_properties()
            print "get physics properties status: ", resp2.status_message
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

        # Setting new physics parameters
        ode_config = resp2.ode_config
        time_step = 0.001
        max_update_rate = 100.0

        gravity = Vector3()
        gravity.x = 0.0
        gravity.y = 0.0
        gravity.z = -9.81

        rospy.wait_for_service('gazebo/set_physics_properties')
        try:
            set_physics_properties = rospy.ServiceProxy('gazebo/set_physics_properties', SetPhysicsProperties)
            resp3 = set_physics_properties(time_step, max_update_rate, gravity, ode_config)
            print "set physics properties status: ", resp3.status_message
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


def callback(message):
    global Force
    fx = message.wrench.force.x
    fy = message.wrench.force.y
    fz = message.wrench.force.z
    nx = message.wrench.torque.x
    ny = message.wrench.torque.y
    nz = message.wrench.torque.z
    # fz = 0
    Force = np.array([fx,fy,fz,nx,ny,nz])


# Subscriber functions:
def force():
    rospy.Subscriber("/ft_sensor1", WrenchStamped, callback)

def joint():
    rospy.Subscriber("/joint_states", JointState, jointCall)

def gazebo_link_states():
    rospy.Subscriber("/gazebo/link_states", LinkStates, callback2) #subscribes to /gazebo/link_states of type LinkStates


# Callback functions for subscriber functions
def callback2(message):
#Obtains 3rd revolute joint position and orientation from gazebo/link_states
    global end_effector_position
    global link4quat
    global end_effector_vel
    global link4ang
    global Force

    link4pose = message.pose[8]
    link4quaternion = link4pose.orientation
    link4quat = [link4quaternion.w, link4quaternion.x, link4quaternion.y, link4quaternion.z]
    q = [link4quaternion.x,link4quaternion.y,link4quaternion.z,link4quaternion.w] #creates list from quaternion since it was not originally
    link4orientation = euler_from_quaternion(q) #transfrom from quaternion to euler angles

    #Maps the end effector from 3rd revolute joint position and orientation
    # end_effector_x = link4pose.position.x + sin(link4orientation[1])
    end_effector_x = link4pose.position.x
    end_effector_y = link4pose.position.y
    end_effector_z = link4pose.position.z
    # end_effector_z = link4pose.position.z - cos(link4orientation[1])
    end_effector_position = [end_effector_x, end_effector_y, end_effector_z]

    link4twist = message.twist[8]
    link4angular = link4twist.angular
    link4ang = [link4angular.x, link4angular.y, link4angular.z]

    link4quat = [link4quaternion.w, link4quaternion.x, link4quaternion.y, link4quaternion.z]
    q = [link4quaternion.x,link4quaternion.y,link4quaternion.z,link4quaternion.w] #creates list from quaternion since it was not originally
    link4orientation = euler_from_quaternion(q) #transfrom from quaternion to euler angles

    #Maps the end effector from 3rd revolute joint position and orientation
    end_effector_xd = link4twist.linear.x
    end_effector_yd = link4twist.linear.y
    end_effector_zd = link4twist.linear.z
    end_effector_vel = [end_effector_xd, end_effector_yd, end_effector_zd]

def jointCall(message):
    global q, qd
    # Todo: make the definitions compact.
    q1 = message.position[0]
    q2 = message.position[1]
    q3 = message.position[2]
    q4 = message.position[3]
    q5 = message.position[4]
    q6 = message.position[5]
    q7 = message.position[6]
    q = np.array([q1, q2, q3, q4, q5, q6, q7])
    # q = np.array([q1, q2, q3, q4, q5, q6])

    qd1 = message.velocity[0]
    qd2 = message.velocity[1]
    qd3 = message.velocity[2]
    qd4 = message.velocity[3]
    qd5 = message.velocity[4]
    qd6 = message.velocity[5]
    qd7 = message.velocity[6]
    qd = np.array([qd1, qd2, qd3, qd4, qd5, qd6, qd7])
    # qd = np.array([qd1, qd2, qd3, qd4, qd5, qd6])
    # print q

def main_simulation():
    global q, qd
    global end_effector_position
    global link4quat
    global end_effector_vel
    global link4ang

    end_effector_position = [0, 0, 0]
    end_effector_vel = [0, 0, 0]
    link4quat = [0, 0, 0, 0]
    link4ang = [0, 0, 0]

    # Defining Robot
    Waifu = Robot('mdh')
    Waifu.robot_init()

######################################################################################
    # ROS Initializations
    rospy.init_node('panda_joint_torque_node', anonymous=True)
    pandaState = stateVector()
    ros_sim = panda_simulation(pandaState)
    ros_sim.init_gazebo_services()

    # Publishers
    pub1 = rospy.Publisher('joint1_torque_controller/command', Float64, queue_size=50)
    pub2 = rospy.Publisher('joint2_torque_controller/command', Float64, queue_size=50)
    pub3 = rospy.Publisher('joint3_torque_controller/command', Float64, queue_size=50)
    pub4 = rospy.Publisher('joint4_torque_controller/command', Float64, queue_size=50)
    pub5 = rospy.Publisher('joint5_torque_controller/command', Float64, queue_size=50)  # 5 candidate
    pub6 = rospy.Publisher('joint6_torque_controller/command', Float64, queue_size=50)
    pub7 = rospy.Publisher('joint7_torque_controller/command', Float64, queue_size=50)  # 7 candidate

    print "ROS publishers initalized."

    rate = rospy.Rate(ros_sim.hz) #100 Hz
######################################################################################

    print "Program Start:\n"
    raw_input()

    t_final = ros_sim.duration
    # q0 = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/2])
    # q0 = np.array([0.0, 0.0, 0.0, -1.5707, 0.0, 1.5707, 1.5707])
    q0 = np.array([0, 0, 0, 0, 0, 0, 0])
    qf = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/2])

    trj = traj(q0, qf, 0, t_final, ros_sim.hz, Waifu)
    # Xp, Xdp, Xddp = trj.pathplanning3()
    # X1, Xd1, Xdd1 = trj.rotationalInterpolation(Xp, Xdp, Xddp, 'quaternion')
    X1, Xd1, Xdd1, Q1, J1, Jd1, Jdd1 = trj.from_jointspace()

######################################################################################

    F = np.array([0,0,0,0,0,0])
    # F = np.array([0,0,0])
    impctrl = impedanceController()
    kx = 300.0
    ky = 300.0
    kz = 300.0
    kr = 3.0
    kp = 3.0
    kya = 3.0

    mx = 2.5
    my = 2.5
    mz = 2.5
    mr = 0.3
    mp = 0.3
    mya = 0.3
    impctrl.nullspace_stiffness_ = 300
    impctrl.Kd = np.diag(np.array([kx,ky,kz, kr, kp, kya]))
    impctrl.Md = np.diag(np.array([mx, my, mz, mr, mp, mya]))
    impctrl.Bd = np.diag( np.array([2*np.sqrt(kx), 2*np.sqrt(ky), 2*np.sqrt(kz),
                                    2*np.sqrt(kr), 2*np.sqrt(kp), 2*np.sqrt(kya)]))

######################################################################################
    # For post trajectory control - to see whether the robot maintains the target position.
    sampcol = trj.samples
    samples = trj.samples
    dx = trj.dx

    data = dataCollect(sampcol)

    # Initial parameters for the impedance controller.
    # The cartesian parameter values:
    Xf = X1[:,0]
    Xfd = Xd1[:,0]
    quatf = quaternion.from_float_array(Q1[:,0])
    # quatf_d = quaternion.from_float_array(Qd1[:,0])

    end_effector_state = stateVector()
    end_effector_state.x = Xf
    end_effector_state.xd = Xfd
    end_effector_state.quat = quatf
    # end_effector_state.quat_d = quatf_d

    # The joint parameter values:
    q = np.array([0, 0, 0, 0, 0, 0, 0])
    # q = np.array([0.0, 0.0, 0.0, -1.5707, 0.0, 1.5707, 1.5707])
    qd = np.array([0, 0, 0, 0, 0, 0, 0])
    qdd = np.array([0, 0, 0, 0, 0, 0, 0])
    Force = np.array([0,0,0,0,0,0])
    jointState = jointStateVector(q, qd, qdd)
    jointState.qnull = q

    desired_state = stateVector()
    desired_state.x = X1[:,0]
    desired_state.xd = Xd1[:,0]
    desired_state.xdd = Xdd1[:,0]
    desired_state.quat = quaternion.from_float_array(Q1[:,0])
    # desired_state.quat_d = quaternion.from_float_array(Qd1[:,0])
    # desired_state.quat_dd = quaternion.from_float_array(Qdd1[:,0])

    errorState = stateVector()
    sim = simulation(desired_state, end_effector_state, jointState, errorState, data)


    F = np.array([0,0,0,0,0,0])
    print "Initializations complete.!"
    raw_input()
    print "samples", sampcol
######################################################################################

    joint()
    ros_sim.unpause()
    print "Go! ->"
    # i = 0
    while not rospy.is_shutdown():
        for i in range(sampcol):
            print i
            if i > (samples-1):
                sim.state_des.x = X1[:,samples-1]
                sim.state_des.xd = Xd1[:,samples-1]
                sim.state_des.xdd = Xdd1[:,samples-1]
                sim.state_des.quat = quaternion.from_float_array(Q1[:,samples-1])
                # sim.state_des.quat_d = quaternion.from_float_array(Qd1[:,samples-1])
                # sim.state_des.quat_dd = quaternion.from_float_array(Qdd1[:,samples-1])
                sim.jointState.qnull = J1[:,samples-1]
                sim.jointState.qdnull = Jd1[:,samples-1]
                sim.jointState.qddnull = Jdd1[:,samples-1]
            else:
                sim.state_des.x = X1[:,i]
                sim.state_des.xd = Xd1[:,i]
                sim.state_des.xdd = Xdd1[:,i]
                sim.state_des.quat = quaternion.from_float_array(Q1[:,i])
                # sim.state_des.quat_d = quaternion.from_float_array(Qd1[:,i])
                # sim.state_des.quat_dd = quaternion.from_float_array(Qdd1[:,i])
                sim.jointState.qnull = J1[:,i]
                sim.jointState.qdnull = Jd1[:,i]
                sim.jointState.qddnull = Jdd1[:,i]

            # Error signal
            sim.feedbackError3(Waifu,'quaternion')

            # Inverse Dynamics
            joint()
            sim.jointState.q = q
            sim.jointState.qd = qd

            force()
            impctrl.F = Force

            tau = sim.spong_impedance_control(impctrl, Waifu)
            # tau = sim.spong_impedance_control(impctrl, Waifu)
            # tau = sim.inertia_avoidance_impedance_control(impctrl, Waifu, 6, 'quaternion')
            pub1.publish(tau[0])
            pub2.publish(tau[1])
            pub3.publish(tau[2])
            pub4.publish(tau[3])
            pub5.publish(tau[4])
            pub6.publish(tau[5])
            pub7.publish(tau[6])
            data.tau[:,i] = tau

            rate.sleep()
            # F = Force
            joint()
            sim.jointState.q = q
            sim.jointState.qd = qd
            # print sim.jointState.q
            #
            # print q
            # print qd

            # ros_sim.ros_output(Waifu)
            sim.outputEndeffector(Waifu, 'quaternion')

            # sim.state_end.x = ros_sim.state.x
            # sim.state_end.xd = ros_sim.state.xd
            # sim.state_end.quat = ros_sim.state.quat
            # sim.state_end.quat_d = ros_sim.state.quat_d

            # Collecting output.
            sim.data.x[:,i] = sim.state_end.x
            sim.data.xd[:,i] = sim.state_end.xd
            sim.data.error[:,i] = sim.error.x
            # i = i+1

        # i = i-1
        ros_sim.plot_data(sim)

        print "Done"
        break


if __name__ == '__main__':
    try: main_simulation()
    except rospy.ROSInterruptException: pass
