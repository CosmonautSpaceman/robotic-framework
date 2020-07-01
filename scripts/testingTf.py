#!/usr/bin/env python

import rospy
import random
import math
import time
import numpy as np

from std_msgs.msg import Float64
from std_srvs.srv import Empty
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import *
# from gazebo_msgs.srv import SetPhysicsProperties
from geometry_msgs.msg import Point, WrenchStamped
from sensor_msgs.msg import JointState
from math import sin,cos,atan2,sqrt,fabs,pi
from robotFunctions import *
import matplotlib.pyplot as plt
import tf

from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# (un)pause = rospy.ServiceProxy('/gazebo/(un)pause_physics', Empty)
#Define a RRRBot joint positions publisher for joint controllers.
def luna_joint_torques_publisher():
    global q, qd

    # Initial values for declaration
    q = np.array([0, 0, 0, 0, 0, 0, 0])
    qd = np.array([0, 0, 0, 0, 0, 0, 0])
    qdd = np.array([0, 0, 0, 0, 0, 0, 0])

    # Robot parameters
    # alpha = [(np.pi/2), (-np.pi/2), (np.pi/2), (-np.pi/2), (np.pi/2), (np.pi/2), 0]
    alpha = [(-np.pi/2), (np.pi/2), (np.pi/2), (-np.pi/2), (np.pi/2), (np.pi/2), 0]
    a = [0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0.107]
    m = [4.970684, 0.646926, 3.228604, 3.587895, 1.225946, 1.666555, 7.35522e-01]
    # r = [-(0.333/2), 0.0, -(0.316/2), 0, -(0.384/2), 0, -(0.107/2)]
    n_dof = 7

    Ipanda1 = np.array([[7.03370e-01,  -1.39000e-04, 6.77200e-03],
            [-1.39000e-04, 7.06610e-01,  1.91690e-02],
            [6.77200e-03, 1.91690e-02,  9.11700e-03]])


    Ipanda2 = np.array([[7.96200e-03, -3.92500e-03, 1.02540e-02],
            [-3.92500e-03, 2.81100e-02,  7.04000e-04],
            [1.02540e-02, 7.04000e-04, 2.59950e-02 ]])


    Ipanda3 = np.array([[3.72420e-02, -4.76100e-03, -1.13960e-02],
            [-4.76100e-03,  3.61550e-02,  -1.28050e-02],
            [-1.13960e-02, -1.28050e-02,  1.08300e-02 ]])


    Ipanda4 = np.array([[2.58530e-02,  7.79600e-03, -1.33200e-03],
              [7.79600e-03,  1.95520e-02,  8.64100e-03],
             [-1.33200e-03,  8.64100e-03,  2.83230e-02 ]])


    Ipanda5 = np.array([[3.55490e-02,  -2.11700e-03, -4.03700e-03],
            [-2.11700e-03,  2.94740e-02,  2.29000e-04],
            [-4.03700e-03,  2.29000e-04,  8.62700e-03]])


    Ipanda6 = np.array([[1.96400e-03,  1.09000e-04,  -1.15800e-03],
            [1.09000e-04,  4.35400e-03,  3.41000e-04],
            [-1.15800e-03,  3.41000e-04,  5.43300e-03]])


    Ipanda7 = np.array([[1.25160e-02,  -4.28000e-04, -1.19600e-03],
            [-4.28000e-04,  1.00270e-02,  -7.41000e-04],
            [-1.19600e-03,  -7.41000e-04,  4.81500e-03]])

    # <inertia ixx="0.00782229414331" ixy="-1.56191622996e-05" ixz="-0.00126005738123" iyy="0.0109027971813" iyz="1.08233858202e-05" izz="0.0102355503949" />

    # <inertia ixx="0.0180416958283" ixy="0.0" ixz="0.0" iyy="0.0159136071891" iyz="0.0046758424612" izz="0.00620690827127" />
    # <inertia ixx="0.0182856182281" ixy="0.0" ixz="0.0" iyy="0.00621358421175" iyz="-0.00472844221905" izz="0.0161514346309" />

    # <inertia ixx="0.00771376630908" ixy="-0.00248490625138" ixz="-0.00332147581033" iyy="0.00989108008727" iyz="-0.00217796151484" izz="0.00811723558464" />

    # <inertia ixx="0.00799663881132" ixy="0.00347095570217" ixz="-0.00241222942995" iyy="0.00825390705278" iyz="0.00235774044121" izz="0.0102515004345" />


    # <inertia ixx="0.030371374513" ixy="6.50283587108e-07" ixz="-1.05129179916e-05" iyy="0.0288752887402" iyz="-0.00775653445787" izz="0.00444134056164" />


    # <inertia ixx="0.00303336450376" ixy="-0.000437276865508" ixz="0.000629257294877" iyy="0.00404479911567" iyz="0.000130472021025" izz="0.00558234286039" />
    # <inertia ixx="0.000888868887021" ixy="-0.00012239074652" ixz="3.98699829666e-05" iyy="0.000888001373233" iyz="-9.33825115206e-05" izz="0.0007176834609" />

    # Ipanda1 = np.array([[0.0180416958283, 0, 0], [0, 0.0159136071891, 0.0046758424612], [0, 0.0046758424612, 0.00620690827127]] )
    # Ipanda2 = np.array([[0.0182856182281, 0, 0], [0, 0.00621358421175, -0.00472844221905], [0, -0.00472844221905, 0.0161514346309]] )
    # Ipanda3 = np.array([[0.00771376630908, -0.00248490625138, -0.00332147581033], [-0.00248490625138, 0.00989108008727, -0.00217796151484], [-0.00332147581033, -0.00217796151484, 0.00811723558464]] )
    # Ipanda4 = np.array([[0.00799663881132, 0.00347095570217, -0.00241222942995], [0.00347095570217, 0.00825390705278, 0.00235774044121], [-0.00241222942995, 0.00235774044121, 0.0102515004345]] )
    # Ipanda5 = np.array([[0.030371374513, 6.50283587108e-07, -1.05129179916e-05], [6.50283587108e-07, 0.0288752887402, -0.00775653445787], [-1.05129179916e-05, -0.00775653445787, 0.00444134056164]] )
    # Ipanda6 = np.array([[0.00303336450376, -0.000437276865508, 0.000629257294877], [-0.000437276865508, 0.00404479911567, 0.000130472021025], [0.000629257294877, 0.000130472021025, 0.00558234286039]] )
    # Ipanda7 = np.array([[0.000888868887021, -0.00012239074652, 3.98699829666e-05], [-0.00012239074652, 0.000888001373233, -9.33825115206e-05], [3.98699829666e-05, -9.33825115206e-05, 0.0007176834609]] )
    # Ixx = 0.3
    # Iyy = 0.3
    # Izz = 0.3
    # Ipanda = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]] )


    # panda.r = [3.875e-03 2.081e-03 0;
    #     -3.141e-03 -2.872e-02 3.495e-03;
    #     2.7518e-02 3.9252e-02 -6.6502e-02;
    #     -5.317e-02 1.04419e-01 2.7454e-02;
    #     -1.1953e-02 4.1065e-02 -3.8437e-02;
    #     6.0149e-02 -1.4117e-02 -1.0517e-02;
    #     1.0517e-02 -4.252e-03 6.1597e-02]';


    # _____________________
    # Parameter order:
    # alpha, a, theta, d, type, inertia, m, r
    Joint1 = rJoint(alpha[0], a[0], 0, d[0], 'R', Ipanda1, m[0], np.array([[3.875e-03],[2.081e-03],[0]]) )
    Joint2 = rJoint(alpha[1], a[1], 0, d[1], 'R', Ipanda2, m[1], np.array([[-3.141e-03],[-2.872e-02],[3.495e-03]]) )
    Joint3 = rJoint(alpha[2], a[2], 0, d[2], 'R', Ipanda3, m[2], np.array([[2.7518e-02],[3.9252e-02],[-6.6502e-02]]) )
    Joint4 = rJoint(alpha[3], a[3], 0, d[3], 'R', Ipanda4, m[3], np.array([[-5.317e-02],[1.04419e-01],[2.7454e-02]]) )
    Joint5 = rJoint(alpha[4], a[4], 0, d[4], 'R', Ipanda5, m[4], np.array([[-1.1953e-02],[4.1065e-02],[-3.8437e-02]]) )
    Joint6 = rJoint(alpha[5], a[5], 0, d[5], 'R', Ipanda6, m[5], np.array([[6.0149e-02],[-1.4117e-02],[-1.0517e-02]]) )
    Joint7 = rJoint(alpha[6], a[6], 0, d[6], 'R', Ipanda7, m[6], np.array([[1.0517e-02],[-4.252e-03],[6.1597e-02]]) )

    # Collecting joints
    JointCol = [Joint1, Joint2, Joint3, Joint4, Joint5, Joint6, Joint7]

    # Defining Robot
    Waifu = Robot(JointCol, n_dof)
    Waifu.joints = JointCol

    grav = np.array([[0],[0],[-9.81]])

    rospy.init_node('panda_joint_torque_node', anonymous=True)

    listener = tf.TransformListener() #This will listen to the tf data later
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    hz = 1000

    # Publishers
    pub1 = rospy.Publisher('joint1_torque_controller/command', Float64, queue_size=100)
    pub2 = rospy.Publisher('joint2_torque_controller/command', Float64, queue_size=100)
    pub3 = rospy.Publisher('joint3_torque_controller/command', Float64, queue_size=100)
    pub4 = rospy.Publisher('joint4_torque_controller/command', Float64, queue_size=100)
    pub5 = rospy.Publisher('joint5_torque_controller/command', Float64, queue_size=100)  # 5 candidate
    pub6 = rospy.Publisher('joint6_torque_controller/command', Float64, queue_size=100)
    pub7 = rospy.Publisher('joint7_torque_controller/command', Float64, queue_size=100)  # 7 candidate


    rate = rospy.Rate(hz) #100 Hz

    print "joints:"
    joint()
    raw_input()
    print q
    raw_input()
    print q
    raw_input()
    for j in range(7):
        Waifu.joints[j].theta = q[j]

    T = Transform(Waifu)
    rot = t2rot(T)
    trans = t2transl(T)
    print "trans: ", trans

    psi0, theta0, phi0 = euler_angles_from_rotation_matrix(rot)
    print(psi0, theta0, phi0)
    # Xf = np.array([trans[0], trans[1], trans[2], psi, theta, phi])

    psi = -1.5708
    theta = -0.0
    phi = np.pi

    # Path planning parameters
    # TODO: Put parameters into datastructures to simplify readability and flexibility.
    p0x, pfx = trans[0], 0.3
    p0y, pfy = trans[1], 0.3
    p0z, pfz = trans[2], 0.926

    duration = 10.0
    # hz = 100.0
    dx = 1.0/hz
    samples = duration*hz
    samples = int(samples)
    X1, Xd1, Xdd1 = pathplanning(p0x, p0y, p0z, pfx, pfy, pfz, psi0, theta0, phi0, psi, theta, phi, duration, hz)

    i = 0
    Xf = X1[:3,0]
    Xfd = Xd1[:3,0]
    Xf_prev = Xf

    # Xf_prev = Xf
    xf = Xf
    xfd = Xfd

    # Force = np.array([0,0,0,0,0,0])
    # F = np.array([0,0,0,0,0,0])

    # Kd = np.diag(np.array([125,125,125,1,1,1]))
    # Bd =  np.diag(np.array([85,85,85,165,165,165]))
    # Md = np.diag(np.array([15,15,15,1,1,1]))
    # ax = np.zeros((6,samples))

    Force = np.array([0,0,0])
    F = np.array([0,0,0])

    Kd = np.diag(np.array([125,125,125]))
    Bd =  np.diag(np.array([85,85,85]))
    Md = np.diag(np.array([15,15,15]))
    ax = np.zeros((3,samples))
    # Kv = np.eye(7,7) * np.array([50,100,50,70,70,50,20])
    # Kp = np.eye(7,7) * np.array([12000,30000,18000,18000,12000,7000,2000])
    # print "This is it:"
    print samples
    sampcol = samples
    Fcollect  = np.zeros((7,sampcol))
    taucollect  = np.zeros((7,sampcol))
    xcollect  = np.zeros((6,sampcol))

    xincollect = X1
    xdincollect = Xd1
    xddincollect = Xdd1
    rho = 0.2
    raw_input()

    joint()
    unpause()
    while not rospy.is_shutdown():
        # print "potato"

        try:
            # print "inside"
            (trans,rot) = listener.lookupTransform('panda_link0', 'panda_link8', rospy.Time(0))
            # print "finished"
            # print trans
            # print rot
            eulers = euler_from_quaternion(rot)

            (lintwist, rottwist) = listener.lookupTwist('panda_link0', 'panda_link8', rospy.Time(0),
                                                                   rospy.Duration(1.0/hz))
            # print lintwist
            print rottwist

            # print "eulers:", eulers[0], eulers[1], eulers[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        #


        # pub2.publish(tau[0])
        # pub3.publish(tau[1])
        # pub4.publish(tau[2])
        # pub5.publish(tau[3])
        # pub6.publish(tau[4])
        # pub7.publish(tau[5])

        # for i in range(sampcol):
        #     if i > (samples-1):
        #         x = X1[:,samples-1]
        #         xd = Xd1[:,samples-1]
        #         xdd = Xdd1[:,samples-1]
        #         print "Done"
        #     else:
        #         x = X1[:,i]
        #         xd = Xd1[:,i]
        #         xdd = Xdd1[:,i]
        #     ax = impedanceXdd(x, xd, xdd, xf, xfd, F, Kd, Bd, Md)
        #     aq_in = calcQdd(ax, q, qd, Waifu)
        #     # print aq_in
        #     q_in = np.array([q])
        #     qd_in = np.array([qd])
        #     aq_in = np.array([aq_in])
        #
        #     for j in range(7):
        #         Waifu.joints[j].theta = q[j]
        #
        #     tau = np.ravel(invdyn(Waifu, q_in, qd_in, aq_in, grav))
        #     taucollect[:,i] = tau
        #
        #     # print "tau:", tau
        #     pub1.publish(tau[0])
        #     pub2.publish(tau[1])
        #     pub3.publish(tau[2])
        #     pub4.publish(tau[3])
        #     pub5.publish(tau[4])
        #     pub6.publish(tau[5])
        #     pub7.publish(tau[6])
        #
        #     joint()
        #     # i = i + 1

        #     # try:
        #     # Get current time
        #     t_cur = rospy.get_rostime().nsecs
        #     F = Force
        #     # Fcollect[:,i] = np.reshape(F,3)
        #     joint()
        #     # Alternative method for computing joint acceleration
        #     for j in range(Waifu.ndof):
        #         Waifu.joints[j].theta = q[j]
        #     T = Transform(Waifu)
        #     Xftr = t2transl(T)
        #     Xftr = Xftr.ravel()
        #     XfR = t2rot(T)
        #     psi, theta, phi = euler_angles_from_rotation_matrix(XfR)
        #     Xfor = np.array([psi,theta,phi])
        #     Xf = np.concatenate([Xftr,Xfor])
        #     Xfd = calcXd(q, qd, Waifu)
        #     # Update previous parameters
        #     t_prev = t_cur
        #     Xf_prev = Xf
        #     Xfd_prev = Xfd
        #     xcollect[:,i] = Xf
        #     # Update final/actual robot position (measured)
        #     xf = Xf
        #     xfd = Xfd
        #     # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, rospy.ROSInterruptException):
        #     #     pass
        #
        # fig, (Cx1,Cx2) = plt.subplots(2)
        # fig.suptitle('Force/Torque')
        #
        # Cx1.plot(Fcollect.T)
        # Cx1.grid()
        # Cx1.legend("xyz 1",loc="upper right")
        # Cx1.set_title('$F$')
        # Cx1.set_xlabel('samples')
        # Cx1.set_ylabel('$[N]$')
        #
        # Cx2.plot(taucollect.T)
        # Cx2.grid()
        # Cx2.legend(['$tau_1$', '$tau_2$'],loc="upper right")
        # Cx2.set_title('$tau$')
        # Cx2.set_xlabel('samples')
        # Cx2.set_ylabel('$[Nm]$')
        # plt.show()
        #
        #
        # fig, Fx1 = plt.subplots()
        #
        # Fx1.plot(xcollect[0,:].T,xcollect[1,:].T)
        # Fx1.plot(xincollect[0,:].T,xincollect[1,:].T)
        # Fx1.grid()
        # Fx1.set_title('Trajectory')
        # Fx1.set_xlabel('X-axis $[m]$')
        # Fx1.set_ylabel('Y-axis $[m]$')
        # Fx1.legend(['$measured$', '$desired$'],loc="upper right")
        # plt.show()


def joint():
    rospy.Subscriber("/panda/joint_states", JointState, jointCall)


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

    qd1 = message.velocity[0]
    qd2 = message.velocity[1]
    qd3 = message.velocity[2]
    qd4 = message.velocity[3]
    qd5 = message.velocity[4]
    qd6 = message.velocity[5]
    qd7 = message.velocity[6]
    qd = np.array([qd1, qd2, qd3, qd4, qd5, qd6, qd7])
    # print q



if __name__ == '__main__':

    try: luna_joint_torques_publisher()
    except rospy.ROSInterruptException: pass
