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
from mpl_toolkits import mplot3d
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
    qd = np.array([0, 0, 0, 0, 0, 0,  0])
    qdd = np.array([0, 0, 0, 0, 0, 0,  0])

    # Robot parameters
    # alpha = [(np.pi/2), (-np.pi/2), (np.pi/2), (-np.pi/2), (np.pi/2), (np.pi/2), 0]
    alpha = [(-np.pi/2), (np.pi/2), (np.pi/2), (-np.pi/2), (np.pi/2), (np.pi/2), 0]
    a = [0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0.107]
    m = [2.34471, 2.36414, 2.38050, 2.42754, 3.49611, 1.46736, 0.45606]
    # r = [-(0.333/2), 0.0, -(0.316/2), 0, -(0.384/2), 0, -(0.107/2)]
    n_dof = 7

    Ipanda = [[0.3, 0, 0],[0, 0.3, 0],[0, 0, 0.3]]


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

    # _____________________
    # Parameter order:
    # alpha, a, theta, d, type, inertia, m, r
    Joint1 = rJoint(alpha[0], a[0], 0, d[0], 'R', Ipanda, m[0], np.array([[3.875e-03],[2.081e-03],[0]]) )
    Joint2 = rJoint(alpha[1], a[1], 0, d[1], 'R', Ipanda, m[1], np.array([[-3.141e-03],[-2.872e-02],[3.495e-03]]) )
    Joint3 = rJoint(alpha[2], a[2], 0, d[2], 'R', Ipanda, m[2], np.array([[2.7518e-02],[3.9252e-02],[-6.6502e-02]]) )
    Joint4 = rJoint(alpha[3], a[3], 0, d[3], 'R', Ipanda, m[3], np.array([[-5.317e-02],[1.04419e-01],[2.7454e-02]]) )
    Joint5 = rJoint(alpha[4], a[4], 0, d[4], 'R', Ipanda, m[4], np.array([[-1.1953e-02],[4.1065e-02],[-3.8437e-02]]) )
    Joint6 = rJoint(alpha[5], a[5], 0, d[5], 'R', Ipanda, m[5], np.array([[6.0149e-02],[-1.4117e-02],[-1.0517e-02]]) )
    Joint7 = rJoint(alpha[6], a[6], 0, d[6], 'R', Ipanda, m[6], np.array([[1.0517e-02],[-4.252e-03],[6.1597e-02]]) )

    # Collecting joints
    JointCol = [Joint1, Joint2, Joint3, Joint4, Joint5, Joint6, Joint7]

    # Defining Robot
    Waifu = Robot(JointCol, n_dof)
    Waifu.joints = JointCol

    grav = np.array([[0],[0],[-9.81]])

    rospy.init_node('panda_joint_torque_node', anonymous=True)

    listener = tf.TransformListener() #This will listen to the tf data later
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

    hz = 100

    # Publishers
    pub1 = rospy.Publisher('joint1_torque_controller/command', Float64, queue_size=100)
    pub2 = rospy.Publisher('joint2_torque_controller/command', Float64, queue_size=100)
    pub3 = rospy.Publisher('joint3_torque_controller/command', Float64, queue_size=100)
    pub4 = rospy.Publisher('joint4_torque_controller/command', Float64, queue_size=100)
    pub5 = rospy.Publisher('joint5_torque_controller/command', Float64, queue_size=100)  # 5 candidate
    pub6 = rospy.Publisher('joint6_torque_controller/command', Float64, queue_size=100)
    pub7 = rospy.Publisher('joint7_torque_controller/command', Float64, queue_size=100)  # 7 candidate


    rate = rospy.Rate(hz) #100 Hz

    unpause()
    print "joints:"
    joint()
    pause()
    raw_input()
    print q
    raw_input()
    print q
    raw_input()
    for j in range(Waifu.ndof):
        Waifu.joints[j].theta = q[j]

    T = Transform(Waifu)
    rot = t2rot(T)
    trans = t2transl(T)
    print "trans: ", trans

    phi0, theta0, psi0 = r2rpy(rot)
    print(psi0, theta0, phi0)
    # Xf = np.array([trans[0], trans[1], trans[2], psi, theta, phi])

    phi = 3.1416
    theta = 0
    psi = 1.5708


    # Path planning parameters
    # TODO: Put parameters into datastructures to simplify readability and flexibility.
    p0x, pfx = trans[0], trans[0]
    p0y, pfy = trans[1], trans[1]
    p0z, pfz = trans[2], trans[2]

    duration = 30.0
    # hz = 100.0
    dx = 1.0/hz
    samples = duration*hz
    samples = int(samples)
    # X1, Xd1, Xdd1 = pathplanning(p0x, p0y, p0z, pfx, pfy, pfz, theta0, phi0, psi0, theta, phi, psi, duration, hz)
    # X1, Xd1, Xdd1 = pathplanning(p0x, p0y, p0z, pfx, pfy, pfz, phi0, theta0, psi0, phi, theta, psi, duration, hz)
    X1, Xd1, Xdd1 = pathplanning(p0x, p0y, p0z, pfx, pfy, pfz, phi0, theta0, psi0, phi, theta, psi, duration, hz)

    i = 0
    Xf = X1[:,0]
    Xfd = Xd1[:,0]
    Xf_prev = Xf

    # Xf_prev = Xf
    xf = np.array([0,0,0,0,0,0])
    xfd = np.array([0,0,0,0,0,0])

    Force = np.array([0,0,0,0,0,0])
    F = np.array([0,0,0,0,0,0])

    Kd = np.diag(np.array([125,125,125,1,1,1]))
    Bd =  np.diag(np.array([85,85,85,165,165,165]))
    Md = np.diag(np.array([1, 1, 1, 1, 1, 1]))
    ax = np.zeros((6,samples))
    # Kv = np.eye(7,7) * np.array([50,100,50,70,70,50,20])
    # Kp = np.eye(7,7) * np.array([12000,30000,18000,18000,12000,7000,2000])
    # print "This is it:"
    print samples
    sampcol = samples + 20
    Fcollect  = np.zeros((6,sampcol))
    taucollect  = np.zeros((Waifu.ndof,sampcol))
    xcollect  = np.zeros((6,sampcol))
    errorcollect  = np.zeros((6,sampcol))
    impcollect = np.zeros((6,sampcol))

    xincollect = X1
    xdincollect = Xd1
    xddincollect = Xdd1
    rho = 0.2
    raw_input()

    joint()
    unpause()
    while not rospy.is_shutdown():
        for i in range(sampcol):
            if i > (samples-1):
                x = X1[:,samples-1]
                xd = Xd1[:,samples-1]
                xdd = Xdd1[:,samples-1]
                print("Desired X: ", x)
                print "Done"
            else:
                x = X1[:,samples-1]
                xd = Xd1[:,samples-1]
                xdd = Xdd1[:,samples-1]
            ax = impedanceXdd(x, xd, xdd, xf, xfd, F, Kd, Bd, Md)
            impcollect[:,i] = ax
            joint()
            for j in range(Waifu.ndof):
                Waifu.joints[j].theta = q[j]

            aq_in = calcQdd(ax, q, qd, Waifu)

            joint()
            for j in range(Waifu.ndof):
                Waifu.joints[j].theta = q[j]

            q_in = np.array([q])
            qd_in = np.array([qd])
            aq_in = np.array([aq_in])
            for j in range(Waifu.ndof):
                Waifu.joints[j].theta = q[j]

            tau = np.ravel(invdyn(Waifu, q_in, qd_in, aq_in, grav))

            J = calcJac(q, Waifu)
            tauin = tau + np.dot(J.T,F)
            taucollect[:,i] = tau

            # print "tau:", tau
            pub1.publish(tauin[0])
            pub2.publish(tauin[1])
            pub3.publish(tauin[2])
            pub4.publish(tauin[3])
            pub5.publish(tauin[4])
            pub6.publish(tauin[5])
            pub7.publish(tauin[6])

            joint()
            rate.sleep()
            try:
                # Get current time
                t_cur = rospy.get_rostime().nsecs
                F = Force
                # Fcollect[:,i] = np.reshape(F,3)
                # Alternative method for computing joint acceleration

                (trans,rot) = listener.lookupTransform('panda_link0', 'panda_link8', rospy.Time(0))

                eulers = euler_from_quaternion(rot)
                Xf = np.array([trans[0], trans[1], trans[2], eulers[0], eulers[1], eulers[2]])

                joint()
                for j in range(Waifu.ndof):
                    Waifu.joints[j].theta = q[j]

                (lintwist, rottwist) = listener.lookupTwist('panda_link0', 'panda_link8', rospy.Time(0),
                                                                       rospy.Duration(1.0/hz))
                Xfd = np.array([lintwist[0], lintwist[1], lintwist[2], rottwist[0], rottwist[1], rottwist[2]])

                # Update previous parameters
                t_prev = t_cur
                Xf_prev = Xf
                Xfd_prev = Xfd
                xcollect[:,i] = Xf

                # Update final/actual robot position (measured)
                xf = Xf
                xfd = Xfd

                errorcollect[:,i] = x-xf

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, rospy.ROSInterruptException):
                pass

        fig, Cx2 = plt.subplots()
        Cx2.plot(taucollect[:,:].T)
        Cx2.grid()
        Cx2.legend(['$tau_1$', '$tau_2$', '$tau_3$', '$tau_4$', '$tau_5$', '$tau_6$', '$tau_7$'],loc="upper right")
        Cx2.set_title('$tau$')
        Cx2.set_xlabel('samples')
        Cx2.set_ylabel('$[Nm]$')
        plt.show()


        # fig, (Bx1,Bx2,Bx3) = plt.subplots(3)
        # fig.suptitle('In/Out')
        #
        # Bx1.plot(xincollect[:3,:].T)
        # Bx1.grid()
        # Bx1.legend("xyz 1",loc="upper right")
        # Bx1.set_title('$xin$')
        # Bx1.set_xlabel('samples')
        # Bx1.set_ylabel('$[m]$')
        #
        # Bx2.plot(xcollect[:3,:].T)
        # Bx2.grid()
        # Bx2.legend(['$x$', '$y$'],loc="upper right")
        # Bx2.set_title('$xout$')
        # Bx2.set_xlabel('samples')
        # Bx2.set_ylabel('$[m]$')
        #
        # Bx3.plot(errorcollect[:3,:].T)
        # Bx3.grid()
        # Bx3.legend(['$x$', '$y$'],loc="upper right")
        # Bx3.set_title('$xout$')
        # Bx3.set_xlabel('samples')
        # Bx3.set_ylabel('$[m]$')
        # plt.show()

        # fig, Fx1 = plt.subplots()
        Fx1 = plt.axes(projection='3d')

        Fx1.plot3D(xcollect[0,:].T,xcollect[1,:].T,xcollect[2,:].T)
        Fx1.plot3D(xincollect[0,:].T, xincollect[1,:].T, xincollect[2,:].T)
        # Fx1.grid()
        # Fx1 = plt.axes(projection='3d')
        # Fx1.set_title('Trajectory')
        # Fx1.set_xlabel('X-axis $[m]$')
        # Fx1.set_ylabel('Y-axis $[m]$')
        # Fx1.legend(['$x$', '$y$', '$z$'],loc="upper right")
        plt.show()


        fig, Fx2 = plt.subplots()

        Fx2.plot(errorcollect[:3,:].T)
        # Fx1.plot(xincollect[0,:].T,xincollect[1,:].T)
        Fx2.grid()
        Fx2.set_title('error')
        # Fx1.set_xlabel('X-axis $[m]$')
        # Fx1.set_ylabel('Y-axis $[m]$')
        Fx2.legend(['$x$', '$y$', '$z$'],loc="upper right")
        plt.show()

        joint()
        for j in range(Waifu.ndof):
            Waifu.joints[j].theta = q[j]

        T = Transform(Waifu)
        rot = t2rot(T)
        trans = t2transl(T)
        print "trans: ", trans

        phi0, theta0, psi0 = r2rpy(rot)
        print(psi0, theta0, phi0)
        p0x, pfx = trans[0], -0.012
        p0y, pfy = trans[1], -0.1
        p0z, pfz = trans[2], 0.826

        X1, Xd1, Xdd1 = pathplanning(p0x, p0y, p0z, pfx, pfy, pfz, phi0, theta0, psi0, phi, theta, psi, duration, hz)


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


def force():
    rospy.Subscriber("/ft_sensor2", WrenchStamped, callback)


def joint():
    rospy.Subscriber("/joint_states", JointState, jointCall)


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



if __name__ == '__main__':

    try: luna_joint_torques_publisher()
    except rospy.ROSInterruptException: pass
