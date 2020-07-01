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
from geometry_msgs.msg import Point, WrenchStamped, Vector3
from sensor_msgs.msg import JointState
from math import sin,cos,atan2,sqrt,fabs,pi
from robot2 import *
import matplotlib.pyplot as plt
import tf

from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import quaternion

# (un)pause = rospy.ServiceProxy('/gazebo/(un)pause_physics', Empty)
def luna_joint_torques_publisher():

    global q, qd
    global end_effector_position
    global link4quat
    global end_effector_vel
    global link4ang

    # Initial values for declaration
    # q = np.array([0, 0, 0, 0, 0, 0, 0])
    q = np.array([0.0, 0.0, 0.0, -1.5707, 0.0, 1.5707, 1.5707])
    qd = np.array([0, 0, 0, 0, 0, 0, 0])
    qdd = np.array([0, 0, 0, 0, 0, 0, 0])

    end_effector_position = [0, 0, 0]
    end_effector_vel = [0, 0, 0]
    link4quat = [0, 0, 0, 0]
    link4ang = [0, 0, 0]

    # Robot parameters
    # alpha = [(np.pi/2), (-np.pi/2), (np.pi/2), (-np.pi/2), (np.pi/2), (np.pi/2), 0]
    alpha = [0, (-np.pi/2), (np.pi/2), (np.pi/2), (-np.pi/2), (np.pi/2), (np.pi/2)]
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
    m = [2.34471, 2.36414, 2.38050, 2.42754, 3.49611, 1.46736, 0.45606, 0]
    # m = [4.970684, 0.646926, 3.228604, 3.587895, 1.225946, 1.666555, 7.35522e-01]
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
    Waifu = Robot(JointCol, n_dof, 'mdh')
    Waifu.joints = JointCol

    # grav = np.array([0,0,9.81])
    grav = np.array([0,0,9.81])

######################################################################################
    # ROS Initializations

    rospy.init_node('panda_joint_torque_node', anonymous=True)

    listener = tf.TransformListener() #This will listen to the tf data later
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

    model_name = 'panda'
    joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
    joint_positions = [0.0, 0.0, 0.0, -1.5707, 0.0, 1.5707, 1.5707]

    rospy.wait_for_service('gazebo/set_model_configuration')
    try:
        set_model_configuration = rospy.ServiceProxy('gazebo/set_model_configuration', SetModelConfiguration)
        resp = set_model_configuration(model_name, "", joint_names, joint_positions)
        print "set model configuration status: ", resp.status_message
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

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
    max_update_rate = 10.0

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

    print "ROS Listeners and Services initialized."

    # Publishers
    pub1 = rospy.Publisher('joint1_torque_controller/command', Float64, queue_size=50)
    pub2 = rospy.Publisher('joint2_torque_controller/command', Float64, queue_size=50)
    pub3 = rospy.Publisher('joint3_torque_controller/command', Float64, queue_size=50)
    pub4 = rospy.Publisher('joint4_torque_controller/command', Float64, queue_size=50)
    pub5 = rospy.Publisher('joint5_torque_controller/command', Float64, queue_size=50)  # 5 candidate
    pub6 = rospy.Publisher('joint6_torque_controller/command', Float64, queue_size=50)
    pub7 = rospy.Publisher('joint7_torque_controller/command', Float64, queue_size=50)  # 7 candidate

    print "ROS publishers initalized."
    hz = 50
    rate = rospy.Rate(hz) #100 Hz
######################################################################################

    unpause()
    print "Program Start:\n"
    # print "joints:", q
    # joint()
    pause()
    raw_input()

    q = np.array([0.0, 0.0, 0.0, -1.5707, 0.0, 1.5707, 1.5707])

    kine = Waifu.forwardKinematics(q)
    ptr = kine.transl
    por = kine.rpy
    pR = kine.R
    print pR
    print

    p0 = np.array([ptr[0], ptr[1], ptr[2], por[0], por[1], por[2]])

    print ptr


    duration = 10
    # hz = 1000.0
    Point0 = cartesian(ptr[0], ptr[1], ptr[2], 3.1416, 0.0, por[2])
    # Point1 = cartesian(0.5545, 0.0, 0.7315, 3.1416, 0.0, por[2])
    Point1 = cartesian(ptr[0], ptr[1], ptr[2], 3.1416, 0.0, por[2])

    Trajectory = traj(Point0, Point1, 0, duration, hz)
    # X1, Xd1, Xdd1 = Trajectory.pathplanning()

    Xp, Xdp, Xddp = Trajectory.pathplanning3()

    it = np.linspace(0,1,Trajectory.samples)
    rot = np.array([[1.0000,         0,         0],
             [0,   -1.0000,   0.0000],
             [0,    0.0000,   -1.0000] ])

    rot2 = np.array([[ 0.0000,   -1.0000,         0],
                     [-1.0000,   -0.0000,   -0.0000],
                     [ 0.0000,    0.0000,   -1.0000]])

    # print
    # print rot
    # print
    # print rot2

    q1et = quaternion.from_rotation_matrix(pR)
    q2et = quaternion.from_rotation_matrix(rot2)
    quatf = q1et
    quatf_d = quatf * np.log(q2et*q1et.inverse())

    # Xfo = np.zeros((3,Trajectory.samples))

    X1 = np.zeros((6,Trajectory.samples))
    Xd1 = np.zeros((6,Trajectory.samples))
    Xdd1 = np.zeros((6,Trajectory.samples))

    # # rot = np.zeros((3,100))
    for i in range(Trajectory.samples):
        quat = quaternion.slerp_evaluate(q1et,q2et,it[i])
        quat_d = quat * np.log(q2et*q1et.inverse())
        quat_dd = quat * np.log(q2et*q1et.inverse())**2

        X1[:,i] = np.array([Xp[0,i],Xp[1,i],Xp[2,i], quat.x, quat.y, quat.z])
        Xd1[:,i] = np.array([Xdp[0,i],Xdp[1,i],Xdp[2,i], quat_d.x, quat_d.y, quat_d.z])
        Xdd1[:,i] = np.array([Xddp[0,i],Xddp[1,i],Xddp[2,i], quat_dd.x, quat_dd.y, quat_dd.z])

    print "Pathplanning: Complete!"

######################################################################################

    Xf = X1[:,0]
    Xfd = Xd1[:,0]
    Xf_prev = Xf

    # Xf_prev = Xf
    xf = np.array([0,0,0,0,0,0])
    xfd = np.array([0,0,0,0,0,0])

    Force = np.array([0,0,0,0,0,0])
    F = np.array([0,0,0,0,0,0])
    impctrl = impedanceController()
    kx = 1000.0
    ky = 1000.0
    kz = 1000.0
    kr = 10.0
    kp = 10.0
    kya = 10.0

    mx = 5.0
    my = 5.0
    mz = 5.0
    mr = 5.0
    mp = 5.0
    mya = 5.0


    impctrl.Kd = np.diag(np.array([kx,ky,kz, kr, kp, kya]))
    impctrl.Md = np.diag(np.array([mx, my, mz, mr, mp, mya]))
    # impctrl.Bd =  np.diag(0.5 *np.array([0.8, 0.8, 0.8, 0.5, 0.5, 0.5]))
    # impctrl.Bd = np.diag(1 * np.array([2*mx*np.sqrt(kx/mx), 2*my*np.sqrt(ky/my), 2*mz*np.sqrt(kz/mz),
    #                                 2*mr*np.sqrt(kr/mr), 2*mp*np.sqrt(kp/mp), 2*mya*np.sqrt(kya/mya)]))
    impctrl.Bd = np.diag(0.2 * np.array([2*np.sqrt(kx/mx), 2*np.sqrt(ky/my), 2*np.sqrt(kz/mz),
                                    2*np.sqrt(kr/mr), 2*np.sqrt(kp/mp), 2*np.sqrt(kya/mya)]))

    eps = np.array([0.8,0.8,0.8,0.5,0.5,0.5])
    D_eps = np.diag(eps)
    Kx = np.sqrt(impctrl.Kd)

    print "natural frequencies: ", np.array([2*mx*np.sqrt(kx/mx), 2*my*np.sqrt(ky/my), 2*mz*np.sqrt(kz/mz),
                                    2*mr*np.sqrt(kr/mr), 2*mp*np.sqrt(kp/mp), 2*mya*np.sqrt(kya/mya)])

    print "damping coefficient: 1"

    print "Control parameters: Initialized."

######################################################################################

    # For post trajectory control - to see whether the robot maintains the target position.
    sampcol = Trajectory.samples
    samples = Trajectory.samples
    dx = Trajectory.dx

    # Initial parameters for the impedance controller.
    # The cartesian parameter values:
    Xf = X1[:,0]
    Xfd = Xd1[:,0]
    Xf_prev = Xf

    # The joint parameter values:

    # Nullspace:
    q_nullspace = q

    # loop parameters (redundant, may fix later), used in case I don't want feedback.
    xf = Xf
    xfd = Xfd

    cdof = 6

######################################################################################

    # Parameters for collecting data.
    # The forces.
    Fcollect  = np.zeros((7,sampcol))
    taucollect  = np.zeros((Waifu.ndof,sampcol))

    # The cartesian positions.
    xcollect  = np.zeros((6,sampcol))
    xdcollect  = np.zeros((6,sampcol))
    qcollect = np.zeros([7,1])
    qdcollect = np.zeros([7,1])
    qddcollect = np.zeros([7,1])
    aqcollect = np.zeros((7,sampcol))

    # The error.
    errorcollect  = np.zeros((6,sampcol))

    # The computed acceleration from the impedance controller.
    impcollect = np.zeros((6,sampcol))

######################################################################################

    # Exponential Filter
    cartesian_stiffness_ = np.zeros((6,6))
    cartesian_damping_ = np.zeros((6,6))
    cartesian_inertia_ = np.zeros((6,6))

    nullspace_stiffness_ = 20
    nullspace_stiffness_target_ = 20
    filter_params_ = 0.0005

    # Fdist = np.array([0.5,0.5,0,0,0,0])
    #
    # eps = np.array([0.3,0.3,0.3,0.3,0.3,0.3])
    # D_eps = np.diag(eps)
    print q.shape[0]

######################################################################################

    print "Initializations complete.!"
    raw_input()
    print "samples", sampcol
######################################################################################
    joint()
    unpause()
    print "Go! ->"
    i = 0
    x = X1[:,0]
    # x_target = X1[:,50]
    while not rospy.is_shutdown():
        # for i in range(sampcol):
        if i > (samples-1):
            x = X1[:cdof,samples-1]
            xd = Xd1[:cdof,samples-1]
            xdd = Xdd1[:cdof,samples-1]
            quat = quaternion.slerp_evaluate(q1et,q2et,0)
            quat_d = quat * np.log(q2et*q1et.inverse())
            quat_dd = quat * np.log(q2et*q1et.inverse())**2
            i = i-1

            fig, Fx1 = plt.subplots()

            Fx1.plot(xcollect[:3,:].T)
            # Fx1.plot(xincollect[0,:].T,xincollect[1,:].T)
            Fx1.grid()
            Fx1.set_title('Trajectory')
            Fx1.set_xlabel('X-axis $[m]$')
            Fx1.set_ylabel('Y-axis $[m]$')
            Fx1.legend(['$x$', '$y$', '$z$'],loc="upper right")
            plt.show()

            fig, Fx3 = plt.subplots()

            Fx3.plot(xcollect[3:,:].T)
            # Fx1.plot(xincollect[0,:].T,xincollect[1,:].T)
            Fx3.grid()
            Fx3.set_title('Unit Quat')
            # Fx1.set_xlabel('X-axis $[m]$')
            # Fx1.set_ylabel('Y-axis $[m]$')
            Fx3.legend(['$x$', '$y$', '$z$'],loc="upper right")
            plt.show()

            fig, Fx2 = plt.subplots()

            Fx2.plot(taucollect.T)
            # Fx1.plot(xincollect[0,:].T,xincollect[1,:].T)
            Fx2.grid()
            Fx2.set_title('Torque')
            # Fx1.set_xlabel('X-axis $[m]$')
            # Fx1.set_ylabel('Y-axis $[m]$')
            Fx2.legend(['$q_1$', '$q_2$', '$q_3$','$q_4$', '$q_5$', '$q_6$', '$q_7$'],loc="upper right")
            plt.show()

            pause()
            print "Done"
        else:
            x = X1[:cdof,1]
            xd = Xd1[:cdof,1]
            xdd = Xdd1[:cdof,1]
            # quat = quaternion.slerp_evaluate(q1et,q2et,it[-1])
            quat = quaternion.slerp_evaluate(q1et,q2et,it[i])
            quat_d = quat * np.log(q2et*q1et.inverse())
            quat_dd = quat * np.log(q2et*q1et.inverse())**2

        # Impedance controller
        # Error signal
        ep = xf[:3] - x[:3]
        eq = quatf * quat.inverse()
        e = np.array([ep[0], ep[1], ep[2], eq.x, eq.y, eq.z])

        # errorcollect[:,i] = e
        # print e

        edp = (xfd - xd)
        edq = quatf_d * quat_d.inverse()
        ed = np.array([edp[0], edp[1], edp[2], edq.x, edq.y, edq.z])

        q_nullspace = q

        #Cartesian Inertia Matrix
        joint()
        Lambda = Waifu.cinertiaComp(q, quat)
        # Lambda = Lambda[:3,:3]

        #Cartesian Coriolis Matrix
        joint()
        C = Waifu.coriolisComp(q, qd)

        #Jointspace gravitational load vector
        joint()
        tg = Waifu.gravloadComp(q, grav)

        # Computing Jacobian:
        joint()
        J = Waifu.calcJac(q)
        A = Waifu.quat2Ja(quatf)
        B = block_diag(np.eye(3), A[1:,:])
        Ja = np.dot(B,J)
        Jpinv = Waifu.pinv(Ja)

        p1 = (np.dot(Ja.T, ( np.dot(Lambda, xdd) )) -
            np.dot(Ja.T, np.dot(np.dot(Lambda, np.linalg.inv(impctrl.Md)),
                               (np.dot(impctrl.Kd, e) + np.dot(impctrl.Bd, ed)))
        ))

        tauc =  p1
        # print tauc
        # J = Waifu.calcJac(q)
        # Jpinv = Waifu.pinv(J)

        tau_nullspace = np.dot((np.eye(7) - np.dot(Ja.T, Jpinv.T) ), np.dot( nullspace_stiffness_, (q_nullspace - q)) -
                         np.dot(np.dot(2, np.sqrt(nullspace_stiffness_)), qd))

        tau = tauc + tau_nullspace + np.dot(C,qd) + tg
        taucollect[:,i] = tau

        # print "tau:", tau
        pub1.publish(tau[0])
        pub2.publish(tau[1])
        pub3.publish(tau[2])
        pub4.publish(tau[3])
        pub5.publish(tau[4])
        pub6.publish(tau[5])
        pub7.publish(tau[6])

        print "iteration", i
        rate.sleep()
    # try:
    #     # Get current time
        t_cur = rospy.get_rostime().nsecs
        F = Force
        # Fcollect[:,i] = np.reshape(F,3)
        # Alternative method for computing joint acceleration

        gazebo_link_states()


        # (trans,rot) = listener.lookupTransform('panda_link0', 'panda_link7', rospy.Time(0))

        # eulers = euler_from_quaternion(rot)

        # print "rot", rot

        Xf = np.array([end_effector_position[0], end_effector_position[1], end_effector_position[2], link4quat[1], link4quat[2], link4quat[3]])

        # (lintwist, rottwist) = listener.lookupTwist('panda_link0', 'panda_link7', rospy.Time(0),
        #                                                        rospy.Duration(1.0/hz))

        print "Xf:", Xf

        quatf = quaternion.from_float_array(link4quat)

        print "quatf", quatf

        A = Waifu.quat2Ja(quatf)
        Xquatd_return = A.dot(np.array([link4ang[0], link4ang[1], link4ang[2]]))
        quatf_d = quaternion.from_float_array(Xquatd_return)

        Xfd = np.array([end_effector_vel[0], end_effector_vel[1], end_effector_vel[2], Xquatd_return[1], Xquatd_return[2], Xquatd_return[3]])

        print
        print
        print "Xfd:", Xfd
        print "quatf_d", quatf_d
        # Collecting output.
        xcollect[:,i] = Xf
        xdcollect[:,i] = Xfd

        # joint()
        # kine = Waifu.forwardKinematics(q)
        # Xftr = kine.transl
        #
        # roll, pitch, yaw = kine.rpy
        # rot = kine.R
        #
        # quatf = quaternion.from_rotation_matrix(rot)
        # Xf = np.array([Xftr[0], Xftr[1], Xftr[2], quatf.x, quatf.y, quatf.z])
        #
        # joint()
        # Xfd_calc = Waifu.calcXd(q, qd)
        # quatf_d = quaternion.from_float_array(Xfd_calc[3:])
        # Xfd = np.array([Xfd_calc[0], Xfd_calc[1], Xfd_calc[2], Xfd_calc[4], Xfd_calc[5], Xfd_calc[6]])


        # Computed output (Parameters used in case I want to remove feedback).
        xf = Xf
        xfd = Xfd

            # Exponential smoothing function.
        # cartesian_stiffness_ = np.dot(filter_params_, impctrl.Kd) + np.dot((1.0 - filter_params_), cartesian_stiffness_)
        # cartesian_damping_ = np.dot(filter_params_, impctrl.Bd) + np.dot((1.0 - filter_params_), cartesian_damping_)
        # # cartesian_inertia_ = np.dot(filter_params_, impctrl.Md) + np.dot((1.0 - filter_params_), cartesian_inertia_)
        # nullspace_stiffness_ = np.dot(filter_params_, nullspace_stiffness_target_) + np.dot((1.0 - filter_params_), nullspace_stiffness_)
        # # x = np.dot( filter_params_, x_target ) +  np.dot( (1.0 - filter_params_), x )

        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, rospy.ROSInterruptException):
        #     pass

        i = i+1
        # fig, Cx2 = plt.subplots()
        # Cx2.plot(taucollect[:,:].T)
        # Cx2.grid()
        # Cx2.legend(['$tau_1$', '$tau_2$', '$tau_3$', '$tau_4$', '$tau_5$', '$tau_6$', '$tau_7$'],loc="upper right")
        # Cx2.set_title('$tau$')
        # Cx2.set_xlabel('samples')
        # Cx2.set_ylabel('$[Nm]$')
        # plt.show()
        #
        #
        # # fig, (Bx1,Bx2,Bx3) = plt.subplots(3)
        # # fig.suptitle('In/Out')
        # #
        # # Bx1.plot(xincollect[:3,:].T)
        # # Bx1.grid()
        # # Bx1.legend("xyz 1",loc="upper right")
        # # Bx1.set_title('$xin$')
        # # Bx1.set_xlabel('samples')
        # # Bx1.set_ylabel('$[m]$')
        # #
        # # Bx2.plot(xcollect[:3,:].T)
        # # Bx2.grid()
        # # Bx2.legend(['$x$', '$y$'],loc="upper right")
        # # Bx2.set_title('$xout$')
        # # Bx2.set_xlabel('samples')
        # # Bx2.set_ylabel('$[m]$')
        # #
        # # Bx3.plot(errorcollect[:3,:].T)
        # # Bx3.grid()
        # # Bx3.legend(['$x$', '$y$'],loc="upper right")
        # # Bx3.set_title('$xout$')
        # # Bx3.set_xlabel('samples')
        # # Bx3.set_ylabel('$[m]$')
        # # plt.show()




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


def gazebo_link_states():
    rospy.Subscriber("/gazebo/link_states", LinkStates, callback2) #subscribes to /gazebo/link_states of type LinkStates


def callback2(message):
#Obtains 3rd revolute joint position and orientation from gazebo/link_states
    global end_effector_position
    global link4quat
    global end_effector_vel
    global link4ang

    # print message.name[8]

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


def qIntegrate(qddcollect,qdcollect,qcollect, qacc, dx):
    # Integrate from acceleration to velocity
    qdtemp = np.concatenate((qddcollect, qacc.reshape(7,1)), axis=1)
    qd = np.trapz(qdtemp, axis=1) * dx

    #Integrate from velocity to position
    qtemp = np.concatenate((qdcollect, qd.reshape(7,1)), axis=1)
    q = np.trapz(qtemp, axis=1) * dx

    qddcollect = np.concatenate((qddcollect, qacc.reshape(7,1)),axis=1)
    qdcollect = np.concatenate((qdcollect, qd.reshape(7,1)),axis=1)
    qcollect = np.concatenate((qcollect, q.reshape(7,1)),axis=1)

    return qd, q, qddcollect, qdcollect, qcollect


if __name__ == '__main__':

    try: luna_joint_torques_publisher()
    except rospy.ROSInterruptException: pass
