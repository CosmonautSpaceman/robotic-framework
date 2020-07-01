import numpy as np
import scipy
from scipy.linalg import block_diag
import math
import quaternion


class rJoint:
    def __init__(self, alpha, a, theta, offset, d, type, inertia, m, r):
        self.alpha = alpha
        self.a = a
        self.offset = offset
        self.theta = theta
        self.d = d
        self.type = type
        self.inertia = inertia
        self.m = m
        self.r = r


class cartesian:
    def __init__(self, x, y, z, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


class jointTarget:
    def __init__(self, q0, qf):
        self.q0 = q0
        self.qf = qf


class jointspaceConfig:
    def __init__(self, joints):
        self.joints = joints


class fkine:
    def __init__(self, T, A, Aout, transl, R, rpy, w_hat):
        self.T = T
        self.A = A
        self.Aout = Aout
        self.transl = transl
        self.R = R
        self.rpy = rpy
        self.w_hat = w_hat


class impedanceController:
    def __init__(self):
        self.Kd = np.diag(np.array([125,125,125,1,1,1]))
        self.Bd = np.diag(np.array([85,85,85,165,165,165]))
        self.Md = np.diag(np.array([15, 15, 15, 1, 1, 1]))
        self.x = np.zeros(6)
        self.x_target = np.zeros(6)
        self.quat = np.quaternion(1,0,0,0)
        self.quat_target = np.quaternion(1,0,0,0)
        self.filter_params_ = 0.0005
        self.cartesian_stiffness_ = np.zeros((6,6))
        self.cartesian_damping_ = np.zeros((6,6))
        self.nullspace_stiffness_ = 20
        self.nullspace_stiffness_target_ = 20
        self.zeta = np.array([1, 1, 1, 1, 1, 1])
        self.F = np.zeros(6)

    # Controllers
    def output(self, x, xd, xdd, xc, xcd, F):

        Mdinv = np.linalg.inv(self.Md)
        damper = np.dot(self.Bd,(xcd - xd))
        spring = np.dot(self.Kd,(xc - x))

        ax = xdd - np.dot(Mdinv,(damper + spring + F))
        return ax

    def output3(self, e, ed, xdd, F):

        Mdinv = np.linalg.inv(self.Md[:3,:3])
        damper = np.dot(self.Bd[:3,:3],(ed[:3]))
        spring = np.dot(self.Kd[:3,:3],(e[:3]))

        ax = xdd[:3] - np.dot(Mdinv,(damper + spring + F[:3]))
        return ax

    def outputquat(self, Kd_b, Bd, e, ed, xdd, F):
        Mdinv = np.linalg.inv(self.Md)
        damper = np.dot(Bd,ed)
        spring = np.dot(Kd_b,e)

        ax = xdd - np.dot(Mdinv,(damper + spring + F))
        return ax

    def exponential_smoothing(self):
        # Exponential smoothing function.
        self.cartesian_stiffness_ = np.dot(self.filter_params_, self.Kd) + np.dot((1.0 - self.filter_params_), self.cartesian_stiffness_)
        self.cartesian_damping_ = np.dot(self.filter_params_, self.Bd) + np.dot((1.0 - self.filter_params_), self.cartesian_damping_)
        self.nullspace_stiffness_ = np.dot(self.filter_params_, self.nullspace_stiffness_target_) + np.dot((1.0 - self.filter_params_), self.nullspace_stiffness_)
        self.x = np.dot( self.filter_params_, self.x_target ) +  np.dot( (1.0 - self.filter_params_), self.x )
        self.quat = quaternion.slerp_evaluate(self.quat, self.quat_target, self.filter_params_)

    def damping_dual_eigen(self, Kd, M):
        # Computing dynamic damping
        B0, Q = scipy.linalg.eigh(Kd, M)
        B0sqrt = np.sqrt(B0)
#         If the resulting eigenvalues are close to zero
        for i in range(6):
            if np.isnan(B0sqrt[i]):
                B0sqrt[i] = 0

        Kd_B = Q.T.dot(B0.dot(Q))
        Bd = 2*Q.T.dot(( np.diag( self.zeta * B0sqrt ).dot(Q)) )
        return Bd, Kd_B

    def damping_dual_eigen2(self, Kd, M):
        # Computing dynamic damping
        B0, Q = scipy.linalg.eigh(Kd, M)
        B0sqrt = np.sqrt(B0)
#         If the resulting eigenvalues are close to zero
        for i in range(6):
            if np.isnan(B0sqrt[i]):
                B0sqrt[i] = 0

        Kd_B = Q.T.dot(B0.dot(Q))
        Bd = 2*Q.T.dot(( np.diag( np.ones(7) * B0sqrt ).dot(Q)) )
        return Bd, Kd_B


    def damping_constant_mass(self):
        # Computing dynamic damping
        K1 = np.sqrt(self.Kd)
        M1 = np.sqrt(self.Md)
        Bd = self.zeta * (M1.dot(K1) + K1.dot(M1) )
        return Bd

    def damping_constant_mass2(self, Kd):
        # Computing dynamic damping
        K1 = np.sqrt(Kd)
        M1 = np.sqrt(self.Md)
        Bd = self.zeta * (M1.dot(K1) + K1.dot(M1) )
        return Bd



class Robot:
    def __init__(self, dh):
        self.joints = 0
        self.ndof = 0
        self.dh = dh
        self.grav = np.array([0,0,9.81])

    def robot_init(self):
        # Initial values for declaration
        q = np.array([0, 0, 0, 0, 0, 0, 0])
        # q = np.array([0.0, 0.0, 0.0, -1.5707, 0.0, 1.5707, 1.5707])
#         q_off = np.array([0.1, 0.1, 0.1, -np.pi/3, 0.1, np.pi/3, np.pi/3])
        q_off = np.array([0, 0, 0, 0, 0, 0, 0])
        # q = np.array([0.1, 0.1, 0.1, -np.pi/3, 0.1, np.pi/3, np.pi/3])
        qd = np.array([0, 0, 0, 0, 0, 0, 0])
        qdd = np.array([0, 0, 0, 0, 0, 0, 0])

        # Robot parameters
        # alpha = [(np.pi/2), (-np.pi/2), (np.pi/2), (-np.pi/2), (np.pi/2), (np.pi/2), 0]
        alpha = [0, (-np.pi/2), (np.pi/2), (np.pi/2), (-np.pi/2), (np.pi/2), (np.pi/2), 0]
        a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
        d = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
        # m = [4.970684, 0.646926, 3.228604, 3.587895, 1.225946, 1.666555, 7.35522e-01]
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
        Joint1 = rJoint(alpha[0], a[0], q[0], q_off[0], d[0], 'R', Ipanda, m[0], np.array([[3.875e-03],[2.081e-03],[0]]) )
        Joint2 = rJoint(alpha[1], a[1], q[1], q_off[1], d[1], 'R', Ipanda, m[1], np.array([[-3.141e-03],[-2.872e-02],[3.495e-03]]) )
        Joint3 = rJoint(alpha[2], a[2], q[2], q_off[2], d[2], 'R', Ipanda, m[2], np.array([[2.7518e-02],[3.9252e-02],[-6.6502e-02]]) )
        Joint4 = rJoint(alpha[3], a[3], q[3], q_off[3], d[3], 'R', Ipanda, m[3], np.array([[-5.317e-02],[1.04419e-01],[2.7454e-02]]) )
        Joint5 = rJoint(alpha[4], a[4], q[4], q_off[4], d[4], 'R', Ipanda, m[4], np.array([[-1.1953e-02],[4.1065e-02],[-3.8437e-02]]) )
        Joint6 = rJoint(alpha[5], a[5], q[5], q_off[5], d[5], 'R', Ipanda, m[5], np.array([[6.0149e-02],[-1.4117e-02],[-1.0517e-02]]) )
        Joint7 = rJoint(alpha[6], a[6], q[6], q_off[6], d[6], 'R', Ipanda, m[6], np.array([[1.0517e-02],[-4.252e-03],[6.1597e-02]]) )

        # Collecting joints
        JointCol = [Joint1, Joint2, Joint3, Joint4, Joint5, Joint6, Joint7]
        self.joints = JointCol
        self.ndof = 7

    def inverseDynamics(self, qc, qcdot, qcddot, grav):

        qc = qc.reshape((1,self.ndof))
        qcdot = qcdot.reshape((1,self.ndof))
        qcddot = qcddot.reshape((1,self.ndof))

        if self.dh.lower() == 'mdh':
            grav = grav.reshape(3)
            Q = np.ravel(self.mdh_invdyn(qc, qcdot, qcddot, grav))
        else:
            grav = grav.reshape((3,1)) # May need to fix this
            Q = np.ravel(self.invdyn(qc, qcdot, qcddot, grav))
        return Q

    def forwardKinematics(self, q, *args):

        if args:
            if args[0].lower() in ['tool']:
                q = np.array([q[0], q[1], q[2], q[3], q[4], q[5], q[6], 0])
                self.n_dof = 8
                if self.dh.lower() == 'mdh':
                    T,A,Aout = self.mdh_Transform(q)
                else:
                    T,A,Aout = self.Transform(q)
                self.n_dof = 7
        else:
            if self.dh.lower() == 'mdh':
                T,A,Aout = self.mdh_Transform(q)
            else:
                T,A,Aout = self.Transform(q)

        transl = self.t2transl(T)
        R = self.t2rot(T)
        r,p,y = self.r2rpy(R)
        rpy = [r,p,y]
        w_hat = self.angleRep(R)

        kinematics = fkine(T, A, Aout, transl, R, rpy, w_hat)

        return kinematics

    def angleRep(self, R):
        theta = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1.0)/2.0)
        vec = np.array([(R[2,1] - R[1,2]),(R[0,2]-R[2,0]),(R[1,0]-R[0,1])])
        w_hat = 1/(2*np.sin(theta)) * vec
        w_hat = np.array([theta * w_hat[0], theta * w_hat[1], theta * w_hat[2]])
        return w_hat

    def jacobian(self, q):
        J = self.calcJac(q)
        return J

    def jacobianDot(self, q, qd):
        Jd = self.calcJacDot(q, qd)
        return Jd

    # Kinematics
    def mdh_Transform(rb, q):
        for j in range(rb.ndof):
            rb.joints[j].theta = q[j]

        A = [0 for i in range(rb.ndof)]

        alp = np.zeros(rb.ndof)
        a = np.zeros(rb.ndof)
        th = np.zeros(rb.ndof)
        d = np.zeros(rb.ndof)

        for i in range(rb.ndof):
            alp[i] = rb.joints[i].alpha
            a[i] = rb.joints[i].a
            th[i] = rb.joints[i].theta + rb.joints[i].offset
            d[i] = rb.joints[i].d

        T = np.identity(4)
        Aout = []
        for i in range(rb.ndof):
            ct = np.cos(th[i])
            st = np.sin(th[i])
            ca = np.cos(alp[i])
            sa = np.sin(alp[i])


            A[i] = np.array([[ct, -st, 0, a[i]],
                      [(st * ca), (ct * ca), -sa, (-d[i] * sa)],
                      [(st * sa), (ct * sa), ca, (d[i] * ca)],
                      [0, 0, 0, 1]])
            Aout.append(np.dot(T, A[i]))
            T = np.dot(T, A[i])

        return T, A, Aout

    def Transform(rb, q):
        for j in range(rb.ndof):
            rb.joints[j].theta = q[j]

        A = [0 for i in range(rb.ndof)]

        alp = np.zeros(rb.ndof)
        a = np.zeros(rb.ndof)
        th = np.zeros(rb.ndof)
        d = np.zeros(rb.ndof)
        # A = np.zeros(2)

        for i in range(rb.ndof):
            alp[i] = rb.joints[i].alpha
            a[i] = rb.joints[i].a
            th[i] = rb.joints[i].theta + rb.joints[i].offset
            d[i] = rb.joints[i].d

        T = np.identity(4)

        Aout = []
        for i in range(rb.ndof):
            A[i] = np.array([[np.cos(th[i]), -np.sin(th[i]) * np.cos(alp[i]), np.sin(th[i]) * np.sin(alp[i]), a[i] * np.cos(th[i])],
                 [np.sin(th[i]), np.cos(th[i]) * np.cos(alp[i]), -np.cos(th[i]) * np.sin(alp[i]), a[i] * np.sin(th[i])],
                 [0, np.sin(alp[i]), np.cos(alp[i]), d[i]],
                 [0, 0, 0, 1]])
            Aout.append(np.dot(T, A[i]))
            T = np.dot(T, A[i])
        return T, A, Aout

    def t2transl(self, T):
        transl = np.ravel(T[:3, 3])
        return transl

    def t2rot(self, T):
        R = T[:3, :3]
        return R

    def r2eul(self, R):
        if (R[0,2] < np.finfo(float).eps and R[1,2] <np.finfo(float).eps):
            theta = 0
            sp = 0
            cp = 1
            phi = np.arctan2(cp*R[0,2] + sp*R[1,2], R[2,2])
            psi = np.arctan2(-sp*R[0,0] + cp*R[1,0], -sp*R[0,1] + cp*R[1,1])
        else:
            # sin(theta) > 0
            #theta = np.arctan2(R[2,2], np.sqrt(1 - (R[2,2]**2)))
            theta = np.arctan2(R[1,2],R[0,2])
            sp = np.sin(theta)
            cp = np.cos(theta)
            phi = np.arctan2(cp*R[0,2] + sp*R[1,2], R[2,2])
            psi = np.arctan2(-sp*R[0,0] + cp*R[1,0], -sp*R[0,1] + cp*R[1,1])
        return theta, phi, psi

    def isclose(self, x, y, rtol=1.e-5, atol=1.e-8):
        return abs(x-y) <= atol + rtol * abs(y)

    def r2rpy(self, R):
        '''
        From a paper by Gregory G. Slabaugh (undated),
        "Computing Euler angles from a rotation matrix
        '''
        phi = 0.0
        if self.isclose(R[2,0],-1.0):
            theta = math.pi/2.0
            psi = math.atan2(R[0,1],R[0,2])
        elif self.isclose(R[2,0],1.0):
            theta = -math.pi/2.0
            psi = math.atan2(-R[0,1],-R[0,2])
        else:
            theta = -math.asin(R[2,0])
            cos_theta = math.cos(theta)
            psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
            phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
        return psi, theta, phi

    def calcInverseKin(self, X):
        # Pre solved for the 2-DOF Planar robot.
        tx = X[0]
        ty = X[1]
        tz = X[2]
        q1 = 2*np.arctan((7*ty + (- 25*tx**4 - 50*tx**2*ty**2 + 49*tx**2 - 25*ty**4 + 49*ty**2)**(1/2))/(5*tx**2 + 7*tx + 5*ty**2))
        q2 = -2*np.arctan((- 25*tx**2 - 25*ty**2 + 49)**(1/2)/(5*(tx**2 + ty**2)**(1/2)))
        if np.isnan(q1):
            q1 = 0
        if np.isnan(q2):
            q2 = 0

        qc = np.array([q1,q2])
        return qc


    def calcQd(rb, Xd, qc, *args):
        J = rb.calcJac(qc)
        Jt = np.transpose(J)
        kine = rb.forwardKinematics(qc,'tool')

        if args:
            if args[0].lower() in ['rpy']:
                rpy = kine.rpy
                A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
                B = block_diag(np.eye(3),np.linalg.inv(A))
                Ja = np.dot(B,J)
                J = Ja

            if args[0].lower() in ['quaternion']:
#                 rot = kine.R
#                 quat = quaternion.from_rotation_matrix(rot)
                quat = quaternion.from_float_array(Xd[3:])
                A = rb.quat2Ja(quat)
                B = block_diag(np.eye(3), A)
                Ja = np.dot(B,J)
                J = Ja

        Jpinv = rb.pinv(J)
        qd = np.dot(Jpinv, Xd)
        return qd


    def calcQdNull(rb, Xd, qc, qn, *args):
        J = rb.calcJac(qc)
        Jt = np.transpose(J)
        kine = rb.forwardKinematics(qc,'tool')

        if args:
            if args[0].lower() in ['rpy']:
                rpy = kine.rpy
                A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
                B = block_diag(np.eye(3),np.linalg.inv(A))
                Ja = np.dot(B,J)
                J = Ja

            if args[0].lower() in ['quaternion']:
#                 rot = kine.R
#                 quat = quaternion.from_rotation_matrix(rot)
                quat = quaternion.from_float_array(Xd[3:])
                A = rb.quat2Ja(quat)
                B = block_diag(np.eye(3), A)
                Ja = np.dot(B,J)
                J = Ja

        Jpinv = rb.pinv(J)
        qd = np.dot(Jpinv, Xd) + np.dot( (np.eye(7) - Jpinv.dot(J)), qn)
        return qd

    def calcQdd(rb, Xdd, qc, qd, *args):
        qin = np.zeros(rb.ndof)
        qdin = np.zeros(rb.ndof)

        for i in range(rb.ndof):
            qin[i] = qc[i]
            qdin[i] = qd[i]

        J = rb.calcJac(qin)
        Jd = rb.calcJacDot(qin, qdin)
        Jdq = np.dot(Jd,qdin)
        kine = rb.forwardKinematics(qin,'tool')

        if args:
            if args[0].lower() in ['rpy']:
                rpy = kine.rpy
                A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
                B = block_diag(np.eye(3),np.linalg.inv(A))
                Ja = np.dot(B,J)
                Jad = np.dot(B,Jd)
                Jadq = np.dot(Jad,qdin)

                J = Ja
                Jdq = Jadq

            if args[0].lower() in ['quaternion']:
                rot = kine.R
                quat = quaternion.from_rotation_matrix(rot)
                A = rb.quat2Ja(quat)
                B = block_diag(np.eye(3), A[1:,:])
                Ja = np.dot(B,J)
                Jad = np.dot(B,Jd)
                Jadq = np.dot(Jad,qdin)

                J = Ja
                Jdq = Jadq

        Jpinv = rb.pinv(J)
        qdd = np.dot(Jpinv, (Xdd - Jdq))
        return qdd

    def calcQddNull(rb, Xdd, qc, qd, qddn,*args):
        qin = np.zeros(rb.ndof)
        qdin = np.zeros(rb.ndof)

        for i in range(rb.ndof):
            qin[i] = qc[i]
            qdin[i] = qd[i]

        J = rb.calcJac(qin)
        Jd = rb.calcJacDot(qin, qdin)
        Jdq = np.dot(Jd,qdin)
        kine = rb.forwardKinematics(qin,'tool')

        if args:
            if args[0].lower() in ['rpy']:
                rpy = kine.rpy
                A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
                B = block_diag(np.eye(3),np.linalg.inv(A))
                Ja = np.dot(B,J)
                Jad = np.dot(B,Jd)
                Jadq = np.dot(Jad,qdin)

                J = Ja
                Jdq = Jadq

            if args[0].lower() in ['quaternion']:
                rot = kine.R
                quat = quaternion.from_rotation_matrix(rot)
                A = rb.quat2Ja(quat)
                B = block_diag(np.eye(3), A[1:,:])
                Ja = np.dot(B,J)
                Jad = np.dot(B,Jd)
                Jadq = np.dot(Jad,qdin)

                J = Ja
                Jdq = Jadq

        Jpinv = rb.pinv(J)
        qdd = np.dot(Jpinv, (Xdd - Jdq))+ np.dot( (np.eye(7) - Jpinv.dot(J)), qddn)
        return qdd

    def calcQdd3(rb, Xdd, qc, qd):
        J = rb.calcJac(qc)
        Jd = rb.calcJacDot(qc, qd)
        Jdq = np.dot(Jd,qd)

#         kine = rb.forwardKinematics(qc)
#         rpy = kine.rpy

#         A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
#         B = block_diag(np.eye(3),np.linalg.inv(A))
#         # Jadq = np.dot(B,Jdq)
#         Ja = np.dot(B,J)
#         Jpinv = rb.pinv(Ja)
        Jpinv = rb.pinv(J)
        qdd = np.dot(Jpinv[:,:3], (Xdd[:3] - Jdq[:3]))
        return qdd

    def calcXd(rb, qc, qd, *args):
        J = rb.calcJac(qc)
        kine = rb.forwardKinematics(qc,'tool')
        if args:
            if args[0].lower() in ['rpy']:
                rpy = kine.rpy
                A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
                B = block_diag(np.eye(3),np.linalg.inv(A))
                Ja = np.dot(B,J)
                J = Ja

            if args[0].lower() in ['quaternion']:
                rot = kine.R
                quat = quaternion.from_rotation_matrix(rot)
#                 quat_R = rb.mat2quat(rot)
#                 quat = quaternion.from_float_array(quat_R)
                quat = rb.rounding_quaternion(quat)
                A = rb.quat2Ja(quat)
                B = block_diag(np.eye(3), A)
                Ja = np.dot(B,J)
                J = Ja

        xd = np.dot(J, qd)
        return xd

    def calcXdd(rb, qc, qd, qdd, *args):
        J = rb.calcJac(qc)
        Jd = rb.calcJacDot(qc, qd)
        Jdq = np.dot(Jd,qd)

        kine = rb.forwardKinematics(qc)
        if args:
            if args[0].lower() in ['rpy']:
                rpy = kine.rpy
                A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
                B = block_diag(np.eye(3),np.linalg.inv(A))
                Ja = np.dot(B,J)
                Jadq = np.dot(B,Jdq)
                J = Ja
                Jd = Jadq

            if args[0].lower() in ['quaternion']:
                rot = kine.R
                quat = quaternion.from_rotation_matrix(rot)
                A = rb.quat2Ja(quat)
                B = block_diag(np.eye(3), A)
                Ja = rb.analyticJacobian(J, quat, 'quaternion')
                Jad = rb.analyticJacobianDot2(J, Jd, qd, quat)
                Jadq = Jad.dot(qd)
                J = Ja
                Jdq = Jadq

        xdd = np.dot(J, qdd) + Jdq
        return xdd

    def calcVd(rb, qc, qd, xd_in, *args):
        kine = rb.forwardKinematics(qc,'tool')
        R_e = block_diag(kine.R,kine.R)
        J = rb.calcJac(qc)
        J = R_e.dot(J)
        if args:
            if args[0].lower() in ['rpy']:
                rpy = kine.rpy
                A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
                B = block_diag(np.eye(3),np.linalg.inv(A))
                Ja = np.dot(B,J)
                J = Ja

            if args[0].lower() in ['quaternion']:
                rot = kine.R
                quat = quaternion.from_rotation_matrix(rot)
#                 quat_R = rb.mat2quat(rot)
#                 quat = quaternion.from_float_array(quat_R)
                quat = rb.rounding_quaternion(quat)
                A = rb.quat2Ja(quat)
                B = block_diag(np.eye(3), A)
                Ja = np.dot(B,J)
                J = Ja

        xd = np.dot(J, qd) - R_e.dot(xd_in)
        return xd

    def calcVdd(rb, qc, qd, qdd, xdd_in, *args):
        kine = rb.forwardKinematics(qc,'tool')
        R_e = block_diag(kine.R, kine.R)
        J = rb.calcJac(qc)
        Jd = rb.calcJacDot(qc, qd)

        J = R_e.dot(J)
        Jd = R_e.dot(Jd)
        Jdq = np.dot(Jd,qd)

        if args:
            if args[0].lower() in ['rpy']:
                rpy = kine.rpy
                A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
                B = block_diag(np.eye(3),np.linalg.inv(A))
                Ja = np.dot(B,J)
                Jadq = np.dot(B,Jdq)
                J = Ja
                Jd = Jadq

            if args[0].lower() in ['quaternion']:
                rot = kine.R
                quat = quaternion.from_rotation_matrix(rot)
                A = rb.quat2Ja(quat)
                B = block_diag(np.eye(3), A)
                Ja = rb.analyticJacobian(J, quat, 'quaternion')
                Jad = rb.analyticJacobianDot2(J, Jd, qd, quat)
                Jadq = Jad.dot(qd)
                J = Ja
                Jdq = Jadq

        xdd = np.dot(J, qdd) + Jdq + R_e.dot(xdd_in)
        return xdd


#     def calcXd3(rb, qc, qd):
#         J = rb.calcJac(qc)
# #         kine = rb.forwardKinematics(qc)
# #         rpy = kine.rpy

# #         A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
# #         B = block_diag(np.eye(3),np.linalg.inv(A))
# #         # Jadq = np.dot(B,Jdq)
# #         Ja = np.dot(B,J)

#         xd = np.dot(J,qd)
#         xd = xd[:3]
#         return xd

    def calcJac(rb, q):
        J = np.zeros((6,rb.ndof))

        kine = rb.forwardKinematics(q,'tool')
        T = kine.T
        Aout = kine.Aout

        # To simplify the readability:
        J1v = np.cross( np.array([0, 0, 1]), T[:3,3])
        J1w = np.array([0, 0, 1])
        J1 = np.concatenate((J1v,J1w))
        J[:,0] = J1

        for i in range(1,rb.ndof):
            Aframe = Aout[i-1]
            Jv = np.cross( (Aframe[:3, 2]), (T[:3, 3] - Aframe[:3, 3]), axis=0)
            Jw = Aframe[:3, 2]
            Jtemp = np.concatenate((Jv, Jw))
            J[:,i] = Jtemp
        return J

    def calcJacDot(rb, q, qd):
        J = np.zeros((6,rb.ndof))
        kine = rb.forwardKinematics(q,'tool')

        T = kine.T
        Aout = kine.Aout

        # To simplify the readability (Jacobian for the first joint):
        J1v = np.cross(np.array([0, 0, 1]), T[:3,3])
        J1w = np.array([0, 0, 1])
        J1 = np.concatenate((J1v,J1w))
        J[:,0] = J1

        # Jacobian computation
        # Declaring variables
        Jvi, Jwi = np.zeros((3,rb.ndof)), np.zeros((3,rb.ndof))
        Jvi[:,0], Jwi[:,0] = J1v, J1w
        w, z = [], []
        z.append( np.array([0, 0, 1]).reshape((3,1)) )
        w.append( np.array([0, 0, 1]).reshape((3,1)) )

        for i in range(1,rb.ndof):
            Aframe = Aout[i-1]
            z.append( np.array(Aframe[:3, 2]).reshape((3,1)) )
            Jv = np.cross( (Aframe[:3, 2]), (T[:3, 3] - Aframe[:3, 3]), axis=0)
            Jw = Aframe[:3, 2]
            Jvi[:,i] = Jv
            Jwi[:,i] = Jw
            Jtemp = np.concatenate((Jv, Jw))
            J[:,i] = Jtemp

            # w and z (used for Jacobian derivative computation)
            # Note to self, be aware of indexing.
            wtemp = w[len(w)-1] + np.dot(z[len(z) - 2], qd[i-1])
            w.append(wtemp)

        # Jacobian derivative computation
        beta = np.array(np.dot(Jvi, qd)).reshape((3,1))
        Jd = np.zeros((6,rb.ndof))

        for i in reversed(range(1, rb.ndof)):
            Aframe = Aout[i-1]
            zd = np.cross(w[i-1], z[i-1], axis = 0)
            alpha = np.array([0, 0, 0]).reshape((3,1))
            for j in range(i):
                alpha = alpha + np.dot( np.cross(z[j+1-1], np.array(T[:3, 3] - Aframe[:3, 3]).reshape((3,1)), axis=0), qd[j])

            # print "alpha", (alpha), "\n\n"
            Jvd = np.cross( zd, (T[:3, 3] - Aframe[:3, 3]), axis=0) + np.cross(z[i-1], (alpha + beta), axis=0)
            Jwd = zd
            Jtemp = np.concatenate((Jvd, Jwd))
            Jd[:,i] = np.ravel(Jtemp)
            beta = beta + np.dot(Jvi[:,i-1], qd[i-1]).reshape((3,1))

        # cross z0 x beta
        Jvd = np.cross(np.array([0, 0, 1]).reshape((3,1)), beta, axis=0)
        Jwd = np.array([0, 0, 0]).reshape((3,1))
        Jtemp = np.concatenate((Jvd, Jwd))
        Jd[:,0] = np.ravel(Jtemp)
        return Jd

    def eul2Ja(self, phi,theta,psi):
        Ja = np.array([[ 0, -np.sin(phi), np.cos(phi) * np.sin(theta)],
                        [0,  np.cos(phi), np.sin(phi) * np.sin(theta)],
                        [1,        0,           np.cos(theta) ]])
        return Ja


    def rpy2Ja(self, r,p,y):
        Ja = np.array([[ 1,          0,              np.sin(p)],
                        [0,  np.cos(r), -np.cos(p) * np.sin(r)],
                        [0,  np.sin(r),  np.cos(p) * np.cos(r)]])
        return Ja

    def quat2Ja_temp(self, q):
        # Method from Robotics Handbook.
        e0 = q.w
        e1 = q.x
        e2 = q.y
        e3 = q.z
        Es = np.array([[-e1, -e2, -e3],[e0, e3, -e2],[-e3, e0, e1], [e2, -e1, e0]])
#         Es = np.array([[e0, e3, -e2],[-e3, e0, e1], [e2, -e1, e0]])
        Eds = (1.0/2.0) * Es
        return Eds

    def quat2Ja(self, q):
        # Method from
        # Modelling and Control of Robot Manipulators
        # Authors: Sciavicco, Lorenzo, Siciliano, Bruno
        eta = q.w
        e = np.array([q.x, q.y, q.z])
        e_skew = np.array([[0, -e[2], e[1]],
                     [e[2], 0, -e[0]],
                     [-e[1], e[0], 0]])

        eta_d = -(1.0/2.0) * e
        e_d = (1.0/2.0)*(eta * np.eye(3) - e_skew)
        Eds = np.vstack((eta_d, e_d))
        return Eds

    def quatprop_E(self, q):
        eta = q.w
        e = np.array([q.x, q.y, q.z])
        e_skew = np.array([[0, -e[2], e[1]],
                     [e[2], 0, -e[0]],
                     [-e[1], e[0], 0]])
        E = (eta * np.eye(3) - e_skew)
        return E


    def analyticJacobian(rb, J, x_orient,*args):
        if args:
            if args[0].lower() in ['rpy']:
                A = rb.rpy2Ja(x_orient[0],x_orient[1],x_orient[2])
                B = block_diag(np.eye(3),np.linalg.inv(A))
                Ja = np.dot(B,J)

            if args[0].lower() in ['quaternion']:
                A = rb.quat2Ja(x_orient)
                B = block_diag(np.eye(3), A)
                Ja = np.dot(B,J)

            if args[0].lower() in ['quaternion6']:
                A = rb.quat2Ja(x_orient)
                B = block_diag(np.eye(3), A[1:,:])
                Ja = np.dot(B,J)
        return Ja

    def analyticJacobianDot(rb, J, Jd, quat, quat_d):
        # What we need:
        # qdd = (1/2)*wd*quat + (1/2)*w*quat_d

        # Optional (fix later for online computation)
        # Compute quaternion derivative
#         A0 = rb.quat2Ja(quat)
#         B0 = block_diag(np.eye(3), A0)
#         Ja = np.dot(B,J)
#         xd = np.dot(Ja, qd)
#         quat_d = xd[:3]
#         np.set_printoptions(suppress=True)

        # Compute (1/2)*w*quat_d
        A_qd = rb.quat2Ja(quat_d)
        B1 = block_diag(np.zeros((3,3)), A_qd[1:,:])
        B_qd = np.dot(B1, J)

        # Compute (1/2)*wd*quat
        A_q = rb.quat2Ja(quat)
        B_q = block_diag(np.eye(3), A_q[1:,:])

        # Computation
        Jad = np.dot(B_q, Jd) + B_qd
        return Jad


    def analyticJacobianDot2(rb, J, Jd, qd, quat):
        # What we need:
        # qdd = (1/2)*wd*quat + (1/2)*w*quat_d

        # Optional (fix later for online computation)
        # Compute quaternion derivative
        A0 = rb.quat2Ja(quat)
        B0 = block_diag(np.eye(3), A0)
        Ja = np.dot(B0,J)
        xd = np.dot(Ja, qd)
        quat_d_float = xd[3:]
        quat_d = quaternion.from_float_array(quat_d_float)
#         np.set_printoptions(suppress=True)

        # Compute (1/2)*w*quat_d
        A_qd = rb.quat2Ja(quat_d)
        B1 = block_diag(np.zeros((3,3)), A_qd[:,:])
        B_qd = np.dot(B1, J)

        # Compute (1/2)*wd*quat
        A_q = rb.quat2Ja(quat)
        B_q = block_diag(np.eye(3), A_q[:,:])

        # Computation
        Jad = np.dot(B_q, Jd) + B_qd
        return Jad



    def pinv(self, J):
        u, s, vh = np.linalg.svd(J.T, full_matrices=True)
        u.shape, s.shape, vh.shape

        rho = 0.2
        S2 = np.dot(J.T,0)
        for i in range(len(s)):
            S2[i,i] = s[i] / (s[i]**2 + rho**2)

        JpinvT = np.dot(np.dot(vh.T,S2.T),u.T)
        Jpinv = JpinvT.T
        return Jpinv

    def rounding_quaternion(self, q0):
        tol = np.finfo(np.float).eps
        q = np.array([q0.w,q0.x,q0.y,q0.z])
        for i in range(4):
            if (q[i] < tol) and (q[i] > -tol):
                q[i] = 0
        return quaternion.from_float_array(q)

    def mat2quat(self, M):
        ''' Calculate quaternion corresponding to given rotation matrix

        Parameters
        ----------
        M : array-like
          3x3 rotation matrix

        Returns
        -------
        q : (4,) array
          closest quaternion to input matrix, having positive q[0]

        Notes
        -----
        Method claimed to be robust to numerical errors in M

        Constructs quaternion by calculating maximum eigenvector for matrix
        K (constructed from input `M`).  Although this is not tested, a
        maximum eigenvalue of 1 corresponds to a valid rotation.

        A quaternion q*-1 corresponds to the same rotation as q; thus the
        sign of the reconstructed quaternion is arbitrary, and we return
        quaternions with positive w (q[0]).

        References
        ----------
        * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
          quaternion from a rotation matrix", AIAA Journal of Guidance,
          Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
          0731-5090

        Examples
        --------
        >>> import numpy as np
        >>> q = mat2quat(np.eye(3)) # Identity rotation
        >>> np.allclose(q, [1, 0, 0, 0])
        True
        >>> q = mat2quat(np.diag([1, -1, -1]))
        >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
        True

        '''
        # Qyx refers to the contribution of the y input vector component to
        # the x output vector component.  Qyx is therefore the same as
        # M[0,1].  The notation is from the Wikipedia article.
        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
        # Fill only lower half of symmetric matrix
        K = np.array([
            [Qxx - Qyy - Qzz, 0,               0,               0              ],
            [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
            [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
            [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
            ) / 3.0
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K)
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[0] < 0:
            q *= -1
        return q


    # Dynamics
    def mdh_calc_transformation(rb, From, to, qc):
        T = np.identity(4)
        From = From
        to = to

        alp = np.zeros(rb.ndof)
        a = np.zeros(rb.ndof)
        th = np.zeros(rb.ndof)
        d = np.zeros(rb.ndof)

        for i in range(rb.ndof):
            alp[i] = rb.joints[i].alpha
            a[i] = rb.joints[i].a
            th[i] = qc[i] + rb.joints[i].offset
            d[i] = rb.joints[i].d

        for i in range(From, to):
            ct = np.cos(th[i] + 0)
            st = np.sin(th[i] + 0)
            ca = np.cos(alp[i])
            sa = np.sin(alp[i])

            A = np.array([[ct, -st, 0, a[i]],
                  [(st * ca), (ct * ca), -sa, (-d[i] * sa)],
                  [(st * sa), (ct * sa), ca, (d[i] * ca)],
                  [0, 0, 0, 1]])
            T = np.dot(T, A)
            # print(A)
        return T

    def mdh_invdyn(rb, qc, qcdot, qcddot, grav):
        z0 = np.array([0, 0, 1])
        R = np.identity(3)
        Q = np.zeros((rb.ndof, 1))
        grav = grav.reshape(3)

        w = np.zeros((3))
        wdot = np.zeros((3))
        vdot = grav

        Fm = np.empty((3,0))
        Nm = np.empty((3,0))

        for k in range(1):
            q = qc[k, :].reshape((rb.ndof,1))
            qdot = qcdot[k, :].reshape((rb.ndof,1))
            qddot = qcddot[k, :].reshape((rb.ndof,1))
            N_DOFS = rb.ndof

        #   Forward recursion
            for i in range(N_DOFS):
                T = rb.mdh_calc_transformation(i, i+1, q)
                R = T[:3,:3]
                p = np.array([rb.joints[i].a,
                              -rb.joints[i].d * np.sin(rb.joints[i].alpha),
                              rb.joints[i].d * np.cos(rb.joints[i].alpha)])

                wdot_ = (np.dot(R.T, wdot) +
                         np.dot(z0,qddot[i,k]) +
                         np.cross(np.dot(R.T,w), np.dot(z0, qdot[i,k])))

                w_ = (np.dot(R.T,w) +
                      np.dot(z0, qdot[i,k]))

                vdot_ = np.dot(R.T, (vdot +
                        np.cross(wdot, p) +
                        np.cross(w, np.cross(w, p))))

                wdot = wdot_
                w = w_
                vdot = vdot_

                vcdot = (vdot + np.cross(wdot, rb.joints[i].r.reshape(3)) +
                         (np.cross(w, np.cross(w, rb.joints[i].r.reshape(3)))) )

                F = np.dot(rb.joints[i].m, vcdot)
                N = np.dot(rb.joints[i].inertia, wdot) + np.cross(w, np.dot(rb.joints[i].inertia, w))

                Fm = np.append(Fm, F.reshape((3,1)), axis=1)
                Nm = np.append(Nm, N.reshape((3,1)), axis=1)

            n = np.zeros(3)
            f = np.zeros(3)

        #   Backward recursion
            for i in reversed(range(N_DOFS)):
                if i+1 < N_DOFS:
                    p = np.array([[rb.joints[i+1].a], [-rb.joints[i+1].d * np.sin(rb.joints[i+1].alpha)],[rb.joints[i+1].d * np.cos(rb.joints[i+1].alpha)]])
                    T = rb.mdh_calc_transformation(i+1, i+2, q)
                    R = T[:3, :3]
                else:
                    R = np.eye(3)
                    p = np.zeros(3).reshape(3,1)

                n_ =(np.dot(R, n) +
                    np.cross(rb.joints[i].r.reshape(3), Fm[:,i]) +
                    np.cross(p.reshape(3), np.dot(R,f)) +
                    Nm[:,i] )

                f_ = np.dot(R, f) + Fm[:,i]

                n = n_
                f = f_
                Q[i,k] = np.dot(n.T, z0)
        return Q

    def calc_transformation(rb, From, to, qc):
        T = np.identity(4)
        From = From +1
        to = to +1

        alp = np.zeros(rb.ndof)
        a = np.zeros(rb.ndof)
        th = np.zeros(rb.ndof)
        d = np.zeros(rb.ndof)

        for i in range(rb.ndof):
            alp[i] = rb.joints[i].alpha
            a[i] = rb.joints[i].a
            # th[i] = rb.joints[i].theta

            # Since it is revolute:
            th[i] = qc[i] + rb.joints[i].offset
            d[i] = rb.joints[i].d

        for i in range(From, to):
            ct = np.cos(th[i] + 0)
            st = np.sin(th[i] + 0)
            ca = np.cos(alp[i])
            sa = np.sin(alp[i])

            A = np.array([[ct, -st * ca, st*sa, a[i]*ct],
                        [st, ct * ca, -ct * sa, a[i] * st],
                        [0, sa, ca, d[i]],
                        [0, 0, 0, 1]])
            T = np.dot(T, A)
            # print(A)
        return T

    def invdyn(rb, qc, qcdot, qcddot, grav):
        z0 = np.array([[0], [0], [1]])
        R = np.identity(3)
        Q = np.zeros((rb.ndof, 1))
        grav = grav.reshape(3,1)

        w = np.dot(np.transpose(R), np.zeros((3, 1)))
        wdot = np.dot(np.transpose(R), np.zeros((3, 1)))
        vdot = np.dot(np.transpose(R), grav)
        Fm = np.empty((3,0))
        Nm = np.empty((3,0))

        n = np.zeros((3,rb.ndof))
        f = np.zeros((3,rb.ndof))

        for k in range(1):
            q = qc[k, :].reshape((rb.ndof,1))
            qdot = qcdot[k, :].reshape((rb.ndof,1))
            qddot = qcddot[k, :].reshape((rb.ndof,1))
            N_DOFS = rb.ndof

        #   Forward recursion
            for i in range(N_DOFS):
                T = rb.calc_transformation(i-1, i, q)
                R = T[:3,:3]
                p = np.array([[rb.joints[i].a], [rb.joints[i].d* np.sin(rb.joints[i].alpha)],[rb.joints[i].d * np.cos(rb.joints[i].alpha)]])

                wdot = np.dot(R.T, (wdot + np.dot(z0,qddot[i,k])) + np.cross(w, np.dot(z0, qdot[i,k]), axis=0))
                w = np.dot(R.T,(w + np.dot(z0, qdot[i,k])))

                vdot = np.dot(R.T, vdot) + np.cross(wdot, p, axis=0) + np.cross(w, np.cross(w, p, axis=0), axis=0)
                vcdot = vdot + np.cross(wdot, rb.joints[i].r, axis=0) + (np.cross(w, np.cross(w, rb.joints[i].r, axis=0), axis=0))

                F = np.dot(rb.joints[i].m, vcdot)
                N = np.dot(rb.joints[i].inertia, wdot) + np.cross(w, np.dot(rb.joints[i].inertia, w))


                Fm = np.append(Fm, F, axis=1)
                # print "line: ",i,"\nFm: ", Fm, "\n"
                Nm = np.append(Nm, N, axis=1)

                # print "line: ",i,"\nNm: ", Nm, "\n"


        #   Backward recursion
            for i in reversed(range(N_DOFS)):
                p = np.array([[rb.joints[i].a], [rb.joints[i].d * np.sin(rb.joints[i].alpha)],
                     [rb.joints[i].d * np.cos(rb.joints[i].alpha)]])

                if i+1 < N_DOFS:
                    T = rb.calc_transformation(i, i+1, q)
                    R = T[:3, :3]

                    a = np.dot(R, (n[:, i + 1].reshape((3,1)) +  np.cross( np.dot(R.T, p), f[:,i+1].reshape((3,1)), axis=0)) )
                    n[:, i] = np.ravel(a + np.cross( (rb.joints[i].r + p), Fm[:,i].reshape((3,1)), axis=0) + Nm[:,i].reshape((3,1)))
                    f[:,i] = np.dot(R, f[:,i+1]) + Fm[:,i]
                else:
                    n[:, i] = np.ravel(np.cross(rb.joints[i].r + p, Fm[:, i].reshape((3,1)), axis=0) + Nm[:, i].reshape((3,1)))
                    f[:, i] = Fm[:, i]

                T = rb.calc_transformation(i-1, i, q)
                R = T[:3,:3]

                # print n[:,i].shape

                a = np.dot(np.transpose(n[:, i].reshape((3,1))), np.transpose(R))
                # print "line: ", i," = ", n[:,1]
                Q[i,k] = np.dot(a, z0)
        return Q


    def inertiaComp(self, qc):

        if qc.shape[0] > self.ndof:
            qc = qc[:self.ndof]

        grav = np.array([[0],[0],[0]])
        qd = np.zeros((1,self.ndof))
        qdd = np.eye(self.ndof)

        q_in = np.array([qc])
        qd_in = np.array([qd])

        M = np.zeros((self.ndof, self.ndof))
        for i in range(self.ndof):
            qdd_in = np.array([qdd[i,:]])
            Q = self.inverseDynamics(q_in, qd_in, qdd_in, grav)
            M[:,i] = Q
        return M

    def cgComp(self, qc,qcd, grav):
#         grav = np.array([[0],[0],[-9.81]])
        qdd = np.zeros((1,self.ndof))

        Q = self.inverseDynamics(qc, qcd, qdd, grav)
        return Q

    def cinertiaComp(rb, q, J):
        if q.shape[0] > rb.ndof:
            q = q[:rb.ndof]

#     #     print q.shape
#         J = rb.calcJac(q)
#     #     Jpinv = rb.pinv(J)


#         A = rb.quat2Ja(quat)
#         B = block_diag(np.eye(3), A[1:,:])
#         Ja = np.dot(B,J)

#     #     Jad = np.dot(B,Jd)
#     #     Ja = np.dot(B,J)
        Jpinv = rb.pinv(J)

        M = rb.inertiaComp(q)

    #     print M.shape
    #     print J.shape
    #     print Jpinv.shape

        Lambda = np.dot(Jpinv.T, np.dot(M, Jpinv))
        return Lambda


    def coriolisComp(rb, q, qd):
        if q.shape[0] > rb.ndof:
            q = q[:rb.ndof]
            qd = qd[:rb.ndof]

        N = rb.ndof
        C = np.zeros((N,N))
        Csq = np.zeros((N,N))
        grav = np.array([0,0,0])

        for j in range(N):
            QD = np.zeros((N))
            QD[j] = 1
            tau = rb.inverseDynamics(q, QD, np.zeros(N), grav)
            Csq[:,j] = Csq[:,j] + tau

        for j in range(N):
            for k in range(j+1,N):
                QD = np.zeros((N))
                QD[j] = 1
                QD[k] = 1
                tau = rb.inverseDynamics(q, QD, np.zeros(N), grav)
                C[:,k] = C[:,k] + np.dot((tau - Csq[:,k] - Csq[:,j]), (qd[j]/2))
                C[:,j] = C[:,j] + np.dot((tau - Csq[:,k] - Csq[:,j]), (qd[k]/2))

        C = (C + np.dot(Csq, np.diag(qd)) )

        return C


    def ccoriolisComp(rb, q, qd, Ja, Jad):
        if q.shape[0] > rb.ndof:
            q = q[:rb.ndof]
            qd = qd[:rb.ndof]

        M = rb.inertiaComp(q)
        C = rb.coriolisComp(q, qd)
        Jpinv = rb.pinv(Ja)
        mu = np.dot(Jpinv.T, np.dot((C - np.dot(M, np.dot(Jpinv, Jad))), Jpinv))
        return mu


    def gravloadComp(rb, q, grav):
        if q.shape[0] > rb.ndof:
            q = q[:rb.ndof]

        qd = np.zeros(rb.ndof)
        qdd = np.zeros(rb.ndof)
        tau_g = rb.inverseDynamics(q, qd, qdd, grav)
        return tau_g


    ############################################################

    def forwardDynamics(self, Q, qc, qcdot, grav):
        M = self.inertiaComp(qc)
        CG = self.cgComp(qc, qcdot, grav)
        qacc = np.dot(np.linalg.inv(M),(Q - CG));
        return qacc

    def plotX(self, X):
        rc('text', usetex=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(X[0, :], X[1, :])
        ax.set_title('${X}_{in}(t)$')

        # Grid lines
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        # removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legends
        ax.legend(["Trajectory"])


class traj:
    def __init__(self, q0, qf, t0, tf, hz, Robot):
        self.q0 = q0
        self.qf = qf
        self.Robot = Robot

        self.p0 = self.initial_position()
        self.pf = self.final_position()
        self.R0 = self.initial_rotation_matrix()
        self.Rf = self.final_rotation_matrix()

        self.t0 = t0
        self.tf = tf
        self.hz = hz
        self.dx = 1.0/hz
        self.samples = int((tf * hz))
        self.samples2 = self.samples*2
        self.it = np.linspace(0, 1, self.samples)

        self.q1et = quaternion.from_rotation_matrix(self.R0)
        self.q2et = quaternion.from_rotation_matrix(self.Rf)
#         self.q1et = Robot.rounding_quaternion(quaternion.from_float_array(Robot.mat2quat(self.R0)))
#         self.q2et = Robot.rounding_quaternion(quaternion.from_float_array(Robot.mat2quat(self.Rf)))

        self.quatf = self.q1et
        self.quatf_d = self.quatf * np.log(self.q2et * self.q1et.inverse())


    def initial_position(self):
#         q = np.array([0, 0, 0, 0, 0, 0, 0])
        kine = self.Robot.forwardKinematics(self.q0)
        rpy = kine.rpy
        p0 = cartesian(kine.transl[0], kine.transl[1], kine.transl[2], rpy[0], rpy[1], rpy[2])
        return p0

    def final_position(self):
        kine = self.Robot.forwardKinematics(self.qf)
        rpy = kine.rpy
        pf = cartesian(kine.transl[0], kine.transl[1], kine.transl[2], rpy[0], rpy[1], rpy[2])
        return pf

    def initial_rotation_matrix(self):
        kine = self.Robot.forwardKinematics(self.q0)
        R0 = kine.R
        return R0

    def final_rotation_matrix(self):
        kine = self.Robot.forwardKinematics(self.qf)
        Rf = kine.R
        return Rf

    # Motion planning:
    def pathplanning(self):
        t0 = self.t0
        tf = self.tf
        hz = self.hz

        samples = self.samples
        dx = self.dx

        v0 = 0 # Starting velocity
        a0 = 0 # Starting acceleration
        vf = 0 # Final velocity
        af = 0 # Final acceleration

        a_mat = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                            [0, 1, 2 ** t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                            [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
                            [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                            [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                            [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]])

        a_mat_1 = np.linalg.inv(a_mat)

        cartesianX = np.array([self.p0.x, v0, a0, self.pf.x, vf, af])
        cartesianY = np.array([self.p0.y, v0, a0, self.pf.y, vf, af])
        cartesianZ = np.array([self.p0.z, v0, a0, self.pf.z, vf, af])

        cartesianRoll = np.array([self.p0.roll, v0, a0, self.pf.roll, vf, af])
        cartesianPitch = np.array([self.p0.pitch, v0, a0, self.pf.pitch, vf, af])
        cartesianYaw = np.array([self.p0.yaw, v0, a0, self.pf.yaw, vf, af])

        px = np.transpose(np.dot(a_mat_1, cartesianX))
        py = np.transpose(np.dot(a_mat_1, cartesianY))
        pz = np.transpose(np.dot(a_mat_1, cartesianZ))

        proll = np.transpose(np.dot(a_mat_1, cartesianRoll))
        ppitch = np.transpose(np.dot(a_mat_1, cartesianPitch))
        pyaw = np.transpose(np.dot(a_mat_1, cartesianYaw))

        pseg = np.fliplr(np.array([px,py,pz, proll, ppitch, pyaw]))

        plen = np.linspace(0,tf,samples)
        time_span = plen

        X1 = np.zeros((samples,6))
        for j in range(len(plen)):
            t = plen[j]
            for i in range(6):
                X1[j,i] = pseg[i,0]*t**5 + pseg[i,1]*t**4 + pseg[i,2]*t**3 + pseg[i,3]*t**2+pseg[i,4]*t + pseg[i,5]

        X1 = np.transpose(X1)

        Xd1 = np.gradient(X1[0, :], dx)
        Xd2 = np.gradient(X1[1, :], dx)
        Xd3 = np.gradient(X1[2, :], dx)
        Xd4 = np.gradient(X1[3, :], dx)
        Xd5 = np.gradient(X1[4, :], dx)
        Xd6 = np.gradient(X1[5, :], dx)
        Xd = np.array([Xd1, Xd2, Xd3, Xd4, Xd5, Xd6])

        Xdd1 = np.gradient(Xd[0, :], dx)
        Xdd2 = np.gradient(Xd[1, :], dx)
        Xdd3 = np.gradient(Xd[2, :], dx)
        Xdd4 = np.gradient(Xd[3, :], dx)
        Xdd5 = np.gradient(Xd[4, :], dx)
        Xdd6 = np.gradient(Xd[5, :], dx)
        Xdd = np.array([Xdd1, Xdd2, Xdd3, Xdd4, Xdd5, Xdd6])

        return X1, Xd, Xdd


    def pathplanning3(self):
        t0 = self.t0
        tf = self.tf
        hz = self.hz

        samples = self.samples
        dx = self.dx

        v0 = 0 # Starting velocity
        a0 = 0 # Starting acceleration
        vf = 0 # Final velocity
        af = 0 # Final acceleration

        a_mat = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                            [0, 1, 2 ** t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                            [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
                            [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                            [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                            [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]])

        a_mat_1 = np.linalg.inv(a_mat)

        cartesianX = np.array([self.p0.x, v0, a0, self.pf.x, vf, af])
        cartesianY = np.array([self.p0.y, v0, a0, self.pf.y, vf, af])
        cartesianZ = np.array([self.p0.z, v0, a0, self.pf.z, vf, af])

#         cartesianRoll = np.array([self.p0.roll, v0, a0, self.pf.roll, vf, af])
#         cartesianPitch = np.array([self.p0.pitch, v0, a0, self.pf.pitch, vf, af])
#         cartesianYaw = np.array([self.p0.yaw, v0, a0, self.pf.yaw, vf, af])

        px = np.transpose(np.dot(a_mat_1, cartesianX))
        py = np.transpose(np.dot(a_mat_1, cartesianY))
        pz = np.transpose(np.dot(a_mat_1, cartesianZ))

#         proll = np.transpose(np.dot(a_mat_1, cartesianRoll))
#         ppitch = np.transpose(np.dot(a_mat_1, cartesianPitch))
#         pyaw = np.transpose(np.dot(a_mat_1, cartesianYaw))

        pseg = np.fliplr(np.array([px,py,pz]))

        plen = np.linspace(0,tf,samples)
        time_span = plen

        X1 = np.zeros((samples,3))
        for j in range(len(plen)):
            t = plen[j]
            for i in range(3):
                X1[j,i] = pseg[i,0]*t**5 + pseg[i,1]*t**4 + pseg[i,2]*t**3 + pseg[i,3]*t**2+pseg[i,4]*t + pseg[i,5]

        X1 = np.transpose(X1)

        cartesianRoll = np.linspace(self.p0.roll, self.pf.roll, samples)
        cartesianPitch = np.linspace(self.p0.pitch, self.pf.pitch, samples)
        cartesianYaw = np.linspace(self.p0.yaw, self.pf.yaw, samples)

        X1 = np.concatenate((X1, cartesianRoll.reshape(1,samples)), axis=0)
        X1 = np.concatenate((X1, cartesianPitch.reshape(1,samples)), axis=0)
        X1 = np.concatenate((X1, cartesianYaw.reshape(1,samples)), axis=0)

        Xd1 = np.gradient(X1[0, :], dx)
        Xd2 = np.gradient(X1[1, :], dx)
        Xd3 = np.gradient(X1[2, :], dx)
        Xd4 = np.gradient(X1[3, :], dx)
        Xd5 = np.gradient(X1[4, :], dx)
        Xd6 = np.gradient(X1[5, :], dx)
        Xd = np.array([Xd1, Xd2, Xd3, Xd4, Xd5, Xd6])

        Xdd1 = np.gradient(Xd[0, :], dx)
        Xdd2 = np.gradient(Xd[1, :], dx)
        Xdd3 = np.gradient(Xd[2, :], dx)
        Xdd4 = np.gradient(Xd[3, :], dx)
        Xdd5 = np.gradient(Xd[4, :], dx)
        Xdd6 = np.gradient(Xd[5, :], dx)
        Xdd = np.array([Xdd1, Xdd2, Xdd3, Xdd4, Xdd5, Xdd6])

        return X1, Xd, Xdd


    def jointpathplanning(self):
        t0 = self.t0
        tf = self.tf
        hz = self.hz

        samples = self.samples
        dx = self.dx

        v0 = 0 # Starting velocity
        a0 = 0 # Starting acceleration
        vf = 0 # Final velocity
        af = 0 # Final acceleration

        a_mat = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                            [0, 1, 2 ** t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                            [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
                            [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                            [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                            [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]])

        a_mat_1 = np.linalg.inv(a_mat)

        jq1 = np.array([self.q0[0], v0, a0, self.qf[0], vf, af])
        jq2 = np.array([self.q0[1], v0, a0, self.qf[1], vf, af])
        jq3 = np.array([self.q0[2], v0, a0, self.qf[2], vf, af])
        jq4 = np.array([self.q0[3], v0, a0, self.qf[3], vf, af])
        jq5 = np.array([self.q0[4], v0, a0, self.qf[4], vf, af])
        jq6 = np.array([self.q0[5], v0, a0, self.qf[5], vf, af])
        jq7 = np.array([self.q0[6], v0, a0, self.qf[6], vf, af])

        q1 = np.transpose(np.dot(a_mat_1, jq1))
        q2 = np.transpose(np.dot(a_mat_1, jq2))
        q3 = np.transpose(np.dot(a_mat_1, jq3))
        q4 = np.transpose(np.dot(a_mat_1, jq4))
        q5 = np.transpose(np.dot(a_mat_1, jq5))
        q6 = np.transpose(np.dot(a_mat_1, jq6))
        q7 = np.transpose(np.dot(a_mat_1, jq7))

        pseg = np.fliplr(np.array([q1, q2, q3, q4, q5, q6, q7]))

        plen = np.linspace(0,tf,samples)
        time_span = plen

        X1 = np.zeros((samples,7))
        for j in range(len(plen)):
            t = plen[j]
            for i in range(7):
                X1[j,i] = pseg[i,0]*t**5 + pseg[i,1]*t**4 + pseg[i,2]*t**3 + pseg[i,3]*t**2+pseg[i,4]*t + pseg[i,5]

        X1 = np.transpose(X1)

        Xd1 = np.gradient(X1[0, :], dx)
        Xd2 = np.gradient(X1[1, :], dx)
        Xd3 = np.gradient(X1[2, :], dx)
        Xd4 = np.gradient(X1[3, :], dx)
        Xd5 = np.gradient(X1[4, :], dx)
        Xd6 = np.gradient(X1[5, :], dx)
        Xd7 = np.gradient(X1[6, :], dx)
        Xd = np.array([Xd1, Xd2, Xd3, Xd4, Xd5, Xd6, Xd7])

        Xdd1 = np.gradient(Xd[0, :], dx)
        Xdd2 = np.gradient(Xd[1, :], dx)
        Xdd3 = np.gradient(Xd[2, :], dx)
        Xdd4 = np.gradient(Xd[3, :], dx)
        Xdd5 = np.gradient(Xd[4, :], dx)
        Xdd6 = np.gradient(Xd[5, :], dx)
        Xdd7 = np.gradient(Xd[5, :], dx)
        Xdd = np.array([Xdd1, Xdd2, Xdd3, Xdd4, Xdd5, Xdd6, Xdd7])

        return X1, Xd, Xdd

    def rotationalInterpolation(self, Xp, Xdp, Xddp, *args):
        X1 = np.zeros((6,self.samples))
        Xd1 = np.zeros((6,self.samples))
        Xdd1 = np.zeros((6,self.samples))

        it = self.it
        q1et = self.q1et
        q2et = self.q2et

        if args:
            if args[0].lower() in ['rpy']:
                for i in range(self.samples):
                    quat = quaternion.slerp_evaluate(q1et,q2et,it[i])
                    quat_d = quat * np.log(q2et*q1et.inverse())
                    quat_dd = quat * np.log(q2et*q1et.inverse())**2

                    R = quat.as_rotation_matrix(quat)
                    Rd = quat.as_rotation_matrix(quat_d)
                    Rdd = quat.as_rotation_matrix(quat_dd)

                    rpy = self.Robot.r2rpy(R)
                    rpy_d = self.Robot.r2rpy(R_d)
                    rpy_dd = self.Robot.r2rpy(R_dd)

                    X1[:,i] = np.array([Xp[0,i],Xp[1,i],Xp[2,i], rpy[0],rpy[1],rpy[2]])
                    Xd1[:,i] = np.array([Xdp[0,i],Xdp[1,i],Xdp[2,i], rpy_d[0],rpy_d[1],rpy_d[2]])
                    Xdd1[:,i] = np.array([Xddp[0,i],Xddp[1,i],Xddp[2,i], rpy_dd[0],rpy_dd[1],rpy_dd[2]])

            if args[0].lower() in ['quaternion']:
                for i in range(self.samples):
#                     quat = quaternion.slerp_evaluate(q1et,q2et,it[i])
#                     quat_d = quat * np.log(q2et*q1et.inverse())
#                     quat_dd = quat * np.log(q2et*q1et.inverse())**2
                    slerp = q1et * (q1et.inverse() * q2et)**it[i]
                    slerp_d = slerp * np.log(q2et * q1et.inverse())
                    slerp_dd = slerp * np.log(q2et * q1et.inverse())**2

                    omega = quaternion.as_float_array(2 * slerp_d * slerp.inverse())
                    omega_d = quaternion.as_float_array(2 * slerp_dd * slerp.inverse())

                    X1[:,i] = np.array([Xp[0,i],Xp[1,i],Xp[2,i], slerp.y, slerp.x, slerp.z])
                    Xd1[:,i] = np.array([Xdp[0,i],Xdp[1,i],Xdp[2,i], omega[1], omega[2], omega[3]])
                    Xdd1[:,i] = np.array([Xddp[0,i],Xddp[1,i],Xddp[2,i],  omega_d[1], omega_d[2], omega_d[3]])

#                     X1[:,i] = np.array([Xp[0,i],Xp[1,i],Xp[2,i], slerp.x, slerp.y, slerp.z])
#                     Xd1[:,i] = np.array([Xdp[0,i],Xdp[1,i],Xdp[2,i], slerp_d.x, slerp_d.y, slerp_d.z])
#                     Xdd1[:,i] = np.array([Xddp[0,i],Xddp[1,i],Xddp[2,i], slerp_dd.x, slerp_dd.y, slerp_dd.z])
        return X1, Xd1, Xdd1

    def rot_then_pos_interpolation(self):
        # Set point trajectory for position, velocity and acceleration:
        Xp, Xdp, Xddp = self.pathplanning3()

        # Empty array for concatenating:
        vec = np.zeros((3,self.samples))

        # Arrays containing initial position, velocity and acceleration:
        X0p = np.full_like(vec,Xp[:3,0].reshape(3,1))
        Xd0p = np.full_like(vec,Xdp[:3,0].reshape(3,1))
        Xdd0p = np.full_like(vec,Xp[:3,0].reshape(3,1))

        # 3D orientation interpolation with SLERP, using "initial condition" arrays
        Xp0o, Xdp0o, Xddp0o = self.rotationalInterpolation(X0p, Xd0p, Xdd0p, 'quaternion')

        # Arrays containing the final angular position, velocity and orientation
        Xfo = np.full_like(vec, Xp0o[3:,-1].reshape(3,1))
        Xdfo = np.full_like(vec, Xdp0o[3:,-1].reshape(3,1))
        Xddfo = np.full_like(vec, Xddp0o[3:,-1].reshape(3,1))

        # Stacking final orientations with poses
        Xpfo = np.vstack((Xp[:3,:],Xfo))
        Xdpfo =  np.vstack((Xdp[:3,:],Xdfo))
        Xddpfo =  np.vstack((Xddp[:3,:],Xddfo))

        # Stacking both orientations and positions together:
        Xpo = np.hstack((Xp0o, Xpfo))
        Xdpo = np.hstack((Xdp0o, Xdpfo))
        Xddpo = np.hstack((Xddp0o, Xddpfo))

        return Xpo, Xdpo, Xddpo


    def rot_trajectory_only(self):
        # Set point trajectory for position, velocity and acceleration:
        Xp, Xdp, Xddp = self.pathplanning3()

        # Empty array for concatenating:
        vec = np.zeros((3,self.samples))

        # Arrays containing initial position, velocity and acceleration:
        X0p = np.full_like(vec,Xp[:3,0].reshape(3,1))
        Xd0p = np.full_like(vec,Xdp[:3,0].reshape(3,1))
        Xdd0p = np.full_like(vec,Xp[:3,0].reshape(3,1))

        # 3D orientation interpolation with SLERP, using "initial condition" arrays
        Xp0o, Xdp0o, Xddp0o = self.rotationalInterpolation(X0p, Xd0p, Xdd0p, 'quaternion')

        return Xp0o, Xdp0o, Xddp0o


    def from_jointspace(self):
        X1 = np.zeros((6,self.samples))
        Xd1 = np.zeros((6,self.samples))
        Xdd1 = np.zeros((6,self.samples))

        Q1 = np.zeros((4,self.samples))
        Qd1 = np.zeros((4,self.samples))
        Qdd1 = np.zeros((4,self.samples))

        J1, Jd1, Jdd1 =self.jointpathplanning()

        for i in range(self.samples):
            kine = self.Robot.forwardKinematics(J1[:,i],'tool')
            p_p = kine.transl
            p_o = quaternion.from_rotation_matrix(kine.R)

            Q1[:,i] = np.array([p_o.w, p_o.x, p_o.y, p_o.z])
            X1[:,i] = np.array([p_p[0], p_p[1], p_p[2], p_o.x, p_o.y, p_o.z])

            Xd_calc = self.Robot.calcXd(J1[:,i], Jd1[:,i])
#             Qd1[:,i] = np.array([ Xd_calc[3], Xd_calc[4], Xd_calc[5], Xd_calc[6]])

            Xd_calc2 = self.Robot.calcXd(J1[:,i], Jd1[:,i])
            Xd1[:,i] = np.array([Xd_calc2[0], Xd_calc2[1], Xd_calc2[2], Xd_calc2[3], Xd_calc2[4], Xd_calc2[5]])

            Xdd_calc = self.Robot.calcXdd(J1[:,i], Jd1[:,i], Jdd1[:,i])
#             Qdd1[:,i] = np.array([ Xdd_calc[3], Xdd_calc[4], Xdd_calc[5], Xdd_calc[6]])

            Xdd_calc2 = self.Robot.calcXdd(J1[:,i], Jd1[:,i], Jdd1[:,i])
            Xdd1[:,i] = np.array([Xdd_calc2[0], Xdd_calc2[1], Xdd_calc2[2], Xdd_calc2[3], Xdd_calc2[4], Xdd_calc2[5]])

        return X1, Xd1, Xdd1, Q1, J1, Jd1, Jdd1


    def quaternion_trajectory(self, X, Xd, Xdd):
        it = self.it
        q1et = self.q1et
        q2et = self.q2et

        Q1 = np.zeros((4,self.samples))
        omega = np.zeros((4, self.samples))
        omega_d = np.zeros((4, self.samples))

        P1 = np.zeros((3,self.samples))
        Pd1 = np.zeros((3,self.samples))
        Pdd1 = np.zeros((3,self.samples))


        for i in range(self.samples):
            quat = quaternion.slerp_evaluate(q1et,q2et,it[i])
            quat_float = quaternion.as_float_array(quat)
            Q1[:,i] = quat_float

        Qd1 = np.gradient(Q1[0, :], self.dx)
        Qd2 = np.gradient(Q1[1, :], self.dx)
        Qd3 = np.gradient(Q1[2, :], self.dx)
        Qd4 = np.gradient(Q1[3, :], self.dx)
        Qd = np.array([Qd1, Qd2, Qd3, Qd4])

        Qdd1 = np.gradient(Qd[0, :], self.dx)
        Qdd2 = np.gradient(Qd[1, :], self.dx)
        Qdd3 = np.gradient(Qd[2, :], self.dx)
        Qdd4 = np.gradient(Qd[3, :], self.dx)
        Qdd = np.array([Qdd1, Qdd2, Qdd3, Qdd4])

        for i in range(self.samples):
            q1 = quaternion.from_float_array(Q1[:,i])
            qd1 = quaternion.from_float_array(Qd[:,i])
            qdd1 = quaternion.from_float_array(Qdd[:,i])

            omega[:,i] = quaternion.as_float_array(2 * qd1 * q1.inverse())
            omega_d[:,i] = quaternion.as_float_array(2 * qdd1 * q1.inverse())

            P1[:,i] = np.dot( np.transpose(quaternion.as_rotation_matrix(q1)),X[:3,i]).reshape(3)
            Pd1[:,i] = np.dot( np.transpose(quaternion.as_rotation_matrix(q1)),Xd[:3,i]).reshape(3)
            Pdd1[:,i] = np.dot( np.transpose(quaternion.as_rotation_matrix(q1)),Xdd[:3,i]).reshape(3)


#             omega[1:,i] = np.dot( np.transpose(quaternion.as_rotation_matrix(q1)),omega[1:,i]).reshape(3)
#             omega_d[1:,i] = np.dot(np.transpose(quaternion.as_rotation_matrix(q1)),omega_d[1:,i]).reshape(3)

        X1 = np.vstack((X[:3,:],Q1[1:,:]))
        Xd1 = np.vstack((Xd[:3,:],omega[1:,:]))
        Xdd1 = np.vstack((Xdd[:3,:],omega_d[1:,:]))

        return X1, Xd1, Xdd1, Q1, Qd, Qdd
