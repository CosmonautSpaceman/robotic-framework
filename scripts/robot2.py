import numpy as np
from scipy.linalg import block_diag
import math
import quaternion


class rJoint:
    def __init__(self, alpha, a, theta, d, type, inertia, m, r):
        self.alpha = alpha
        self.a = a
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

    # Controllers
    def output(self, x, xd, xdd, xc, xcd, F):

        Mdinv = np.linalg.inv(self.Md)
        damper = np.dot(self.Bd,(xcd - xd))
        spring = np.dot(self.Kd,(xc - x))

        ax = xdd - np.dot(Mdinv,(damper + spring + F))
        return ax

    def outputquat(self, e, ed, xdd, F):

        Mdinv = np.linalg.inv(self.Md)
        damper = np.dot(self.Bd,ed)
        spring = np.dot(self.Kd,e)

        ax = xdd - np.dot(Mdinv,(damper + spring + F))
        return ax


class Robot:
    def __init__(self, joints, ndof, dh):
        self.joints = joints
        self.ndof = ndof
        self.dh = dh

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

    def forwardKinematics(self, q):
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
            th[i] = rb.joints[i].theta
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
            th[i] = rb.joints[i].theta
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

    def calcQd(rb, Xd, qc):
        J = rb.calcJac(qc)
        Jt = np.transpose(J)

        kine = rb.forwardKinematics(qc)
        rpy = kine.rpy

        A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
        B = block_diag(np.eye(3),np.linalg.inv(A))
#         Jadq = np.dot(B,Jdq)
        Ja = np.dot(B,J)
        Jpinv = rb.pinv(Ja)


        qd = np.dot(Jpinv, Xd)

#         print qd

        return qd

    def calcQdd(rb, Xdd, qc, qd):
        J = rb.calcJac(qc)
        Jd = rb.calcJacDot(qc, qd)


        kine = rb.forwardKinematics(qc)
#         rpy = kine.rpy
        rot = kine.R

        quat = quaternion.from_rotation_matrix(rot)
        A = rb.quat2Ja(quat)
        B = block_diag(np.eye(3), A[1:,:])
        Ja = np.dot(B,J)
        Jad = np.dot(B,Jd)
#         Ja = np.dot(B,J)
        Jadq = np.dot(Jd,qd)

        Jpinv = rb.pinv(Ja)

        qdd = np.dot(Jpinv, (Xdd - Jadq))
        return qdd

    def calcQdd3(rb, Xdd, qc, qd):
        J = rb.calcJac(qc)
        Jd = rb.calcJacDot(qc, qd)
        Jdq = np.dot(Jd,qd)

        kine = rb.forwardKinematics(qc)
        rpy = kine.rpy

        A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
        B = block_diag(np.eye(3),np.linalg.inv(A))
        # Jadq = np.dot(B,Jdq)
        Ja = np.dot(B,J)
        Jpinv = rb.pinv(Ja)

        qdd = np.dot(Jpinv[:,:3], (Xdd - Jdq[:3]))
        return qdd

    def calcXd(rb, qc, qd):
        J = rb.calcJac(qc)
        kine = rb.forwardKinematics(qc)
        rot = kine.R
#         rpy = kine.rpy

        quat = quaternion.from_rotation_matrix(rot)
        A = rb.quat2Ja(quat)
        B = block_diag(np.eye(3), A)
        Ja = np.dot(B,J)
#         A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
#         B = block_diag(np.eye(3),np.linalg.inv(A))

#         Jadq = np.dot(B,Jdq)
#         Ja = np.dot(B,J)

        xd = np.dot(Ja, qd)
#         w_hat = xd[3:]
#         xd = np.dot(Ja,qd)

#         e0 = quat.w
#         e1 = quat.x
#         e2 = quat.y
#         e3 = quat.z
#         Es = np.array([[-e1, -e2, -e3],[e0, e3, -e2],[-e3, e0, e1], [e2, -e1, e0]])
#         xdquat = (1.0/2.0) * np.dot(Es, w_hat)
        return xd


    def calcXd3(rb, qc, qd):
        J = rb.calcJac(qc)
        xd = np.dot(J,qd)
        return xd


    def calcXdd(rb, qc, qd, qdd):
        J = rb.calcJac(qc)
        kine = rb.forwardKinematics(qc)
        rot = kine.R
        rpy = kine.rpy
        quat = quaternion.from_rotation_matrix(rot)

        A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
        B = block_diag(np.eye(3),np.linalg.inv(A))

#         Jadq = np.dot(B,Jdq)
        Ja = np.dot(B,J)
        Jd = rb.calcJacDot(qc, qd)
        Jdq = np.dot(Jd,qd)
        Jadq = np.dot(B,Jdq)

        xdd = np.dot(J, qdd) + Jadq
#         xd = np.dot(J,qd)

#         w_hat = xd[3:]
#         wd_hat = xdd[3:]

#         e0 = quat.w
#         e1 = quat.x
#         e2 = quat.y
#         e3 = quat.z
#         Es = np.array([[-e1, -e2, -e3],[e0, e3, -e2],[-e3, e0, e1], [e2, -e1, e0]])
#         xddquat = (1.0/2.0) * np.dot(Es, w_hat**2) + (1.0/4.0) * np.dot(Es, wd_hat)

        return xdd


    def calcXd3(rb, qc, qd):
        J = rb.calcJac(qc)
#         kine = rb.forwardKinematics(qc)
#         rpy = kine.rpy

#         A = rb.rpy2Ja(rpy[0],rpy[1],rpy[2])
#         B = block_diag(np.eye(3),np.linalg.inv(A))
#         # Jadq = np.dot(B,Jdq)
#         Ja = np.dot(B,J)

        xd = np.dot(J,qd)
        xd = xd[:3]
        return xd

    def calcJac(rb, q):
        J = np.zeros((6,rb.ndof))

        kine = rb.forwardKinematics(q)
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
        kine = rb.forwardKinematics(q)

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

    def quat2Ja(self, q):
        e0 = q.w
        e1 = q.x
        e2 = q.y
        e3 = q.z
        Es = np.array([[-e1, -e2, -e3],[e0, e3, -e2],[-e3, e0, e1], [e2, -e1, e0]])
#         Es = np.array([[e0, e3, -e2],[-e3, e0, e1], [e2, -e1, e0]])
        Eds = (1.0/2.0) * Es
        return Eds


    def pinv(self, J):
        u, s, vh = np.linalg.svd(J.T, full_matrices=True)
        u.shape, s.shape, vh.shape

        rho = 4
        S2 = np.dot(J.T,0)
        for i in range(len(s)):
            S2[i,i] = s[i] / (s[i]**2 + rho**2)

        JpinvT = np.dot(np.dot(vh.T,S2.T),u.T)
        Jpinv = JpinvT.T
        return Jpinv

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
            th[i] = qc[i]
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
            th[i] = qc[i]
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


    def cinertiaComp(self, q, quat):
        if q.shape[0] > self.ndof:
            q = q[:self.ndof]

    #     print q.shape
        J = self.calcJac(q)
    #     Jpinv = rb.pinv(J)


        A = self.quat2Ja(quat)
        B = block_diag(np.eye(3), A[1:,:])
        Ja = np.dot(B,J)

    #     Jad = np.dot(B,Jd)
    #     Ja = np.dot(B,J)
        Jpinv = self.pinv(Ja)

        M = self.inertiaComp(q)

    #     print M.shape
    #     print J.shape
    #     print Jpinv.shape

        Lambda = np.dot(Jpinv.T, np.dot(M, Jpinv))
        return Lambda


    def coriolisComp(self, q, qd):
        if q.shape[0] > self.ndof:
            q = q[:self.ndof]
            qd = qd[:self.ndof]

        N = self.ndof
        C = np.zeros((N,N))
        Csq = np.zeros((N,N))
        grav = np.array([0,0,0])

        for j in range(N):
            QD = np.zeros((N))
            QD[j] = 1
            tau = self.inverseDynamics(q, QD, np.zeros(N), grav)
            Csq[:,j] = Csq[:,j] + tau

        for j in range(N):
            for k in range(j+1,N):
                QD = np.zeros((N))
                QD[j] = 1
                QD[k] = 1
                tau = self.inverseDynamics(q, QD, np.zeros(N), grav)
                C[:,k] = C[:,k] + np.dot((tau - Csq[:,k] - Csq[:,j]), (qd[j]/2))
                C[:,j] = C[:,j] + np.dot((tau - Csq[:,k] - Csq[:,j]), (qd[k]/2))

        C = (C + np.dot(Csq, np.diag(qd)) )

        return C


    def ccoriolisComp(self, q, qd, rpy):
        if q.shape[0] > self.ndof:
            q = q[:self.ndof]
            qd = qd[:self.ndof]

        J = self.calcJac(q)
        Jd = self.calcJacDot(q, qd)
        M = self.inertiaComp(q)
        C = self.coriolisComp(q, qd)

        A = self.rpy2Ja(rpy[0],rpy[1],rpy[2])
        B = block_diag(np.eye(3),np.linalg.inv(A))
        Jad = np.dot(B,Jd)
        Ja = np.dot(B,J)
        Jpinv = self.pinv(Ja)

        mu = np.dot(Jpinv.T, np.dot((C - np.dot(M, np.dot(Jpinv, Jd))), Jpinv))

        return mu


    def gravloadComp(self, q, grav):
        if q.shape[0] > self.ndof:
            q = q[:self.ndof]

        qd = np.zeros(self.ndof)
        qdd = np.zeros(self.ndof)
        tau_g = self.inverseDynamics(q, qd, qdd, grav)
        return tau_g


    def quatVelocity(q,w_hat):
        e0 = q.w
        e1 = q.x
        e2 = q.y
        e3 = q.z
        Es = np.array([[-e1, -e2, -e3],[e0, e3, -e2],[-e3, e0, e1], [e2, -e1, e0]])
        Eds = (1.0/2.0) * np.dot(Es, w_hat)
        qd = quaternion.from_float_array(Eds)
        return qd


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
    def __init__(self, p0, pf, t0, tf, hz):
        self.p0 = p0
        self.pf = pf
        self.t0 = t0
        self.tf = tf
        self.hz = hz
        self.samples = int((tf*hz))
        self.dx = 1.0/hz

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


    def jointpathplanning(self, joints):
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

        jq1 = np.array([joints[0].q0, v0, a0, joints[0].qf, vf, af])
        jq2 = np.array([joints[1].q0, v0, a0, joints[1].qf, vf, af])
        jq3 = np.array([joints[2].q0, v0, a0, joints[2].qf, vf, af])
        jq4 = np.array([joints[3].q0, v0, a0, joints[3].qf, vf, af])
        jq5 = np.array([joints[4].q0, v0, a0, joints[4].qf, vf, af])
        jq6 = np.array([joints[5].q0, v0, a0, joints[5].qf, vf, af])
        jq7 = np.array([joints[6].q0, v0, a0, joints[6].qf, vf, af])

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
