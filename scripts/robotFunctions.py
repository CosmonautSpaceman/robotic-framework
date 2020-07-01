import numpy as np
import math
from scipy.linalg import block_diag


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


class Robot:
    def __init__(self, joints, ndof):
        self.joints = joints
        self.ndof = ndof


def calc_transformation(From, to, rb, qc):
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


# def inertiaMatrix(rb, qc, grav):
#     mqcd = np.array([[0, 0, 0]])
#     mqcdd1 = np.array([[1, 0, 0]])
#     mqcdd2 = np.array([[0, 1, 0]])
#
#     M1 = invdyn(rb, qc, mqcd, mqcdd1, grav)
#     M2 = invdyn(rb, qc, mqcd, mqcdd2, grav)
#     M = np.concatenate((M1[:2, :], M2[:2, :]), axis=1)
#     return M
#
#
# def coriolisGravVector(rb, qc, qcdot, grav):
#     CGqcdd = np.array([[0, 0, 0]])
#
#     CG = invdyn(rb, qc, qcdot, CGqcdd, grav)
#     CG = np.delete(CG, 2, 0)
#     return CG
#
#
# def forwardDyn(M, Q, CG):
#     rotForces = (Q - CG)
#     Minv = np.linalg.inv(M)
#     qddr = np.dot(Minv,rotForces)
#     return qddr


def invdyn(rb, qc, qcdot, qcddot, grav):
    z0 = np.array([[0], [0], [1]])
    R = np.identity(3)
    Q = np.zeros((rb.ndof, 1))

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
            T = calc_transformation(i-1, i, rb, q)
            R = T[:3,:3]
            p = np.array([[rb.joints[i].a], [rb.joints[i].d* np.sin(rb.joints[i].alpha)],[rb.joints[i].d * np.cos(rb.joints[i].alpha)]])

            wdot = np.dot(R.T, (wdot + np.dot(z0,qddot[i,k]))  + np.cross(w, np.dot(z0, qdot[i,k]), axis=0))
            w = np.dot(R.T,(w + np.dot(z0, qdot[i,k])))

            vdot = np.dot(R.T, vdot) + np.cross(wdot, p, axis=0) + np.cross(w, np.cross(w, p, axis=0), axis=0)
            vcdot = vdot + np.cross(wdot, rb.joints[i].r, axis=0) + (np.cross(w, np.cross(w, rb.joints[i].r, axis=0), axis=0))

            F = np.dot(rb.joints[i].m, vcdot)
            N = np.dot(rb.joints[i].inertia, wdot) + np.cross(w, np.dot(rb.joints[i].inertia, w),axis=0)

            Fm = np.append(Fm, F, axis=1)
            # print "line: ",i,"\nFm: ", Fm, "\n"
            Nm = np.append(Nm, N, axis=1)
            # print "line: ",i,"\nNm: ", Nm, "\n"


    #   Backward recursion
        for i in reversed(range(N_DOFS)):
            p = np.array([[rb.joints[i].a], [rb.joints[i].d * np.sin(rb.joints[i].alpha)],
                 [rb.joints[i].d * np.cos(rb.joints[i].alpha)]])

            if i+1 < N_DOFS:
                T = calc_transformation(i, i+1, rb, q)
                R = T[:3, :3]

                a = np.dot(R, (n[:, i + 1].reshape((3,1)) +  np.cross( np.dot(R.T, p), f[:,i+1].reshape((3,1)), axis=0)) )
                n[:, i] = np.ravel(a + np.cross( (rb.joints[i].r + p), Fm[:,i].reshape((3,1)), axis=0) + Nm[:,i].reshape((3,1)))
                f[:,i] = np.dot(R, f[:,i+1]) + Fm[:,i]
            else:
                n[:, i] = np.ravel(np.cross(rb.joints[i].r + p, Fm[:, i].reshape((3,1)), axis=0) + Nm[:, i].reshape((3,1)))
                f[:, i] = Fm[:, i]

            T = calc_transformation(i-1, i, rb, q)
            R = T[:3,:3]

            # print n[:,i].shape
            a = np.dot(np.transpose(n[:, i].reshape((3,1))), np.transpose(R))
            # print "line: ", i," = ", n[:,1]
            Q[i,k] = np.dot(a, z0)
    return Q


def jointpathplanning(q0, qf, tf, hz):
    t0 = 0
    # tf = 30
    v0 = 0
    a0 = 0
    vf = 0
    af = 0
    vec = []
    qvec = []

    a_mat = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                        [0, 1, 2 ** t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                        [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
                        [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                        [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                        [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]])

    a_mat_1 = np.linalg.inv(a_mat)

    for i in range(7):
        vec.append(np.array([q0[i], v0, a0, qf[i], vf, af]))

    for i in range(7):
        qvec.append(np.transpose(np.dot(a_mat_1, vec[i])))

    qseg = np.fliplr(np.array([qvec[0], qvec[1], qvec[2], qvec[3], qvec[4], qvec[5], qvec[6]]))
    samples = int((tf*hz))
    plen = np.linspace(0,tf,(tf*hz))
    time_span = plen
    dx = (1.0/hz)

    Q1 = np.zeros((samples,7))
    for j in range(len(plen)):
        t = plen[j]
        for i in range(7):
            Q1[j,i] = qseg[i,0]*t**5 + qseg[i,1]*t**4 + qseg[i,2]*t**3 + qseg[i,3]*t**2+qseg[i,4]*t + qseg[i,5]

    # X1 =np.concatenate((X1,pya), axis=1)
    Q1 = np.transpose(Q1)

    Qd1 = np.gradient(Q1[0, :], dx)
    Qd2 = np.gradient(Q1[1, :], dx)
    Qd3 = np.gradient(Q1[2, :], dx)
    Qd4 = np.gradient(Q1[3, :], dx)
    Qd5 = np.gradient(Q1[4, :], dx)
    Qd6 = np.gradient(Q1[5, :], dx)
    Qd7 = np.gradient(Q1[5, :], dx)

    Qd = np.array([Qd1, Qd2, Qd3, Qd4, Qd5, Qd6, Qd7])

    Qdd1 = np.gradient(Qd[0, :], dx)
    Qdd2 = np.gradient(Qd[1, :], dx)
    Qdd3 = np.gradient(Qd[2, :], dx)
    Qdd4 = np.gradient(Qd[3, :], dx)
    Qdd5 = np.gradient(Qd[4, :], dx)
    Qdd6 = np.gradient(Qd[5, :], dx)
    Qdd7 = np.gradient(Qd[5, :], dx)
    Qdd = np.array([Qdd1, Qdd2, Qdd3, Qdd4, Qdd5, Qdd6, Qdd7])

    return Q1, Qd, Qdd


def pathplanning(p0x, p0y, p0z, pfx, pfy, pfz, theta0, phi0, psi0, theta, phi, psi, tf, hz):
    t0 = 0
    # tf = 30
    v0 = 0
    a0 = 0
    vf = 0
    af = 0

    a_mat = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                        [0, 1, 2 ** t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                        [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
                        [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                        [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                        [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]])

    a_mat_1 = np.linalg.inv(a_mat)

    cartesianX = np.array([p0x, v0, a0, pfx, vf, af])
    cartesianY = np.array([p0y, v0, a0, pfy, vf, af])
    cartesianZ = np.array([p0z, v0, a0, pfz, vf, af])

    cartesianTHETA = np.array([0, v0, a0, theta, vf, af])
    cartesianPHI = np.array([0, v0, a0, phi, vf, af])
    cartesianPSI = np.array([0, v0, a0, psi, vf, af])

    px = np.transpose(np.dot(a_mat_1, cartesianX))
    py = np.transpose(np.dot(a_mat_1, cartesianY))
    pz = np.transpose(np.dot(a_mat_1, cartesianZ))

    pps = np.transpose(np.dot(a_mat_1, cartesianTHETA))
    pth = np.transpose(np.dot(a_mat_1, cartesianPHI))
    pph = np.transpose(np.dot(a_mat_1, cartesianPSI))

    pseg = np.fliplr(np.array([px,py,pz, pph, pth, pps]))

    samples = int((tf*hz))
    plen = np.linspace(0,tf,(tf*hz))
    time_span = plen
    # dx = time_span[1] - time_span[0]
    dx = (1.0/hz)

    X1 = np.zeros((samples,6))
    # print "X1[0,0] is:", X1[0,1]

    # pro = np.ones((samples, 1)) * psi
    # pth = np.ones((samples, 1)) * theta
    # pya = np.ones((samples, 1)) * phi

    for j in range(len(plen)):
        t = plen[j]
        for i in range(6):
            X1[j,i] = pseg[i,0]*t**5 + pseg[i,1]*t**4 + pseg[i,2]*t**3 + pseg[i,3]*t**2+pseg[i,4]*t + pseg[i,5]

    # zaxis = np.zeros((len(plen),1))

    # Concatenating the roll pitch yaw.
    # X1 =np.concatenate((X1,pro), axis=1)
    # X1 =np.concatenate((X1,pth), axis=1)
    # X1 =np.concatenate((X1,pya), axis=1)
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


def plotX(X):
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


def calcJac(q, rb):
    A = [0 for i in range(rb.ndof)]
    alp = np.zeros(rb.ndof)
    a = np.zeros(rb.ndof)
    th = np.zeros(rb.ndof)
    d = np.zeros(rb.ndof)
    J = np.zeros((6,rb.ndof))

    for i in range(rb.ndof):
        alp[i] = rb.joints[i].alpha
        a[i] = rb.joints[i].a
        th[i] = q[i]
        d[i] = rb.joints[i].d

    T = np.identity(4)

    Aout = []
    for i in range(rb.ndof):
        A[i] = np.array(
            [[np.cos(th[i]), -np.sin(th[i]) * np.cos(alp[i]), np.sin(th[i]) * np.sin(alp[i]), a[i] * np.cos(th[i])],
             [np.sin(th[i]), np.cos(th[i]) * np.cos(alp[i]), -np.cos(th[i]) * np.sin(alp[i]), a[i] * np.sin(th[i])],
             [0, np.sin(alp[i]), np.cos(alp[i]), d[i]],
             [0, 0, 0, 1]])
        Aout.append(np.dot(T, A[i]))
        T = np.dot(T, A[i])

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


def calcJacDot(q, qd, rb):
    A = [0 for i in range(rb.ndof)]
    alp = np.zeros(rb.ndof)
    a = np.zeros(rb.ndof)
    th = np.zeros(rb.ndof)
    d = np.zeros(rb.ndof)
    J = np.zeros((6,7))

    for i in range(rb.ndof):
        alp[i] = rb.joints[i].alpha
        a[i] = rb.joints[i].a
        th[i] = q[i]
        d[i] = rb.joints[i].d

    T = np.identity(4)

    Aout = []
    for i in range(rb.ndof):
        A[i] = np.array(
            [[np.cos(th[i]), -np.sin(th[i]) * np.cos(alp[i]), np.sin(th[i]) * np.sin(alp[i]), a[i] * np.cos(th[i])],
             [np.sin(th[i]), np.cos(th[i]) * np.cos(alp[i]), -np.cos(th[i]) * np.sin(alp[i]), a[i] * np.sin(th[i])],
             [0, np.sin(alp[i]), np.cos(alp[i]), d[i]],
             [0, 0, 0, 1]])
        Aout.append(np.dot(T, A[i]))
        T = np.dot(T, A[i])

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


def calcInverseKin(X):
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


def calcQd(Xd, qc, rho, rb):
    J = calcJac(qc, rb)
    Jt = np.transpose(J)

    inner = np.linalg.inv( np.dot(J,Jt) + np.dot(np.identity(6),rho) )
    Jinv = np.dot(Jt,inner)
    TestSingularity = np.linalg.det(J[:2,:2])

    if(TestSingularity < (1e-9)) and (TestSingularity > -(1e-9)):
        qd = np.array([0,0])
        # print "in here qd"
    else:
        qd = np.dot(Jinv[:2,:2],Xd[:2])

    return qd


def calcQdd(Xdd, qc, qd, rb):
    J = calcJac(qc, rb)
    Jd = calcJacDot(qc, qd, rb)
    Jdq = np.dot(Jd,qd)

    T = Transform(rb)
    R = t2rot(T)
    r,p,y = r2rpy(R)
    A = rpy2Ja(r,p,y)
    B = block_diag(np.eye(3),np.linalg.inv(A))
    # Jadq = np.dot(B,Jdq)
    Ja = np.dot(B,J)

    Jpinv = pinv(Ja)
    qdd = np.dot(Jpinv, (Xdd - Jdq))
    return qdd


def calcQdd3(Xdd, qc, qd, rb):
    J = calcJac(qc, rb)
    J = J[:,:3]
    Jd = calcJacDot(qc, qd, rb)
    Jdq = np.dot(Jd,qd)
    # Jt = np.transpose(J)

    Jpinv = pinv2(J)

    # inner = np.linalg.inv(np.dot(J, Jt) + np.dot(np.identity(6), rho))
    # Jinv = np.dot(Jt, inner)
    # TestSingularity = np.linalg.det(J[:2,:2])

    # if(TestSingularity < (1e-9)) and (TestSingularity > -(1e-9)):
    #     qdd = np.array([0, 0, 0, 0, 0, 0, 0])
        # print "Singularity!"
    # else:
    qdd = np.dot(Jpinv, (Xdd - Jdq))
    return qdd



def calcXd(qc, qd, rb):
    J = calcJac(qc, rb)
    # TestSingularity = np.linalg.det(J[:2,:2])
    #
    # if(TestSingularity < (1e-9)) and (TestSingularity > -(1e-9)):
    #     xd = np.array([0,0,0])
    #     print "in here xd"
    # else:
    xd = np.dot(J,qd)
    return xd


def pinv(J):
    u, s, vh = np.linalg.svd(J.T, full_matrices=True)
    u.shape, s.shape, vh.shape

    rho = 4
    S2 = np.dot(J.T,0)
    for i in range(len(s)):
        S2[i,i] = s[i] / (s[i]**2 + rho**2)

    JpinvT = np.dot(np.dot(vh.T,S2.T),u.T)
    Jpinv = JpinvT.T
    return Jpinv

def pinv2(J):
    u, s, vh = np.linalg.svd(J.T, full_matrices=True)
    u.shape, s.shape, vh.shape

    rho = 5
    S2 = np.dot(J.T,0)
    for i in range(len(s)):
        S2[i,i] = s[i] / (s[i]**2 + rho**2)

    JpinvT = np.dot(np.dot(u,S2),vh.T)
    Jpinv = JpinvT.T
    return Jpinv



def impedanceXdd(x,xd,xdd,xc,xcd, F,Kd,Bd,Md):

    Mdinv = np.linalg.inv(Md)
    # Md_temp = np.zeros((3,3))
    # Md_temp[:-1,:-1] = Mdinv
    # Md = Md_temp
    # Md = np.concatenate((Md, np.array([0, 0])), axis=0)
    # Md = np.concatenate((Md, np.array([0, 0, 0])), axis=1)

    damper = np.dot(Bd,(xcd - xd))
    # print(xd)
    # print(xcd)
    # print(damper)
    # print("xcd - xd: ",xcd-xd)


    spring = np.dot(Kd,(xc - x))
    # print(x)
    # print(xc)
    # print("xc - x: ", xc-x)


    # print(damper + spring + F)

    ax = xdd - np.dot(Md,(damper + spring + F))
    # print("ax: ", ax)
    return ax


def Transform(rb):
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

    for i in range(rb.ndof):
        A[i] = np.array([[np.cos(th[i]), -np.sin(th[i]) * np.cos(alp[i]), np.sin(th[i]) * np.sin(alp[i]), a[i] * np.cos(th[i])],
             [np.sin(th[i]), np.cos(th[i]) * np.cos(alp[i]), -np.cos(th[i]) * np.sin(alp[i]), a[i] * np.sin(th[i])],
             [0, np.sin(alp[i]), np.cos(alp[i]), d[i]],
             [0, 0, 0, 1]])
        T = np.dot(T, A[i])
    return T


def t2transl(T):
    transl = T[:3, 3]
    return transl


def t2rot(T):
    R = T[:3, :3]
    return R


def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)


def r2rpy(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2,0],-1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif isclose(R[2,0],1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return psi, theta, phi


def r2eul(R):
    # Theta, Phi, Psi
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

    # print("phi:", phi)
    # print("theta:", theta)
    # print("psi:", psi)
    return theta, phi, psi


def eul2Ja(phi,theta,psi):
    Ja = np.array([[ 0, -np.sin(phi), np.cos(phi) * np.sin(theta)],
                    [0,  np.cos(phi), np.sin(phi) * np.sin(theta)],
                    [1,        0,           np.cos(theta) ]])
    return Ja


def rpy2Ja(r,p,y):
    Ja = np.array([[ 1,          0,              np.sin(p)],
                    [0,  np.cos(r), -np.cos(p) * np.sin(r)],
                    [0,  np.sin(r),  np.cos(p) * np.cos(r)]])
    return Ja


#A = eul2Ja(r,p,y)



# def force2Torque(F, q, rb):
#     J = calcJac(q, rb)
#     Jt = np.transpose(J)
#
#     tau = np.transpose(np.dot(Jt[:,:3],F))
#
#     return tau
