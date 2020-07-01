from robot3 import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import scipy
mpl.style.use('seaborn')

class stateVector:
    def __init__(self):
        self.x = np.zeros(6)
        self.xd = np.zeros(6)
        self.xdd = np.zeros(6)
        self.quat = np.quaternion(1,0,0,0)
        self.quat_d = np.quaternion(1,0,0,0)
        self.quat_dd = np.quaternion(1,0,0,0)
        self.x3 = self.x[:3]
        self.xd3 = self.x[:3]
        self.xdd3 = self.x[:3]

class jointStateVector:
    def __init__(self, q, qd, qdd):
        self.q = q
        self.qd = qd
        self.qdd = qdd
        self.qnull = self.q
        self.qdnull = self.qd
        self.qddnull = self.qdd

class dataCollect:
    def __init__(self, sampcol):
        self.x = np.zeros((6,sampcol))
        self.xd = np.zeros((6,sampcol))
        self.q = np.zeros([7,1])
        self.qd = np.zeros([7,1])
        self.qdd = np.zeros([7,1])
        self.aq = np.zeros((7,sampcol))
        self.error  = np.zeros((6,sampcol))
        self.imp = np.zeros((6,sampcol))
        self.F  = np.zeros((7,sampcol))
        self.tau  = np.zeros((7,sampcol))


class simulation:
    def __init__(self, state_des, state_end, jointState, error, data):
        self.state_des = state_des
        self.state_end = state_end
        self.jointState = jointState
        self.error = error
        self.data = data
        self.Bn = np.zeros((7,7))
        self.Kn = np.zeros((7,7))

    def spong_impedance_control(self, impctrl, rb):
        # Double integrator impedance control strategy
        kine = rb.forwardKinematics(self.jointState.q)
        rot = kine.R
#         rot2 = quaternion.from_rotation_matrix(self.state_des.quat)
#         R_e = block_diag(rot, rot)
#         xdd_in = np.dot(R_e.T, self.state_des.xdd)

#         E = rb.quatprop_E(self.error.quat)
        Ko =  rot.dot(impctrl.Kd[3:, 3:])
        Kd_b = block_diag(impctrl.Kd[:3,:3], Ko)
#         Kd_b = impctrl.Kd
        Bd = impctrl.damping_constant_mass()
#         impctrl.Kd[3:, 3:] = Ko



#         impctrl.Bd = impctrl.damping_constant_mass()
#         Kd_b = impctrl.Kd

        ax = impctrl.outputquat(Kd_b, Bd, self.error.x, self.error.xd, self.state_des.xdd, impctrl.F)
#         self.data.imp[:3,i] = ax

        aq_in = rb.calcQddNull(ax, self.jointState.q, self.jointState.qd,
                               self.jointState.qddnull)

        # Inverse dynamics
        tauc = rb.inverseDynamics(self.jointState.q, self.jointState.qd, aq_in, rb.grav)

        # Computing Jacobian for external forces:
        J = rb.calcJac(self.jointState.q)
        Jpinv = rb.pinv(J)

        # Nullspace torque (projection matrix * torque_0)
        tau_nullspace = np.dot((np.eye(7) - np.dot(J.T, Jpinv.T) ),
                                   (np.dot( impctrl.nullspace_stiffness_, (self.jointState.qnull - self.jointState.q)) -
                                    np.dot(np.dot(2, np.sqrt(impctrl.nullspace_stiffness_)), self.jointState.qd)))

        # q_null update
        self.jointState.q_null = self.jointState.q

        # Torque collected
        tau = tauc + tau_nullspace

        return tau

    def classical_impedance_control(self, impctrl, rb, ndof, *args):
        # Frame Selection
        kine = rb.forwardKinematics(self.jointState.q)
        rot = kine.R
        R_e = block_diag(rot, rot)

        # Frame variables
        vd_error = rb.calcVd(self.jointState.q, self.jointState.qd, self.state_des.xd)
        xdd_in = R_e.dot(self.state_des.xdd)
        xd_end = R_e.dot(self.state_end.xd)

        if args:
            rb.ndof = 7
            if args[0].lower() in ['rpy']:
                J = rb.calcJac(qin)
                Ja = rb.analyticJacobian(J, self.state_end.x[3:], 'rpy')
                # Fix for jacobian dot later

            if args[0].lower() in ['quaternion']:
                J = rb.calcJac(self.jointState.q)
                Jd = rb.calcJacDot(self.jointState.q, self.jointState.qd)
#                 Ja = J
                Ja = R_e.dot(J)
                Jad = R_e.dot(Jd)
#                 Ja = rb.analyticJacobian(J, self.state_end.quat, 'quaternion6')
#                 Jad = rb.analyticJacobianDot(J, Jd, self.state_end.quat, self.state_end.quat_d)

        # Cartesian Inertia Matrix
#         M = rb.inertiaComp(self.jointState.q)
        Lambda = rb.cinertiaComp(self.jointState.q, Ja)
#         print Lambda.shape

        # Cartesian Coriolis Matrix
        mu = rb.ccoriolisComp(self.jointState.q, self.jointState.qd, Ja, Jad)

        # For the classical impedance controller without redundancy
        if ndof is 6:
            qin = np.zeros(6)
            qdin = np.zeros(6)

            qin[0] = self.jointState.q[0]
            qin[1] = self.jointState.q[2]
            qin[2] = self.jointState.q[3]
            qin[3] = self.jointState.q[4]
            qin[4] = self.jointState.q[5]
            qin[5] = self.jointState.q[6]

            qdin[0] = self.jointState.qd[0]
            qdin[1] = self.jointState.qd[2]
            qdin[2] = self.jointState.qd[3]
            qdin[3] = self.jointState.qd[4]
            qdin[4] = self.jointState.qd[5]
            qdin[5] = self.jointState.qd[6]
            Ja = np.delete(Ja, 1, 1)
            Jad = np.delete(Jad, 1, 1)
#             Lambda = np.delete(Lambda, 1, 1)
#             mu = np.delete(Lambda,1,1)
#             rb.ndof = 6
#         else:
#             qin = self.jointState.q
#             qdin = self.jointState.qd

        rb.ndof = 7
        # Jointspace Coriolis Matrix
#         C = rb.coriolisComp(self.jointState.q, self.jointState.qd)

        # Jointspace gravitational load vector
        tg = rb.gravloadComp(self.jointState.q, rb.grav)

        # Computing Jacobian:
#         Jnull = rb.calcJac(self.jointState.q)
        Jpinv = rb.pinv(Ja)

        # Computing dynamic damping
        E = rb.quatprop_E(self.error.quat)
        Ko = 2 * E.T.dot( rot.dot( impctrl.Kd[3:, 3:]) )
        Bd = impctrl.Bd
#         Bd = impctrl.damping_constant_mass()
#         impctrl.Kd[3:, 3:] = Ko


#         u, s, vh = np.linalg.svd(M.T, full_matrices=True)
#         u.shape, s.shape, vh.shape

#         rho = 0.2
#         S2 = np.dot(M.T,0)
#         for i in range(len(s)):
#             S2[i,i] = s[i] / (s[i]**2 + rho**2)

#         MpinvT = np.dot(np.dot(vh.T,S2.T),u.T)
#         MpinvT = MpinvT.T


#         impctrl.zeta = np.array([1, 1, 1, 1, 1, 1])

# #         # Dual eigen:
#         Bd, Kd_b3 = impctrl.damping_dual_eigen(impctrl.Kd,impctrl.Md)

        # Test
#         Kd_b = impctrl.Kd

        # Quaternion stiffness and translational stiffness
        Kd_b = block_diag(impctrl.Kd[:3,:3], Ko)

        # Compute torque "cartesian"
#         tauc = (np.dot(Ja.T, ( Lambda.dot(self.state_des.xdd) + mu.dot(self.state_end.xd))) -
#                 np.dot(Ja.T, np.dot(np.dot(Lambda, np.linalg.inv(impctrl.Md)),
#                                    (np.dot(Kd_b, self.error.x) + np.dot(Bd,  self.error.xd)))
#             ))
#
        tauc = (np.dot(Ja.T, ( Lambda.dot(xdd_in) + mu.dot(xd_end) )) -
                np.dot(Ja.T, np.dot(np.dot(Lambda, np.linalg.inv(impctrl.Md)),
                (np.dot(Kd_b, self.error.x) + np.dot(Bd,  vd_error))))   +
                np.dot( Ja.T.dot( Lambda.dot( np.linalg.inv(impctrl.Md) ) - np.eye(6)),
                impctrl.F))

        # Append to torque if ndof 6 sim
        if ndof is 6:
            # At the beginning
            tauc = np.array([tauc[0], 0,tauc[1],tauc[2],tauc[3],tauc[4],tauc[5]])


        # Dual eigen:
#         ns = impctrl.nullspace_stiffness_
#         K_n = np.diag(np.array([ns,ns,ns,ns,ns,ns,ns]))

#         B_n, Kd_b2 = impctrl.damping_dual_eigen2(K_n,M)

        # Compute torque nullspace
#         tau_nullspace = np.dot((np.eye(7) - np.dot(Jnull.T, Jpinv.T) ),
#                                (np.dot( K_n, (self.jointState.qnull - self.jointState.q)) -
#                                 np.dot(B_n, self.jointState.qd)))

#         Jpinv2 = np.dot(Lambda,J.dot(MpinvT))

        tau_nullspace = np.dot((np.eye(7) - np.dot(Ja.T, Jpinv.T) ),
                               (np.dot( self.Kn, (self.jointState.qnull - self.jointState.q)) -
                                np.dot(self.Bn, self.jointState.qd)))


#         tau =  tauc + tau_nullspace + tg + np.dot(C, self.jointState.qd)
        tau =  tauc + tau_nullspace + tg
        return tau


    def inertia_avoidance_impedance_control(self, impctrl, rb, ndof, *args):
        # For the classical impedance controller without redundancy
        if ndof is 6:
            qin = self.jointState.q[:6]
            qdin = self.jointState.qd[:6]
            rb.ndof = 6
        else:
            qin = self.jointState.q
            qdin = self.jointState.qd

        # Computing analytical Jacobian
        if args:
            if args[0].lower() in ['rpy']:
                J = rb.calcJac(qin)
                Ja = rb.analyticJacobian(J, self.state_end.x[3:], 'rpy')

            if args[0].lower() in ['quaternion']:
                J = rb.calcJac(qin)
                Ja = rb.analyticJacobian(J, self.state_end.quat, 'quaternion6')

        # Cartesian Inertia Matrix
        M = rb.inertiaComp(qin)
        Lambda = rb.cinertiaComp(qin, Ja)

        # Cartesian Coriolis Matrix
        mu = rb.ccoriolisComp(qin, qdin, Ja, Jad)

        # Reset DoF
        rb.ndof = 7

        # Cartesian Coriolis Matrix
#         C = rb.coriolisComp(self.jointState.q, self.jointState.qd)

        # Jointspace gravitational load vector
        tg = rb.gravloadComp(self.jointState.q, rb.grav)

        # Computing Jacobian:
        Jnull = rb.calcJac(self.jointState.q)
        Jpinv = rb.pinv(Jnull)

        # Computing dynamic damping
        impctrl.zeta = np.array([1, 1, 1, 1, 1, 1])

        # Dual eigen:
        Bd, Kd_b = impctrl.damping_dual_eigen(impctrl.Kd,Lambda)

        # Test
        Kd_b = impctrl.Kd

        # Compute torque "cartesian"
        tauc = tg + np.dot(Ja.T, ( np.dot(Lambda, self.state_des.xdd) +
                    mu.dot(self.state_des.xd) - np.dot(Kd_B, self.error.x) -
                    np.dot(Bd, self.error.xd) ))

        # Append to torque if ndof 6 sim
        if ndof is 6:
            # At the beginning
            tauc = np.append(0,tauc)

            # insert
            #tauc = np.hstack((a[0:4], np.zeros(12), a[4:]))

        # Compute torque nullspace
        tau_nullspace = np.dot((np.eye(7) - np.dot(Jnull.T, Jpinv.T) ),
                               (np.dot( self.Kn, (self.jointState.qnull - self.jointState.q)) -
                                np.dot(self.Bn, self.jointState.qd)))

        tau =  tauc + tau_nullspace + tg
        return tau

    def impedance_control_equilibrium(self, impctrl, rb, ndof, *args):
        # For the classical impedance controller without redundancy
        if args:
            rb.ndof = 7
            if args[0].lower() in ['rpy']:
                J = rb.calcJac(qin)
                Ja = rb.analyticJacobian(J, self.state_end.x[3:], 'rpy')
                # Fix for jacobian dot later

            if args[0].lower() in ['quaternion']:
                J = rb.calcJac(self.jointState.q)
                Jd = rb.calcJacDot(self.jointState.q, self.jointState.qd)
                Ja = rb.analyticJacobian(J, self.state_end.quat, 'quaternion6')
                Jad = rb.analyticJacobianDot(J, Jd, self.state_end.quat, self.state_end.quat_d)


        # For the classical impedance controller without redundancy
        if ndof is 6:
            qin = np.zeros(6)
            qdin = np.zeros(6)

            qin[0] = self.jointState.q[1]
            qin[1] = self.jointState.q[2]
            qin[2] = self.jointState.q[3]
            qin[3] = self.jointState.q[4]
            qin[4] = self.jointState.q[5]
            qin[5] = self.jointState.q[6]

            qdin[0] = self.jointState.qd[1]
            qdin[1] = self.jointState.qd[2]
            qdin[2] = self.jointState.qd[3]
            qdin[3] = self.jointState.qd[4]
            qdin[4] = self.jointState.qd[5]
            qdin[5] = self.jointState.qd[6]
            Ja = np.delete(Ja, 0, 1)
            Jad = np.delete(Jad, 0, 1)
            rb.ndof = 6
        else:
            qin = self.jointState.q
            qdin = self.jointState.qd

        # Reset DoF
        rb.ndof = 7

        # Cartesian Coriolis Matrix
#         mu = rb.ccoriolisComp()
        C = rb.coriolisComp(self.jointState.q, self.jointState.qd)

        # Jointspace gravitational load vector
        tg = rb.gravloadComp(self.jointState.q, rb.grav)

        # Computing Jacobian:
        Jnull = rb.calcJac(self.jointState.q)
        Jpinv = rb.pinv(Jnull)

        # Compute torque "cartesian"
        tauc = np.dot(J.T, ( - np.dot(impctrl.cartesian_stiffness_, self.error.x) -
                                  np.dot(impctrl.cartesian_damping_, self.error.xd) ))

        # Append to torque if ndof 6 sim
        if ndof is 6:
            # At the beginning
            tauc = np.array([0,tauc[0],tauc[1],tauc[2],tauc[3],tauc[4],tauc[5]])

        # Compute torque nullspace
        tau_nullspace = np.dot((np.eye(7) - np.dot(Jnull.T, Jpinv.T) ),
                               (np.dot( impctrl.nullspace_stiffness_, (self.jointState.qnull - self.jointState.q)) -
                                np.dot(np.dot(2, np.sqrt(impctrl.nullspace_stiffness_)), self.jointState.qd)))

        tau =  tauc + tau_nullspace + np.dot(C, self.jointState.qd) + tg
        return tau


    def outputEndeffector(self, rb, *args):
        kine = rb.forwardKinematics(self.jointState.q)
        Xftr = kine.transl

        if args:
            if args[0].lower() in ['rpy']:
                rpy = kine.rpy
                Xf = np.array([Xftr[0], Xftr[1], Xftr[2], rpy[0], rpy[1], rpy[2]])
                Xfd_calc = rb.calcXd(self.jointState.q, self.jointState.qd, 'rpy')
                Xfd = np.array([Xfd_calc[0], Xfd_calc[1], Xfd_calc[2], Xfd_calc[3], Xfd_calc[4], Xfd_calc[5]])

            if args[0].lower() in ['quaternion']:
                rot = kine.R
                quatf = quaternion.from_rotation_matrix(rot)
#                 quatf_float = rb.mat2quat(rot)
#                 quatf = quaternion.from_float_array(quatf_float)
                Xf = np.array([Xftr[0], Xftr[1], Xftr[2], quatf.x, quatf.y, quatf.z])

                Xfd_calc = rb.calcXd(self.jointState.q, self.jointState.qd)
#                 quatf_d = quaternion.from_float_array(Xfd_calc[3:])
                Xfd = np.array([Xfd_calc[0], Xfd_calc[1], Xfd_calc[2], Xfd_calc[3], Xfd_calc[4], Xfd_calc[5]])

                self.state_end.quat = quatf
#                 self.state_end.quat_d = quatf_d

        self.state_end.x = Xf
        self.state_end.xd = Xfd

    def output_equilibrium_update(self, rb):
        kine = rb.forwardKinematics(self.jointState.q)
        Xftr = kine.transl
        rot = kine.R
        quatf = quaternion.from_rotation_matrix(rot)

        Xf = np.array([Xftr[0], Xftr[1], Xftr[2], quatf.x, quatf.y, quatf.z])

        self.state_end.quat = quatf
        self.state_end.x = Xf

    def quat_subtract(self, quat_des, quat_end):
        orientation_d = quaternion.as_float_array(quat_des)
        orientation = quaternion.as_float_array(quat_end)

        # Sign Ambiguity
        if (orientation_d[1:].dot(orientation[1:]) < 0.0):
            quat_end.x = -orientation[1]
            quat_end.y = -orientation[2]
            quat_end.z = -orientation[3]

        eq = quat_end.inverse() * quat_des

#         eq = np.array([eq_t.x, eq_t.y, eq_t.z])

#         eta_e = quat_end.w
#         eps_e = np.array([quat_end.x, quat_end.y, quat_end.z])

#         eta_d = quat_des.w
#         eps_d = np.array([quat_des.x, quat_des.y, quat_des.z])
#         eps_d_skew = np.array([[0, -eps_d[2], eps_d[1]],
#                          [eps_d[2], 0, -eps_d[0]],
#                          [-eps_d[1], eps_d[0], 0]])

#         eq = (eta_d * eps_e) - (eta_e * eps_d) - eps_d_skew.dot(eps_e)

        return eq

    def quat_subtract2(self, quat_des, quat_end):
        orientation_d = quaternion.as_float_array(quat_des)
        orientation = quaternion.as_float_array(quat_end)

        # Sign Ambiguity
        if (orientation_d[1:].dot(orientation[1:]) < 0.0):
            quat_end.x = -orientation[1]
            quat_end.y = -orientation[2]
            quat_end.z = -orientation[3]

#         eq = np.array([eq_t.x, eq_t.y, eq_t.z])

        eta_e = quat_end.w
        eps_e = np.array([quat_end.x, quat_end.y, quat_end.z])

        eta_d = quat_des.w
        eps_d = np.array([quat_des.x, quat_des.y, quat_des.z])
        eps_d_skew = np.array([[0, -eps_d[2], eps_d[1]],
                         [eps_d[2], 0, -eps_d[0]],
                         [-eps_d[1], eps_d[0], 0]])

        eq = (eta_d * eps_e) - (eta_e * eps_d) - eps_d_skew.dot(eps_e)
        return eq


    def feedbackError3(self, rb, *args):
        kine = rb.forwardKinematics(self.jointState.q, 'tool')
        rot = kine.R
        R_e = block_diag(rot, rot)
        if args:
            if args[0].lower() in ['rpy']:
                e = self.state_end.x - self.state_des.x
                e = rot.dot(e)
                ed = self.state_end.xd - self.state_des.xd

            if args[0].lower() in ['quaternion']:
                ep = self.state_end.x[:3] - self.state_des.x[:3]
                eq = self.quat_subtract(self.state_des.quat, self.state_end.quat)
                eqt = quaternion.as_float_array(eq)
                eqt = 2*eqt[1:]
#                 eqt = 2*np.dot(rot, eqt)

                e = np.array([ep[0], ep[1], ep[2], eqt[0], eqt[1], eqt[2]])
#                 e = np.dot(R_e, e)

                edp = (self.state_end.xd - self.state_des.xd)
                ed = np.array([edp[0], edp[1], edp[2], edp[3], edp[4], edp[5]])
#                 ed = np.dot(R_e, e)
                # self.error.quat = eq
                # self.error.quat_d = edq

        self.error.x = e
        self.error.xd = ed
        self.error.quat = eq
#         self.error.quat_d = edq


    def feedback_equilibrium(self, rb):
        kine = rb.forwardKinematics(self.jointState.q)
        rot = kine.R
        ep = self.state_end.x[:3] - self.state_des.x[:3]
        ep = np.dot(rot, ep)
        eq = self.quat_subtract(self.state_des.quat, self.state_end.quat)
        eqt = quaternion.as_float_array(eq)
        eqt = eqt[1:]
        eqt = np.dot(rot, eqt)

        e = np.array([ep[0], ep[1], ep[2], eqt[0], eqt[1], eqt[2]])
        self.error.x = e


    def qIntegrate(self, qacc, dx):
        # Integrate from acceleration to velocity
        qdtemp = np.concatenate((self.data.qdd, qacc.reshape(7,1)), axis=1)
        qd = np.trapz(qdtemp, axis=1) * dx

        #Integrate from velocity to position
        qtemp = np.concatenate((self.data.qd, qd.reshape(7,1)), axis=1)
        q = np.trapz(qtemp, axis=1) * dx

        self.data.qdd = np.concatenate(( self.data.qdd, qacc.reshape(7,1)),axis=1)
        self.data.qd = np.concatenate(( self.data.qd, qd.reshape(7,1)),axis=1)
        self.data.q = np.concatenate(( self.data.q, q.reshape(7,1)),axis=1)

        self.jointState.q = q
        self.jointState.qd = qd
