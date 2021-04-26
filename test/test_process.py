#simple mujoco_py visualization
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState
import copy
import os
import rospy
import numpy as np
from VR.msg import matrix_3_4
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
import toml
from qpsolvers import solve_qp
num_tracker = 6
state = np.zeros((num_tracker + 2) * 7) # last 2 for shoulder

base_postion = np.array([0, 0, 0.4])
base_raw_pos = np.zeros(3)
pos = np.zeros(3)
quat = np.array([1,0,0,0])
tracker = 0
def tracker_callback(msg):
    global state, base_postion, pos, quat, base_raw_pos, tracker
    tracker += 1
    topic_id = int(msg._connection_header["topic"][-1])
    
    T_raw = np.vstack([np.array(msg.firstRow),
                       np.array(msg.secondRow),
                       np.array(msg.thirdRow),
                       np.array([0 , 0, 0, 1])])
    
    if topic_id == 0:
        base_raw_pos = T_raw[:3, 3]
    
    pos = T_raw[:3, 3] - base_raw_pos + base_postion
    quat = R.from_matrix(T_raw[:3, :3]).as_quat()
    quat = np.hstack([quat[3:4], quat[:3]])
    
    state[7 * topic_id : 7 * topic_id + 7] = np.hstack([pos, quat])

def posquat2posrot(pos_quat):
    pos = pos_quat[:3]
    quat = pos_quat[3:]
    quat = np.hstack([quat[1:4], quat[0:1]])
    return pos, R.from_quat(quat).as_matrix()

def shoulder_estimation(old, new):
    global state
    chest = 1
    left = 2
    right = 4

    old_chest = old[7 * chest : 7 * chest + 7]
    old_left = old[7 * left : 7 * left + 7]
    old_right = old[7 * right : 7 * right + 7]
    p_c, R_c = posquat2posrot(old_chest)
    p_l, R_l = posquat2posrot(old_left)
    p_r, R_r = posquat2posrot(old_right)

    new_chest = new[7 * chest : 7 * chest + 7]
    new_left = new[7 * left : 7 * left + 7]
    new_right = new[7 * right : 7 * right + 7]

    p_cn, R_cn = posquat2posrot(new_chest)
    p_ln, R_ln = posquat2posrot(new_left)
    p_rn, R_rn = posquat2posrot(new_right)

    # A_l = np.block([[R_l, -R_c],[R_ln, -R_cn]]) + 0.0001*np.identity(6)
    # A_r = np.block([[R_r, -R_c],[R_rn, -R_rn]]) + 0.0001*np.identity(6)

    # b_l = np.block([-p_l + p_c, -p_ln + p_cn]).reshape(-1, 1)
    # b_r = np.block([-p_r + p_c, -p_rn + p_cn]).reshape(-1, 1)

    A_l = np.block([R_ln, -R_cn])
    A_r = np.block([R_rn, -R_rn])

    b_l = (-p_ln + p_cn).reshape(-1, 1)
    b_r = (-p_rn + p_cn).reshape(-1, 1)


    P = A_l.T @ A_l + 0.001*np.identity(6)
    q = (b_l.T @ A_l).reshape(-1)
    G = np.array([0,0,-1,0,0,0])
    h = -np.ones(1)*0.2
    x_l = solve_qp(P, q,G=G, h=h, lb=-np.ones(6), ub=np.ones(6))

    P = A_r.T @ A_r + np.identity(6)
    q = (b_r.T @ A_r).reshape(-1)
    G = np.array([0,0,-1,0,0,0])
    h = -np.ones(1)*0.2
    x_r = solve_qp(P, q,G=G, h=h, lb=-np.ones(6), ub=np.ones(6))
    # print("QP solution: x = {}".format(x))

    # x_l = np.linalg.solve(A_l, b_l) #el->sh , chest -> sh
    # print("MA solution: x = {}".format(x_l))
    # x_r = np.linalg.solve(A_r, b_r)

    # print("In elbow ", (p_l.reshape(-1, 1) + R_l @ x_l[:3]).reshape(-1))
    # print("In chest ", (p_c.reshape(-1, 1) + R_c @ x_l[3:]).reshape(-1))
    # print(b_l)
    # print(x_l.reshape(-1))
    # print("---------------------------------------------------")

    if x_l is not None:
        x_l = x_l.reshape(-1,1)
        x_r = x_r.reshape(-1,1)
        
        lsh = (p_l.reshape(-1, 1) + R_l @ x_l[:3]).reshape(-1)
        rsh = (p_r.reshape(-1, 1) + R_r @ x_r[:3]).reshape(-1)

        state[7 * 6 : 7 * 6 + 3] = lsh
        state[7 * 7 : 7 * 7 + 3] = rsh

    # quat = R.from_matrix(R_cn[:3, :3]).as_quat()
    # quat = np.hstack([quat[3:4], quat[:3]])
    
    # state[7 * chest : 7 * chest + 7] = np.hstack([p_cn, quat])

def main():
    global state
    rospy.init_node('raw_tracker_node')
    rospy.Subscriber("/TRACKER0", matrix_3_4, tracker_callback)
    rospy.Subscriber("/TRACKER1", matrix_3_4, tracker_callback)
    rospy.Subscriber("/TRACKER2", matrix_3_4, tracker_callback)
    rospy.Subscriber("/TRACKER3", matrix_3_4, tracker_callback)
    rospy.Subscriber("/TRACKER4", matrix_3_4, tracker_callback)
    rospy.Subscriber("/TRACKER5", matrix_3_4, tracker_callback)

    model = load_model_from_path(os.path.dirname(__file__) + "/../simple_human_model/test.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)

    state = copy.copy(sim.get_state()).qpos
    qvel = copy.copy(sim.get_state()).qvel
    while True:
        old_state = copy.copy(sim.get_state())
        # if tracker > 50:
            # print(tracker)
        shoulder_estimation(old_state.qpos, state)
        new_state = MjSimState(old_state.time, state, qvel,
                                         old_state.act, old_state.udd_state)
        # print(state)
        sim.set_state(new_state)
        sim.forward()
        viewer.render()

    #     if os.getenv('TESTING') is not None:
    #         break


if __name__ == "__main__":
    main()