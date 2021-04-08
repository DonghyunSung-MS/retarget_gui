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

num_tracker = 6
state = np.zeros((num_tracker + 2) * 7) # last 2 for shoulder

base_postion = np.array([0, 0, 0.4])
base_raw_pos = np.zeros(3)
pos = np.zeros(3)
quat = np.array([1,0,0,0])

def tracker_callback(msg):
    global state, base_postion, pos, quat, base_raw_pos
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

def main():
    global state
    rospy.init_node('node_name')

    for i in range(8):
        rospy.Subscriber("/retarget/calib_tracker" + str(i), matrix_3_4, tracker_callback)

    model = load_model_from_path(os.path.dirname(__file__) + "/test.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)

    state = copy.copy(sim.get_state()).qpos
    qvel = copy.copy(sim.get_state()).qvel
    while True:
        old_state = sim.get_state()
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