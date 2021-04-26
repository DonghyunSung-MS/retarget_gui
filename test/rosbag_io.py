import os
import rosbag
import h5py
import numpy as np

from scipy.spatial.transform import Rotation as R

home_dir = os.path.expanduser("~")
bag = rosbag.Bag(os.path.join(home_dir, "exp_data/bag_file/test1.bag"))

hf5 = h5py.File("test1.h5","w")

data_dict = dict()
for i in range(6):
    data_dict[i] = []

for topic, msg, t in bag.read_messages():

    if topic[-1].isdigit():
        topic_id = int(topic[-1])

        T_raw = np.vstack([np.array(msg.firstRow),
                        np.array(msg.secondRow),
                        np.array(msg.thirdRow),
                        np.array([0 , 0, 0, 1])])
        pos = T_raw[:3, 3]
        quat = R.from_matrix(T_raw[:3, :3]).as_quat()
        quat = np.hstack([quat[3:4], quat[:3]])

        state = np.hstack([pos, quat]) #xyz wxyz
        data_dict[topic_id].append(state)

for i in range(6):
    data_dict[i] = np.vstack(data_dict[i])
    dset = hf5.create_dataset(f"tracker{i}", shape=data_dict[i].shape, dtype=data_dict[i].dtype,data=data_dict[i])

bag.close()
