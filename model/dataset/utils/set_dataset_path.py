import json

dataset_path = {}
dataset_path['robot_push_jpgs_h5_train'] = '/datasets/robot_push_h5/robot_push_jpgs.h5'
dataset_path['robot_push_jpgs_h5_test'] = '/datasets/robot_push_h5/robot_push_testnovel_jpgs.h5'
dataset_path['robot_push_traj_h5_train'] = '/trajectories/rp/traj_stor_train.h5'
dataset_path['robot_push_traj_h5_test'] = '/trajectories/rp/traj_stor_test.h5'
dataset_path['ucf101_jpgs'] = '/datasets/UCF101_seq/UCF-101'
dataset_path['ucf101_traj_h5_train'] = '/trajectories/ucf/traj_stor_train.h5'
dataset_path['ucf101_traj_h5_test'] = '/trajectories/ucf/traj_stor_test.h5'
dataset_path['kitti_traj_h5_train'] = '/trajectories/kitti/traj_stor_train.h5'
dataset_path['kitti_traj_h5_test'] = '/trajectories/kitti/traj_stor_test_dense.h5'
dataset_path['kitti_png'] = '/datasets/KITTI/dataset/sequences'
dataset_path['kitti_bmp'] = '/datasets/KITTI_bmp/dataset/sequences'

with open('../dataset_path.json', 'w') as f:
    json.dump(dataset_path, f)
    
with open('../dataset_path.json', 'r') as f:
     data = json.load(f)
     
print(data)
