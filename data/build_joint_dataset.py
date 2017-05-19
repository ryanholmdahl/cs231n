from dataset_builder import Dataset

d = Dataset((64, 64), [0.8, 0.1, 0.1])
d.read_pairs("joint_data")
d.save_sets("joint_pairs_64")
