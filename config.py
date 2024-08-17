import multiprocessing
#1.DataSet
dataset_name = "consep"
train_dir = "data/consep/toy_train/"
test_dir = "data/consep/toy_test/"

experiment = "experiments/consep/results/"
scgccm_experiment = "experiments/consep/scgccm/"

####
assemble_dir = "experiments/"

checkpoints_dir = "checkpoints"
tensorboard_logs = "run_logs"

# CoNSep Weights
inference_para_weights = "weight.pth"

device = "cuda"
batch_size = 32
epochs = 10
learning_rate = 0.01
watershed_threshold = 0.17
vis = True
num_workers = 1 #multiprocessing.cpu_count()