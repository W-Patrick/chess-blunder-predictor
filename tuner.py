import os

training_command = 'python model.py --batch-size {} --epochs 15 --train "https://pwalanki-blunder-predictor-data.s3.us-east-1.amazonaws.com/output-large" --dense-layers {} --dropout {} --learning-rate {} --num-nodes {} --parts 9 --tensorboard True'


# CHANGE THESE VALUES DEPENDING ON
# WHAT HYPERPARAMETERS YOU WANT TO 
# TRAIN AND OBSERVE
dense_layers = [4]
num_nodes = [2048]
batch_sizes = [50, 100, 1000]
learning_rates = [.000001, .0000001]
dropouts = [0.2]

for dense_layer in dense_layers:
	for num_node in num_nodes:
		for batch_size in batch_sizes:
			for learning_rate in learning_rates:
				for dropout in dropouts:
					command = training_command.format(batch_size, dense_layer, dropout, learning_rate, num_node)
					os.system(command)
