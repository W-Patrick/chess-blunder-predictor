import os

training_command = 'python model.py --batch-size {} --epochs 15 --train "{}" --dense-layers {} --dropout {} --learning-rate {} --num-nodes {} --parts 9 --tensorboard True'
S3_INPUT_URL = 'https://pwalanki-blunder-predictor-data.s3.us-east-1.amazonaws.com/output-large'

# CHANGE THESE VALUES DEPENDING ON
# WHAT HYPERPARAMETERS YOU WANT TO 
# TRAIN AND OBSERVE
dense_layers = [4]
num_nodes = [2048]
batch_sizes = [50]
learning_rates = [.0000001]
dropouts = [0.0]

for dense_layer in dense_layers:
	for num_node in num_nodes:
		for batch_size in batch_sizes:
			for learning_rate in learning_rates:
				for dropout in dropouts:
					command = training_command.format(S3_INPUT_URL, batch_size, dense_layer, dropout, learning_rate, num_node)
					os.system(command)
