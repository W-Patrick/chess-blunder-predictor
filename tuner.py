import os

# learning_rates = [.0000001, .000001, .00001]
# batch_sizes = [1000, 3000, 5000]
# dropouts = [0.0, 0.2, 0.4]

dense_layers = [4, 5]
num_nodes = [2048, 4096]
batch_sizes = [1000, 5000, 10000]
learning_rates = [.0000001, .000001, .00001, .0001]
dropouts = [0.2]

training_command = 'python model.py --batch-size {} --epochs 15 --train "https://pwalanki-blunder-predictor-data.s3.us-east-1.amazonaws.com/output-large" --dense-layers {} --dropout {} --learning-rate {} --num-nodes {} --parts 8 --tensorboard True'

for dense_layer in dense_layers:
	for num_node in num_nodes:
		for batch_size in batch_sizes:
			for learning_rate in learning_rates:
				for dropout in dropouts:
					command = training_command.format(batch_size, dense_layer, dropout, learning_rate, num_node)
					os.system(command)
