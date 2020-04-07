import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import argparse
import time


def model(training_data, validation_data, dense_layers, num_nodes, epochs, learning_rate, callback=None):
	feature_columns = [
		tf.feature_column.numeric_column('position', shape=(1, 8, 8, 12), dtype=tf.dtypes.int64),
		tf.feature_column.numeric_column('turn', shape=(1,), dtype=tf.dtypes.int64),
		tf.feature_column.numeric_column('elo', shape=(1,), dtype=tf.dtypes.float32)
	]

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.DenseFeatures(feature_columns))

	for i in range(dense_layers):
		model.add(tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu))

	model.add(tf.keras.layers.Dropout(0.2))

	model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
				  loss='binary_crossentropy',
				  metrics=['accuracy', tf.metrics.Recall(), tf.metrics.Precision()])

	model.fit(training_data,
			  validation_data=validation_data,
			  epochs=epochs,
			  callbacks=callbacks)

	return model


def _parse_function(raw_data):
	feature_description = {
		'position': tf.io.FixedLenFeature((), tf.string),
		'turn': tf.io.FixedLenFeature((), tf.int64),
		'elo': tf.io.FixedLenFeature((), tf.float32),
		'label': tf.io.FixedLenFeature((), tf.int64)
	}

	example = tf.io.parse_single_example(raw_data, feature_description)

	raw_position = example['position']

	position = tf.io.decode_raw(raw_position, tf.int64)
	position = tf.reshape(position, tf.stack([8, 8, 12]))

	turn = example['turn']
	elo = example['elo']
	label = example['label']

	return dict({'position': [position], 'elo': [elo], 'turn': [turn]}), [label]


def load_datasets(training_data_paths, validation_data_paths, test_data_paths):
	raw_dataset = tf.data.TFRecordDataset(training_data_paths)
	train_ds = raw_dataset.map(_parse_function)
	train_ds = train_ds.shuffle(buffer_size=256)

	raw_validation_set = tf.data.TFRecordDataset(validation_data_paths)
	val_ds = raw_validation_set.map(_parse_function)
	val_ds = val_ds.shuffle(buffer_size=256)

	raw_test_dataset = tf.data.TFRecordDataset(test_data_paths)
	test_ds = raw_test_dataset.map(_parse_function)
	test_ds = test_ds.shuffle(buffer_size=256)

	return train_ds, val_ds, test_ds


def download_part(part, s3_url, namespace):
	part_prefix = 'part-r-'
	part_suffix = str(part)
	while len(part_suffix) < 5: part_suffix = '0' + part_suffix
	part_file = part_prefix + part_suffix

	url = '{}/{}'.format(s3_url, part_file)
	file_path = tf.keras.utils.get_file('{}.{}'.format(namespace, part_file), url)
	return file_path


def load_remote_training_data(s3_url, parts, namespace='main'):
	if parts < 3:
		raise ValueError('There must be at least 3 parts in the dataset')

	test_part = parts - 1
	val_part = parts - 2
	test_parts = range(parts - 2)

	training_data = []
	for part in test_parts:
		file_path = download_part(part, s3_url, namespace)
		training_data.append(file_path)

	validation_data = []
	val_file_path = download_part(val_part, s3_url, namespace)
	validation_data.append(val_file_path)

	test_data = []
	test_file_path = download_part(test_part, s3_url, namespace)
	test_data.append(test_file_path)

	return load_datasets(training_data, validation_data, test_data)


def load_local_training_data():
	training_data_files = ['input-medium/part-r-00000']
	validation_data_files = ['input-medium/part-r-00001']
	test_data_files = ['input-medium/part-r-00002']

	return load_datasets(training_data_files, validation_data_files, test_data_files)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', type=int, default=1000)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--parts', type=str, default='3')
	parser.add_argument('--learning-rate', type=int, default=.00001)
	parser.add_argument('--dense-layers', type=int, default=2)
	parser.add_argument('--num-nodes', type=int, default=1024)

	parser.add_argument('--train', type=str)
	parser.add_argument('--tensorboard', type=bool, default=False)

	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()

	# load all datasets
	if args.train is None:
		train_ds, val_ds, test_ds = load_local_training_data()
	else:
		train_ds, val_ds, test_ds = load_remote_training_data(args.train, int(args.parts))

	# batch the datasets
	train_ds = train_ds.batch(args.batch_size)
	val_ds = val_ds.batch(args.batch_size)
	test_ds = test_ds.batch(args.batch_size)

	if args.tensorboard:
		name = 'LARGE-SET-blunder-predictor-{}-batch-{}-dense-{}-nodes-0.4-dropout-{}-learning-rate-{}'.format(
			args.batch_size, args.dense_layers, args.num_nodes, args.learning_rate, int(time.time()))

		tensorboard = TensorBoard(log_dir='C:\\logs\\{}'.format(name))
		callbacks = [tensorboard]
	else:
		callbacks = None

	# create and train the model
	model = model(train_ds, val_ds, args.dense_layers, args.num_nodes, args.epochs, args.learning_rate, callbacks)

	model.summary()

	model.save('blunder-predictor.model')

	# evaluate model
	model.evaluate(test_ds)
