import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np
import argparse
import time
import os
import json
import math
import tarfile
import requests


def sigmoid_loss_with_weight(labels, logits, pos_weight):
	return tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight)


def compile_model(model, weighted_loss, weight, learning_rate):
	if weighted_loss:
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
					  loss=lambda labels, logits: sigmoid_loss_with_weight(labels, logits, weight),
					  metrics=['accuracy', tf.metrics.Recall(), tf.metrics.Precision()])
	else:
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
					  loss='binary_crossentropy',
					  metrics=['accuracy', tf.metrics.Recall(), tf.metrics.Precision()])


def model(training_data, validation_data, dense_layers, num_nodes, epochs, learning_rate, dropout, weighted_loss=False, weight=1.0, callbacks=None):
	feature_columns = [
		tf.feature_column.numeric_column('position', shape=(1, 8, 8, 12), dtype=tf.dtypes.int64),
		tf.feature_column.numeric_column('turn', shape=(1,), dtype=tf.dtypes.int64),
		tf.feature_column.numeric_column('elo', shape=(1,), dtype=tf.dtypes.float32)
	]

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.DenseFeatures(feature_columns))

	for i in range(dense_layers):
		model.add(tf.keras.layers.Dense(num_nodes, activation=tf.nn.relu))

	model.add(tf.keras.layers.Dropout(dropout))

	model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

	compile_model(model, weighted_loss, weight, learning_rate)

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
	label = tf.cast(example['label'], tf.float32)

	return dict({'position': [position], 'elo': [elo], 'turn': [turn]}), [label]


def load_datasets(training_data_paths, validation_data_paths, test_data_paths, compressed):
	compression_type = 'GZIP' if compressed else None

	raw_dataset = tf.data.TFRecordDataset(training_data_paths, compression_type=compression_type)
	train_ds = raw_dataset.map(_parse_function)
	train_ds = train_ds.shuffle(buffer_size=256)

	raw_validation_set = tf.data.TFRecordDataset(validation_data_paths, compression_type=compression_type)
	val_ds = raw_validation_set.map(_parse_function)
	val_ds = val_ds.shuffle(buffer_size=256)

	raw_test_dataset = tf.data.TFRecordDataset(test_data_paths, compression_type=compression_type)
	test_ds = raw_test_dataset.map(_parse_function)
	test_ds = test_ds.shuffle(buffer_size=256)

	return train_ds, val_ds, test_ds


def load_part(part, s3_url, aws, namespace, compressed):
	part_prefix = 'part-r-'
	part_suffix = str(part)
	while len(part_suffix) < 5: part_suffix = '0' + part_suffix
	part_file = part_prefix + part_suffix
	if compressed:
		part_file += '.gz'

	url = '{}/{}'.format(s3_url, part_file)

	# check if we are on sagemaker
	if aws:
		file_path = url
	else:
		file_path = tf.keras.utils.get_file('{}.{}'.format(namespace, part_file), url)

	return file_path


def load_remote_training_data(s3_url, parts, aws, namespace='main', compressed=True):
	if parts < 3:
		raise ValueError('There must be at least 3 parts in the dataset')

	training_split = .7

	training_parts_upper = int(parts * training_split)
	train_parts = range(training_parts_upper)

	remaining_parts = parts - len(train_parts)
	num_validation_parts = math.ceil(remaining_parts / 2)
	validation_parts_upper = training_parts_upper + num_validation_parts

	val_parts = range(training_parts_upper, validation_parts_upper)
	test_parts = range(validation_parts_upper, parts)

	training_data = []
	for part in train_parts:
		file_path = load_part(part, s3_url, aws, namespace, compressed)
		training_data.append(file_path)

	validation_data = []
	for part in val_parts:
		val_file_path = load_part(part, s3_url, aws, namespace, compressed)
		validation_data.append(val_file_path)

	test_data = []
	for part in test_parts:
		test_file_path = load_part(part, s3_url, aws, namespace, compressed)
		test_data.append(test_file_path)

	return load_datasets(training_data, validation_data, test_data, compressed)


def evaluate_model(model, test_ds, model_name='model'):
	test_loss, test_acc, test_recall, test_precision = model.evaluate(test_ds)
	print('\n{}: test_loss={} test_acc={} test_recall={} test_precision={}\n'.format(model_name, test_loss, test_acc, test_recall, test_precision))


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--aws', type=bool, default=False)

	parser.add_argument('--batch-size', type=int, default=1000)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--parts', type=str, default='3')
	parser.add_argument('--learning-rate', type=float, default=.00001)
	parser.add_argument('--dense-layers', type=int, default=2)
	parser.add_argument('--num-nodes', type=int, default=1024)
	parser.add_argument('--dropout', type=float, default=0.2)
	parser.add_argument('--tensorboard', type=bool, default=False)

	parser.add_argument('--model_dir', type=str)
	parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR') if 'SM_MODEL_DIR' in os.environ else None)
	parser.add_argument('--train', type=str)
	parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')) if 'SM_HOSTS' in os.environ else None)
	parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST') if 'SM_CURRENT_HOST' in os.environ else None)

	parser.add_argument('--early-stopping', type=bool, default=False)
	parser.add_argument('--patience', type=int, default=100)

	parser.add_argument('--compressed', type=bool, default=True)
	parser.add_argument('--continue-training', type=bool, default=False)
	parser.add_argument('--model-location', type=str)
	parser.add_argument('--model-name', type=str)

	parser.add_argument('--weighted-loss', type=bool, default=False)
	parser.add_argument('--weight', type=float, default=1.0)

	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()

	model_output_dir = args.sm_model_dir if args.sm_model_dir is not None else 'models'
	model_output_fp = os.path.join(model_output_dir, 'blunder-predictor.model')
	model_best_acc_fp = os.path.join(model_output_dir, 'best-accuracy.model')
	model_best_recall_fp = os.path.join(model_output_dir, 'best-recall.model')
	model_best_precision_fp = os.path.join(model_output_dir, 'best-precision.model')

	# need to check the environment variables here because they are slightly different depending
	# on whether you are running a regular training or a tuning job
	if args.train is None:
		if 'SM_CHANNEL_TRAINING' in os.environ:
			train = os.environ.get('SM_CHANNEL_TRAINING')
		elif 'SM_CHANNEL_TRAIN' in os.environ:
			train = os.environ.get('SM_CHANNEL_TRAIN')
		else:
			train = args.train
	else:
		train = args.train

	split_train_name = args.train.split('/')
	namespace = 'main' if len(split_train_name) == 0 else split_train_name[len(split_train_name) - 1]
	# load all datasets
	train_ds, val_ds, test_ds = load_remote_training_data(train, int(args.parts), args.aws, namespace=namespace, compressed=args.compressed)

	# batch the datasets
	train_ds = train_ds.batch(args.batch_size)
	val_ds = val_ds.batch(args.batch_size)
	test_ds = test_ds.batch(args.batch_size)

	# define callbacks
	callbacks = []

	if args.early_stopping:
		print('EARLY STOPPING IS TURNED ON')
		early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.patience)
		
		accuracy_checkpoint = ModelCheckpoint(
			model_best_acc_fp, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True) 

		recall_checkpoint = ModelCheckpoint(
			model_best_recall_fp, monitor='val_recall', mode='max', verbose=1, save_best_only=True)

		precision_checkpoint = ModelCheckpoint(
			model_best_precision_fp, monitor='val_precision', mode='max', verbose=1, save_best_only=True)

		callbacks.append(early_stopping)
		callbacks.append(accuracy_checkpoint)
		callbacks.append(recall_checkpoint)
		callbacks.append(precision_checkpoint)

	if args.tensorboard:
		name = 'WEIGHT_REDUCTION_TRAINING-{}-batch-{}-dense-{}-nodes-{}-dropout-{}-learning-rate-{}-weighted_loss-{}-weight-{}-{}'.format(
			args.batch_size, args.dense_layers, args.num_nodes, args.dropout, args.learning_rate, args.weighted_loss, args.weight, args.model_name, int(time.time()))

		tensorboard = TensorBoard(log_dir='C:\\logs\\{}'.format(name))
		callbacks.append(tensorboard)

	if args.continue_training:
		if args.model_location is None:
			raise ValueError('You need to specify a model location to continue training')

		print('Downloading: {}'.format(args.model_location))
		tarred_model = requests.get(args.model_location)
		open('models.tar.gz', 'wb').write(tarred_model.content)

		tarfile.open('models.tar.gz').extractall()
	
		model = tf.keras.models.load_model(args.model_name)
		compile_model(model, args.weighted_loss, args.weight, args.learning_rate)

		model.fit(train_ds,
			validation_data=val_ds,
			epochs=args.epochs,
			callbacks=callbacks)
	else:
		# create and train the model
		model = model(train_ds, val_ds, args.dense_layers, args.num_nodes, args.epochs, args.learning_rate, args.dropout, args.weighted_loss, args.weight, callbacks)
		
	if args.early_stopping:
		# load models and evaluate
		acc_model = tf.keras.models.load_model(model_best_acc_fp)
		recall_model = tf.keras.models.load_model(model_best_recall_fp)

		evaluate_model(acc_model, test_ds, 'acc_model')
		evaluate_model(recall_model, test_ds, 'recall_model')
	else:
		# evaluate model
		evaluate_model(model, test_ds)
	
	# save the final model
	model.save(model_output_fp)
