import tensorflow as tf
import numpy as np


def model(training_data, validation_data):
	feature_columns = [
		tf.feature_column.numeric_column('position', shape=(1, 8, 8, 12), dtype=tf.dtypes.int64),
		tf.feature_column.numeric_column('turn', shape=(1,), dtype=tf.dtypes.int64),
		tf.feature_column.numeric_column('elo', shape=(1,), dtype=tf.dtypes.float32)
	]

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.DenseFeatures(feature_columns))
	model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

	model.compile(optimizer='adam',
				  loss='binary_crossentropy',
				  metrics=['accuracy', tf.metrics.Recall(), tf.metrics.Precision()])

	model.fit(training_data, validation_data=validation_data, epochs=5)
	return model

def load_training_data(path):
	pass

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


if __name__ == '__main__':
	filenames = ['input-medium/part-r-00000']
	raw_dataset = tf.data.TFRecordDataset(filenames)

	dataset = raw_dataset.map(_parse_function)

	dataset = dataset.shuffle(buffer_size=256).batch(32)

	raw_validation_set = tf.data.TFRecordDataset('input-medium/part-r-00001')
	val_ds = raw_validation_set.map(_parse_function)
	val_ds = dataset.shuffle(buffer_size=256)

	raw_test_dataset = tf.data.TFRecordDataset('input-medium/part-r-00002')
	test_ds = raw_test_dataset.map(_parse_function)
	test_ds = dataset.shuffle(buffer_size=256)

	model = model(dataset, val_ds)
	model.save('my_first_model.model')

	loss, accuracy, recall, precision = model.evaluate(test_ds, verbose=0)
	print('Loss: {}'.format(loss))
	print('Accuracy: {}'.format(accuracy))
	print('Recall: {}'.format(recall))
	print('Precision: {}'.format(precision))
