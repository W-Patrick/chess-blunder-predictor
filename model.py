import tensorflow as tf
import numpy as np


def model(dataset):
	feature_columns = []
	feature_columns.append(tf.feature_column.numeric_column('position', shape=(1, 8, 8, 12), dtype=tf.dtypes.int64))
	feature_columns.append(tf.feature_column.numeric_column('turn', shape=(1,), dtype=tf.dtypes.int64))
	feature_columns.append(tf.feature_column.numeric_column('elo', shape=(1,), dtype=tf.dtypes.float32))

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.DenseFeatures(feature_columns))
	model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

	model.compile(optimizer='adam',
				  loss='binary_crossentropy',
				  metrics=["accuracy"])

	model.fit(dataset, epochs=10)
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
	filenames = ['part-r-00000', 'part-r-00001']
	raw_dataset = tf.data.TFRecordDataset(filenames)

	dataset = raw_dataset.map(_parse_function)

	# dataset = dataset.shuffle(buffer_size=256)
	# dataset = dataset.repeat(3)
	# dataset = dataset.batch(15)

	training_dataset = dataset.take(2000)
	test_dataset = dataset.skip(2000).take(677)

	model = model(training_dataset)
	model.save('my_first_model.model')

	loss, accuracy = model.evaluate(test_dataset, verbose=0)
	print('LOSS:')
	print(loss)
	print('ACCURACY:')
	print(accuracy)
