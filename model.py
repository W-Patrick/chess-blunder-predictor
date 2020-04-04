import tensorflow as tf
import numpy as np


def model(dataset):
	model = tf.keras.models.Sequential()

	feature_columns = []
	feature_columns.append(tf.feature_column.numeric_column('position', shape=(768,), dtype=tf.dtypes.int64))
	feature_columns.append(tf.feature_column.numeric_column('turn', shape=(1,), dtype=tf.dtypes.int64))
	feature_columns.append(tf.feature_column.numeric_column('elo', shape=(1,), dtype=tf.dtypes.float32))

	model.add(tf.keras.layers.DenseFeatures(feature_columns))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(2310, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(2310, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

	model.compile(optimizer='adam',
				  loss='binary_crossentropy',
				  metrics=['accuracy'])

	model.fit(dataset)
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

	turn = example['turn']
	elo = example['elo']
	label = example['label']

	return dict({'position': [position], 'turn': [turn], 'elo': [elo]}), [label]


if __name__ == '__main__':
	filenames = ['part-r-00000', 'part-r-00001']
	raw_dataset = tf.data.TFRecordDataset(filenames)

	dataset = raw_dataset.map(_parse_function)

	model = model(dataset)
	model.save('my_first_model.model')
	

	
