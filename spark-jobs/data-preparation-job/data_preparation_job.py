import sys
from pyspark import SparkContext, AccumulatorParam
from pyspark.sql import SQLContext
import io
import math
import datetime
import numpy as np
import chess.pgn
import random


def round_elo(elo):
	return math.floor(elo / 100.0) * 100


def get_elo_from_str(elo):
	try:
		return round_elo(int(elo))
	except Exception:
		return -1 # -1 will represent an unknown elo


def get_time_allocated(time_control):
	split_time_control = time_control.split('+')
	if len(split_time_control) > 1:
		return int(split_time_control[0])
	else:
		return -1 # -1 will represent a missing time allocation


def get_one_hot_encoding_of_board(board):
	encoding_template = [0 for i in range(12)]

	position = []
	for square in chess.SQUARES:
		piece = board.piece_at(square)

		encoding = encoding_template.copy()
		if piece is not None:
			piece_type = piece.piece_type
			# add six if its a black piece
			if not piece.color:
				piece_type += 6

			encoding[piece_type - 1] = 1

		position.append(encoding)

	return position


# gets the clock time
def get_clock_time_in_seconds(comment):
	pass
	sp = comment.split('clk')
	if len(sp) > 1:
		t = sp[1].strip().strip(']')
		hours, minutes, seconds = t.split(':')
		return datetime.timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds)).total_seconds()
	else:
		return -1 # -1 will represent an unknown clock time


WHITE_ELO = 'WhiteElo'
BLACK_ELO = 'BlackElo'
TIME_CONTROL = 'TimeControl'
TERMINATION = 'Termination'
VARIANT = 'Variant'
EVAL = 'eval'

MIN_ELO = 1200
MAX_ELO = 2200

BALANCE = True

def normalize_elo(elo):
	return ((elo - MIN_ELO) * 1.0) / (MAX_ELO - MIN_ELO)

# range of elos we will include in the dataset
VALID_ELOS = set(range(MIN_ELO, MAX_ELO + 100, 100))

# lowest time format we will consider
TIME_FORMAT_CUTOFF = 600

# all moves played with less than this amount of time will not be considered
CLOCK_CUTOFF = 120


if __name__ == '__main__':
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]

	sc = SparkContext(appName="PgnCount")
	sc._jsc.hadoopConfiguration().set("textinputformat.record.delimiter", "\n\n[Event")

	sqlContext = SQLContext(sc)

	# we need to use \n\n[Event as our delimiter because of PGN's specific format
	# then we need to fix the records so that an individual game is a single record
	# that a mapper processes
	def fix_record(record):
		if record.startswith('[Event'):
			return record + '\n'
		else:
			return ''.join(['[Event', record, '\n'])

	def contains_headers(game):
		return (WHITE_ELO in game.headers
			and BLACK_ELO in game.headers
			and TIME_CONTROL in game.headers)

	records_analyzed = 0
	batch_size = 10

	# function used to label all the positions in the game
	def label_game(record):
		pgn = io.StringIO(record)
		game = chess.pgn.read_game(pgn)

		if not contains_headers(game):
			return []

		if not len(game.variations) > 0:
			return []

		comment = game.variations[0].comment

		# game has engine evaluation
		if not '%eval' in comment:
			return []

		white_elo = get_elo_from_str(game.headers[WHITE_ELO])
		black_elo = get_elo_from_str(game.headers[BLACK_ELO])

		# if both players are not in the elo range skip the game
		if not white_elo in VALID_ELOS and not black_elo in VALID_ELOS:
			return []

		time_allocated = get_time_allocated(game.headers[TIME_CONTROL])

		# if the time control doesnt meet the cutoff skip the game
		if time_allocated < TIME_FORMAT_CUTOFF:
			return []

		blunders = []
		non_blunders = []

		# loop through the mainline of the game
		board = game.board()

		for node in game.mainline():
			# get previous positions turn
			turn = board.turn

			# get the previous turns elo
			elo = white_elo if turn else black_elo

			if elo in VALID_ELOS:

				clock_time = get_clock_time_in_seconds(node.comment)
				# was there enough time on the clock?
				if clock_time >= CLOCK_CUTOFF:

					# get the position as a one hot encoding
					position = get_one_hot_encoding_of_board(board)
					position = bytearray(np.array(position).tobytes())
	
					if chess.pgn.NAG_BLUNDER in node.nags:
						blunders.append((position, int(turn), normalize_elo(elo), 1))
					else:
						non_blunders.append((position, int(turn), normalize_elo(elo), 0))

			board.push(node.move)

		if BALANCE and len(non_blunders) > len(blunders):
			num_blunders = len(blunders)
			included_non_blunders = random.sample(non_blunders, num_blunders)

			dataset = blunders + included_non_blunders
		else:
			dataset = blunders + non_blunders

		global records_analyzed
		global batch_size
		records_analyzed += 1
		if records_analyzed % batch_size == 0:
			print('Number of records processed: {}'.format(records_analyzed))

		return dataset

	text_file = sc.textFile(input_dir)
	res = text_file.map(fix_record).flatMap(label_game)

	df = sqlContext.createDataFrame(res, ['position', 'turn', 'elo', 'label'])
	df.write.format("tfrecords").option("codec", "org.apache.hadoop.io.compress.GzipCodec").mode("overwrite").save(output_dir)
