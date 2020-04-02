import sys
from pyspark import SparkContext, AccumulatorParam
import chess.pgn
import io
import math
import logging

# class created to accumulate counts of
# dynamic names, for example this will
# help us count how much of each time format
# we have in the dataset without actually
# knowing what all the time formats are
class DictAccumulator(AccumulatorParam):	
	def zero(self, initialValue):
		return initialValue
	
	def addInPlace(self, v1, v2):	
		for k in v2:
			if k not in v1:
				v1[k] = 0
			
			v1[k] += v2[k]

		return v1


def round_elo(elo):
	return math.floor(elo / 100.0) * 100

WHITE_ELO = 'WhiteElo'
BLACK_ELO = 'BlackElo'
TIME_CONTROL = 'TimeControl'
TERMINATION = 'Termination'
VARIANT = 'Variant'
EVAL = 'eval'

if __name__ == '__main__':
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]

	sc = SparkContext(appName="PgnCount")
	sc._jsc.hadoopConfiguration().set("textinputformat.record.delimiter", "\n\n[Event")

	text_file = sc.textFile(input_dir)

	# accumulators
	time_control_counts = sc.accumulator({}, DictAccumulator())
	elo_counts = sc.accumulator({}, DictAccumulator())
	termination_counts = sc.accumulator({}, DictAccumulator())
	variant_counts = sc.accumulator({}, DictAccumulator())
	eval_counts = sc.accumulator({}, DictAccumulator())

	records_analyzed = 0
	batch_size = 10

	def analyze_game(record):
		pgn = io.StringIO(record)
		game = chess.pgn.read_game(pgn)

		if WHITE_ELO in game.headers:
			try:
				white_elo = round_elo(int(game.headers[WHITE_ELO]))
				elo_counts.add({white_elo: 1})
			except Exception:
				white_elo = game.headers[WHITE_ELO]
				elo_counts.add({game.headers[WHITE_ELO]: 1})

		if BLACK_ELO in game.headers:
			try:
				black_elo = round_elo(int(game.headers[BLACK_ELO]))
				elo_counts.add({black_elo: 1})
			except Exception:
				black_elo = game.headers[BLACK_ELO]
				elo_counts.add({game.headers[BLACK_ELO]: 1})

		if TIME_CONTROL in game.headers:
			time_control = game.headers[TIME_CONTROL]
			time_control_counts.add({time_control: 1})

		if TERMINATION in game.headers:
			termination = game.headers[TERMINATION]
			termination_counts.add({termination: 1})

		if VARIANT in game.headers:
			variant = game.headers[VARIANT]
			variant_counts.add({variant: 1})

		if len(game.variations) > 0:
			comment = game.variations[0].comment

			# game has engine evaluation
			if '%eval' in comment:
				eval_counts.add({EVAL: 1})
				eval_counts.add({EVAL + '-' + time_control: 1})

				if WHITE_ELO in game.headers:
					eval_counts.add({EVAL + '-' + str(white_elo): 1})

				if BLACK_ELO in game.headers:
					eval_counts.add({EVAL + '-' + str(black_elo): 1})

		global records_analyzed
		global batch_size
		records_analyzed += 1
		if records_analyzed % batch_size == 0:
			print('Number of records processed: {}'.format(records_analyzed))

	# we need to use \n\n[Event as our delimiter because of PGN's specific format
	# then we need to fix the records so that an individual game is a single record
	# that a mapper processes
	def fix_record(record):
		if record.startswith('[Event'):
			return record + '\n'
		else:
			return ''.join(['[Event', record, '\n'])

	res = text_file.map(fix_record)
	res.foreach(analyze_game)

	def transform_counter(name, counter, result):
		result.append(','.join([name, 'Count']))
		for k in counter.value:
			result.append(','.join([str(k), str(counter.value[k])]))

	def transform_counter_to_csv():
		csv_lines = []
		transform_counter('Elo', elo_counts, csv_lines)
		transform_counter('TimeControl', time_control_counts, csv_lines)
		transform_counter('Termination', termination_counts, csv_lines)
		transform_counter('Eval', eval_counts, csv_lines)
		transform_counter('Variant', variant_counts, csv_lines)
		return csv_lines

	sc.parallelize(transform_counter_to_csv(), 1).saveAsTextFile(output_dir)
