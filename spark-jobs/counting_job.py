import sys
from pyspark import SparkContext
import chess.pgn
import io

if __name__ == '__main__':
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]

	sc = SparkContext(appName="PgnCount")
	sc._jsc.hadoopConfiguration().set("textinputformat.record.delimiter", "\n\n[Event")
	text_file = sc.textFile(input_dir)

	# we need to use \n\n[Event as our delimiter because of PGN's specific format
	# then we need to fix the records so that an individual game is a single record
	# that a mapper processes
	def fix_record(record):
		if record.startswith('[Event'):
			return record + '\n'
		else:
			return ''.join(['[Event', record, '\n'])

	def analyze_game(record):
		pgn = io.StringIO(record)
		game = chess.pgn.read_game(pgn)

		analysis = []
		analysis.append((game.headers['Result'], 1))

		board = game.board()
		for move in first_game.mainline_moves():
			board.push(move)

		return analysis

	res = text_file.map(fix_record).flatmap(analyze_game).reduceByKey(lambda a,b : a + b)
	res.saveAsTextFile(output_dir)