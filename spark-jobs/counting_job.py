from pyspark import SparkContext
import chess

if __name__ == '__main__':
	sc = SparkContext(appName="PgnCount")
	sc._jsc.hadoopConfiguration().set("textinputformat.record.delimiter", "\n\n[Event")
	text_file = sc.textFile("counting-input")

	# we need to use \n\n[Event as our delimiter because of PGN's specific format
	# then we need to fix the records so that an individual game is a single record
	# that a mapper processes
	def fix_record(record):
		if record.startswith('[Event'):
			return record + '\n'
		else:
			return ''.join(['[Event', record, '\n'])

	res = text_file.map(fix_record)
	res.saveAsTextFile("output")