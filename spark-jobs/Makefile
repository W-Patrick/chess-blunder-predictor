input.dir = counting-input
output.dir = output

clean-output:
	rm -rf output

run: clean-output
	spark-submit counting_job.py ${input.dir} ${output.dir}
