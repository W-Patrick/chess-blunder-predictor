# Local Execution
spark.root=/home/patrick/tools/spark-2.4.5-bin-without-hadoop
hadoop.root=/home/patrick/tools/hadoop-3.2.1
script.name=counting_job.py
input.dir=counting-input
output.dir=output
bootstrap.script.name=bootstrap.sh

# AWS EMR Execution
aws.emr.release=emr-5.17.0
aws.bucket.name=pwalanki-blunder-predictor-data
aws.input=input
aws.output=output
aws.log.dir=log
aws.num.nodes=2
aws.instance.type=m4.large

clean-local-output:
	rm -rf output

run: clean-local-output
	spark-submit ${script.name} ${input.dir} ${output.dir}

cp-config:
	cp config/standalone/*.xml ${hadoop.root}/etc/hadoop

upload-input:
	aws s3 sync ${input.dir} s3://${aws.bucket.name}/${aws.input}

delete-output-aws:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.output}*"

upload-app:
	aws s3 cp ${script.name} s3://${aws.bucket.name}

upload-bootstrap:
	aws s3 cp ${bootstrap.script.name} s3://${aws.bucket.name}

download-output-aws: clean-local-output
	mkdir ${output.dir}
	aws s3 sync s3://${aws.bucket.name}/${aws.output} ${output.dir}

aws: upload-app upload-bootstrap delete-output-aws
	aws emr create-cluster \
		--name "Counting Job Spark Cluster" \
		--release-label ${aws.emr.release} \
		--instance-groups '[{"InstanceCount":${aws.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
	    --applications Name=Hadoop Name=Spark \
		--steps Type=CUSTOM_JAR,Name="Counting Job",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","s3://${aws.bucket.name}/${script.name}","s3://${aws.bucket.name}/${aws.input}","s3://${aws.bucket.name}/${aws.output}"] \
		--bootstrap-actions Path=s3://${aws.bucket.name}/${bootstrap.script.name},Name=InstallLibs \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--configurations '[{"Classification": "spark-env","Configurations": [{"Classification": "export","Properties": {"PYSPARK_PYTHON": "/usr/bin/python3"}}]}]' \
		--use-default-roles \
		--enable-debugging \
		--auto-terminate
		