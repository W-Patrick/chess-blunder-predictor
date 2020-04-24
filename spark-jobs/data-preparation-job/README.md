# data-preparation-job

This is a spark job written in python that will prepare and label the data processed in the input PGN file and output them as
Tensorflow records.

## Setup

You could adjust how aggressive the filtering is of the script by updating the global variables at the top of the script.
Before attempting to run the job you should first update the Make file and update all of the paramaters for your local system.

## Execution

To run the job locally just run `make run`. If you would like to run the job on AWS run `make aws`, note that you will have to have an AWS
account as well as the aws cli configured in order to run this. You also need to create an S3 bucket and set the proper parameters in the 
AWS section and upload your input via `make upload-input`.
