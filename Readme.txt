#It is recommended that you set this up on an AWS instance with GPU computing power. We used the p2.8xlarge EC2 instance. It has 8 GPU's and charges ~$7.2/hour, but there is a cheaper version called the p2.xlarge that has 1 GPU and is charged at ~$0.90/hour. By default, new AWS users are not allowed to use either of these GPU instances. You will have to submit a request to Amazon to increase the allowed limit of how many of these instances you want to be able to use, as these limits are initially 0.
#We used the Deep Learning AMI (Ubuntu) Version 2.0.

#After getting permission to use the p2.8x large instance, you can ssh into it as follows:
ssh -i <key_file>.pem ubnutu@<PublicDNS>

Where the key file was given to you when you set up the instance, and the Public DNS is stated in your EC2 console. ssh from the same directory containing the .pem file.

Use df -h to see how much memory the instance has. It will probably be too little, in which case you should follow the instructions at this link to increase the disk size:
https://n2ws.com/how-to-guides/how-to-increase-the-size-of-an-aws-ebs-cloud-volume-attached-to-a-linux-machine.html

#Run each of these commands line by line

#Clone Tensorflow Models
git clone https://github.com/tensorflow/models

#Install emacs, a text editor. Or use vim if that's what you prefer.
sudo apt-get install emacs

#Edit ~/.bashrc file
emacs ~/.bashrc

#Add the following lines to the end of ~/.bashrc
cd models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ~

#Save and close

#Re-source the .bashrc file like this
source ~/.bashrc

#Edit line 780 of the following file
emacs ~/models/research/object_detection/core/box_list_ops.py
#Set maximum_normalized_coordinate=1500.0 to prevent errors in training

#Clone repo
git clone https://github.com/ajaayc/Car_Detection

#Install the project dependencies
pip install -r requirements.txt

#Install Cuda for GPU support. Instructions loosely based off of this link:
#https://docs.devicehive.com/v2.0/blog/using-gpus-for-training-tensorflow-models
pip install tensorflow-gpu
#Install Cuda Toolkit
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb"
sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda

#Next step:
#Download cuDNN Library for Linux from https://developer.nvidia.com/cudnn (note that you will need to register for the Accelerated Computing Developer Program) and copy it to your EC2 instance using scp command. It should be a .tgz file:
#We had lots of issues with version compability. We found out that we needed Cuda V6.0 in order for our code to work.

#After transferring the .tgz file to your AWS instance with scp or rync, run:
sudo tar -xvf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
export PATH=/usr/local/cuda/bin:$PATH
export  LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda

#The steps above should complete the set up process. You can test the Cuda and GPU installation with:
python
>>> import tensorflow as tf
>>> sess = tf.Session()
You should see “Found device 0 with properties: name: GRID K520”
#If errors show up, you may need to make some Google searches and/or lurk on Stackoverflow to learn how to fix them.

#Download the training data and unzip
wget http://umich.edu/~fcav/rob599_dataset_deploy.zip
unzip rob599_dataset_deploy.zip -d rob599_dataset_deploy

#Create a TFRecord file called train.record from the training data. The train.record file contains all of the data required to 
python createTFRecord_Py3.py 1 no_stats

#Runs training
python train.py --train_dir=./models/train --pipeline_config_path=test_frcnn_resnet50_coco.config

#To run training in the background even when you're not connected to the AWS instance, use nohup as follows:
nohup python train.py --train_dir=./models/train --pipeline_config_path=test_frcnn_resnet50_coco.config&

#The following lets you see the status of the training
tail -n 5 nohup.out

#To kill training
pgrep python | xargs kill -9

#When training is running in the background, you can run this command to evaluate on some of your test data. You may need to change the evaluation section of the test_frcnn_resnet50_coco.config file to set up what tfrecord file you want to evaluate the model on. You can use the createTFRecord_Py3.py script as above to create a .record file for any amount of the Grand Theft Auto data.
python eval.py --checkpoint_dir=models/train --eval_dir=eval --pipeline_config_path=test_frcnn_resnet50_coco.config&

#Run tensorboard in the background. This lets you monitor the evaluation results in real time.
tensorboard --logdir=eval --host=0.0.0.0 --port=6006&
#To view the Tensorboard results live from the AWS instance, run the following in another terminal on your local computer:
ssh -i aws_ubuntu1.pem -NL 6006:localhost:6006 ubuntu@ec2-18-217-250-20.us-east-2.compute.amazonaws.com

#Then visit localhost:6006 in your web browser to see the evaluation results

#After training is done, create, frozen_inference_graph.pb, a frozen version of the neural network that can be used to classify images. frozen_inference_graph.pb is stored in fine_tuned_model folder after running this command.
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./test_frcnn_resnet50_coco.config --trained_checkpoint_prefix ./models/train/model.ckpt-<Number of Steps you used from .config file> --output_directory ./fine_tuned_model

#Then to create the Kaggle submission, use
python eval_test.py

#Submission file is result.txt. The eval_test.py script uses the neural network stored in ./fine_tuned_model/frozen_inference_graph.pb to count the number of cars in each of the test images in rob599_dataset_deploy/test.
