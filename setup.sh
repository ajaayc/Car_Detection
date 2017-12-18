conda info --envs
conda create -n venv
source activate venv
conda install spyder tensorflow jupyter matplotlib PIL pillow
conda install -c conda-forge tensorflow
conda list
sudo apt-get install protobuf-compiler
#source deactivate
#conda env remove test_env

#Follow setup instructions at. Need protoc
#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

#Change the ./core/box_list_ops.py maximum_box_variable to prevent assertions
