#Creates the TFRecord file, check if it doesn't exist first.
python createTFRecord_Py3.py 0.99 no_stats

#Runs training
python train.py --train_dir=./models/train --pipeline_config_path=test_frcnn_resnet50_coco.config

tensorboard --logdir=eventData/DIR

#Creates frozen_inference_graph.pb, a frozen version of the neural network that can be used to classify images
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./test_frcnn_resnet50_coco.config --trained_checkpoint_prefix ./models/train/model.ckpt-<Number of Steps you used> --output_directory ./fine_tuned_model
