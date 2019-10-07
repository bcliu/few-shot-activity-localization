### Preparation:

1. Download the ground truth annatations and videos in ActivityNet dataset.

	```Shell
	cd ./preprocess/activityNet/
	
	# Download the groud truth annotations in ActivityNet dataset.
	wget http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json
	
	# Download the videos in ActivityNet dataset into ./preprocess/activityNet/videos.
	python download_video.py
	```

2. Extract frames from downloaded videos in 25 fps.

	```Shell
	# training video frames are saved in ./preprocess/activityNet/frames/training/
	# validation video frames are saved in ./preprocess/activityNet/frames/validation/ 
	python generate_frames.py
	```

3. Generate the pickle data for training and testing R-C3D model.

	```Shell
  	# generate training data
	python generate_roidb_training.py
  	# generate validation data
	python generate_roidb_validation.py
  	```