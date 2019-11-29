Prereqs:
	pip3 install moviepy
	pip3 install torch==0.3.1 torchvision==0.2.0
	pip3 install ffmpeg
Notes:
	- I provided the zip files in addition to folders containing code. You can use either, but the instructions below discuss using the zip files since you'll need to move them to a VM instance anyway, and it's much easier to copy a zip file to an instance instead of a folder

1) Assume you are located at the root directory
	- I created a Google Cloud instance, and my username is cs8803throwaway2
	- The root directory is /home/cs8803throwaway2/
2) There should be a disk2/ folder containing the THUMOS dataset (within disk2/ there is only the THUMOS14/ folder)
	- This disk2/ folder is obtained by using Chen's THUMOS image on Google Cloud and copying all of his contents into the root directory
		- i.e. use the following command after cloning his VM instance: "cp -avr ../vltava/disk2/ ./" 
	- It's structure should be as follows:
		- /home/cs8803throwaway2/disk2/THUMOS14/videos/ : there are three subfolders named test/, UCF101/, and val/
		- /home/cs8803throwaway2/disk2/THUMOS14/frames/ : there are three subfolders named test/, UCF101/, and val/
		- /home/cs8803throwaway2/disk2/THUMOS14/annotations_test/ : some annotations
		- /home/cs8803throwaway2/disk2/THUMOS14/annotations_val/ : some annotations
		- /home/cs8803throwaway2/disk2/THUMOS14/annotations_val_readme.txt : some readme
		- /home/cs8803throwaway2/disk2/THUMOS14/category.txt : not really important right now
	- Once you have the /home/cs8803throwaway2/disk2/THUMOS14/ directory, extract all the contents of thumos-files.zip into /home/cs8803throwaway2/disk2/THUMOS14
		- I think the following command should work (assuming you're at the root directory): "unzip thumos-files.zip -d /home/cs8803throwaway2/disk2/THUMOS14/"
3) Navigate to /home/cs8803throwaway2/disk2/THUMOS14/ if not there already
	- We need to transform the given THUMOS14 dataset into a dataset that's compatible with our TRN
	- Call "python3 transform_thumos_dataset.py" from this directory
	- This will extract the subclips containing actual activities from the longer untrimmed videos
		- This creates a new directory at /home/cs8803throwaway2/disk2/THUMOS14/new_videos/
	- This program also creates a couple new files that aren't necessarily that important, but just required to train the TRN
4) Run "python3 extract_thumos_frames.py"
	- This uses the /home/cs8803throwaway2/disk2/THUMOS14/new_videos/ directory to create /home/cs8803throwaway2/disk2/THUMOS14/new_frames/
5) Extract TRN-pytorch-new.zip into /home/cs8803throwaway2/ so you now have the /home/cs8803throwaway2/TRN-pytorch/ directory
	- Run the following command: "unzip TRN-pytorch.zip -d ./TRN-pytorch/"
6) Navigate to /home/cs8803throwaway2/TRN-pytorch/
7) Open the datasets_video.py file
	- i.e. run the following command: "vim datasets_video.py"
8) Only look at the return_thumos_fs_dataset(...) and return_thumos_dataset(...) functions. In both functions, modify the username in the 'root_data' variable to be your account's username instead of cs8803throwaway2
	- For example, if your google cloud username is wendi5, then set 'root_data' to /home/wendi5/disk2/THUMOS14/new_frames/fs
	- Do a similar thing for the ROOT_DATASET variable at the very top of the program (should be 7th file)
9) Run the following command to train the program, and run it from the TRN-pytorch directory:
	- "CUDA_VISIBLE_DEVICES=0,1 python3 main.py thumos RGB --arch BNInception --num_segments 3 --consensus_type TRN --batch-size 16 --epochs 200 --learning-rate 0.0001 --dropout 0.5"
	- Remove the quotes from the above command, and only change the hyperparameters
		- You can find more hyperparameter choices in /home/cs8803throwaway2/TRN-pytorch/opts.py
	- One thing to note is I found using a batch size greater than 16 crashes the program due to memory allocation issues, but if you're using a lot of RAM you could probably have batch size of 32, 64, or 128. I think I'm using 8GB RAM or something like that
	- I don't know what the num_segments parameter does, but one thing to note is don't change the "thumos", "RGB", --arch, or --consensus-type command-line arguments

NOTES:
	- Included in this zip file is the process_dataset.py file. All this does is generate the ss_val_videofolder.txt, ss_train_videofolder.txt, fs_train_videofolder.txt, and fs_val_videofolder.txt files, which are provided in this thumos-files.zip file already. If those files don't exist, then run "python3 process_dataset.py" from the /home/cs8803throwaway2/disk2/THUMOS14/ directory
	- In the TRN-pytorch/ directory you extracted above, you'll notice a VERY_IMPORTANT_OUT.txt file. You can pretty much just ignore that. I was using that when I was redirecting the output of the model's training because there's no formal logging system set up. If you want, you can do the same thing as I did by running "CUDA_VISIBLE_DEVICES=0,1 python3 main.py thumos RGB --arch BNInception --num_segments 3 --consensus_type TRN --batch-size 16 --epochs 200 --learning-rate 0.0001 --dropout 0.5 > VERY_IMPORTANT_OUTPUT.txt" instead of the command provided in instruction 9 above