# get dependencies
pip install tqdm
pip install moviepy
sudo apt-get install ffmpeg

# get project
git clone https://github.com/mwolfram/CarND-Semantic-Segmentation.git

# prep kitti data
cd CarND-Semantic-Segmentation
cd data
wget http://kitti.is.tue.mpg.de/kitti/data_road.zip
unzip data_road.zip

mkdir test_videos
cd test_videos
wget https://www.dropbox.com/s/9iiy7u4nv5l77ww/hart1.mp4?dl=1

cd ..
cd ..

# run project
python main.py
