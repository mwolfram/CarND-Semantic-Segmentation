# get dependencies
pip install tqdm

# get project
git clone https://github.com/mwolfram/CarND-Semantic-Segmentation.git

# prep kitti data
cd CarND-Semantic-Segmentation
cd data
wget http://kitti.is.tue.mpg.de/kitti/data_road.zip
unzip data_road.zip

cd ..

# run project
python main.py
