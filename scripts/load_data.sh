# get 101 categories data
wget -P data http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
tar -xvzf data/101_ObjectCategories.tar.gz -C data
rm data/101_ObjectCategories.tar.gz

# get drawn data
wget -O data/drawings.tar.gz https://drive.google.com/uc?id=0B5qJKyeNZGIbSUtlQzBoLTN3Q2M
tar -xvzf data/drawings.tar.gz -C data
rm data/drawings.tar.gz
