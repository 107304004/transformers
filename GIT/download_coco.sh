mkdir ../data/COCO2017

wget http://images.cocodataset.org/zips/train2017.zip -P ../data/COCO2017/
wget http://images.cocodataset.org/zips/val2017.zip -P ../data/COCO2017/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ../data/COCO2017/

unzip ../data/COCO2017/train2017.zip -d ../data/COCO2017/
unzip ../data/COCO2017/val2017.zip -d ../data/COCO2017/
unzip ../data/COCO2017/annotations_trainval2017.zip -d ../data/COCO2017/
