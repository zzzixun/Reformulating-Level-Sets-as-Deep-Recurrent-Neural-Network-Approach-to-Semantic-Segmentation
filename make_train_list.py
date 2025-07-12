import os
ids = open('/home/cbel/Desktop/zixun/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt').read().split()
with open('train_list.txt','w') as f:
    for id in ids:
        img = os.path.join(IMG_DIR, id + '.jpg')
        lbl = os.path.join(LABEL_DIR, id + '.png')
        f.write(f"{img} {lbl}\n")
