import os

image_dir = 'VOC2012/JPEGImages'
label_dir = 'VOC2012/SegmentationClassRaw'  # 建議先轉成灰階 index label
list_file = 'train_list.txt'

with open(list_file, 'w') as f:
    for fname in os.listdir(image_dir):
        if not fname.endswith('.jpg'):
            continue
        name = fname[:-4]
        image_path = os.path.join(image_dir, name + '.jpg')
        label_path = os.path.join(label_dir, name + '.png')
        if os.path.exists(label_path):
            f.write(f'{image_path} {label_path}\n')
