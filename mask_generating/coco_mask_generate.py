from tqdm import tqdm
from threadpool import ThreadPool, makeRequests
import numpy as np
from glob import glob
import cv2
import os

coco_train_stuff = '../../../source/COCO_stuff_maps/train2017/*'
coco_val_stuff = '../../../source/COCO_stuff_maps/val2017/*'
min_rate = 0.05
max_rate = 0.4
target_size = 256
output_path = '../../coco_mask'

os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)

train_files = glob(coco_train_stuff)
val_files = glob(coco_val_stuff)
print('COCO train masks:', len(train_files))
print('COCO val masks:', len(val_files))

# 1-91 thing categories (not used for stuff segmentation) (0~90)
# 92-182 stuff categories (91~181)
# 183 other category (all thing pixels) (182~)
# labels refer to https://github.com/nightrome/cocostuff/blob/master/labels.md
# 所有label要-1偏移一位

remain_list = [93, 94, 118, 121, 127, 128, 129, 142, 167]
ban_list = []
for i in range(91, 256):
    if i not in remain_list:
        ban_list.append(i)
ban_list = set(ban_list)
print('BAN Labels', ban_list)


def mask_convert(f):
    if 'train2017' in f:
        mode = 'train'
    elif 'val2017' in f:
        mode = 'val'
    else:
        raise NotImplementedError

    name = f.split('/')[-1].split('.png')[0]
    mask = cv2.imread(f)
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    unique_mask = np.unique(mask)
    for i, m in enumerate(unique_mask):
        if m in ban_list:
            continue
        sub_mask = mask.copy().astype(np.int32)
        sub_mask[sub_mask != m] = -1
        sub_mask[sub_mask == m] = 1
        sub_mask[sub_mask == -1] = 0
        sub_mask = sub_mask.astype(np.uint8)

        # 计算比例
        rate = np.sum(sub_mask) / np.prod(sub_mask.shape)
        if rate < min_rate or rate > max_rate:
            continue

        sub_mask *= 255
        cv2.imwrite(os.path.join(output_path, mode, name + '_{}.png'.format(i)), sub_mask)


with tqdm(total=len(val_files), desc='Converting val images') as pbar:
    def callback(_, x):
        pbar.update()


    t_pool = ThreadPool(12)
    requests = makeRequests(mask_convert, val_files, callback=callback)
    for req in requests:
        t_pool.putRequest(req)
    t_pool.wait()

with tqdm(total=len(train_files), desc='Converting train images') as pbar:
    def callback(_, x):
        pbar.update()


    t_pool = ThreadPool(12)
    requests = makeRequests(mask_convert, train_files, callback=callback)
    for req in requests:
        t_pool.putRequest(req)
    t_pool.wait()

print('Final train mask num', len(glob(os.path.join(output_path, 'train', '*'))))
print('Final val mask num', len(glob(os.path.join(output_path, 'val', '*'))))
