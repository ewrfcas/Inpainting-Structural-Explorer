from glob import glob
# random mask with COCO mask and brush mask (50%)
from PIL import Image, ImageDraw
import math
import numpy as np

H = 256
W = 256

min_num_vertex = 4
max_num_vertex = 12
mean_angle = 2 * math.pi / 5
angle_range = 2 * math.pi / 15
min_width = 12 * (H // 256)
max_width = 40 * (H // 256)

min_brush = 2
max_brush = 5

random_rate = 0.5

input_path = 'celeba/val_list.txt'
output_path = 'celeba/test_mask'

import os

os.makedirs(output_path, exist_ok=True)

valid_list = []
with open(input_path, 'r') as f:
    for line in f:
        valid_list.append(line.strip())

coco_valmask_list = glob('../../coco_mask/val/*')
print(len(coco_valmask_list))


def generate_mask(H, W):
    average_radius = math.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(min_brush, max_brush + 1)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2,
                          v[1] - width // 2,
                          v[0] + width // 2,
                          v[1] + width // 2),
                         fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (H, W, 1))
    return mask


import random

random.seed(556)
np.random.seed(556)

random.shuffle(coco_valmask_list)

from tqdm import tqdm
import cv2
import os

i = 0
for p in tqdm(valid_list):
    if random.random() <= random_rate:
        mask = generate_mask(H, W)
    else:
        mask = cv2.imread(coco_valmask_list[i]) / 255
        mask = cv2.resize(mask, (H, W), interpolation=cv2.INTER_NEAREST)
        i += 1

    mask = np.ones((H, W, 3), dtype=np.uint8) * 255 * mask
    cv2.imwrite(os.path.join(output_path, p.split('/')[-1].replace('.jpg', '.png')), mask)
