"""
Генерирует три JSON‑файла:
  data_bccd/BCCD_train.json
  data_bccd/BCCD_val.json
  data_bccd/BCCD_test.json
в формате, который ждёт оригинальный train.py из репозитория SSD.
"""
import sys, pathlib
sys.path.append('/content/ssd')   # путь к корню проекта
import json, random, xml.etree.ElementTree as ET
from pathlib import Path
from utils_bccd import label_map

root = Path('bccd_raw/BCCD')
imgs   = sorted((root / 'JPEGImages').glob('*.[jJpP][pPnN][gG]'))
annots = sorted((root / 'Annotations').glob('*.xml'))
assert len(imgs) == len(annots), "число картинок и xml не совпадает"

idx = list(range(len(imgs)))
random.shuffle(idx)

n = len(idx)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)

splits = {
    'train': idx[:n_train],
    'val'  : idx[n_train:n_train+n_val],
    'test' : idx[n_train+n_val:],
}

def export(name, indices):
    images, objects = [], []
    for i in indices:
        img_p = imgs[i]
        xml_p = annots[i]

        boxes, labels, diff = [], [], []
        tree = ET.parse(xml_p).getroot()
        for obj in tree.iter('object'):
            cls = obj.find('name').text
            if cls not in label_map:          # пропускаем лишние классы
                continue

            bnd = obj.find('bndbox')
            xmin = int(float(bnd.find('xmin').text))
            ymin = int(float(bnd.find('ymin').text))
            xmax = int(float(bnd.find('xmax').text))
            ymax = int(float(bnd.find('ymax').text))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_map[cls])
            diff.append(0)                    # BCCD — все объекты «легкие»

        images.append(str(img_p.resolve()))
        objects.append({'boxes': boxes,
                        'labels': labels,
                        'difficulties': diff})

    out = Path('ssd/data_bccd')
    out.mkdir(exist_ok=True)

    with open(out / f'{name.upper()}_images.json', 'w') as f:
      json.dump(images, f)

    with open(out / f'{name.upper()}_objects.json', 'w') as f:
      json.dump(objects, f)
    print(f'✓ {name}: {len(images)} imgs')

for split_name, split_idx in splits.items():
    export(split_name, split_idx)
