import argparse
import pathlib
import json
from pprint import PrettyPrinter

import torch
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader
from datasets import PascalVOCDataset
from utils import calculate_mAP
from tqdm import tqdm

# Подключаем модель и разрешаем классы
from model import SSD300, VGGBase, AuxiliaryConvolutions, PredictionConvolutions

add_safe_globals([SSD300, VGGBase, AuxiliaryConvolutions, PredictionConvolutions])

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', required=True, type=str, help='Path to .pth checkpoint')
parser.add_argument('--data_folder', required=True, type=str, help='Folder with *_images.json')
parser.add_argument('--label_map', required=True, type=str, help='utils_bccd.py with label_map')
args = parser.parse_args()

# Загрузка label_map
import importlib.util
label_path = pathlib.Path(args.label_map).expanduser()
spec = importlib.util.spec_from_file_location("label_mod", label_path)
label_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label_mod)
label_map = label_mod.label_map
rev_label_map = label_mod.rev_label_map

# Параметры
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
workers = 4
keep_difficult = True
pp = PrettyPrinter()

# Загрузка модели
checkpoint = torch.load(args.trained_model, map_location=device, weights_only=False)
model = checkpoint['model']
model = model.to(device)
model.eval()
print(f"✅ Model loaded: {args.trained_model}")

# Загрузка тестового датасета
test_dataset = PascalVOCDataset(args.data_folder, split='test', keep_difficult=keep_difficult)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

# Оценка
def evaluate(test_loader, model):
    model.eval()
    det_boxes, det_labels, det_scores = [], [], []
    true_boxes, true_labels, true_difficulties = [], [], []

    with torch.no_grad():
        for images, boxes, labels, difficulties in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            predicted_locs, predicted_scores = model(images)
            boxes_batch, labels_batch, scores_batch = model.detect_objects(
                predicted_locs, predicted_scores, min_score=0.01, max_overlap=0.45, top_k=200)

            det_boxes.extend(boxes_batch)
            det_labels.extend(labels_batch)
            det_scores.extend(scores_batch)
            true_boxes.extend([b.to(device) for b in boxes])
            true_labels.extend([l.to(device) for l in labels])
            true_difficulties.extend([d.to(device) for d in difficulties])

    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    print('\n📊 AP по классам:')
    pp.pprint(APs)
    print(f'\n✅ mAP: {mAP:.4f}')

if __name__ == '__main__':
    evaluate(test_loader, model)
