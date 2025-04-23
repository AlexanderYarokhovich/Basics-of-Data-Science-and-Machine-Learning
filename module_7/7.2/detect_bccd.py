import argparse, pathlib, json, subprocess, sys

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', default='ssd/data_bccd', type=str)
parser.add_argument('--trained_model', required=True,        type=str)
parser.add_argument('--label_map',     default='utils_bccd.py')
parser.add_argument('--min_score',     default='0.2')
parser.add_argument('--max_overlap', default='0.35')
parser.add_argument('--top_k', default='35')
args = parser.parse_args()

# путь к TEST_images.json
images_path = pathlib.Path(args.data_folder) / 'TEST_images.json'
test_images = json.load(open(images_path))

if not test_images:
    raise ValueError("❌ TEST_images.json пустой или не найдено изображений.")

img_path = pathlib.Path(test_images[0])  # ← путь к первой картинке

# команда вызова detect.py
cmd = [
    sys.executable,
    '/content/ssd/detect.py',
    '--trained_model', args.trained_model,
    '--input',         str(img_path),
    '--label_map',     args.label_map,
    '--min_score',     args.min_score,
    '--max_overlap', args.max_overlap,
    '--top_k', args.top_k,
]

print('👉 Запускаем:\n', ' '.join(cmd))
subprocess.run(cmd, check=True)
