import argparse
import torch
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import importlib.util
import pathlib

# -------- аргументы командной строки --------
parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', required=True, type=str,
                    help='Path to trained .pth model checkpoint')
parser.add_argument('--input', required=True, type=str,
                    help='Path to input image')
parser.add_argument('--label_map', required=True, type=str,
                    help='Path to utils_bccd.py with label_map inside')
parser.add_argument('--min_score', default=0.2, type=float,
                    help='Minimum confidence score threshold')
parser.add_argument('--max_overlap', default=0.35, type=float)
parser.add_argument('--top_k', default=35, type=int)
args = parser.parse_args()
# --------------------------------------------

# Load label_map from external file (dynamically)
label_path = pathlib.Path(args.label_map).expanduser()
spec = importlib.util.spec_from_file_location("label_mod", label_path)
label_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label_mod)

label_map = label_mod.label_map
rev_label_map = label_mod.rev_label_map

# (опционально — если ты определял label_color_map в utils_bccd.py)
try:
    label_color_map = label_mod.label_color_map
except AttributeError:
    label_color_map = {cls: 'red' for cls in label_map}  # fallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = torch.load(args.trained_model, map_location=device, weights_only=False)
model = checkpoint['model']
model = model.to(device)
model.eval()
print(f"✅ Model loaded: {args.trained_model}")

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, predicted_scores,
        min_score=min_score,
        max_overlap=max_overlap,
        top_k=top_k
    )

    det_boxes = det_boxes[0].to('cpu')
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]
    ).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    if det_labels == ['background']:
        return original_image

    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    try:
        font = ImageFont.truetype("./calibril.ttf", 15)
    except:
        font = ImageFont.load_default()

    for i in range(det_boxes.size(0)):
        if suppress is not None and det_labels[i] in suppress:
            continue

        box_location = det_boxes[i].tolist()
        color = label_color_map.get(det_labels[i], 'red')
        draw.rectangle(xy=box_location, outline=color, width=3)

        text = det_labels[i].upper()
        text_size = font.getsize(text)
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1],
                            box_location[0] + text_size[0] + 4., box_location[1]]

        draw.rectangle(xy=textbox_location, fill=color)
        draw.text(xy=text_location, text=text, fill='white', font=font)

    del draw
    return annotated_image

if __name__ == '__main__':
    img_path = args.input
    original_image = Image.open(img_path, mode='r').convert('RGB')
    result = detect(original_image,
                    min_score=args.min_score,
                    max_overlap=args.max_overlap,
                    top_k=args.top_k).show()

    if result is not None:
      result.save('prediction.jpg')
      print('✅ Результат сохранён в prediction.jpg')
    else:
      print("❌ Объекты не найдены — изображение не сохранено.")

    
