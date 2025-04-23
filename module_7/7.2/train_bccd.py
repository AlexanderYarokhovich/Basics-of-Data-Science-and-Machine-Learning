import subprocess, sys, pathlib

root = pathlib.Path('.').resolve()   # текущая папка Colab
data_folder = root/'data_bccd'
ckpt_dir    = root/'checkpoints'
ckpt_dir.mkdir(exist_ok=True)

cmd = [
    sys.executable,
    '/content/ssd/train.py',              # абсолютный путь
    '--data_folder', '/content/ssd/data_bccd',
    '--label_map',   '/content/utils_bccd.py',
    '--checkpoint',  '/content/checkpoints/ssd300_bccd.pth',
    '--num_epochs',  '25',
    '--batch_size',  '8',
    '--validation',
]

print(' '.join(cmd))
subprocess.run(cmd, check=True)
