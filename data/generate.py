import os
from pathlib import Path
import glob


if __name__ == '__main__':
    root = '/Volumes/INTEL/Data/CIFAR100'
    train_files = glob.glob(os.path.join(root, 'train/*/*.jpg'))
    val_files = glob.glob(os.path.join(root, 'val/*/*.jpg'))
    train_files = list(map(lambda x: f'{Path(x).relative_to(Path(x).parent.parent.parent)},{Path(x).parent.stem}', train_files))
    val_files = list(
        map(lambda x: f'{Path(x).relative_to(Path(x).parent.parent.parent)},{Path(x).parent.stem}', val_files))
    with open('cifar100_train.txt', 'w') as f:
        f.write('\n'.join(train_files))
    with open('cifar100_val.txt', 'w') as f:
        f.write('\n'.join(val_files))
