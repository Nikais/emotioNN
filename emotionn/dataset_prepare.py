import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# extract dataset
def extract_targz(fname, to_directory='data'):
    fname = Path.cwd().parent / fname
    try:
        print(f'Распаковка {fname.name}')
        targz = tarfile.open(fname, 'r:gz')
        targz.extractall(fname.parent / to_directory)
        targz.close()
    except tarfile.ReadError:
        print(f'{fname.name} - не файл tar.gz')


root = Path(__file__).parent
extract_targz(root / 'fer2013.tar.gz')

# making folders
outer_names = ['train', 'test']
emotions_train = {'angry': 0, 'disgusted': 0, 'fearful': 0, 'happy': 0, 'sad': 0, 'surprised': 0, 'neutral': 0}
emotions_test = emotions_train.copy()

for outer_name in outer_names:
    for inner_name in emotions_test.keys():
        directory = root / 'data/fer2013' / outer_name / inner_name
        directory.mkdir(parents=True, exist_ok=True)

print('Чтение датасета')

df = pd.read_csv(root / 'data/fer2013/fer2013.csv')
mat = np.zeros((48, 48))

print('Saving images')
len_df = len(df)

# read the csv file line by line
for i in tqdm(range(len_df)):
    txt = df['pixels'][i]
    words = txt.split(' ')

    # the image size is 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = int(words[j])

    img = Image.fromarray(mat)
    # Convert img to grayscale
    if img.mode != 'L':
        img = img.convert('L')
    if df['Usage'][i] == 'Training':
        for id_emotion, emotion in enumerate(emotions_train.items()):
            if df['emotion'][i] == id_emotion:
                img.save(str(root / f'data/fer2013/train/{emotion[0]}/img_{emotion[1]}.png'))
                emotions_train[emotion[0]] += 1
                break
    else:
        for id_emotion, emotion in enumerate(emotions_test.items()):
            if df['emotion'][i] == id_emotion:
                img.save(str(root / f'data/fer2013/test/{emotion[0]}/img_{emotion[1]}.png'))
                emotions_test[emotion[0]] += 1
                break

print('Done!')
