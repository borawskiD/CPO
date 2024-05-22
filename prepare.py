
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from tqdm import tqdm
import stow
from mltu.dataProvider import DataProvider
from mltu.augmentors import RandomBrightness, RandomErodeDilate, RandomSharpen
configs = {
    'batch_size': 32,
    'width': 128,
    'height': 32,
    'vocab': 'abcdefghijklmnopqrstuvwxyz ',
    'max_text_length': 32,
    'model_path': 'Models/04_sentence_recognition/202301131202',
    'image_class': ImageResizer,
}

sentences_txt_path = stow.join('Datasets', 'IAM_Sentences', 'ascii', 'sentences.txt')
sentences_folder_path = stow.join('Datasets', 'IAM_Sentences', 'sentences')

dataset, vocab, max_len = [], set(), 0

words = open(sentences_txt_path, "r").readlines()
for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.split(" ")
    if line_split[2] == "err":
        continue

    folder1 = line_split[0][:3]
    folder2 = line_split[0][:8]
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip('\n')

    label = label.replace('|', ' ')

    rel_path = stow.join(sentences_folder_path, folder1, folder2, file_name)
    if not stow.exists(rel_path):
        continue

    dataset.append([rel_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs['batch_size'],
    data_preprocessors=[ImageReader(image_class=configs['image_class'])],
    transformers=[
        ImageResizer(configs['width'], configs['height'], keep_aspect_ratio=True),
        LabelIndexer(configs['vocab'].split()),
        LabelPadding(max_word_length=configs['max_text_length'], padding_value=len(configs['vocab'])),
    ],
)

train_data_provider, val_data_provider = data_provider.split(split=0.9)

train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
]

