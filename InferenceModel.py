import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

def test_single_example(image_path, label, model):
    image = cv2.imread(image_path.replace("\\", "/"))
    if image is None:
        print(f"Image at path {image_path} could not be loaded.")
        return

    try:
        prediction_text = model.predict(image)
        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print("Image:", image_path)
        print("Label:", label)
        prediction_str = "".join(prediction_text)
        print("Prediction:", prediction_str)
        print(f"CER: {cer}; WER: {wer}")
    except TypeError as e:
        # Handle the error gracefully
        print(f"Error occurred for image: {image_path}. Skipping this image.")
        print(f"Error message: {e}")

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


if __name__ == "__main__":
    import cv2
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from mltu.utils.text_utils import get_cer, get_wer
    import matplotlib.pyplot as plt
    from mltu.configs import BaseModelConfigs
    import os


    sentences_txt_path = os.path.join('Datasets', 'IAM_Sentences', 'ascii', 'sentences.txt')
    sentences_folder_path = os.path.join('Datasets', 'IAM_Sentences', 'sentences')

    dataset, vocab, max_len = [], set(), 0

    with open(sentences_txt_path, "r") as file:
        words = file.readlines()

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

        rel_path = os.path.join(sentences_folder_path, folder1, folder2, file_name)
        if not os.path.exists(rel_path):
            continue

        dataset.append([rel_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))

    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Maximum label length: {max_len}")

    configs = BaseModelConfigs.load("Models/04_sentence_recognition/202301131202/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("Models/04_sentence_recognition/202301131202/val.csv").values.tolist()

    accum_cer, accum_wer = [], []
    cer_per_image, wer_per_image = [], []

    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))
        print(image_path)
        try:
            prediction_text = model.predict(image)

            if isinstance(prediction_text, float):
                prediction_text = [prediction_text]

            prediction_text = [str(token) for token in prediction_text]

            cer = get_cer(prediction_text, label)
            wer = get_wer(prediction_text, label)
            print("Image: ", image_path)
            print("Label:", label)
            prediction_str = "".join(prediction_text)
            print("Prediction: ", prediction_str)
            print(f"CER: {cer}; WER: {wer}")

            accum_cer.append(cer)
            accum_wer.append(wer)

            cer_per_image.append(cer)
            wer_per_image.append(wer)
        except TypeError as e:
            print(f"Error occurred for image: {image_path}. Skipping this image.")
            print(f"Error message: {e}")
            continue

    avg_cer = np.average(accum_cer)
    avg_wer = np.average(accum_wer)
    print(f"Average CER: {avg_cer}, Average WER: {avg_wer}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(cer_per_image)), cer_per_image, color='blue', label='CER')
    plt.plot(range(len(wer_per_image)), wer_per_image, color='red', label='WER')
    plt.xlabel('Image Index')
    plt.ylabel('Correctness')
    plt.title('Correctness Metrics for Each Image')
    plt.legend()
    plt.show()

    own_examples = [
        ('CustomExamples/Obrazek1.jpeg', 'Dominik Borawski'),
        ('CustomExamples/Obrazek2.jpeg', 'DOMINIK BORAWSKI'),
        ('CustomExamples/Obrazek3.jpeg', 'Dominik Borawski'),
        ('CustomExamples/Obrazek4.jpeg', 'Akademia marynarki wojennej w Gdyni'),
        ('CustomExamples/Obrazek5.jpeg', 'Akademia marynarki wojennej w Gdyni'),
        ('CustomExamples/Obrazek6.jpeg', 'AMW'),
        ('CustomExamples/Obrazek7.jpeg', 'Akademia MW'),

    ]

    for example in own_examples:
        image_path, label = example
        test_single_example(image_path, label, model)
