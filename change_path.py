from pathlib import Path

import matplotlib.pyplot as plt

from lbl.dataset import DatasetContainer

container = DatasetContainer.from_json('5577.json')

#path = "/home/jon/Documents/LBL/all/dataset/6300/teeest"
#path = r"C:\Users\Krist\Documents\dataset\6300"
path = r"C:\Users\Krist\Documents\dataset\5577"

for entry in container:
    filename = Path(Path(entry.image_path).name)
    entry.image_path = str(Path(path) / filename)

container.to_json(path='5577_k.json')


Test = False
if Test:
    for entry in container:
        img = entry.open()

        plt.imshow(img)
        plt.show()
