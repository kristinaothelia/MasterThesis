from pathlib import Path

import matplotlib.pyplot as plt

from lbl.dataset import DatasetContainer

container = DatasetContainer.from_json('datasets/Full_aurora.json')

path_red = "/home/jon/Documents/LBL/dataset/6300"
path_green = "/home/jon/Documents/LBL/dataset/5577"


# path = r"C:\Users\Krist\Documents\dataset\6300"
# path = r"C:\Users\Krist\Documents\dataset\5577"

for entry in container:
    if entry.datasetname == "green":
        filename = Path(Path(entry.image_path).name)
        entry.image_path = str(Path(path_green) / filename)
    else:
        filename = Path(Path(entry.image_path).name)
        entry.image_path = str(Path(path_red) / filename)

container.to_json(path='datasets/Full_aurora_jon.json')


Test = False
if Test:
    for entry in container:
        img = entry.open()

        plt.imshow(img)
        plt.show()
