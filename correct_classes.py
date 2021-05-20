from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer

container = DatasetContainer.from_json('6300.json')

corrector = ClassCorrector(container=container)

corrector.correct_class(label='arc', prediction_level=1, save_path='test.json')
