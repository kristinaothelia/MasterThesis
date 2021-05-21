from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer

container = DatasetContainer.from_json('6300_k.json')

corrector = ClassCorrector(container=container)

# Prediction level 0: checks human_prediction = False
# Prediction level 1: checks human_prediction = None
# Prediction level 2: checks human_prediction = True

#corrector.correct_class(label='arc', prediction_level=1, save_path='6300_k.json')
#corrector.correct_class(label='diffuse', prediction_level=1, save_path='6300_k.json')
#corrector.correct_class(label='discrete', prediction_level=1, save_path='6300_k.json')
corrector.correct_class(label='aurora-less', prediction_level=1, save_path='6300_k.json')
