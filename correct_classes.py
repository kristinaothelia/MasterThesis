from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer

# Prediction level 0: checks human_prediction = False
# Prediction level 1: checks human_prediction = None
# Prediction level 2: checks human_prediction = True

def correct_6300():
    container = DatasetContainer.from_json('6300_k.json')
    corrector = ClassCorrector(container=container)

    #corrector.correct_class(label='arc', prediction_level=1, save_path='6300_k.json')
    #corrector.correct_class(label='diffuse', prediction_level=1, save_path='6300_k.json')
    #corrector.correct_class(label='discrete', prediction_level=1, save_path='6300_k.json')
    corrector.correct_class(label='aurora-less', prediction_level=2, save_path='test_k.json')


def correct_5577():
    container = DatasetContainer.from_json('5577_k.json')
    corrector = ClassCorrector(container=container)

    #corrector.correct_class(label='arc', prediction_level=1, save_path='5577_k.json')
    #corrector.correct_class(label='diffuse', prediction_level=1, save_path='5577_k.json')
    #corrector.correct_class(label='discrete', prediction_level=1, save_path='5577_k.json')
    corrector.correct_class(label='aurora-less', prediction_level=1, save_path='5577_k.json')

# Correct 630.0nm, green aurora
#correct_6300()

# Correct 557.7nm, red aurora
correct_5577()


# nya6_20150126_000907_5577_cal arc?
# nya6_20171116_011208_5577_cal
