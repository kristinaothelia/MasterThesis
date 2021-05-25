from lbl.class_corrector import ClassCorrector
from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer

# Prediction level 0: checks human_prediction = False
# Prediction level 1: checks human_prediction = None
# Prediction level 2: checks human_prediction = True

def class_correction(pred_level, json_from='', json_to='',
                     arc=False, diff=False, disc=False, noa=False):

    if pred_level < 0 or pred_level > 2:
        print("Unvalid prediction level. [0,1,2]")
        print("Prediction level 0: checks human_prediction = False")
        print("Prediction level 1: checks human_prediction = None")
        print("Prediction level 2: checks human_prediction = True")
        pred_level = input("New prediction level: ")


    if arc == True:
        label = 'arc'
    elif diff == True:
        label = 'diffuse'
    elif disc == True:
        label = 'discrete'
    elif diff == True:
        label = 'aurora-less'
    else:
        print("No label selected")
    print(label)

    container = DatasetContainer.from_json(json_from)
    corrector = ClassCorrector(container=container)

    corrector.correct_class(label, prediction_level=pred_level, save_path=json_to)


class_correction(pred_level=0, json_from='datasets/5577_k_aurora.json', json_to='datasets/5577_k_aurora_correct.json',
                 arc=True, diff=False, disc=False, noa=False)


'''
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
#correct_5577()
'''

# nya6_20150126_000907_5577_cal arc?
# nya6_20171116_011208_5577_cal
