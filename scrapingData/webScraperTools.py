from collections import Counter
from shutil import copyfile
#Note that this fxn is reversed from removeIncorrectGPS since the GPS from the
#api was actually reversed
def removeIncorrectLatLng(listToRemove, key = 'lnglat'):

    correctGPS = [x for x in listToRemove if x[key][0] > 0]
    correctGPS = [x for x in correctGPS if x[key][1] < 0]

    return correctGPS


def count_states(list):

    extract_state = [x['us_state'] for x in list]

    count_state = Counter(extract_state)

    return count_state
    
def add_url(list):

    string_to_add = 'https://www.mountainproject.com/area/'

    for _, data in enumerate(list):

        url = string_to_add + data['unique-id'] + '/' + data['area_name']

        data['url'] = url

    return None


def extract_for_manual(neighbors_list):

    neighbor_exist = []
    final_output = []
    for _, data in enumerate(neighbors_list):

        if data['index'] in neighbor_exist:
            continue

        else:
            neighbor_exist = neighbor_exist + data['neighbors']
            final_output.append(data)


    return final_output

def extract_file_name(one_area):
    stringie = 'train_' + one_area['area_name'] + '__' +  one_area['url'].split('/')[-2]

    png_stringie = stringie + '.png'

    return png_stringie

def find_and_move(input_dir, output_dir, to_move):

    for _, data in enumerate(to_move):

        full_input = input_dir + data
        full_output = output_dir + data

        copyfile(full_input,full_output)

    return None