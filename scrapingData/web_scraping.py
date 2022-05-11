import requests
import json
from bs4 import BeautifulSoup
import re

def grabURL(urlToStart):

    rawHTML = requests.get(urlToStart).text

    htmlFormatted = BeautifulSoup(rawHTML,'html.parser')

    return htmlFormatted

def remove_trailing_characters(list_to_clean):

    cleaned_up = [re.sub('\n',' ',x) for x in list_to_clean]

    return cleaned_up

def extract_info(html, attribute_to_find):

    sub_info = html.find(attrs  = attribute_to_find)

    return sub_info


#Grab other metadata from topHTML: state, gps, name, parent_id
def extract_urls(htmlToSearch, urlPosition = 0):

    all_links = htmlToSearch.find_all('a')

    extract_sidebar_url = all_links[urlPosition].get('href')

    return extract_sidebar_url


def grab_state(topHTML):

    grab_links = extract_info(topHTML, {'class':'mb-half small text-warm'})

    state_url = extract_urls(grab_links, urlPosition= 1)

    state = state_url.split('/')[-1]

    return state


def grab_area_name(topHTML):

    sidebar_page = extract_info(topHTML, {'class':'mp-sidebar'}) # there are routes so do something

    area_url = extract_urls(sidebar_page, urlPosition= 0)

    areaName = area_url.split('/')[-1]

    return areaName

def grab_unique_id(topHTML):

    sidebar_page = extract_info(topHTML, {'class':'mp-sidebar'}) # there are routes so do something

    area_url = extract_urls(sidebar_page, urlPosition= 0)

    unique_id = area_url.split('/')[-2]

    return unique_id

def grab_major_area_name(topHTML):

    grab_links = extract_info(topHTML, {'class':'mb-half small text-warm'})
    
    area_url = extract_urls(grab_links, urlPosition= 3)

    grandparent_name = area_url.split('/')[-1]

    return grandparent_name

def grab_parent_id(topHTML):

    grab_links = extract_info(topHTML, {'class':'mb-half small text-warm'})

    parent_url = extract_urls(grab_links, urlPosition= -1)

    parent_id = parent_url.split('/')[-1]

    return parent_id


def findGPSInString(gpsString, gps_identifier = 'maps?q='):

    find_identifier = gpsString.find(gps_identifier)

    if(find_identifier != -1):
        return True

    return False

def uglyGPSFormatter(uglyGPSString):

    splitBySlash = uglyGPSString.split('/')

    gps_loc = [findGPSInString(x) for x in splitBySlash].index(True)

    gps_query = splitBySlash[gps_loc]

    extract_gps = re.sub('&t=h&hl=en','',gps_query)
    extract_gps = extract_gps.split('=')[1]

    convert_to_list = extract_gps.split(',')

    convert_to_int = [float(x) for x in convert_to_list]

    return convert_to_int

def grab_gmap_coords(topHTML):

    grab_links = extract_info(topHTML, {'class':'description-details'})

    parent_url = extract_urls(grab_links, urlPosition= 0)

    gps_output = uglyGPSFormatter(parent_url)

    return gps_output

def create_area(html, listOfFxns = [grab_parent_id, grab_unique_id, grab_area_name, grab_gmap_coords, grab_state, grab_major_area_name]):

    area_output = [f(html) for f in listOfFxns]

    area_dict = {'parent-id':area_output[0],
                 'unique-id':area_output[1],
                 'area_name':area_output[2],
                 'lnglat':area_output[3],
                 'us_state':area_output[4],
                 'major_area':area_output[5]}

    return area_dict

def find_map_in_url(url_string):

    extract_type_of_url = url_string.split('/')[-3]

    if 'map' in extract_type_of_url:
        return True

    return False

def extract_https(string_to_check):

    if 'https' in string_to_check:
        return True
    
    if 'http' in string_to_check:
        return True

    return False

def remove_empties(list_to_modify):

    no_empty = [x for x in list_to_modify if x != {}]

    return no_empty

#------------------------------------------------------------------------------#
#Find the head

def scrap_process(data):

    html_process = grabURL(urlToStart=data)

    routes = extract_info(html_process, {'class':'mp-sidebar'})

    subAreas_exist = extract_info(routes, {'class':'lef-nav-row'})
    boulders_exist = extract_info(routes, {'class':'route-type Boulder'})
    alpine_boulders_exist = extract_info(routes, {'class':'route-type Boulder Alpine'})

    if subAreas_exist is None:

        if (boulders_exist is None) & (alpine_boulders_exist is None):
            area = {}
            return area

        else: 
            area = create_area(html_process)
            return area

    else: 
         #call fxn again - keep going down further
        subAreas_url = []
        for link in routes.find_all('a'):
            subAreas_url.append(link.get('href'))

        #extract https only
        https_only = [x for x in subAreas_url if extract_https(x)]
        areas_only = [x for x in https_only if not find_map_in_url(x)]

        recursion_list = []
        for _, sub_data in enumerate(areas_only):

            sub_info = scrap_process(sub_data)
            recursion_list.append(sub_info)

        recursion_no_empties = remove_empties(recursion_list)

        return recursion_no_empties

def remove_empties(list_to_modify):

    no_empties = [x for x in list_to_modify if x != []]

    return no_empties

def list_dict_type(list):

    list_of_types = [True if type(x) is dict else None for x in list]

    return list_of_types

def unlist_lists(lists):

    remove_empty = remove_empties(lists)

    types = list_dict_type(remove_empty)
    #If its a list -> recurse
    #If its a dict -> append 
    for data, type in zip(remove_empty, types):

        if type:
            global_list.append(data)
        else: 
            unlist_lists(data)

    return None

all_states = ['https://www.mountainproject.com/area/105708961/nevada',
    'https://www.mountainproject.com/area/105708959/california',
    'https://www.mountainproject.com/area/105708965/oregon',
    'https://www.mountainproject.com/area/105708966/washington',
    'https://www.mountainproject.com/area/105708962/arizona',
    'https://www.mountainproject.com/area/105708964/new-mexico',
    'https://www.mountainproject.com/area/105835804/texas',
    'https://www.mountainproject.com/area/105708960/wyoming',
    'https://www.mountainproject.com/area/105907492/montana',
    'https://www.mountainproject.com/area/105708958/idaho',
    'https://www.mountainproject.com/area/105708957/utah',
    'https://www.mountainproject.com/area/105708963/south-dakota',
    'https://www.mountainproject.com/area/105708956/colorado']


accurate_data = ['https://www.mountainproject.com/area/121567471/nine-mile-bouldering',
'https://www.mountainproject.com/area/105937608/kraft-boulders',
'https://www.mountainproject.com/area/120108791/hidden-valley-area-bouldering',
'https://www.mountainproject.com/area/105870979/stoney-point',
'https://www.mountainproject.com/area/106116317/culp-valley',
'https://www.mountainproject.com/area/105991060/the-tramway',
'https://www.mountainproject.com/area/106079037/druid-stones',
'https://www.mountainproject.com/area/106094717/volcanic-tablelands-happysad-boulders',
'https://www.mountainproject.com/area/106132808/buttermilks-main',
'https://www.mountainproject.com/area/112199974/narrows-blocs',
'https://www.mountainproject.com/area/113198209/bristlecone-bouldering']

##Add and less trustworth not done

less_trustworth = ['https://www.mountainproject.com/area/105920760/horse-flats',
'https://www.mountainproject.com/area/105895938/triassic',
'https://www.mountainproject.com/area/107238496/garth-rocks',
'https://www.mountainproject.com/area/119201309/canals',
'https://www.mountainproject.com/area/116365038/pinkland',
'https://www.mountainproject.com/area/106071063/way-lake']

all_areas_list = []

for _, data in enumerate(less_trustworth):
    
    interim_data = scrap_process(data)
    all_areas_list.append(interim_data)


global_list = [] 
unlist_lists(all_areas_list)

global_list_complete = [x for x in global_list if x != {}] 

def save_json_zip(global_list):
    with open('hand_picked_medium_quality.json', 'w', encoding='utf-8') as f:
        json.dump(global_list, f, ensure_ascii=False, indent=4)

save_json_zip(global_list_complete)

# def read_json(filename):

#     with open(filename) as f:
#         data = json.load(f)

#     return data

# co = read_json('colorado.json')

# co_no_empty = [x for x in co if x != {}]
