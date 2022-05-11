import ee
import folium
from folium.features import ColorLine

# ee.Authenticate()
ee.Initialize()
# ndvi = ee.ImageCollection('LANDSAT/LC8_L1T_8DAY_NDVI')
mbToken = 'pk.eyJ1IjoiamxvbWJhcmQzMTQxNTkiLCJhIjoiY2t4ZG4wbDQ0M2h0dDJvcG12NWxkN2ljcCJ9.fNlPsPAoumrKbHKgXS2umw'

# Add custom base maps to folium
basemaps = {
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'ESRI': folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
    ),
    'MB': folium.TileLayer(
        'https://api.mapbox.com/v4/mapbox.naip/{z}/{x}/{y}@2x.png?access_token=' + str(mbToken),
        attr = 'Mapbox',
        name = 'Mapbox',
        overlay = False,
        control = False
    )
}

# Define the center of our map.
lat, lon = 45.77, 4.855

lat, lon = 39.491, -106.0501

my_map = folium.Map(location=[lat, lon], zoom_start=20,
    tiles = None)

# #Add polyline to guesstimate length of map
# locLine_vertical = [(39.489640, -106.0501),
# (39.49235, -106.0501)]

# folium.PolyLine(locLine_vertical, color='red', weight = 15, opacity = 0.6).add_to(my_map)

locLine_horiz = [(39.490, -106.05031), (39.490,-106.0503109),
(39.489999,-106.05031), (39.489999,-106.0503109)]

testLine = [(39.49, -106.05031),(39.49,-106.05032),
(39.48,-106.05031),(39.48,-106.05032)]

folium.Polygon(testLine, color='#1414F0', opacity = 1).add_to(my_map)


##INCLUDE THE # FOR HEX
#1414F0 = 20, 20, 240
#EE4B2B = 238, 75, 43
# folium.CircleMarker(location = [39.490, -106.0503], fill=True,
#     color = '#EE4B2B', radius = 1, fillOpacity = 1).add_to(my_map)

# Add custom basemaps
# basemaps['Google Satellite'].add_to(my_map)

def add_ee_layer(self, ee_object, vis_params, name):
    
    try:    
        # display ee.ImageCollection()
        if isinstance(ee_object, ee.imagecollection.ImageCollection):    
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)

    except:
        print("Could not display {}".format(name))

ndvi = ee.ImageCollection("SKYSAT/GEN-A/PUBLIC/ORTHO/RGB")

rgb = ndvi.select(['R','G','B'])

visParm = {'min':11.0,
           'max':190.0}

folium.Map.add_ee_layer = add_ee_layer

my_map.add_ee_layer(rgb,visParm,'uh')


import io
from PIL import Image
import numpy as np

img_data = my_map._to_png(5)
img = Image.open(io.BytesIO(img_data))

temp = np.asarray(img)

#Produce each pixel:
ydim = temp.shape[0]
xdim = temp.shape[1]

pixelList = []

def arrayNPEquality(array1, array2):

    if (array1 == array2).all():

        return True

    return False

for i in range(0,xdim-1):
    for j in range(0,ydim-1):

        pixelSlice = temp[j,i,:]

        if arrayNPEquality(pixelSlice[0:3],[238,75,43]):

            pixelList.append((i,j))

#The fill I used is 238 75 43[]

img.save('image_ee_rgb.png')


my_map.save('test.html')


#---------------------------------------------
#Calculations for JAL

#With zoom = 20, this line spans the entire vertical aspect.

locLine_vertical = [[39.489640, -106.0501],
[39.491, -106.0501]]

vertHeight = locLine_vertical[0][0] - locLine_vertical[1][0]

locLine_hoiz = [[39.491,-106.04646],
[39.491,-106.053711]]

horizHeight = locLine_hoiz[0][1] - locLine_hoiz[1][1]

# from html2image import Html2Image
# hti = Html2Image(size=(1800,900),
#     custom_flags=['--virtual-time-budget=10000'])
# hti.screenshot(html_file='test.html', save_as='test_1800_900.png')

#----------------------------------------------------------------
#Derive a bounding box based on the image.
##Model JAL is feeding is based on 300x300 so start with that

##TODO: Implement this as a class (?)
##TODO: final product: generate a xmin - ymax + class + uniqueID (row of a pd DF)
size=(1800,900)

midPoint = (683,322)

boxLength = 30

xMin = midPoint[0]-boxLength/2
xMax = midPoint[0]+boxLength/2
yMin = midPoint[1]-boxLength/2
yMax = midPoint[1]+boxLength/2