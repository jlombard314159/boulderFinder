import folium
import io
from PIL import Image
from folium.vector_layers import Polygon

##TODO: potentially use some other type of image for multi band data.
# ndvi = ee.ImageCollection("SKYSAT/GEN-A/PUBLIC/ORTHO/RGB")

# Add custom base maps to folium
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

def boxCreator(coordOfBoulder):

    xTol = 0.000001
    yTol = 0.000001

    polyShape = [tuple(coordOfBoulder),
        (coordOfBoulder[0],coordOfBoulder[1] - xTol),
        (coordOfBoulder[0] - yTol, coordOfBoulder[1]),
        (coordOfBoulder[0] - yTol, coordOfBoulder[1] - xTol)]


    return polyShape

def polygonCreator(coordOfBoulder, map):

    polygonShape = boxCreator(coordOfBoulder)

    folium.Polygon(polygonShape, color="#EE4B2B", opacity = 1).add_to(map)

    return None

def generateMap(gpsCoord, customMap = basemaps, mapOption = 'MB'):

    my_map = folium.Map(location=gpsCoord, zoom_start=20,zoom_control=False,
               scrollWheelZoom=False,
               dragging=False, tiles =None)

    customMap[mapOption].add_to(my_map)

    return my_map

def generateMapWithBoulders(gpsCoord, listOfNeighbors, customMap = basemaps):

    my_map = folium.Map(location=gpsCoord, zoom_start=20,zoom_control=False,
               scrollWheelZoom=False,
               dragging=False, tiles =None)

    for _, data in enumerate(listOfNeighbors):

        polygonCreator(data,my_map)

    customMap['MB'].add_to(my_map)

    return my_map

def saveConvert(createdMap, imageLabel,
    localFolder = 'C:/Users/jlomb/Documents/Personal Coding/Python/MP/MPExtensions/testOutput/'):

    img_data = createdMap._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    img.save(localFolder + imageLabel + '.png')
        
    return None
