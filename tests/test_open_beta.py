import pytest

from scrapingData.openBetaScraper import extractRoutesWithBoulders, extractStates, \
syncUpRoutesWithArea

areasMadeUp = [
    {'us_state':'Iowa',
    'area_name':'Bob',
    'lnglat':[-10,10]},

    {'us_state':'Wyoming',
    'area_name':'Cool Dude Ranch',
    'lnglat':[-20,20]}
]

routesMadeUp = [
    {'type':{'boulder'},
    'metadata':{'parent_lnglat':[-20,20],
    'parent_sector':'Bob'}},
    {'type':{'weewoo'},
    'metadata':{'parent_lnglat':[-20.2,20.1],
    'parent_sector':'Bob'}}
]

class TestScrapingData:

    def testBouldersOnly(self):

        boulders = extractRoutesWithBoulders(routesMadeUp)[0]
        bouldersOnly = routesMadeUp[0]['metadata']

        assert boulders == bouldersOnly

    def testStatesOnly(self):

        wyo = extractStates(routesMadeUp)

        assert wyo == areasMadeUp[1]

    def testRoutesAndAreaSync(self):

        boulders = extractRoutesWithBoulders(routesMadeUp)
        finalRoutes = syncUpRoutesWithArea(boulders, areasMadeUp)[0]

        assert finalRoutes == routesMadeUp[0]['metadata']

