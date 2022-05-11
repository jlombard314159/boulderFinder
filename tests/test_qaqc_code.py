import pytest
from scrapingData.qaqcOpenBetaData import duplicateGPSRemover, matchByKeyword, modifyNameForOutput,  \
     removeIncorrectGPS, removeByKeyword

gpsCheckAreas = [{'lnglat':[-10,10], 'id':'correct'},
{'lnglat':[-10,-10]},
{'lnglat':[10,10]},
{'lnglat':[10,-10]}]

duplGPS = [{'lnglat':[10,10]},
{'lnglat':[10,10]},
{'lnglat':[10,100]}]

keyWord = [{'area_name': 'wyoming'},
{'area_name':'Wyoming'},
{'area_name':'bob'}]

class TestQAQCCode:

    def testRemoveIncorrect(self):

        area = removeIncorrectGPS(gpsCheckAreas)[0]

        assert area['id'] == 'correct'

    def testDuplicateGPSRemover(self):

        noDupl = duplicateGPSRemover(duplGPS)

        assert len(noDupl) == 2

    def testMatchByKeyword(self):

        testLower = matchByKeyword('wyoming', keyWord[0]['area_name'])
        testUpper = matchByKeyword('Wyoming', keyWord[1]['area_name'])

        errors = []
    # replace assertions by conditions
        if not testLower:
            errors.append("Lower case failed")
        if not testUpper:
            errors.append("Upper case failed")

        # assert no error message has been registered, else print messages
        assert not errors, "errors occured:\n{}".format("\n".join(errors))

    def testRemoveByKeyword(self):

        testCase = removeByKeyword(keyWord, keywords = ['wyoming','bob'])

        assert len(testCase) == 1

    def testModifyName(self):

        testData = [{'area_name': 'bob is cool'}]

        testSpaceRemover = modifyNameForOutput(testData)

        assert testSpaceRemover[0]['area_name'] == 'bobiscool'