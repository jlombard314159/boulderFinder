import pytest

from generatingData.trainingDataCreation import cleanUpCoords

class TestCoords:

    testSet = [(10,10),(11,11),(200,201),(180,180),(40,41),(42,40),(9,9),(8,8)]

    def testCleanUpCoords(self,testSet = testSet):

        goodResult = [(10,10),(200,201),(180,180),(40,41)]
        noClosePixels = cleanUpCoords(testSet)

        assert goodResult == noClosePixels