import unittest
import itemData
import pyConTextGraph as pyConText

class imageTools_test(unittest.TestCase):
    def setUp(self):
        # create a sample image in memory
        self.context = pyConText.pyConText()
        self.su1 = u'kanso <Diagnosis>**diabetes**</Diagnosis> utesl\xf6t eller diabetes men inte s\xe4kert. Vi siktar p\xe5 en r\xf6ntgenkontroll. kan det vara nej panik\xe5ngesten\n?'
        self.su2 =  u'IMPRESSION: 1. LIMITED STUDY DEMONSTRATING NO GROSS EVIDENCE OF SIGNIFICANT PULMONARY EMBOLISM.'
    def tearDown(self):
        self.context = 0
        self.su1 = 0
    #def testSource(self):
        #assert self.context.__file__ == 'pyConTextGraph.pyc'
    def test_setRawText(self):
        self.context.setRawText(self.su1)
        assert self.context.getRawText() == self.su1
    def test_scrub_preserve_unicode(self):
        self.context.setRawText(self.su1)
        self.context.cleanText()
        assert self.context.getText().index(u'\xf6') == 40
    def test_scrub_text(self):
        self.context.setRawText(self.su2)
        self.context.cleanText()
        assert self.context.getText().rfind(u'.') == -1
def run():
    pass
    
