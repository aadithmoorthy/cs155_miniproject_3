import hmmTools
import warnings
from sklearn.preprocessing import LabelEncoder#

# Scikit learn has a bug. Fixed in unreleased version
warnings.filterwarnings("ignore", category=DeprecationWarning)


def generateNaiveSonnet(HMMparams, encoder, syllableDict):
    sonnet = []
    # Generate 14 sonnet lines
    for j in range(14):
        line = hmmTools.generateNewLine(HMMparams, encoder, syllableDict)
        sonnet.append(line)
    return sonnet

def generateRhymingSonnet(HMMparams, encoder, sylDict, rDict):
    sonnet = []
    # Generate three stanzas of four lines each
    for i in range(3):
        lineEndings = []
        for j in range(4):
            if (j==0) or (j==1):
                # This line sets the rhyme for later lines
                line = hmmTools.generateNewLine(HMMparams, encoder, sylDict, \
                                                                   rhyDict=rDict)
                finalWord = line[-1]
                sonnet.append(line)
                lineEndings.append(finalWord)
            else:
                # Need to rhyme with second to last line
                rhyming = lineEndings[j-2]
                line = hmmTools.generateNewLine(HMMparams, encoder, sylDict, \
                                               rhyDict=rDict, rhymingWord=rhyming)
                sonnet.append(line)
    # Generate the couplet
    line = hmmTools.generateNewLine(HMMparams, encoder, sylDict, rhyDict=rDict)
    finalWord = line[-1]
    sonnet.append(line)
    line = hmmTools.generateNewLine(HMMparams, encoder, sylDict, \
                                rhyDict=rDict, rhymingWord=finalWord)
    sonnet.append(line)
    return sonnet

def formatSonnetToText(sonnet):
    textSonnet = ''
    for lineInd in range(len(sonnet)):
        line = sonnet[lineInd]
        strLine = ' '.join(line)
        if lineInd%2 == 0:
            # Even index lines are the start of sentences
            strLine = strLine.capitalize() + ',\n'
        else:
            # Odd index lines are the end of sentences
            strLine = strLine + '.\n'
        textSonnet = textSonnet + strLine
    return textSonnet

def generateHaiku(HMMparams, encoder, sylDict):
    haiku = []
    # Generate haiku
    lineSyllables = [5,7,5]
    for count in lineSyllables:
        line = hmmTools.generateNewLine(HMMparams, encoder, sylDict, \
                                                   lineSylTarget=count)
        haiku.append(line)
    return haiku

def generateRhymingHaiku(HMMparams, encoder, sylDict, rDict):
    line1 = hmmTools.generateNewLine(HMMparams, encoder, sylDict, \
                                      rhyDict=rDict, lineSylTarget=5)
    rWord = line1[-1]
    line2 = hmmTools.generateNewLine(HMMparams, encoder, sylDict, \
                                                     lineSylTarget=7)
    line3 = hmmTools.generateNewLine(HMMparams, encoder, sylDict, \
                                     lineSylTarget=5, rhyDict=rDict, rhymingWord=rWord)
    haiku = [line1, line2, line3]
    return haiku

def formatHaikuToText(haiku):
    textHaiku = ''
    for line in haiku:
        strLine = ' '.join(line)
        strLine = strLine + '\n'
        textHaiku = textHaiku + strLine
    return textHaiku
