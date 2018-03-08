import numpy as np
import string


def readInSonnets(filepath, syllableDict):
    # Open file and read in lines
    f = open(filepath, 'r')
    lines = f.readlines()
    # Initialize lists for sonnets and the current sonnet
    sonnets = []
    currentSonnet = []
    for line in lines[1:]:
        # Split line into words
        lineContents = _processSonnetLine(line, syllableDict)
        # Check if line is non-empty
        if lineContents != []:
            # If line is just a number, this is 
            # the start of a new sonnet
            if all([char in string.digits for char in lineContents[0]]):
                # Add previous sonnet to the list
                sonnets.append(currentSonnet)
                # Start new sonnet
                currentSonnet = []
            else:
                # Otherwise append current line to the sonnet
                currentSonnet.append(lineContents)
    return sonnets

def readInSyllableCounts(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    syllableDict = {}
    for line in lines:
        elements = line.split()
        syllableDict[elements[0]] = elements[1:]
    return syllableDict

def readInWords(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    wordStateBiDict = {}
    ind = 0
    for line in lines:
        word = line.split()[0]
        wordStateBiDict[word] = ind
        wordStateBiDict[ind] = word
        ind +=1
    return wordStateBiDict

def labelSonnetSyllables(sonnet, syllableDict):
    syllablesByLine = []
    for line in sonnet:
        lineSyllables = labelLineSyllables(line, syllableDict)
        syllablesByLine.append(lineSyllables)
    return syllablesByLine

def labelLineSyllables(line, syllableDict):
    # Find each possible syllable count for each word
    lineOptions = [syllableDict[element] for element in line]
    # For every non-end word, remove any special 
    # ending syllable counts
    for ind in range(len(lineOptions)-1):
        wordOptions = lineOptions[ind]
        reducedSet = [count for count in wordOptions if 'E' not in count]
        lineOptions[ind] = reducedSet
    # For ending word, use special ending syllable
    # count if it exists
    if any( ['E' in count for count in lineOptions[-1]] ):
        lineOptions[-1] = [count.lstrip('E') for count in lineOptions[-1] if 'E' in count]
    # Convert all strings to integers
    lineOptions = [[int(count) for count in wordOptions] for wordOptions in lineOptions]
    # If only one option exists for each word,
    # no need for further processing. Otherwise,
    # need to determine which syllable counts to
    # use
    if any([len(wordOptions)!=1 for wordOptions in lineOptions]):
        # Pass to helper function
        bestCounts = _findSyllablesLine(lineOptions)
        # Helper function either returns the unique syllable counts
        # or return none if there is no combination that gives
        # ten syllables in a line
        if bestCounts != None:
            lineOptions = bestCounts
        else:
            # If none is returned, the line has more than ten
            # syllables. Return the lowest possible syllable count
            lineOptions = [[option[0]] for option in lineOptions]
    # Return as a single list rather than list of lists
    syllableCounts = [wordOptions[0] for wordOptions in lineOptions]
    return syllableCounts

def _processSonnetLine(line, syllableDict):
    # Strip all punctuation except hyphens and apostraphes
    punctuation = string.punctuation.replace('-', '').replace('\'', '')
    line = line.translate(str.maketrans('', '', punctuation))
    # Make all characters lower case
    line = line.lower()
    # Split line into its contents
    lineContents = line.split()
    for ind in range(len(lineContents)):
        # Some words have leading apostrophes from being
        # in quotations
        word = lineContents[ind]
        if word not in syllableDict:
            lineContents[ind] = word.strip("'")
    return lineContents

def _findSyllablesLine(lineOptions, remainingCount=10):
    # Work recursively to find the correct syllable counts.
    # This works right to left across the line.
    # Base Case: lineOptions is just a single word
    if len(lineOptions)==1:
        wordOptions = lineOptions[0]
        # Check if there is a syllable count which
        # matches the remaining syllables
        if all([counts!=remainingCount for counts in wordOptions]):
            # If no option matches the remaining
            # count, return None as there is
            # no valid choice
            return None
        else:
            # There is a choice which exactly
            # matches remainingCount
            return [[remainingCount]]
    # Recursive Step: lineOptions is several words
    nextWordOptions = lineOptions[-1]
    remainingLine = lineOptions[:-1]
    for possibleCount in nextWordOptions:
        # For each possible syllable count for the word,
        # see if the remaining sentence can be constructed
        r = remainingCount - possibleCount
        lineCounts = _findSyllablesLine(remainingLine, r)
        if lineCounts != None:
            # A valid combination of syllable counts was
            # found. Append the current syllable choice
            # and return
            lineCounts.append([possibleCount])
            return lineCounts
    # Otherwise, no valid combination of counts
    # was found for the remaining fragment. Thus,
    # there is no valid choice.
    return None
