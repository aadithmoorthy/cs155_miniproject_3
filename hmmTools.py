import numpy as np
from hmmlearn import hmm
import warnings
from sklearn.preprocessing import LabelEncoder

# Scikit learn has a bug. Fixed in unreleased version
warnings.filterwarnings("ignore", category=DeprecationWarning)


def trainHMM(preparedSonnets, sonnetLengths, n_states, n_steps=10):
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=n_steps)
    model.fit(preparedSonnets, lengths=sonnetLengths)
    return model

def extractModelParams(HMModel):
    A_start = HMModel.startprob_ # 1 x n_hidden
    A = HMModel.transmat_ # n_hidden x n_hidden
    O = HMModel.emissionprob_ # n_hidden x n_output
    n_hidden = HMModel.n_components
    n_output = HMModel.n_features
    return A_start, A, O, n_hidden, n_output

def generateHiddenSequence(HMMparams, length=10):
    # Unpack model parameters
    A_start = HMMparams[0]
    A = HMMparams[1]
    L = HMMparams[3]
    states = []
    for m in range(length):
        if (m==0):
            # Initialize hidden state from the start state
            p_new = A_start
            l_new = np.random.choice([l for l in range(L)], p=p_new)
            states.append(l_new)
        else:
            # Load previous hidden state
            l_old = states[-1]
            # Transition to new hidden state
            p_new = A[l_old]
            l_new = np.random.choice([l for l in range(L)], p=p_new)
            states.append(l_new)
    return states

def generateEmission(HMMparams, encoder, currentState, rhyDict=None, rhymingWord=None):
    # Unpack model parameters
    O = HMMparams[2]
    n_output = HMMparams[4]
    # Load emission probabilities for hidden state
    p_out = np.copy(O[currentState])
    if rhyDict != None:
        # If the rhyming dictionary has been provided,
        # the emission must be a rhymable word. This 
        # is done by setting all non-rhymable
        # emission probabilities to zero and
        # normalizing.
        if rhymingWord != None:
            # If the rhyming word has been specified,
            # the emission must rhyme with this 
            # specified word.
            options = rhyDict[rhymingWord]
            optionList = [option for option in options]
        else:
            # Otherwise, the emission must simply come
            # from the rhyming dictionary
            optionList = [option for option in rhyDict]
        # Convert to hmm encoding
        optionInd = encoder.transform(optionList)
        mask = np.zeros(p_out.shape,dtype=bool)
        mask[optionInd] = True
        # Zero out all non-rhyming options.
        p_out[~mask] = 0
        # Normalize the array.
        norm = np.sum(p_out[mask])
        p_out[mask] /= norm
    # Generate the emission index
    x_out = np.random.choice([x for x in range(n_output)], p=p_out)
    # Convert the index to a word
    emission = encoder.inverse_transform(x_out)
    return emission

def generateNewLine(HMMparams, encoder, sylDict, lineSylTarget=10, rhyDict=None, rhymingWord=None):
    generatingLine = True
    while generatingLine:
        # Generate potential lines
        potentialLines = generatePotentialLines(HMMparams, encoder, \
                                                lineSylTarget, rhyDict, rhymingWord)
        # Check to see if any of the generated lines are valid
        isValidLine = [checkLine(line, sylDict, lineSylTarget) \
                                        for line in potentialLines]
        if any(isValidLine):
            # If there is a valid line, terminate the loop 
            # and return the line
            validInds = [ind for ind in range(len(isValidLine)) \
                                                 if isValidLine[ind]]
            newLine = potentialLines[validInds[0]]
            generatingLine = False
    return newLine

def generatePotentialLines(HMMparams, encoder, lineSylTarget=10, rhyDict=None, rhymingWord=None):
    # Each word has at least one syllable. Thus, we need at most
    # lineSylTarget number of words. We generate a sequence of
    # hidden states of this length.
    hiddenSequence = generateHiddenSequence(HMMparams, lineSylTarget)
    # We now generate a sequence of possible words from the
    # sequence of hidden states.
    possibleLines = []
    currentLine = []
    for ind in range(len(hiddenSequence)):
        currentState = hiddenSequence[ind]
        if (rhyDict == None):
            # Generate new word for the line
            newWord = generateEmission(HMMparams, encoder, currentState)
            currentLine.append(newWord)
            possibleLines.append(currentLine)
        else:
            # If the rhyming dictionary is provided,
            # try to end the line with a rhymable word
            rhymingEmission = generateEmission(HMMparams, encoder, \
                                               currentState, rhyDict, rhymingWord)
            possibleLines.append(currentLine + [rhymingEmission])
            # If the line isn't finished, append a potential
            # word to the current line
            if (ind != len(hiddenSequence)-1):
                newWord = generateEmission(HMMparams, encoder, currentState)
                currentLine.append(newWord)
    # We now have a list of possible lines
    # of varying length. Each ends in the
    # rhyming word if specified.
    return possibleLines

def checkLine(line, sylDict, lineSylTarget=10):
    currentPossibleSylCounts = [0]
    for ind in range(len(line)):
        currentWord = line[ind]
        # Find possible syllable counts for the word
        possibleSyls = sylDict[currentWord]
        # Given all possible previous syllable counts, 
        # find all possible syllable counts using this
        # current word.
        newPossibleSylCounts = []
        for lineSylCount in currentPossibleSylCounts:
            for wordSyls in possibleSyls:
                # If 'E' is in the possible word 
                # syllable count, it indicates a 
                # syllable count which is only
                # valid if this is the final word.
                if ('E' in wordSyls):
                    if (ind == len(line)-1):
                        count = int(wordSyls.strip('E'))
                        newPossibleSylCounts.append(lineSylCount+count)
                else:
                    # Otherwise, syllable count can be used 
                    # anywhere in the line
                    count = int(wordSyls)
                    newPossibleSylCounts.append(lineSylCount+count)
        currentPossibleSylCounts = newPossibleSylCounts
    # We have now enumerated all possible syllable counts
    # for this line. The line is considered valid if there
    # exists some combination of syllables which matches
    # the target syllable count
    isValid = any([count == lineSylTarget \
                   for count in currentPossibleSylCounts])
    return isValid
