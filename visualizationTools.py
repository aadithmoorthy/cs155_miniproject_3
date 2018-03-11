import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt

# Scikit learn has a bug. Fixed in unreleased version
warnings.filterwarnings("ignore", category=DeprecationWarning)

def visualizeTransitions(modelParams, cutA=.001, cutO=.0001):
    # Load transition matrices
    A = np.copy(modelParams[1])
    O = np.copy(modelParams[2])
    # Cut off low probability transitions
    A[A<cutA] = cutA
    O[O<cutO] = cutO
    # Define log transition matrices
    lnA = np.log(A)
    lnO = np.log(O)
    # Clear existing plots and set colormap
    plt.close('all')
    plt.set_cmap('viridis')
    # Visualization of A.
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(A, vmax=1, vmin=0)
    cbar = plt.colorbar()
    cbar.set_label('Transition Probability')
    plt.title('Visualization of A matrix')
    plt.xlabel('Final State')
    plt.ylabel('Initial State')
    plt.show()
    # Visualization of log(A).
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(lnA, vmax=0, vmin=np.log(cutA))
    cbar = plt.colorbar()
    cbar.set_label('Log Probability')
    plt.title('Visualization of log(A) matrix')
    plt.xlabel('Final State')
    plt.ylabel('Initial State')
    plt.show()
    # Visualization of O.
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(O, vmax=.05, vmin=0, aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Emission Probability')
    plt.title('Visualization of O matrix')
    plt.xlabel('Emission State')
    plt.ylabel('Hidden State')
    plt.show()
    # Visualization of log(O).
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(lnO, vmax=0, vmin=np.log(cutO), aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Log Probability')
    plt.title('Visualization of log(O) matrix')
    plt.xlabel('Emission State')
    plt.ylabel('Hidden State')
    plt.show()
    return

def visualizeWordClouds(modelParams, encoder):
    # Unpack model parameters
    O = modelParams[2]
    n_hidden = modelParams[3]
    n_output = modelParams[4]
    for i in range(n_hidden):
        # Load emission probabilities for hidden state
        p_out = np.copy(O[i])
        # Create array of words
        words = encoder.inverse_transform(np.arange(n_output))
        # Find ten most frequent words for this state
        topInds = np.argsort(p_out)[::-1][:10]
        topWords = words[topInds].tolist()
        # Build word-frequency dictionary for word cloud
        wfDict = {word: frequency for (word, frequency) in zip(words, p_out)}
        # Convert the indices to words
        newWordCloud(wfDict, topWords, title='State %d' % i)
    return

def newWordCloud(wfDict, topWords, title=''):
    plt.close('all')
    # Generate a wordcloud
    newCloud = WordCloud(random_state=0, max_words=50,
                          background_color='white').generate_from_frequencies(wfDict)
    # Show wordcloud
    plt.imshow(newCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=24)
    plt.show()
    print("Top Words: " + repr(topWords))
    return newCloud
