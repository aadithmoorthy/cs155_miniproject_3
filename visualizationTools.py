import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Scikit learn has a bug. Fixed in unreleased version
warnings.filterwarnings("ignore", category=DeprecationWarning)


def createWordclouds(modelParams, encoder):
    # Unpack model parameters
    O = modelParams[2] #
    n_hidden = modelParams[3] #
    n_output = modelParams[4] #
    # Initialize word clouds
    M_sample = 100000
    wordclouds = []
    # For each state, generate a list of outputs
    outCount = []
    for i in range(n_hidden):
        # Load emission probabilities for hidden state
        p_out = np.copy(O[i])
        # Generate the emission indices
        x_out = np.random.choice(np.arange(n_output), size=M_sample, p=p_out)
        # Convert the indices to words
        emissions = encoder.inverse_transform(x_out)
        sentence = [word for word in emissions]
        sentence_str = ' '.join(sentence)
        wordclouds.append(newWordcloud(sentence_str, title='State %d' % i))
    return wordclouds

def newWordcloud(text, title=''):
    plt.close('all')
    # Generate a wordcloud
    wordcloud = WordCloud(random_state=0, max_words=50,
                          background_color='white').generate(text)
    # Show
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=24)
    plt.show()
    return wordcloud
