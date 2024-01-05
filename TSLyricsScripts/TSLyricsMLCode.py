# Psych 2085 Final Project
# Camille Phaneuf (cphaneuf@g.harvard.edu)

# ------------------------------ project overview -----------------------------

# --- RESEARCH QUESTION ---
# How has Taylor Swift's music changed across time?

# --- DATA DESCRIPTION ---
# Lyrics from Taylor Swift's albums, sourced from a heroic Swiftie: https://www.reddit.com/r/TaylorSwift/comments/16eo7va/every_taylor_swift_song_lyric_in_order/?rdt=52684

# ------------------------------ import packages ------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os

# -------------------------- import and examine data --------------------------

# set working directories
abs_path = '/Users/camillephaneuf/Desktop/ANDL/G3Courses/Psy2085/' # absolute path to personal machine
inputs = 'taylor_swift/TSLyricsInputs/'
outputs = 'taylor_swift/TSLyricsOutputs/'
curr_wd = abs_path + inputs 
os.chdir(curr_wd)

# read in text files
TaylorSwift = open("TaylorSwift.txt", "r")
TaylorSwift_Lyrics_raw = TaylorSwift.read()
FearlessTV = open("FearlessTV.txt", "r")
FearlessTV_Lyrics_raw = FearlessTV.read()
SpeakNowTV = open("SpeakNowTV.txt", "r")
SpeakNowTV_Lyrics_raw = SpeakNowTV.read()
RedTV = open("RedTV.txt", "r")
RedTV_Lyrics_raw = RedTV.read()
NineteenEightyNine = open("1989.txt", "r")
NineteenEightyNine_Lyrics_raw = NineteenEightyNine.read()
Reputation = open("Reputation.txt", "r")
Reputation_Lyrics_raw = Reputation.read()
Lover = open("Lover.txt", "r")
Lover_Lyrics_raw = Lover.read()
Folklore = open("Folklore.txt", "r")
Folklore_Lyrics_raw = Folklore.read()
Evermore = open("Evermore.txt", "r")
Evermore_Lyrics_raw = Evermore.read()
Midnights = open("Midnights.txt", "r")
Midnights_Lyrics_raw = Midnights.read()

# set random number for use throughout project
rand = 42

# ---------------------------- text preprocessing -----------------------------

# import additional packages
import inflect
import string
import nltk 
from nltk.tokenize import word_tokenize
nltk.download('punkt') # tokenizer
from nltk.corpus import stopwords
nltk.download('stopwords') # stopword list
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') # lemmatizer

'''
The following functions were originally written by Patrick Mair, PhD and were adapted
slightly for this project. 
'''

# lowercase conversion
def text_lowercase(text):
    return text.lower()
TaylorSwift_Lyrics = text_lowercase(TaylorSwift_Lyrics_raw)
FearlessTV_Lyrics = text_lowercase(FearlessTV_Lyrics_raw)
SpeakNowTV_Lyrics = text_lowercase(SpeakNowTV_Lyrics_raw)
RedTV_Lyrics = text_lowercase(RedTV_Lyrics_raw)
NineteenEightyNine_Lyrics = text_lowercase(NineteenEightyNine_Lyrics_raw)
Reputation_Lyrics = text_lowercase(Reputation_Lyrics_raw)
Lover_Lyrics = text_lowercase(Lover_Lyrics_raw)
Folklore_Lyrics = text_lowercase(Folklore_Lyrics_raw)
Evermore_Lyrics = text_lowercase(Evermore_Lyrics_raw)
Midnights_Lyrics = text_lowercase(Midnights_Lyrics_raw)

# convert numbers to words
def numbers_to_words(text):
    con = inflect.engine()
    words = text.split()
    new_words = []
    for word in words:
        if word.isdigit():
            word = con.number_to_words(word)
        new_words.append(word)
    return ' '.join(new_words)
TaylorSwift_Lyrics = numbers_to_words(TaylorSwift_Lyrics)
FearlessTV_Lyrics = numbers_to_words(FearlessTV_Lyrics)
SpeakNowTV_Lyrics = numbers_to_words(SpeakNowTV_Lyrics)
RedTV_Lyrics = numbers_to_words(RedTV_Lyrics)
NineteenEightyNine_Lyrics = numbers_to_words(NineteenEightyNine_Lyrics)
Reputation_Lyrics = numbers_to_words(Reputation_Lyrics)
Lover_Lyrics = numbers_to_words(Lover_Lyrics)
Folklore_Lyrics = numbers_to_words(Folklore_Lyrics)
Evermore_Lyrics = numbers_to_words(Evermore_Lyrics)
Midnights_Lyrics = numbers_to_words(Midnights_Lyrics)

# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
TaylorSwift_Lyrics = remove_punctuation(TaylorSwift_Lyrics)
FearlessTV_Lyrics = remove_punctuation(FearlessTV_Lyrics)
SpeakNowTV_Lyrics = remove_punctuation(SpeakNowTV_Lyrics)
RedTV_Lyrics = remove_punctuation(RedTV_Lyrics)
NineteenEightyNine_Lyrics = remove_punctuation(NineteenEightyNine_Lyrics)
Reputation_Lyrics = remove_punctuation(Reputation_Lyrics)
Lover_Lyrics = remove_punctuation(Lover_Lyrics)
Folklore_Lyrics = remove_punctuation(Folklore_Lyrics)
Evermore_Lyrics = remove_punctuation(Evermore_Lyrics)
Midnights_Lyrics = remove_punctuation(Midnights_Lyrics)

# strip whitespaces
def remove_whitespace(text):
    return  " ".join(text.split())
TaylorSwift_Lyrics = remove_whitespace(TaylorSwift_Lyrics)
FearlessTV_Lyrics = remove_whitespace(FearlessTV_Lyrics)
SpeakNowTV_Lyrics = remove_whitespace(SpeakNowTV_Lyrics)
RedTV_Lyrics = remove_whitespace(RedTV_Lyrics)
NineteenEightyNine_Lyrics = remove_whitespace(NineteenEightyNine_Lyrics)
Reputation_Lyrics = remove_whitespace(Reputation_Lyrics)
Lover_Lyrics = remove_whitespace(Lover_Lyrics)
Folklore_Lyrics = remove_whitespace(Folklore_Lyrics)
Evermore_Lyrics = remove_whitespace(Evermore_Lyrics)
Midnights_Lyrics = remove_whitespace(Midnights_Lyrics)

# remove stopwords and tokenization
stop_words = stopwords.words('english')
print(stop_words) # this does not serve our purposes very well (e.g., pronouns could be informative)
len(stop_words)
custom_stop_words = ['oh', 'mm-mm', 'la', 'mmm-mmm', 'eh', 'ah', 'ha', 'ooh', 'oooh', 'ohoh'] # make a custom stopword list instead
len(custom_stop_words)
def remove_stopwords(text):
    stop_words = set(custom_stop_words)                             
    word_tokens = word_tokenize(text) # word tokenizer
    filtered_text = [word for word in word_tokens if word not in stop_words] # remove stopwords
    return filtered_text # turns text into a list
TaylorSwift_Lyrics_list = remove_stopwords(TaylorSwift_Lyrics)
TaylorSwift_Lyrics = TreebankWordDetokenizer().detokenize(TaylorSwift_Lyrics_list)
FearlessTV_Lyrics_list = remove_stopwords(FearlessTV_Lyrics)
FearlessTV_Lyrics = TreebankWordDetokenizer().detokenize(FearlessTV_Lyrics_list)
SpeakNowTV_Lyrics_list = remove_stopwords(SpeakNowTV_Lyrics)
SpeakNowTV_Lyrics = TreebankWordDetokenizer().detokenize(SpeakNowTV_Lyrics_list)
RedTV_Lyrics_list = remove_stopwords(RedTV_Lyrics)
RedTV_Lyrics = TreebankWordDetokenizer().detokenize(RedTV_Lyrics_list)
NineteenEightyNine_Lyrics_list = remove_stopwords(NineteenEightyNine_Lyrics)
NineteenEightyNine_Lyrics = TreebankWordDetokenizer().detokenize(NineteenEightyNine_Lyrics_list)
Reputation_Lyrics_list = remove_stopwords(Reputation_Lyrics)
Reputation_Lyrics = TreebankWordDetokenizer().detokenize(Reputation_Lyrics_list)
Lover_Lyrics_list = remove_stopwords(Lover_Lyrics)
Lover_Lyrics = TreebankWordDetokenizer().detokenize(Lover_Lyrics_list)
Folklore_Lyrics_list = remove_stopwords(Folklore_Lyrics)
Folklore_Lyrics = TreebankWordDetokenizer().detokenize(Folklore_Lyrics_list)
Evermore_Lyrics_list = remove_stopwords(Evermore_Lyrics)
Evermore_Lyrics = TreebankWordDetokenizer().detokenize(Evermore_Lyrics_list)
Midnights_Lyrics_list = remove_stopwords(Midnights_Lyrics)
Midnights_Lyrics = TreebankWordDetokenizer().detokenize(Midnights_Lyrics_list)

# lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word, pos = 'n') for word in word_tokens]
    return lemmas
print(lemmatize_word(TaylorSwift_Lyrics)) # this splits words like 'wanna' --> do not apply lemmatizer to the lyrics

# ----------------------------- text visualization ----------------------------

# import additional packages
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

# set plotting colors corresponding to album covers
TS_col = "#44CFB6"
Fear_col = "#CF9D2D"
SN_col = "#6D1AAD"
Red_col = "#B70202"
NEN_col = "#3DD5CE"
Rep_col = "#000000"
Love_col = "#FDA3DA"
Folk_col = "#A9A9A9"
Ever_col = "#EDE3C9"
Mid_col = "#282276"

# create word cloud for Taylor Swift album
TaylorSwift_wc = WordCloud(background_color = TS_col, random_state = rand)
TaylorSwift_wc.generate(TaylorSwift_Lyrics)
TaylorSwift_coloring = np.array(Image.open("TaylorSwift.png")) # color the word cloud with the album cover colors
TaylorSwift_colors = ImageColorGenerator(TaylorSwift_coloring)
plt.imshow(TaylorSwift_wc, interpolation = "bilinear")
plt.imshow(TaylorSwift_wc.recolor(color_func = TaylorSwift_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()

# create word cloud for Fearless (Taylor's Version) album
FearlessTV_wc = WordCloud(background_color = Fear_col, random_state = rand)
FearlessTV_wc.generate(FearlessTV_Lyrics)
FearlessTV_coloring = np.array(Image.open("FearlessTV.png")) # color the word cloud with the album cover colors
FearlessTV_colors = ImageColorGenerator(FearlessTV_coloring)
plt.imshow(FearlessTV_wc, interpolation = "bilinear")
plt.imshow(FearlessTV_wc.recolor(color_func = FearlessTV_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()

# create word cloud for Speak Now (Taylor's Version) album
SpeakNowTV_wc = WordCloud(background_color = SN_col, random_state = rand)
SpeakNowTV_wc.generate(SpeakNowTV_Lyrics)
SpeakNowTV_coloring = np.array(Image.open("SpeakNowTV.png")) # color the word cloud with the album cover colors
SpeakNowTV_colors = ImageColorGenerator(SpeakNowTV_coloring)
plt.imshow(SpeakNowTV_wc, interpolation = "bilinear")
plt.imshow(SpeakNowTV_wc.recolor(color_func = SpeakNowTV_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()

# create word cloud for Red (Taylor's Version) album
RedTV_wc = WordCloud(background_color = Red_col, random_state = rand)
RedTV_wc.generate(FearlessTV_Lyrics)
RedTV_coloring = np.array(Image.open("RedTV.png")) # color the word cloud with the album cover colors
RedTV_colors = ImageColorGenerator(RedTV_coloring)
plt.imshow(RedTV_wc, interpolation = "bilinear")
plt.imshow(RedTV_wc.recolor(color_func = RedTV_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()

# create word cloud for 1989 album
NineteenEightyNine_wc = WordCloud(background_color = NEN_col, random_state = rand)
NineteenEightyNine_wc.generate(NineteenEightyNine_Lyrics)
NineteenEightyNine_coloring = np.array(Image.open("1989.png")) # color the word cloud with the album cover colors
NineteenEightyNine_colors = ImageColorGenerator(NineteenEightyNine_coloring)
plt.imshow(NineteenEightyNine_wc, interpolation = "bilinear")
plt.imshow(NineteenEightyNine_wc.recolor(color_func = NineteenEightyNine_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()

# create word cloud for Reputation album
Reputation_wc = WordCloud(background_color = "black", random_state = rand)
Reputation_wc.generate(Reputation_Lyrics)
Reputation_coloring = np.array(Image.open("Reputation.png")) # color the word cloud with the album cover colors
Reputation_colors = ImageColorGenerator(Reputation_coloring)
plt.imshow(Reputation_wc, interpolation = "bilinear")
plt.imshow(Reputation_wc.recolor(color_func = Reputation_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()

# create word cloud for Lover album
Lover_wc = WordCloud(background_color = "white", random_state = rand)
Lover_wc.generate(Lover_Lyrics)
Lover_coloring = np.array(Image.open("Lover.png")) # color the word cloud with the album cover colors
Lover_colors = ImageColorGenerator(Lover_coloring)
plt.imshow(Lover_wc, interpolation = "bilinear")
plt.imshow(Lover_wc.recolor(color_func = Lover_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()

# create word cloud for Folklore album
Folklore_wc = WordCloud(background_color = "black", random_state = rand)
Folklore_wc.generate(Folklore_Lyrics)
Folklore_coloring = np.array(Image.open("Folklore.png")) # color the word cloud with the album cover colors
Folklore_colors = ImageColorGenerator(Folklore_coloring)
plt.imshow(Folklore_wc, interpolation = "bilinear")
plt.imshow(Folklore_wc.recolor(color_func = Folklore_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()

# create word cloud for Evermore album
Evermore_wc = WordCloud(background_color = Ever_col, random_state = rand)
Evermore_wc.generate(Evermore_Lyrics)
Evermore_coloring = np.array(Image.open("Evermore.png")) # color the word cloud with the album cover colors
Evermore_colors = ImageColorGenerator(Evermore_coloring)
plt.imshow(Evermore_wc, interpolation = "bilinear")
plt.imshow(Evermore_wc.recolor(color_func = Evermore_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()

# create word cloud for Midnights album
Midnights_wc = WordCloud(background_color = Mid_col, random_state = rand)
Midnights_wc.generate(Midnights_Lyrics)
Midnights_coloring = np.array(Image.open("Midnights.png")) # color the word cloud with the album cover colors
Midnights_colors = ImageColorGenerator(Midnights_coloring)
plt.imshow(Midnights_wc, interpolation = "bilinear")
plt.imshow(Midnights_wc.recolor(color_func = Midnights_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()

# change working directory and save word clouds
curr_wd = abs_path + outputs 
os.chdir(curr_wd)
TaylorSwift_wc.to_file('TaylorSwift_wc.png')
FearlessTV_wc.to_file('FearlessTV_wc.png')
SpeakNowTV_wc.to_file('SpeakNowTV_wc.png')
RedTV_wc.to_file('RedTV_wc.png')
NineteenEightyNine_wc.to_file('1989_wc.png')
Reputation_wc.to_file('Reputation_wc.png')
Lover_wc.to_file('Lover_wc.png')
Folklore_wc.to_file('Folklore_wc.png')
Evermore_wc.to_file('Evermore_wc.png')
Midnights_wc.to_file('Midnights_wc.png')

'''
Even from these simple word clouds, which display the most frequent words in each 
album, we see both common and temporary themes across time.

For example, 'youre' shows up again and again, indicating that Taylor is often describing
another person in her songs. But, this pattern breaks for, e.g., 1989, Reputation, and
Midnights, where she is perhaps centering her own identity and experiences. At first blush,
this is a bit surprising for 1989, which is supposedly about Harry Styles. But, the original
release of the album (included here, since 'Taylor's Version' came out after starting this
project) discusses him less than the vault tracks do, so this may make sense after all.

Folklore and Evermore seem to have the most similar word clouds, which is logical given
that they are the two indie albums, released as 'sisters' during 2020.
'''

# --- text "importance": term frequency-inverse document frequency (tf-idf) ---

# import additional packages
from sklearn.feature_extraction.text import TfidfVectorizer

# make corpus
lyric_corpus = [TaylorSwift_Lyrics, FearlessTV_Lyrics, SpeakNowTV_Lyrics, RedTV_Lyrics,
                NineteenEightyNine_Lyrics, Reputation_Lyrics, Lover_Lyrics, Folklore_Lyrics,
                Evermore_Lyrics, Midnights_Lyrics]
len(lyric_corpus)
lyric_corpus[0]

# build vocabulary and vectorize 
#   - ngram_range: use single words, bigrams, and trigrams
#   - max_df: ignore terms that have a document frequency higher than 0.6
#   - min_df: ignore terms that have a document frequency lower than 0.01
# vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (1,3), max_df = .6, min_df = .01) # this does not serve our purposes very well when using N > 1 n-grams
vectorizer = TfidfVectorizer(stop_words = custom_stop_words, ngram_range = (1,3), max_df = .6, min_df = .01) # use custom stopword list instead

# create the tf-idf weighted DTM
X = vectorizer.fit_transform(lyric_corpus)
type(X)
X.shape # 10 albums, 67000+ features

# get all single word, bigram, and trigram features
feature_names = vectorizer.get_feature_names_out()
print(feature_names)
len(feature_names)

# convert the DTM into a data frame
dense = X.todense()
dense.shape # 10 albums, 67000+ features
num_albums = dense.shape[0]
denselist = dense.tolist()                      
df = pd.DataFrame(denselist, columns = feature_names)
df.head() # not sure how '158' was introduced, as it was not in any of the original txt files; hopefully it will not matter, because there is so much data

# extract top 15 most important words for each album
Lyrics = df.transpose()
Lyrics.columns = ['Taylor Swift', 'Fearless (Taylor\'s Version)', 'Speak Now (Taylor\'s Version',
                  'Red (Taylor\'s Version)', '1989', 'Reputation', 'Lover', 'Folklore',
                  'Evermore', 'Midnights']
print(Lyrics)
top_dict = {}
for i in range(num_albums):
    top = Lyrics.iloc[:,i].sort_values(ascending = False).head(15) # find top 15 words in each album
    top_dict[Lyrics.columns[i]] = list(zip(top.index, top.values)) # convert top 15 words into a dictionary
for album, top_words in top_dict.items():
    print(album)
    print(', '.join([word for word, count in top_words]))
    print('---')
    
'''
Considering word importance (with the above print-outs) yields slightly different 
information than the word clouds (produced previously). For example, 'fairytale' emerges
as an important word for Fearless (Taylor's Version), but is not frequent enough to
be displayed in large font in that album's word cloud. The same can be said for 'grow'
in Speak Now (Taylor's Version), 'bless' in Lover, or 'hope' in Folklore.
'''

# aggregate tf-idf across all lyrics and produce a tf-idf based word cloud
#   - the bigger a word, the more *important* it is
#   - focus on the top 50 aggregated words
print(Lyrics.mean(axis = 1).sort_values(ascending = False).head(50))
Lyrics_wc = WordCloud(width = 800, height = 800, random_state = rand, max_words = 50).generate_from_frequencies(Lyrics.mean(axis = 1))
plt.imshow(Lyrics_wc, interpolation = 'bilinear')
plt.axis("off")
plt.title("Word Importance Across\nTaylor Swift Albums")
plt.show()
Lyrics_wc.to_file('Lyrics_wc.png')

'''
From the list of the top 50 most important words and their visualization in the word
cloud, we can see how even tf-idf is not the best measure: it is still quite sensitive to
mere word repetitions. For example, 'red, red, red' is frequently repeated in Red (Taylor's 
Version), 'I shake it off' is frequently repeated in 'Shake it Off', and 'Are we out of the 
woods yet?' is frequently repeated in 'Out of the Woods'. That does not mean that, e.g., 
the feeling of being 'out of the woods' is encompassed throughout Taylor's music.
'''

# ------------------------------ word embeddings ------------------------------

# import additional packages
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import MDS

# tokenize corpus
print(lyric_corpus)
lyric_corpus_tokenized = [sentence.split() for sentence in lyric_corpus]  
print(lyric_corpus_tokenized)

# fit the embedding (i.e., train word2vec)
#   - 100D embeddings (default)
#   - window size of 5 (5 words left, 5 words right)
#   - do not consider words that appear < 5 times
#   - use skip-gram approach
lyric_model = Word2Vec(lyric_corpus_tokenized, vector_size = 100, window = 5, min_count = 5, sg = 1)
lyric_embeddings = lyric_model.wv
lyric_vectors = lyric_embeddings.vectors 
lyric_vectors.shape # 1287 words, 100 dimensions

# access a particular word vector present in the corpus
word1  = 'love'
print(lyric_embeddings[word1]) # vector of length 100 for the word "love"

# query: what are the 5 most similar words for "love" within this corpus?
similar_words = lyric_embeddings.most_similar(word1, topn = 5)
for word, score in similar_words:
    print(f'Word: {word}, Cosine Similarity Score: {score:.5f}')

# query: how similar is "you" to "love"?
word2 = 'you'
lyric_embeddings.similarity(word1, word2) # medium similarity score

# query: how similar is "you" to "hate"?
word3 = 'hate'
lyric_embeddings.similarity(word3, word2) # slightly higher similarity score

'''
Interestingly, the most similar words to 'love' are other words that often appear 
with it (e.g., 'falling in love', 'this love', or 'true love') rather than other
words that are semantically similar (e.g., 'adore', 'enchantment', 'affection').
To address this issue, future iterations of this project should consider removing 
more stopwords. Regardless, 'you' is slightly more similar to 'hate' than 'love'.
To the subjects of Taylor's songs - yikes!
'''

# query (sanity check within Fearless (Taylor's Version)): how similar is "today" to "fairytale" vs. "romeo"?
word4 = 'today'
word5 = 'fairytale'
word6 = 'romeo'
lyric_embeddings.similarity(word4, word5) # should be high similarity (i.e., "today was a fairytale" from Today Was a Fairytale (Taylor's Version))
lyric_embeddings.similarity(word4, word6) # should be lower similarity (i.e., "romeo" is only written in Love Story (Taylor's Version))

# extract words for plot labeling
lyric_tsne = TSNE(n_components = 2, random_state = rand)
lyric_vectors_2d = lyric_tsne.fit_transform(lyric_vectors)
lyric_words = lyric_embeddings.index_to_key

# visualize embeddings with t-SNE generated scatter plot
plt.figure(figsize = (10, 10))
plt.scatter(lyric_vectors_2d[:, 0], lyric_vectors_2d[:, 1], marker = 'o') # plot points
for i, word in enumerate(lyric_words):
    x, y = lyric_vectors_2d[i]
    plt.annotate(word, (x, y), alpha = 0.5, fontsize = 8) # annotate points with word labels
plt.title('Word2Vec Embeddings Visualization for\nTaylor Swift Lyrics using t-SNE')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid()
plt.savefig('Embeddings.png')
plt.clf()

'''
There do not appear to be any clear clusters; most of the words 'hang together'.
This means that Taylor's key themes and vernacular are likely repeated across albums.
Zooming in on the plot though, (e.g.) 'remember' and 'memories' are nearby, as are 
'red' and 'burning', which both make sense!
'''

# create sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2') # fast, general purpose embedding from https://www.sbert.net/docs/pretrained_models.html
model

# encode each album as embedding vectors
TaylorSwift_embeddings = model.encode(TaylorSwift_Lyrics, convert_to_tensor = True)
FearlessTV_embeddings = model.encode(FearlessTV_Lyrics, convert_to_tensor = True)
SpeakNowTV_embeddings = model.encode(SpeakNowTV_Lyrics, convert_to_tensor = True)
RedTV_embeddings = model.encode(RedTV_Lyrics, convert_to_tensor = True)
NineteenEightyNine_embeddings = model.encode(NineteenEightyNine_Lyrics, convert_to_tensor = True)
Reputation_embeddings = model.encode(Reputation_Lyrics, convert_to_tensor = True)
Lover_embeddings = model.encode(Lover_Lyrics, convert_to_tensor = True)
Folklore_embeddings = model.encode(Folklore_Lyrics, convert_to_tensor = True)
Evermore_embeddings = model.encode(Evermore_Lyrics, convert_to_tensor = True)
Midnights_embeddings = model.encode(Midnights_Lyrics, convert_to_tensor = True)

# compute cosine similarities between every album and display in a matrix
embeddings = [TaylorSwift_embeddings, FearlessTV_embeddings, SpeakNowTV_embeddings,
              RedTV_embeddings, NineteenEightyNine_embeddings, Reputation_embeddings,
              Lover_embeddings, Folklore_embeddings, Evermore_embeddings, Midnights_embeddings]
cos_sims = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
cos_sims_df = pd.DataFrame(cos_sims, columns = ['TS', 'Fear', 'SN', 'Red', '1989', 'Rep',
                                                'Love', 'Folk', 'Ever', 'Mid'])
num_albums = len(embeddings)
for i in range(num_albums):
    embed = embeddings[i]
    for j in range(num_albums):
        embed2 = embeddings[j]
        tens_val = util.cos_sim(embed, embed2)
        val = tens_val.numpy()[0, 0]
        cos_sims_df.iloc[i, j] = val
new_index = {0: 'TS', 1: 'Fear', 2: 'SN', 3: 'Red', 4: '1989', 5: 'Rep', 6: 'Love',
             7: 'Folk', 8: 'Ever', 9: 'Mid'}
cos_sims_df_renamed = cos_sims_df.rename(index = new_index)
print(cos_sims_df_renamed)

'''
We see that 1989 is the most different from the other albums. I am a bit surprised
by this, as I would have expected this pop album to have a high cosine similarity to Lover,
another pop album, but it does not. Expectedly though, the sister indie albums Folklore 
and Evermore share among the greatest cosine similarities, at .615233.
'''

# examine how albums are associated with each other using multidimensional scaling
mds = MDS(n_components = 2, random_state = rand, normalized_stress = 'auto')
mds_fit = mds.fit_transform(cos_sims_df_renamed)
short_album_labels = ['TS', 'Fear', 'SN', 'Red', '1989', 'Rep', 'Love', 'Folk', 'Ever', 'Mid']
plt.scatter(mds_fit[:, 0], mds_fit[:, 1])
plt.xlabel('Coordinate 1')
plt.ylabel('Coordinate 2')
for i, txt in enumerate(df.index):
    lab = short_album_labels[txt]
    plt.annotate(lab, (mds_fit[:, 0][i], mds_fit[:, 1][i]))
plt.savefig('MDS.png')
plt.clf()

'''
Multidimensional scaling corroborates the conclusions above: 1989, positioned far
to the right along Coordinate 1, is quite different from the other albums, especially
(e.g.) Evermore, Speak Now (Taylor's Version), and Folklore. Multidimensional scaling
also reveals Taylor Swift to be quite differnt from the other albums as it is positioned
far to the top along Coordinate 2. The latter observation is less surprising, as Taylor
Swift was the first album recorded.
'''

# ------------------------------ topic modeling -------------------------------

# import additional packages
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# create a dictionary from the preprocessed lyrics
lyric_list = [TaylorSwift_Lyrics_list, FearlessTV_Lyrics_list, SpeakNowTV_Lyrics_list,
              RedTV_Lyrics_list, NineteenEightyNine_Lyrics_list, Reputation_Lyrics_list,
              Lover_Lyrics_list, Folklore_Lyrics_list, Evermore_Lyrics_list, Midnights_Lyrics_list]
lyric_series = pd.Series(lyric_list)
lyric_dict = Dictionary(lyric_series)
lyric_dict

# filter out words that appear in fewer than 2 albums or more than 80% of albums
lyric_dict.filter_extremes(no_below = 2, no_above = 0.8)
lyric_bow = [lyric_dict.doc2bow(text) for text in lyric_series] # create bag-of-words representation (tf-weighted DTM)

# train a latent dirichlet allocation model with K = 3 topics
#   - passes: how often we want to train the model on the entire corpus for convergence
#   - alpha: document-topic density (lower values of alpha --> returns documents with fewer topics)
#   - eta: topic word density (lower value of beta --> topics contains few words)
K = 3
n_words = 15
lyric_lda = LdaModel(lyric_bow, num_topics = K, id2word = lyric_dict, passes = 20, alpha = 'auto', eta = 'auto', random_state = rand)
lyric_lda.print_topics(num_words = n_words) # show 15 words with weight for each topic

# extract the topics (take top 15 words per topic)
lyric_topics = lyric_lda.show_topics(num_topics = K, num_words = n_words, log = False, formatted = False)

# print the topics with their top 15 words
for topic_id, topic in lyric_topics:
    print("Topic: {}".format(topic_id))
    print("Words: {}".format([word for word, _ in topic]))
    
# plot a word cloud for each topic (including the top 15 words)
for topic_id, topic in enumerate(lyric_lda.print_topics(num_topics = K, num_words = n_words)):
    topic_words = " ".join([word.split("*")[1].strip() for word in topic[1].split(" + ")])
    wordcloud = WordCloud(width = 800, height = 800, random_state = rand, max_font_size = 110).generate(topic_words)
    plt.figure()
    plt.imshow(wordcloud, interpolation = "bilinear")
    plt.axis("off")
    plt.title("Topic: {}".format(topic_id + 1))
    plt.show()
    wordcloud.to_file('Topic_{}.png'.format(topic_id + 1))
    
# print the topic associations with the albums
albums = ['Taylor Swift', 'Fearless (Taylor\'s Version)', 'Speak Now (Taylor\'s Version',
          'Red (Taylor\'s Version)', '1989', 'Reputation', 'Lover', 'Folklore', 'Evermore',
          'Midnights']
count = 0
for assoc in lyric_lda[lyric_bow]:
    print('Album: ', albums[count], assoc)
    count += 1

'''
The 1st topic (index 0) is strongly associated with the 1989 and Reputation albums,
and seems to emphasize fun or entertainment ('new york', 'dancing', 'game').

The 2nd topic (index 1) is strongly associated with the Folklore, Evermore, and Midnights
albums, and seems to emphasize contemplation or counterfactuals ('karma', 'wouldve', 
'different'). Interestingly, these are also the three most recent releases.

The 3rd topic (index 2) is strongly associated with the Taylor Swift, Fearless (Taylor's
Version), Speak Now (Taylor's Version), Red (Taylor's Version), and Lover albums, and
seems to emphasize nostalgia or longing ('feeling', 'wish', 'remember').
'''

# ------------------------ sentiment analysis: valence ------------------------

# import additional packages
from transformers import pipeline # Hugging Face transformers library
import pickle

# initialize model
sentiment_pipeline = pipeline(task = 'sentiment-analysis') # use default 'distilbert-base-uncased-finetuned-sst-2-english' model (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) 

# turn each album into lists of lyrics
tempTS = TaylorSwift_Lyrics_raw.split('\n')
tempTS = [i for i in tempTS if i]
tempFear = FearlessTV_Lyrics_raw.split('\n')
tempFear = [i for i in tempFear if i]
tempSN = SpeakNowTV_Lyrics_raw.split('\n')
tempSN = [i for i in tempSN if i]
tempRed = RedTV_Lyrics_raw.split('\n')
tempRed = [i for i in tempRed if i]
temp1989 = NineteenEightyNine_Lyrics_raw.split('\n')
temp1989 = [i for i in temp1989 if i]
tempRep = Reputation_Lyrics_raw.split('\n')
tempRep = [i for i in tempRep if i]
tempLove = Lover_Lyrics_raw.split('\n')
tempLove = [i for i in tempLove if i]
tempFolk = Folklore_Lyrics_raw.split('\n')
tempFolk = [i for i in tempFolk if i]
tempEver = Evermore_Lyrics_raw.split('\n')
tempEver = [i for i in tempEver if i]
tempMid = Midnights_Lyrics_raw.split('\n')
tempMid = [i for i in tempMid if i]

# fit sentiment model for each album OR read in fitted models from earlier pickle dump (fitting takes ~20 seconds per model, otherwise)
# sent_TS_pick = sentiment_pipeline(tempTS)
# pickle.dump(sent_TS_pick, file = open("sent_TS.file", "wb"))
sent_TS = pickle.load(open("sent_TS.file", "rb"))
# sent_Fear_pick = sentiment_pipeline(tempFear)
# pickle.dump(sent_Fear_pick, file = open("sent_Fear.file", "wb"))
sent_Fear = pickle.load(open("sent_Fear.file", "rb"))
# sent_SN_pick = sentiment_pipeline(tempSN)
# pickle.dump(sent_SN_pick, file = open("sent_SN.file", "wb"))
sent_SN = pickle.load(open("sent_SN.file", "rb"))
# sent_Red_pick = sentiment_pipeline(tempRed)
# pickle.dump(sent_Red_pick, file = open("sent_Red.file", "wb"))
sent_Red = pickle.load(open("sent_Red.file", "rb"))
# sent_1989_pick = sentiment_pipeline(temp1989)
# pickle.dump(sent_1989_pick, file = open("sent_1989.file", "wb"))
sent_1989 = pickle.load(open("sent_1989.file", "rb"))
# sent_Rep_pick = sentiment_pipeline(tempRep)
# pickle.dump(sent_Rep_pick, file = open("sent_Rep.file", "wb"))
sent_Rep = pickle.load(open("sent_Rep.file", "rb"))
# sent_Love_pick = sentiment_pipeline(tempLove)
# pickle.dump(sent_Love_pick, file = open("sent_Love.file", "wb"))
sent_Love = pickle.load(open("sent_Love.file", "rb"))
# sent_Folk_pick = sentiment_pipeline(tempFolk)
# pickle.dump(sent_Folk_pick, file = open("sent_Folk.file", "wb"))
sent_Folk = pickle.load(open("sent_Folk.file", "rb"))
# sent_Ever_pick = sentiment_pipeline(tempEver)
# pickle.dump(sent_Ever_pick, file = open("sent_Ever.file", "wb"))
sent_Ever = pickle.load(open("sent_Ever.file", "rb"))
# sent_Mid_pick = sentiment_pipeline(tempMid)
# pickle.dump(sent_Mid_pick, file = open("sent_Mid.file", "wb"))
sent_Mid = pickle.load(open("sent_Mid.file", "rb"))

# visualize output for Taylor Swift 
df_TS = pd.DataFrame(sent_TS)
df_TS['texts'] = tempTS
print(df_TS)
df_TS['label'].value_counts()
propTS = df_TS['label'].value_counts()[0] / (df_TS['label'].value_counts()[0] + df_TS['label'].value_counts()[1])
print(f'Positivity Proportion for Taylor Swift: {propTS}')

# visualize output for Fearless (Taylor's Version)
df_Fear = pd.DataFrame(sent_Fear)
df_Fear['texts'] = tempFear
print(df_Fear)
df_Fear['label'].value_counts()
propFear = df_Fear['label'].value_counts()[0] / (df_Fear['label'].value_counts()[0] + df_Fear['label'].value_counts()[1])
print(f'Positivity Proportion for Fearless (Taylor\'s Version): {propFear}')

# visualize output for Speak Now (Taylor's Version)
df_SN = pd.DataFrame(sent_SN)
df_SN['texts'] = tempSN
print(df_SN)
df_SN['label'].value_counts()
propSN = df_SN['label'].value_counts()[0] / (df_SN['label'].value_counts()[0] + df_SN['label'].value_counts()[1])
print(f'Positivity Proportion for Speak Now (Taylor\'s Version): {propSN}')

# visualize output for Red (Taylor's Version)
df_Red = pd.DataFrame(sent_Red)
df_Red['texts'] = tempRed
print(df_Red)
df_Red['label'].value_counts()
propRed = df_Red['label'].value_counts()[0] / (df_Red['label'].value_counts()[0] + df_Red['label'].value_counts()[1])
print(f'Positivity Proportion for Red (Taylor\'s Version): {propRed}')

# visualize output for 1989
df_1989 = pd.DataFrame(sent_1989)
df_1989['texts'] = temp1989
print(df_1989)
df_1989['label'].value_counts()
prop1989 = df_1989['label'].value_counts()[0] / (df_1989['label'].value_counts()[0] + df_1989['label'].value_counts()[1])
print(f'Positivity Proportion for 1989: {prop1989}')

# visualize output for Reputation
df_Rep = pd.DataFrame(sent_Rep)
df_Rep['texts'] = tempRep
print(df_Rep)
df_Rep['label'].value_counts()
propRep = df_Rep['label'].value_counts()[0] / (df_Rep['label'].value_counts()[0] + df_Rep['label'].value_counts()[1])
print(f'Positivity Proportion for Reputation: {propRep}')

# visualize output for Lover
df_Love = pd.DataFrame(sent_Love)
df_Love['texts'] = tempLove
print(df_Love)
df_Love['label'].value_counts()
propLove = df_Love['label'].value_counts()[0] / (df_Love['label'].value_counts()[0] + df_Love['label'].value_counts()[1])
print(f'Positivity Proportion for Lover: {propLove}')

# visualize output for Folklore
df_Folk = pd.DataFrame(sent_Folk)
df_Folk['texts'] = tempFolk
print(df_Folk)
df_Folk['label'].value_counts()
propFolk = df_Folk['label'].value_counts()[0] / (df_Folk['label'].value_counts()[0] + df_Folk['label'].value_counts()[1])
print(f'Positivity Proportion for Folklore: {propFolk}')

# visualize output for Evermore
df_Ever = pd.DataFrame(sent_Ever)
df_Ever['texts'] = tempEver
print(df_Ever)
df_Ever['label'].value_counts()
propEver = df_Ever['label'].value_counts()[0] / (df_Ever['label'].value_counts()[0] + df_Ever['label'].value_counts()[1])
print(f'Positivity Proportion for Evermore: {propEver}')

# visualize output for Midnights
df_Mid = pd.DataFrame(sent_Mid)
df_Mid['texts'] = tempMid
print(df_Mid)
df_Mid['label'].value_counts()
propMid = df_Mid['label'].value_counts()[0] / (df_Mid['label'].value_counts()[0] + df_Mid['label'].value_counts()[1])
print(f'Positivity Proportion for Midnights: {propMid}')

'''
Lover is the most positive album, which is expected; it is thought to encapsulate
a very happy period of Taylor's relationship with Joe Alywn and reflect a pivot from
Reputation, which described her frustrations with the media. Again, not surprisingly,
Folklore and Evermore are the least positive. Both are full of moody indie tracks and 
were released during the height of the COVID-19 pandemic.
'''

# ------------------------ sentiment analysis: emotion ------------------------

'''
The following function was originally written by Patrick Mair, PhD and Daniel Low
and was adapted slightly for this project. 
'''

# credit: https://github.com/danielmlow/tutorials/blob/main/text/sentiment_analysis_emotion.ipynb
def huggingface_to_df(output_dict):
	feature_names = [n.get('label') for n in output_dict[0]]
	feature_vectors = []
	for doc in output_dict:
		feature_vectors_doc = []
		for feature in doc:
			feature_vectors_doc.append(feature.get('score'))
		feature_vectors.append(feature_vectors_doc)
	feature_vectors = pd.DataFrame(feature_vectors, columns = feature_names)
	return feature_vectors

# initialize model (to detect sadness, joy, love, anger, fear, surprise) 
emotion_pipeline = pipeline(model = "bhadresh-savani/distilbert-base-uncased-emotion") # use emotion classification model (https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)

# fit sentiment model for each album OR read in fitted models from earlier pickle dump (fitting takes ~20 seconds per model, otherwise)
# emo_TS_pick = emotion_pipeline(tempTS, return_all_scores = True) # show all emotion scores, not just the top one
# pickle.dump(emo_TS_pick, file = open("emo_TS.file", "wb"))
emo_TS = pickle.load(open("emo_TS.file", "rb"))
# emo_Fear_pick = emotion_pipeline(tempFear, return_all_scores = True) # show all emotion scores, not just the top one
# pickle.dump(emo_Fear_pick, file = open("emo_Fear.file", "wb"))
emo_Fear = pickle.load(open("emo_Fear.file", "rb"))
# emo_SN_pick = emotion_pipeline(tempSN, return_all_scores = True) # show all emotion scores, not just the top one
# pickle.dump(emo_SN_pick, file = open("emo_SN.file", "wb"))
emo_SN = pickle.load(open("emo_SN.file", "rb"))
# emo_Red_pick = emotion_pipeline(tempRed, return_all_scores = True) # show all emotion scores, not just the top one
# pickle.dump(emo_Red_pick, file = open("emo_Red.file", "wb"))
emo_Red = pickle.load(open("emo_Red.file", "rb"))
# emo_1989_pick = emotion_pipeline(temp1989, return_all_scores = True) # show all emotion scores, not just the top one
# pickle.dump(emo_1989_pick, file = open("emo_1989.file", "wb"))
emo_1989 = pickle.load(open("emo_1989.file", "rb"))
# emo_Rep_pick = emotion_pipeline(tempRep, return_all_scores = True) # show all emotion scores, not just the top one
# pickle.dump(emo_Rep_pick, file = open("emo_Rep.file", "wb"))
emo_Rep = pickle.load(open("emo_Rep.file", "rb"))
# emo_Love_pick = emotion_pipeline(tempLove, return_all_scores = True) # show all emotion scores, not just the top one
# pickle.dump(emo_Love_pick, file = open("emo_Love.file", "wb"))
emo_Love = pickle.load(open("emo_Love.file", "rb"))
# emo_Folk_pick = emotion_pipeline(tempFolk, return_all_scores = True) # show all emotion scores, not just the top one
# pickle.dump(emo_Folk_pick, file = open("emo_Folk.file", "wb"))
emo_Folk = pickle.load(open("emo_Folk.file", "rb"))
# emo_Ever_pick = emotion_pipeline(tempEver, return_all_scores = True) # show all emotion scores, not just the top one
# pickle.dump(emo_Ever_pick, file = open("emo_Ever.file", "wb"))
emo_Ever = pickle.load(open("emo_Ever.file", "rb"))
# emo_Mid_pick = emotion_pipeline(tempMid, return_all_scores = True) # show all emotion scores, not just the top one
# pickle.dump(emo_Mid_pick, file = open("emo_Mid.file", "wb"))
emo_Mid = pickle.load(open("emo_Mid.file", "rb"))

# quickly visualize output for Taylor Swift (rows are the statements, columns are the emotion scores)
emo_df_TS = huggingface_to_df(emo_TS)
print(emo_df_TS)
ex1 = 0
ex2 = 250
ex3 = 500               
tempTS[ex1]      
emo_df_TS.iloc[ex1] # joy
tempTS[ex2]
emo_df_TS.iloc[ex2] # sadness
tempTS[ex3]
emo_df_TS.iloc[ex3] # joy

# comprehensively visualize output for Taylor Swift in a table and in a bar chart
emo_df_TS['highest scoring emotion'] = emo_df_TS.apply(lambda row: row.idxmax(), axis = 1)
print(emo_df_TS)
TS_value_counts = emo_df_TS['highest scoring emotion'].value_counts()
plt.bar(TS_value_counts.index, TS_value_counts.values, color = TS_col)
plt.xlabel('emotions', fontweight = 'bold')
plt.ylabel('number of lyric lines', fontweight = 'bold')
plt.title('What emotions are most frequently associated\nwith Taylor Swift lyrics?', fontweight = 'bold')
plt.savefig('TaylorSwift_bar.png')
plt.clf()

# comprehensively visualize output for Fearless (Taylor's Version) in a table and in a bar chart
emo_df_Fear = huggingface_to_df(emo_Fear)
emo_df_Fear['highest scoring emotion'] = emo_df_Fear.apply(lambda row: row.idxmax(), axis = 1)
print(emo_df_Fear)
Fear_value_counts = emo_df_Fear['highest scoring emotion'].value_counts()
plt.bar(Fear_value_counts.index, Fear_value_counts.values, color = Fear_col)
plt.xlabel('emotions', fontweight = 'bold')
plt.ylabel('number of lyric lines', fontweight = 'bold')
plt.title('What emotions are most frequently associated\nwith Fearless (Taylor\'s Version) lyrics?', fontweight = 'bold')
plt.savefig('FearlessTV_bar.png')
plt.clf()

# comprehensively visualize output for Speak Now (Taylor's Version) in a table and in a bar chart
emo_df_SN = huggingface_to_df(emo_SN)
emo_df_SN['highest scoring emotion'] = emo_df_SN.apply(lambda row: row.idxmax(), axis = 1)
print(emo_df_SN)
SN_value_counts = emo_df_SN['highest scoring emotion'].value_counts()
plt.bar(SN_value_counts.index, SN_value_counts.values, color = SN_col)
plt.xlabel('emotions', fontweight = 'bold')
plt.ylabel('number of lyric lines', fontweight = 'bold')
plt.title('What emotions are most frequently associated\nwith Speak Now (Taylor\'s Version) lyrics?', fontweight = 'bold')
plt.savefig('SpeakNowTV_bar.png')
plt.clf()

# comprehensively visualize output for Red (Taylor's Version) in a table and in a bar chart
emo_df_Red = huggingface_to_df(emo_Red)
emo_df_Red['highest scoring emotion'] = emo_df_Red.apply(lambda row: row.idxmax(), axis = 1)
print(emo_df_Red)
Red_value_counts = emo_df_Red['highest scoring emotion'].value_counts()
plt.bar(Red_value_counts.index, Red_value_counts.values, color = Red_col)
plt.xlabel('emotions', fontweight = 'bold')
plt.ylabel('number of lyric lines', fontweight = 'bold')
plt.title('What emotions are most frequently associated\nwith Red (Taylor\'s Version) lyrics?', fontweight = 'bold')
plt.savefig('RedTV_bar.png')
plt.clf()

# comprehensively visualize output for 1989 in a table and in a bar chart
emo_df_1989 = huggingface_to_df(emo_1989)
emo_df_1989['highest scoring emotion'] = emo_df_1989.apply(lambda row: row.idxmax(), axis = 1)
print(emo_df_1989)
NEN_value_counts = emo_df_1989['highest scoring emotion'].value_counts()
plt.bar(NEN_value_counts.index, NEN_value_counts.values, color = NEN_col)
plt.xlabel('emotions', fontweight = 'bold')
plt.ylabel('number of lyric lines', fontweight = 'bold')
plt.title('What emotions are most frequently associated\nwith 1989 lyrics?', fontweight = 'bold')
plt.savefig('1989_bar.png')
plt.clf()

# comprehensively visualize output for Reputation in a table and in a bar chart
emo_df_rep = huggingface_to_df(emo_Rep)
emo_df_rep['highest scoring emotion'] = emo_df_rep.apply(lambda row: row.idxmax(), axis = 1)
print(emo_df_rep)
rep_value_counts = emo_df_rep['highest scoring emotion'].value_counts()
plt.bar(rep_value_counts.index, rep_value_counts.values, color = Rep_col)
plt.xlabel('emotions', fontweight = 'bold')
plt.ylabel('number of lyric lines', fontweight = 'bold')
plt.title('What emotions are most frequently associated\nwith Reputation lyrics?', fontweight = 'bold')
plt.savefig('Reputation_bar.png')
plt.clf()

# comprehensively visualize output for Lover in a table and in a bar chart
emo_df_love = huggingface_to_df(emo_Love)
emo_df_love['highest scoring emotion'] = emo_df_love.apply(lambda row: row.idxmax(), axis = 1)
print(emo_df_love)
love_value_counts = emo_df_love['highest scoring emotion'].value_counts()
plt.bar(love_value_counts.index, love_value_counts.values, color = Love_col)
plt.xlabel('emotions', fontweight = 'bold')
plt.ylabel('number of lyric lines', fontweight = 'bold')
plt.title('What emotions are most frequently associated\nwith Lover lyrics?', fontweight = 'bold')
plt.savefig('Lover_bar.png')
plt.clf()

# comprehensively visualize output for Folklore in a table and in a bar chart
emo_df_folk = huggingface_to_df(emo_Folk)
emo_df_folk['highest scoring emotion'] = emo_df_folk.apply(lambda row: row.idxmax(), axis = 1)
print(emo_df_folk)
folk_value_counts = emo_df_folk['highest scoring emotion'].value_counts()
plt.bar(folk_value_counts.index, folk_value_counts.values, color = Folk_col)
plt.xlabel('emotions', fontweight = 'bold')
plt.ylabel('number of lyric lines', fontweight = 'bold')
plt.title('What emotions are most frequently associated\nwith Folklore lyrics?', fontweight = 'bold')
plt.savefig('Folklore_bar.png')
plt.clf()

# comprehensively visualize output for Evermore in a table and in a bar chart
emo_df_ever = huggingface_to_df(emo_Ever)
emo_df_ever['highest scoring emotion'] = emo_df_ever.apply(lambda row: row.idxmax(), axis = 1)
print(emo_df_ever)
ever_value_counts = emo_df_ever['highest scoring emotion'].value_counts()
plt.bar(ever_value_counts.index, ever_value_counts.values, color = Ever_col)
plt.xlabel('emotions', fontweight = 'bold')
plt.ylabel('number of lyric lines', fontweight = 'bold')
plt.title('What emotions are most frequently associated\nwith Evermore lyrics?', fontweight = 'bold')
plt.savefig('Evermore_bar.png')
plt.clf()

# comprehensively visualize output for Midnights in a table and in a bar chart
emo_df_mid = huggingface_to_df(emo_Mid)
emo_df_mid['highest scoring emotion'] = emo_df_mid.apply(lambda row: row.idxmax(), axis = 1)
print(emo_df_mid)
mid_value_counts = emo_df_mid['highest scoring emotion'].value_counts()
plt.bar(mid_value_counts.index, mid_value_counts.values, color = Mid_col)
plt.xlabel('emotions', fontweight = 'bold')
plt.ylabel('number of lyric lines', fontweight = 'bold')
plt.title('What emotions are most frequently associated\nwith Midnights lyrics?', fontweight = 'bold')
plt.savefig('Midnights_bar.png')
plt.clf()

'''
Wow, I did not anticipate that anger would be the most prevalent emotion for so many
albums! This is not surprising for Reputation, but very surprising for Taylor Swift.
I also thought that love would dominate more of the albums, especially for the 1st
four albums, Lover, and even the indie sisters. Perhaps this is such a coarse parsing
of emotional experience that 'anger' is capturing 'extreme feelings' rather than 'being
mad' - which Taylor does often express, but from listening to her music, seemingly not 
to this extent.
'''
