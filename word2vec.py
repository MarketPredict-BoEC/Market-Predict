from gensim.models import Word2Vec
import gensim




from sklearn.decomposition import PCA
from matplotlib import pyplot
import nltk
from nltk.corpus import stopwords
#step 3 build word2vec model 

nltk.download('punkt')

fileHandle = open('Preprocess MyNewsCourpus non lemmatized version.txt','r')
data = fileHandle.readlines()

text = [nltk.word_tokenize(w) for w in data]
conceptwindow = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
filenames = [ 'ForexNews'+ str(i)+'.embedding' for i in conceptwindow]
def removeStopwords(s):
    stop_words = set(stopwords.words('english'))
    
    s= [w for w in s if not w in stop_words]
    return s

text= [removeStopwords(w) for w in text ] 
# train model
i = 0
for item in filenames:
    #model = Word2Vec(text, size=100, window=item, min_count=3, workers=4 )
    #model.save(filenames[i]);
    #model.wv.save_word2vec_format('ForexNewslemmatize.txt', binary=False)
    
    new_model = gensim.models.Word2Vec.load(item)
    vectors = new_model.wv

    print(vectors.most_similar('war'))
    print(vectors.most_similar('Iran'))
    print('+----------------------------------------+')



