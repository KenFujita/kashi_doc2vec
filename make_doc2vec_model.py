import MeCab
import numpy as np
import nltk
import re
import sys
from sklearn.feature_extraction import stop_words
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# 意味のない単語を削除
stop_words_sklearn = stop_words.ENGLISH_STOP_WORDS
nltk.download('stopwords')
stop_words_nltk = nltk.corpus.stopwords.words('english')
stop_words_nltk.extend(stop_words_sklearn)
stop_words_all = stop_words_nltk
stop_words_jp = ["それ","てる","よう","こと","の","し","い","ん","れ"]
stop_words_all.extend(stop_words_jp)

# 形態素解析の設定
hinshi = ("名詞","形容詞","動詞")
sub_hinshi = ("接尾","数")             #意味のない単語を更に消すため(99.9%の原因?)
tagger = MeCab.Tagger("-Ochasen")
tagger.parse('')                    #入れないと一部surfaceが表示されない

traning_docs = []                   # モデル作成のための文書のリスト

def create_countvectorizer_from_texts(name,kashi):
    sent = TaggedDocument(words=kashi,tags=[name])
    traning_docs.append(sent)

def parse_by_mecab(k_name,docs):
    for i,kashi in enumerate(docs):
        keitaiso = []
        kashi = re.sub('[!-/:-@[-`{-~]','',kashi)   # 半角記号を取り除く
        result = tagger.parseToNode(kashi)
        while result:
            #print("feature: {}, surface: {}".format(result.feature,result.surface))
            if(result.feature.split(",")[0] in hinshi and result.feature.split(",")[1] not in sub_hinshi and result.surface not in stop_words_all):
                keitaiso.append(result.surface)
            result = result.next
        create_countvectorizer_from_texts(k_name[i],keitaiso)

def prepare_data(fname):
    kyoku_li = []                       # 曲リスト
    kyoku_name = []                     # 曲のタイトルリスト
    kashi_list = []                     # 全曲の歌詞リスト
    with open(fname,"r") as f:          
        text_list = f.readlines()       # テキストデータの中身は「title,lyrics」が1行につき1曲分書き込まれている

    for text in text_list:
        kyoku_li.append(text.split(","))

    for kyoku in kyoku_li:
        kyoku_name.append(kyoku[0])
        kashi_list.append(kyoku[1].replace('\n',''))

    return [kyoku_name,kashi_list]

def create_doc2vec_model():
    # 正直パラメータについてはサンプルを見て設定してうまくいった
    # URL: https://deepage.net/machine_learning/2017/01/08/doc2vec.html
    # しっかり勉強しないとうまく調整はできない
    model = Doc2Vec(min_cnt=1,dm=1,vector_size=300,window=15,alpha=.045,min_alpha=.0001,sample=1e-5,epochs=500)
    model.build_vocab(traning_docs)
    model.train(traning_docs,total_examples=model.corpus_count,epochs=model.epochs)
    model.save("/path/to/model")                            # 保存する名前を指定
    #model = Doc2Vec(documents=traning_docs,min_cnt=1,dm=1)  #類似TOP3ほぼ99.9%の一致率
    return model

if __name__ == '__main__':
    title = []                          # 曲のタイトルリスト
    lyrics = []                         # 各曲の歌詞リスト

    title,lyrics = prepare_data(sys.argv[1])
    parse_by_mecab(title,lyrics)
    m = create_doc2vec_model()

    for t in title:
        print(t+": ")
        print(m.docvecs.most_similar(t,topn=1))             # それぞれの歌詞に一番近しい曲のタイトルを表示
    #print(model.docvecs.similarity(1,1))   確認用
