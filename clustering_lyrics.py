from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

#モデルの読み込み
m = Doc2Vec.load("./model/JAM_model.model")

#モデルデータの各ベクトル、それに対応する曲名をリストに格納
vectors_li = []
vectitle_li = []
for n in range(len(m.docvecs)):
    vectors_li.append(m.docvecs[n])
    vectitle_li.append(m.docvecs.index_to_doctag(n))

#ドキュメント番号のリスト
doc_nums = range(0,len(m.docvecs))

#クラスタリングの設定
n_clusters = 8
#KMeans()...verboseは謎、n_jobsは何コアを使って並列処理をするかのパラメータっぽい(-1を指定すると落ちる)
kmeans_model = KMeans(n_clusters=n_clusters,verbose=0,random_state=1)

#クラスタリング実行
kmeans_model.fit(vectors_li)

#ラベルづけ
labels = kmeans_model.labels_
print(str(len(labels)))

#ラベル+ドキュメント番号の辞書を作成
cluster_to_docs = defaultdict(list)
for cluster_id,doc_num in zip(labels,doc_nums):
    cluster_to_docs[cluster_id].append(vectitle_li[doc_num])

#クラスター結果を出力
for docs in cluster_to_docs.values():
    print(docs)
