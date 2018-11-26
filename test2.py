from gensim.models import word2vec
#import gensim;

model=word2vec.Word2Vec.load("G:/SCUAI/word/wordvec300.bin")
#model = gensim.models.KeyedVectors.load_word2vec_format('G:/SCUAI/news_12g_baidubaike_20g_novel_90g_embedding_64.bin', binary=True)

print(model.most_similar("不好", topn=10))
#print(model.most_similar("？"),topn=10)
#print(model.most_similar("可以"),topn=10)
print(model.most_similar("恶心",topn=10))
print(model.most_similar("差",topn=10))
print(model.most_similar("垃圾",topn=10))
print(model.most_similar("棒",topn=10))