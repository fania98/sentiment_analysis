import jieba;
import json;
from gensim.models import word2vec
import pickle;
import numpy as np;
import gensim
import random



stop_word=[]
label_to_id={'positive':0,'neutral':1,'negative':2}
def load_stopwords():
    with open('stop_words','r',encoding='utf-8-sig') as f:
        for line in f.readlines():
            word=line.strip()
            stop_word.append(word)

#读取原本的json文件
def collect_contents():
    content_dict = {}
    label_dict={}
    with open("dataset_C/data_train.json",encoding='utf-8') as f:
         contents=json.load(f)
         for content in contents:
            content_dict[content['id']]=content['news_comment'];
            label_dict[content['id']]=label_to_id[content['polarity']];
    return content_dict,label_dict;

def collect_valid_contents():
    content_dict = {}
    label_dict={}
    with open("dataset_C/data_valid.json",encoding='utf-8') as f:
         contents=json.load(f)
         for content in contents:
            content_dict[content['id']]=content['news_comment'];
            label_dict[content['id']]=label_to_id[content['polarity']];
    return content_dict,label_dict;


def collect_content_test():
    content_dict = {}
    with open("dataset_C/data_test.json", encoding='utf-8') as f:
        contents = json.load(f)
        for content in contents:
            content_dict[content['id']] = content['news_comment'];
            #label_dict[content['id']] = label_to_id[content['polarity']];
    return content_dict;

def collect_contents_for_vocab():
    content_list=[]
    with open("dataset_C/data_test.json",encoding='utf-8') as f:
        contents=json.load(f);
        for content in contents:
            content_list.append(content['news_comment']);
    with open("dataset_C/data_valid.json",encoding='utf-8') as f:
        contents = json.load(f);
        for content in contents:
            content_list.append(content['news_comment']);

    with open("dataset_C/data_test.json",encoding='utf-8') as f:
        contents = json.load(f);
        for content in contents:
            content_list.append(content['news_comment']);
    return content_list

def cut_content_for_vocab():
    content_list=collect_contents_for_vocab();
    content_cut_list=[];
    for value in content_list:
        value=list(value);
        value=[c for c in value if c not in stop_word];
        value=''.join(value);
        content = ' '.join(jieba.cut(value))
        content=content.split(' ');
        content=[w for w in content if w !=''and w not in stop_word]
        content_cut_list.append(content)
    return content_cut_list;


#将评论内容分词
def cut_content(content_dict):
    content_cut={}
    for key,value in content_dict.items():
        value=list(value);
        value=[c for c in value if c not in stop_word];
        value=''.join(value);
        content = ' '.join(jieba.cut(value))
        content=content.split(' ');
        content=[w for w in content if w !=''and w not in stop_word]
        content_cut[key]=content;
    return content_cut;

#将分词内容储存在文件中
def save_words():
    load_stopwords();
    #content_list=collect_contents_for_vocab();
    content_cut_list=cut_content_for_vocab();
    with open("word_C.txt",'a+',encoding='utf-8')as f:
        for value in content_cut_list:
            for word in value:
                f.write(word+" ")

def embedding():
    sentences = word2vec.Text8Corpus(r'word.txt')
    model=word2vec.Word2Vec(sentences,window=6,size=150)

#建立词到id的词典
def build_vocabdict():
    vocab_dict={}
    max_size=30000
    with open("word_C.txt",'r',encoding='utf-8') as f:
        for line in f:
            wordlist=line.split(" ")
            for word in wordlist:
                if word in vocab_dict:
                    vocab_dict[word]+=1;
                else:
                    vocab_dict[word]=1;
    print(len(vocab_dict))
    vocab_size = min(max_size, len(vocab_dict))
    vocab_list = sorted(vocab_dict, key=vocab_dict.get, reverse=True)
    vocab_list = vocab_list[1:vocab_size]
    print(vocab_list)
    vocab_dict.clear()
    for index,word in enumerate(vocab_list):
        vocab_dict[word]=index
    print(len(vocab_dict))
    print(vocab_dict)
    with open("vocab_dict_C.pkl",'wb')as f:
        pickle.dump(vocab_dict,f);

#建立id到词向量的映射矩阵
def embedding_matrix():
    word_vector = gensim.models.KeyedVectors.load_word2vec_format('G:/SCUAI/news_12g_baidubaike_20g_novel_90g_embedding_64.bin', binary=True)
    with open('vocab_dict_C.pkl', 'rb') as vocabf:
        vocab_dict = pickle.load(vocabf);
    word_embedding = np.zeros((len(vocab_dict), 64));
    for word, id in vocab_dict.items():
        if word in word_vector:
            word_embedding[id] = word_vector[word];
        elif word=='！':
            word_embedding[id,:]=0.1;
        elif word=='？':
            word_embedding[id,:]=0.2;
    with open('word_embedding_matrix_C.pkl', 'wb') as matrixf:
        pickle.dump(word_embedding, matrixf);

def generate_batch(batchsize=25):
    seq_len=20;
    index=0;
    id_batch = np.zeros([batchsize, seq_len], dtype='int32');
    label_batch = np.zeros([batchsize], dtype='int32');
    content_dict,label_dict = collect_contents();
    content_cut = cut_content(content_dict);
    with open('vocab_dict_C.pkl','rb') as f:
        vocab_dict=pickle.load(f);
        #print(vocab_dict)
    while 1:
        keys=content_cut.keys();
        keys=list(keys)
        random.shuffle(keys)
        for key in keys:
            value=content_cut[key]
            #print(key)
            #print(content_dict[key])
            #print(value)
            #print(label_dict[key])
            for i,word in enumerate(value):
                if i<seq_len and word in vocab_dict:
                    id_batch[index][i]=vocab_dict[word];

            label_batch[index]=label_dict[key]
            index+=1;
            if index==batchsize:
                #print(content_dict)
                #print(label_batch)
                yield id_batch,label_batch;
                index=0;
                id_batch = np.zeros([batchsize, seq_len], dtype='int32');
                label_batch = np.zeros([batchsize], dtype='int32');



def convertjson():
    json_list=[]
    with open("dataset_C/trian",encoding='utf-8') as f:
        for line in f:
            content=json.loads(line);
            json_list.append(content)
    with open('dataset_C/data_train.json','w+',encoding='utf-8') as f:
        json.dump(json_list,f,ensure_ascii=False)

def see_batch(batchsize=25):
    seq_len=20;
    index=0;
    content_list=[]
    content_dict = collect_contents();
    content_cut = cut_content(content_dict);
    with open('vocab_dict.pkl','rb') as f:
        vocab_dict=pickle.load(f);
        #print(vocab_dict)
    while 1:
        keys=content_cut.keys();
        keys=list(keys)
        random.shuffle(keys)
        for key in keys:
            value=content_cut[key]
            #print(key)
            #print(content_dict[key])
            #print(value)
            #print(label_dict[key])
            for i,word in enumerate(value):
                if i<seq_len and word in vocab_dict:
                    content_list.append(word);
            print(content_list);
            content_list.clear();
            #print(label_dict[key]);
            index+=1;
            if index==batchsize:
                index=0;
                yield content_list;

def generate_valid_batch(batchsize=25):
    load_stopwords();
    seq_len=20;
    index=0;
    id_batch = np.zeros([batchsize, seq_len], dtype='int32');
    label_batch = np.zeros([batchsize], dtype='int32');
    content_dict,label_dict = collect_valid_contents();
    content_cut = cut_content(content_dict);
    with open('vocab_dict_C.pkl','rb') as f:
        vocab_dict=pickle.load(f);
        #print(vocab_dict)
    while 1:
        keys=content_cut.keys();
        keys=list(keys)
        random.shuffle(keys)
        for key in keys:
            value=content_cut[key]
            #print(key)
            #print(content_dict[key])
            #print(value)
            #print(label_dict[key])
            for i,word in enumerate(value):
                if i<seq_len and word in vocab_dict:
                    id_batch[index][i]=vocab_dict[word];

            label_batch[index]=label_dict[key]
            index+=1;
            if index==batchsize:
                #print(content_dict)
                #print(label_batch)
                yield id_batch,label_batch;
                index=0;
                id_batch = np.zeros([batchsize, seq_len], dtype='int32');
                label_batch = np.zeros([batchsize], dtype='int32');

