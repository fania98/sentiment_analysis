import json;
import numpy as np;
import jieba
import pickle
class save:
    def __init__(self):
        self.stop_word=[]
        self.contentids=[]
        self.predict_list=[]
        self.label=['positive','neutral','negative']
    def load_stopwords(self):
        with open('stop_words', 'r', encoding='utf-8-sig') as f:
            for line in f.readlines():
                word = line.strip()
                self.stop_word.append(word)

    def collect_content_test(self):
        content_dict = {}
        with open("dataset/data_test.json", encoding='utf-8') as f:
            contents = json.load(f)
            for content in contents:
                content_dict[content['id']] = content['news_comment'];
                # label_dict[content['id']] = label_to_id[content['polarity']];
        return content_dict;

    def cut_content(self,content_dict):
        content_cut = {}
        for key, value in content_dict.items():
            value = list(value);
            value = [c for c in value if c not in self.stop_word];
            value = ''.join(value);
            content = ' '.join(jieba.cut(value))
            content = content.split(' ');
            content = [w for w in content if w != '' and w not in self.stop_word]
            content_cut[key] = content;
        return content_cut;

    def generate_test_batch(self,batchsize=200):
        self.load_stopwords();
        seq_len = 20;
        index = 0;
        seq_num=0
        id_batch = np.zeros([batchsize, seq_len], dtype='int32');
        # label_batch = np.zeros([batchsize], dtype='int32');
        content_dict = self.collect_content_test();
        content_cut = self.cut_content(content_dict);
        with open('vocab_dict.pkl', 'rb') as f:
            vocab_dict = pickle.load(f);
            for key, value in content_cut.items():
                for j, word in enumerate(value):
                    if j < seq_len and word in vocab_dict:
                        id_batch[index][j] = vocab_dict[word];
                self.contentids.append(key)
                index += 1;
                seq_num+=1
                if index == batchsize or seq_num==3641:
                    print(len(self.contentids))
                    print(seq_num)
                    yield id_batch
                    index = 0;
                    id_batch = np.zeros([batchsize, seq_len], dtype='int32');

    # def loadjson(self,predicts,turn):
    #     with open('predict_json.json','a+',encoding='utf-8') as f:
    #         contentnum=len(self.contentids)
    #         for index in range(0,len(predicts)):
    #             if index<contentnum:
    #                 predict=self.label[predicts[index]]
    #                 contentid=self.contentids[index];
    #                 prediction={}
    #                 prediction['id']=contentid;
    #                 prediction['polarity']=predict
    #                 json.dump(prediction,f,ensure_ascii=False)
    #                 self.predict_list.append(prediction)
    #                 f.write('\n')
    #     self.contentids.clear()

    def loadjson(self,predicts,turn):
        contentnum=len(self.contentids)
        for index in range(0,len(predicts)):
            if index<contentnum:
                predict=self.label[predicts[index]]
                contentid=self.contentids[index];
                prediction={}
                prediction['id']=contentid;
                prediction['polarity']=predict

                self.predict_list.append(prediction)
        if turn==12:
            with open('predict_json.json', 'a+', encoding='utf-8') as f:
                json.dump(self.predict_list, f, ensure_ascii=False)
        self.contentids.clear()
