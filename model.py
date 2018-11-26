import tensorflow as tf;
from load_data import generate_batch;
import numpy as np;
import pickle;
import save;
import os;
import re;



class model:
    def __init__(self):
        self.seq_len=20;
        self.word_embedding = self.load_wordembedding();
        self.label_onehot=self.generate_label();
        self.x_input=tf.placeholder(tf.float32,[None,self.seq_len,64,1]);
        self.y_=tf.placeholder(tf.float32,shape=[None,3]);
        self.keep_prob=tf.placeholder(tf.float32);
        #卷积与池化
        self.conv=self.convoltion([2,3],self.seq_len);
        #全连接层
        self.y=self.full_connection();
        #交叉熵
        self.loss=self.calculate_loss();
        correct_prediction=tf.equal(tf.argmax(self.y,1),tf.arg_max(self.y_,1));

        self.predict=tf.arg_max(self.y,1)
        self.trainstep=tf.train.AdamOptimizer(1e-4).minimize(self.loss);
        self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32));
        self.saver = tf.train.Saver(max_to_keep=20);

    def generate_label(self):
        return np.identity(3);

    def load_wordembedding(self):
        with open('word_embedding_matrix.pkl','rb') as f:
            word_embedding=pickle.load(f);
        return word_embedding;

    def convoltion(self,size,seq_len):
        pooled_output=[];
        for i in size:
            W=tf.Variable(tf.truncated_normal([i,64,1,50],mean=0,stddev=0.01),name='W');
            b=tf.Variable(tf.truncated_normal([50],mean=0,stddev=0.01),name='b');
            h_conv1=tf.nn.relu(tf.nn.conv2d(self.x_input,W,strides=[1,1,1,1],padding='VALID')+b);
            pool=tf.nn.max_pool(h_conv1,[1,seq_len-i+1,1,1],strides=[1,1,1,1],padding='VALID');
            pooled_output.append(pool);

        h_pool=tf.concat(pooled_output,3);#每个文档的pool值排放在一起
        h_pool_dropout=tf.nn.dropout(h_pool,self.keep_prob);
        h_pool_flat=tf.reshape(h_pool_dropout,[-1,100]);
        return h_pool_flat

    def full_connection(self):
        W_fc1=tf.Variable(tf.truncated_normal([100,30],mean=0,stddev=0.01),name='W_fc1');
        b_fc1=tf.Variable(tf.truncated_normal([30],mean=0,stddev=0.1),name='b_fc1');
        h_fc1=tf.nn.relu(tf.matmul(self.conv,W_fc1)+b_fc1);
        h_fc1_drop=tf.nn.dropout(h_fc1,self.keep_prob);

        #W_fc2=tf.Variable(tf.truncated_normal([100,10],mean=0,stddev=0.01),name='W_fc2');
        #b_fc2=tf.Variable(tf.truncated_normal([30],mean=0,stddev=0.01),name='b_fc2');
        #h_fc2=tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2);
        #h_fc2_drop=tf.nn.dropout(h_fc2,self.keep_prob);

        W_fc3 = tf.Variable(tf.truncated_normal([30, 3], mean=0, stddev=0.01), name='W_fc2');
        b_fc3 = tf.Variable(tf.truncated_normal([3], mean=0, stddev=0.01), name='b_fc2');
        y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3);
        return y;

    def calculate_loss(self):
        loss=tf.reduce_mean(-tf.reduce_sum(self.y_*tf.log(self.y),reduction_indices=[1]))
        return loss;

    def train(self):
        #导入已转成id形式的语料
        data=generate_batch(batchsize=10);
        valid_data=generate_batch(batchsize=300)
        sess=tf.Session();
        sess1=tf.Session();
        batchx = tf.placeholder(tf.int32, shape=[None,20]);
        batchy = tf.placeholder(tf.int32, shape=[None])
        # 将id形式的语料embedding
        embedx = tf.nn.embedding_lookup(self.word_embedding, batchx);
        trainX = tf.expand_dims(embedx, -1);
        labelY = tf.nn.embedding_lookup(self.label_onehot, batchy);
        sess.run(tf.global_variables_initializer());
        turn = 1
        max_accuracy = 0;
        while(True):
            if (turn % 10 != 0):
                batch_X,batch_Y=next(data);
                X=sess1.run(trainX,feed_dict={batchx:batch_X})
                Y=sess1.run(labelY,feed_dict={batchy:batch_Y});
                #print(Y);
                #print(sess.run(self.y, feed_dict={self.x_input: X, self.y_: Y, self.keep_prob: 0.75}));
                sess.run(self.trainstep,feed_dict={self.x_input:X,self.y_:Y,self.keep_prob:0.75});
                print("正在训练，i= %g"%turn);
            else:
                batch_X, batch_Y = next(valid_data);
                X = sess1.run(trainX, feed_dict={batchx: batch_X})
                Y = sess1.run(labelY, feed_dict={batchy: batch_Y});
                #print(Y);
                #y = sess.run(self.y, feed_dict={self.x_input: X, self.y_: Y, self.keep_prob: 1})
                #print(y)
                accuracy=sess.run(self.accuracy, feed_dict={self.x_input: X, self.y_: Y, self.keep_prob: 1})
                if (accuracy > 0.8):
                    if accuracy > max_accuracy:
                        self.saver.save(sess, "ckpt1/", global_step=turn);
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                print("accuracy: %g"%accuracy);
                print("max_accuracy: %g"%max_accuracy);
            turn+=1;

    def valid(self):
        sess = tf.Session();
        model_file = tf.train.latest_checkpoint('ckpt1/')
        self.saver.restore(sess, model_file)
        #导入已转成id形式的语料
        data=generate_batch(batchsize=300);
        #valid_data=generate_batch(batchsize=300)
        sess1=tf.Session();
        batchx = tf.placeholder(tf.int32, shape=[None,20]);
        batchy = tf.placeholder(tf.int32, shape=[None])
        # 将id形式的语料embedding
        embedx = tf.nn.embedding_lookup(self.word_embedding, batchx);
        trainX = tf.expand_dims(embedx, -1);
        labelY = tf.nn.embedding_lookup(self.label_onehot, batchy);
        turn = 1
        #max_accuracy = 0;
        for i in range(0,10):
                batch_X, batch_Y = next(data);
                X = sess1.run(trainX, feed_dict={batchx: batch_X})
                Y = sess1.run(labelY, feed_dict={batchy: batch_Y});
                #print(Y);
                #y=sess.run(self.y,feed_dict={self.x_input: X, self.y_: Y, self.keep_prob: 1})
                #print(y)
                accuracy=sess.run(self.accuracy, feed_dict={self.x_input: X, self.y_: Y, self.keep_prob: 1})
                #if accuracy > max_accuracy:
                   # max_accuracy = accuracy
                #if(max_accuracy>0.99):
                    #if accuracy == max_accuracy:
                        #self.saver.save(sess, "ckpt1/", global_step=turn);
                print("accuracy: %g"%accuracy);
                #print("max_accuracy: %g"%max_accuracy);
            #turn+=1

    def test(self):
        sess = tf.Session();
        model_file = tf.train.latest_checkpoint('ckpt1/')
        self.saver.restore(sess, model_file)
        # 导入已转成id形式的语料
        saver=save.save();
        data = saver.generate_test_batch(batchsize=300);
        # valid_data=generate_batch(batchsize=300)
        sess1 = tf.Session();
        batchx = tf.placeholder(tf.int32, shape=[None, 20]);
        # 将id形式的语料embedding
        embedx = tf.nn.embedding_lookup(self.word_embedding, batchx);
        trainX = tf.expand_dims(embedx, -1);
        turn = 1
        # max_accuracy = 0;
        for i in range(0, 13):
            batch_X = next(data);
            X = sess1.run(trainX, feed_dict={batchx: batch_X})
            predict=sess.run(self.predict, feed_dict={self.x_input: X, self.keep_prob: 1})
            saver.loadjson(predict,i)
            # if accuracy > max_accuracy:
            # max_accuracy = accuracy
            # if(max_accuracy>0.99):
            # if accuracy == max_accuracy:
            # self.saver.save(sess, "ckpt1/", global_step=turn);
            #print("accuracy: %g" % accuracy);



Model=model();
Model.test();