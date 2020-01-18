import tensorflow as tf
import numpy as np
import pickle
from load_data import generate_batch
from load_data import generate_valid_batch
import save
class rnn:
    def __init__(self):
        self.batch_size = 32;
        self.num_steps = 20;
        self.hidden_size=128;
        self.max_grad_norm=5
        self.word_embedding=self.load_wordembedding();
        self.label_onehot=self.generate_label()
        self.keep_prob=tf.placeholder(tf.float32)
        self.input_x = tf.placeholder(tf.int32,[None,self.num_steps],name='input_x')
        self.input_y = tf.placeholder(tf.int32,[None]);
        self.x=tf.nn.embedding_lookup(self.word_embedding,self.input_x)
        self.x=tf.cast(self.x, tf.float32)

        #print(self.x)
        self.y_=tf.cast(tf.nn.embedding_lookup(self.label_onehot,self.input_y),tf.float32)
        #self.lstmCell_layer1=self.BIRNN(self.hidden_size)
        #output1,_=tf.nn.dynamic_rnn(cell=self.lstmCell_layer1,inputs=self.x,dtype=tf.float32)
        #print(output1)
        #self.lstmCell_layer2=self.BIRNN(self.hidden_size);
        #output2, _ = tf.nn.dynamic_rnn(cell=self.lstmCell_layer2, inputs=output1, dtype=tf.float32)
        self.rnnOutput= self.BIRNN(self.hidden_size,self.x)
        #self.rnnOutput=tf.concat(outputs,2)
        #print(rnnOutput1)
        #self.rnnOutput=self.RNN(self.hidden_size,rnnOutput1)
       # print(self.rnnOutput)
        self.rnnOutput=self.rnnOutput[-1]
        #self.rnnOutput=output[:,self.num_steps-1,:]
        self.y=self.full_connection(self.rnnOutput)
        self.cost =  self.calculate_loss();
        tvar=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(self.cost,tvar),self.max_grad_norm)
        optimizer=tf.train.AdamOptimizer();
        self.train_op=optimizer.apply_gradients(zip(grads,tvar))

        self.predict=tf.arg_max(self.y,1)
        correct_prediction = tf.equal(self.predict, tf.arg_max(self.y_, 1));
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
        self.saver = tf.train.Saver(max_to_keep=10);

    def BIRNN(self,size,input):
        x = tf.transpose(input, [1, 0, 2])
        x = tf.reshape(x, [-1, 64])
        x = tf.split(x, self.num_steps)
        #self.input_x=tf.transpose(self.input_x,[1,0,2])
        #w=tf.Variable(tf.truncated_normal([2*self.hidden_size,128], mean=0, stddev=0.01), name='w_combine')
        #b=tf.Variable(tf.truncated_normal([128], mean=0, stddev=0.01), name='w_combine')
        fwcell=tf.nn.rnn_cell.BasicLSTMCell(size,forget_bias=0.6,state_is_tuple=True)
        bwcell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.6, state_is_tuple=True)
        drop1=tf.nn.rnn_cell.DropoutWrapper(fwcell,output_keep_prob=self.keep_prob)
        drop2=tf.nn.rnn_cell.DropoutWrapper(bwcell,output_keep_prob=self.keep_prob)
        #output1,_=tf.nn.dynamic_rnn(cell=cell1,inputs=self.x,dtype=tf.float32)
        #cell2=tf.nn.rnn_cell.BasicLSTMCell(size,forget_bias=0.0,state_is_tuple=True)
        #bwcell=tf.nn.rnn_cell.BasicLSTMCell(size,forget_bias=0.0,state_is_tuple=True)
        outputs,_,_=tf.nn.static_bidirectional_rnn(drop1, drop2, inputs=x, dtype=tf.float32)

        return outputs;
        #return tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=self.keep_prob)

    def RNN(self,size,input):
        cell=tf.nn.rnn_cell.BasicLSTMCell(size,forget_bias=0.5,state_is_tuple=True)
        drop1 = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        output,_=tf.nn.static_rnn(drop1,inputs=input,dtype=tf.float32);
        return output

    def generate_label(self):
        return np.identity(3);


    def load_wordembedding(self):
        with open('word_embedding_matrix_C.pkl', 'rb') as f:
            word_embedding = pickle.load(f);
        return word_embedding;


    def convoltion(self, size, seq_len):
        pooled_output = [];
        for i in size:
            W = tf.Variable(tf.truncated_normal([i, 64, 1, 32], mean=0, stddev=0.01), name='W');
            b = tf.Variable(tf.truncated_normal([32], mean=0, stddev=0.01), name='b');
            h_conv1 = tf.nn.relu(tf.nn.conv2d(self.x_input, W, strides=[1, 1, 1, 1], padding='VALID') + b);
            # print(h_conv1)
            # print(seq_len-i)
            pool = tf.nn.max_pool(h_conv1, [1, (seq_len - i + 1) / 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID');
            # print(pool)
            # topkpool=tf.nn.top_k(h_conv1,k=2);
            # print(topkpool[0])
            # print(topkpool[1])
            pooled_output.append(pool);
            print(pooled_output)
            # reg_loss=tf.contrib.layers.l2_regularizer(0.5)(W);
            # tf.add_to_collection("reg_losses",reg_loss);

        h_pool = tf.concat(pooled_output, 3);  # 每个文档的pool值排放在一起
        print(h_pool)
        h_pool_dropout = tf.nn.dropout(h_pool, self.keep_prob);
        h_pool_flat = tf.reshape(h_pool_dropout, [-1, 384]);
        return h_pool_flat


    def full_connection(self,input):
        W_fc1 = tf.Variable(tf.truncated_normal([256, 64], mean=0, stddev=0.01), name='W_fc1');
        b_fc1 = tf.Variable(tf.truncated_normal([64], mean=0, stddev=0.1), name='b_fc1');
        h_fc1 = tf.nn.relu(tf.matmul(input, W_fc1) + b_fc1);
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob);
        # reg_loss = tf.contrib.layers.l2_regularizer(0.5)(W_fc1);
        # tf.add_to_collection("reg_losses", reg_loss);
        # W_fc2=tf.Variable(tf.truncated_normal([128,32],mean=0,stddev=0.01),name='W_fc2');
        # b_fc2=tf.Variable(tf.truncated_normal([32],mean=0,stddev=0.01),name='b_fc2');
        # h_fc2=tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2);
        # h_fc2_drop=tf.nn.dropout(h_fc2,self.keep_prob);

        W_fc3 = tf.Variable(tf.truncated_normal([64, 3], mean=0, stddev=0.01), name='W_fc2');
        b_fc3 = tf.Variable(tf.truncated_normal([3], mean=0, stddev=0.01), name='b_fc2');
        # reg_loss = tf.contrib.layers.l2_regularizer(0.5)(W_fc3);
        # tf.add_to_collection("reg_losses", reg_loss);
        y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3);
        return y;


    def calculate_loss(self):
        # reg_loss=tf.add_n(tf.get_collection('reg_losses'));
        loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.cast(tf.log(self.y),tf.float32), reduction_indices=[1]));
        return loss;


    def train(self):
        # 导入已转成id形式的语料
        data = generate_batch(batchsize=self.batch_size);
        valid_data=generate_batch(batchsize=1000)
        data_valid = generate_valid_batch(batchsize=3979);
        sess = tf.Session();
        sess.run(tf.global_variables_initializer());
        turn = 1
        max_accuracy = 0;
        while (True):
            if (turn % 50 != 0):
                batch_X, batch_Y = next(data);

                # print(Y);
                # print(sess.run(self.y, feed_dict={self.x_input: X, self.y_: Y, self.keep_prob: 0.75}));
                sess.run(self.train_op, feed_dict={self.input_x: batch_X, self.input_y: batch_Y, self.keep_prob: 0.7});
                # print("正在训练，i= %g"%turn);
            else:
                batch_X1,batchY1=next(valid_data)
                batch_X, batch_Y = next(data_valid)
                #batchx_valid, batchy_valid = next(data_valid);
                # print(Y);
                # y = sess.run(self.y, feed_dict={self.x_input: X, self.y_: Y, self.keep_prob: 1})
                # print(y)
                accuracy = sess.run(self.accuracy,feed_dict={self.input_x: batch_X, self.input_y: batch_Y, self.keep_prob: 1})
                accuracy2 = sess.run(self.accuracy, feed_dict={self.input_x: batch_X1, self.input_y: batchY1, self.keep_prob: 1})
                if (accuracy > 0.70 and accuracy2>0.72):
                    if accuracy > max_accuracy:
                        self.saver.save(sess, "ckpt2_C/", global_step=turn);
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                print("valid accuracy: %g" % accuracy)
                print("train accuracy: %g" %accuracy2)
                print("max_accuracy: %g" % max_accuracy);
               # print("valid_accuracy: %g" % accuracy2);
            turn += 1;


    def valid(self):
        sess = tf.Session();
        model_file = tf.train.latest_checkpoint('ckpt3_C/')
        self.saver.restore(sess, model_file)
        # 导入已转成id形式的语料
        data = generate_valid_batch(batchsize=1000);
        # valid_data=generate_batch(batchsize=300)

        # max_accuracy = 0;
        for i in range(0, 10):
            batch_X, batch_Y = next(data);
            # print(Y);
            # y=sess.run(self.y,feed_dict={self.x_input: X, self.y_: Y, self.keep_prob: 1})
            # print(y)
            accuracy = sess.run(self.accuracy, feed_dict={self.input_x: batch_X, self.input_y: batch_Y, self.keep_prob: 1})
            # if accuracy > max_accuracy:
            # max_accuracy = accuracy
            # if(max_accuracy>0.99):
            # if accuracy == max_accuracy:
            # self.saver.save(sess, "ckpt1/", global_step=turn);
            print("valid_accuracy: %g" % accuracy);
            # print("max_accuracy: %g"%max_accuracy);
        # turn+=1


    def test(self):
        sess = tf.Session();
        model_file = tf.train.latest_checkpoint('ckpt3_C/')
        self.saver.restore(sess, model_file)
        # 导入已转成id形式的语料
        saver = save.save();
        data = saver.generate_test_batch(batchsize=300);
        # valid_data=generate_batch(batchsize=300)
        # max_accuracy = 0;
        for i in range(0, 29):
            batch_X = next(data);
            predict = sess.run(self.predict, feed_dict={self.input_x: batch_X, self.keep_prob: 1})
            saver.loadjson(predict, i)
            # if accuracy > max_accuracy:
            # max_accuracy = accuracy
            # if(max_accuracy>0.99):
            # if accuracy == max_accuracy:
            # self.saver.save(sess, "ckpt1/", global_step=turn);
            # print("accuracy: %g" % accuracy);

Model=rnn();
Model.train();