import tensorflow as tf
import numpy as np
import math
import os
from pylab import *
from data import *
from evaluation import *
import argparse


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepRank.")
    parser.add_argument('--path', nargs='?', default='datasets',
                        help='Input data path.')
    parser.add_argument('--data_name', nargs='?', default='ml-100k/u.data',
                        help='Choose a dataset.')
    parser.add_argument('--epoches', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--user_factors', type=int, default=16,
                        help='Embedding size of users.')
    parser.add_argument('--item_factors', type=int, default=16,
                        help='Embedding size of items.')
    parser.add_argument('--layers', nargs='?', default='[16,8]',
                        help="Size of each layer. Note that the first hidden layer is the interaction layer.")
    parser.add_argument('--reg', type=float, default=0.00001,
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--list_length', type=int, default=5,
                        help='Length of list for training.')
    parser.add_argument('--num_positive', type=int, default=2,
                        help='Number of positive instances in train list.')
    parser.add_argument('--sample_time', type=int, default=2,
                        help='Times of sample from instances.')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top_n list for recommendation.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--min_loss', type=float, default=0.01,
                        help='The minimum loss value for stopping training.')
    parser.add_argument('--path_model', nargs='?', default='model',
                        help='Output path for saving pre_trained model.')
    return parser.parse_args()


class DeepRank():
    def __init__(self,               
                 users_num = None,             # users number
                 items_num = None,             # items number
                 batch_size = 512,             # batch size
                 embedding_size_users = 16,       # embedding size of users
                 embedding_size_items = 16,       # embedding size of items
                 hidden_size = [16,8],          # hidden layers
                 list_length = 5,             # list lenth
                 learning_rate = 1e-3,          # learning rate
                 lamda_regularizer = 1e-4,       # regularization coefficient for L2
                 model_path = 'model'          # path for saving pre_trained model
                ):
        self.users_num = users_num
        self.items_num = items_num
        self.batch_size = batch_size
        self.embedding_size_users = embedding_size_users
        self.embedding_size_items = embedding_size_items
        self.hidden_size = hidden_size
        self.list_length = list_length
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.model_path = model_path

        # loss records
        self.train_loss_records, self.results = [],[]   
        self.build_graph()   

        
    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            # _________ input data _________
            self.user_inputs = tf.placeholder(tf.int32, shape = [None,self.list_length], name='user_inputs')
            self.item_inputs = tf.placeholder(tf.int32, shape = [None, self.list_length], name='item_inputs')
            self.train_labels = tf.placeholder(tf.float32, shape = [None,self.list_length], name='train_labels') 
            self.user_ids = tf.placeholder(tf.int32, shape = [None], name='user_predict')
            self.item_ids = tf.placeholder(tf.int32, shape = [None], name='item_predict')
            
            # _________ variables _________
            self.weights = self._initialize_weights()
            
            # _________ train _____________
            self.y_ = self.inference(user_inputs=self.user_inputs, item_inputs=self.item_inputs)
            self.loss_train = self.loss_function(true_labels=self.train_labels, 
                                                 predicted_labels=tf.reshape(self.y_,shape=[-1,self.list_length]),
                                                 lamda_regularizer=self.lamda_regularizer)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss_train) 

            # _________ prediction _____________
            self.predictions = self.inference(user_inputs=self.user_ids, item_inputs=self.item_ids)
        
            # Variable initialization
            self.saver = tf.train.Saver() #  
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
    
    
    def _init_session(self):
        # adaptively growing memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)
    
    
    def _initialize_weights(self):
        all_weights = dict()

        # -----embedding layer------
        all_weights['embedding_users'] = tf.Variable(tf.random_normal([self.users_num, self.embedding_size_users], 
                                                                      0, 0.1),name='embedding_users')
        all_weights['embedding_items'] = tf.Variable(tf.random_normal([self.items_num, self.embedding_size_items], 
                                                                      0, 0.1),name='embedding_items') 
        
        # ------hidden layer------
        all_weights['weight_0'] = tf.Variable(tf.random_normal([(self.embedding_size_users+self.embedding_size_items),
                                                                self.hidden_size[0]], 0.0, 0.1),name='weight_0')
        all_weights['bias_0'] = tf.Variable(tf.zeros([self.hidden_size[0]]), name='bias_0')
        all_weights['weight_1'] = tf.Variable(tf.random_normal([self.hidden_size[0],self.hidden_size[1]], 
                                                               0.0, 0.1), name='weight_1')
        all_weights['bias_1'] = tf.Variable(tf.zeros([self.hidden_size[1]]), name='bias_1')
        
        # ------output layer-----
        all_weights['weight_n'] = tf.Variable(tf.random_normal([self.hidden_size[-1], 1], 0, 0.1), name='weight_n')
        all_weights['bias_n'] = tf.Variable(tf.zeros([1]), name='bias_n')

        return all_weights
        
    
    def train(self, data_sequence):
        train_size = len(data_sequence)
        
        np.random.shuffle(data_sequence)
        batch_size = self.batch_size
        total_batch = math.ceil(train_size/batch_size)

        for batch in range(total_batch):
            start = (batch*batch_size)% train_size
            end = min(start+batch_size, train_size)
            data_array = np.array(data_sequence[start:end])

            feed_dict = {self.user_inputs: data_array[:,:,0], 
                     self.item_inputs: data_array[:,:,1],
                     self.train_labels: data_array[:,:,-1]}  
            loss, opt = self.sess.run([self.loss_train,self.train_op], feed_dict=feed_dict)
            self.train_loss_records.append(loss)
            
        return self.train_loss_records

        
    def inference(self, user_inputs, item_inputs):
        embed_users = tf.reshape(tf.nn.embedding_lookup(self.weights['embedding_users'], user_inputs),
                                 shape=[-1, self.embedding_size_users])
        embed_items = tf.reshape(tf.nn.embedding_lookup(self.weights['embedding_items'], item_inputs),
                                 shape=[-1, self.embedding_size_items])
            
        layer0 = tf.nn.relu(tf.matmul(tf.concat([embed_items,embed_users],1), self.weights['weight_0']) + self.weights['bias_0'])
        layer1 = tf.nn.relu(tf.matmul(layer0, self.weights['weight_1']) + self.weights['bias_1']) 
        y_ = tf.matmul(layer1,self.weights['weight_n']) + self.weights['bias_n']
        return y_         
        
        
    def loss_function(self, true_labels, predicted_labels,lamda_regularizer=1e-4):   
        cross_entropy_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=predicted_labels))
        regularizer_1 = tf.contrib.layers.l2_regularizer(lamda_regularizer)
        regularization = regularizer_1(
            self.weights['embedding_users']) + regularizer_1(
            self.weights['embedding_items'])+ regularizer_1(
            self.weights['weight_0']) + regularizer_1(
            self.weights['weight_1']) + regularizer_1(
            self.weights['weight_n'])
        cost = cross_entropy_mean

        return cost    
 
    
    # save model and its parameters
    def save_model(self, save_path):
        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save tf model
        tf_path = os.path.join(save_path, 'trained_model')
            
        if os.path.exists(tf_path):
            os.remove(tf_path)
            
        self.saver.save(self.sess,tf_path)
        
        
    def evaluate(self, test_sequence, topK=10):
        score = np.zeros([self.users_num, self.items_num])
        users = np.array([u for u in range(self.users_num)])
        items = np.array([i for i in range(self.items_num)])    
        for u in range(self.users_num):
            user_ids = u * np.ones([self.items_num])
            feed_dict = {self.user_ids: user_ids,self.item_ids:items}
            out = self.sess.run([self.predictions],feed_dict=feed_dict)
            score[u,:] = np.reshape(out,(-1,self.items_num))
            
        ranklist = get_topk(prediction=score,test_sequence=np.array(test_sequence), topK=topK)
        hits,ndcgs = hit_ndcg(test_sequence=np.array(test_sequence), ranklist=ranklist)
        hr,ndcg = np.array(hits).mean(),np.array(ndcgs).mean()
        return hr,ndcg

    
def train(model, train_list, test_list, users_num, items_num, 
          list_length=5, positive_size=2, sample_size=2, epoches=50, topK=10):
    train_mat= sequence2mat(sequence=train_list, N=users_num, M=items_num) # train data : user-item matrix
    
    hr_list=[]
    ndcg_list=[]
    hr,ndcg = model.evaluate(test_sequence=test_list, topK=topK)
    hr_list.append(hr)
    ndcg_list.append(ndcg)
    print('Init: HR = %.4f, NDCG = %.4f' %(hr, ndcg))
    best_hr, best_ndcg = hr, ndcg
    for epoch in range(epoches):
        data_sequence = generate_list(train_mat=train_mat, positive_size=positive_size, list_length=list_length, sample_size=sample_size)
        loss_records = model.train(data_sequence=data_sequence)
        hr,ndcg = model.evaluate(test_sequence=test_list, topK=topK)
        hr_list.append(hr)
        ndcg_list.append(ndcg)
        print('epoch=%d, loss=%.4f, HR=%.4f, NDCG=%.4f' %(epoch,loss_records[-1],hr,ndcg))
        if ndcg_list[-1]<ndcg_list[-2] and ndcg_list[-2]<ndcg_list[-3]:
            best_hr, best_ndcg = hr_list[-3], ndcg_list[-3]
            break
        best_hr, best_ndcg = hr, ndcg
            
    print("End. Best HR = %.4f, NDCG = %.4f. " %(best_hr, best_ndcg))
    model.save_model(save_path=model_path)
    

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    data_name = args.data_name
    embedding_size_users = args.user_factors
    embedding_size_items = args.item_factors
    hidden_size = eval(args.layers)
    lamda_regularizer = args.reg
    sample_size = args.sample_time
    learning_rate = args.lr
    list_length = args.list_length
    positive_size = args.num_positive
    batch_size = args.batch_size
    epoches = args.epoches
    min_loss = args.min_loss
    topK = args.top_n
    model_path = args.path                   

    data_dir = path + '/' + data_name 
    users_num, items_num, data_list, _ = load_data(file_dir=data_dir)
    print(' data length: %d \n user number: %d \n item number: %d' %(len(data_list),users_num,items_num)) 

    rating_mat = sequence2mat(data_list, users_num, items_num)
    train_list,test_list,negative_items, _ = get_train_test(rating_mat=rating_mat)
    test_list = np.c_[np.array(test_list),np.array(negative_items)]
    
    # build model
    model = DeepRank(users_num = users_num,
               items_num = items_num,
               batch_size = batch_size,
               embedding_size_users = embedding_size_users,
               embedding_size_items = embedding_size_items,
               hidden_size = hidden_size,
               list_length = list_length,
               learning_rate = learning_rate,
               lamda_regularizer=lamda_regularizer,   
               model_path = model_path
               )

    train(model = model, 
        train_list = train_list,
        test_list = test_list,
        users_num = users_num, 
        items_num = items_num,
        list_length = list_length,
        positive_size = positive_size,
        sample_size = sample_size,
        epoches = epoches, 
        topK = topK)
