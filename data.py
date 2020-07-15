import numpy as np

# Read original records
def load_data(file_dir):
    # output: 
    # N: the number of user;
    # M: the number of item
    # data: the list of rating information
    user_ids_dict, rated_item_ids_dict = {},{}
    N, M, u_idx, i_idx = 0,0,0,0 
    data = []
    f = open(file_dir)
    for line in f.readlines():
        if '::' in line:
            u, i, r, _ = line.split('::')
        else:
            u, i, r, _ = line.split()
    
        if int(u) not in user_ids_dict:
            user_ids_dict[int(u)]=u_idx
            u_idx+=1
        if int(i) not in rated_item_ids_dict:
            rated_item_ids_dict[int(i)]=i_idx
            i_idx+=1
        data.append([user_ids_dict[int(u)],rated_item_ids_dict[int(i)],float(r)])
    
    f.close()
    N = u_idx
    M = i_idx

    return N, M, data, rated_item_ids_dict


# Convert the list data to array
def sequence2mat(sequence, N, M):
    # input:
    # sequence: the list of rating information
    # N: row number, i.e. the number of users
    # M: column number, i.e. the number of items
    # output:
    # mat: user-item rating matrix
    records_array = np.array(sequence)
    row = records_array[:,0].astype(int)
    col = records_array[:,1].astype(int)
    values = records_array[:,2].astype(np.float32)
    mat = np.zeros([N,M])
    mat[row,col]=values
    
    return mat


# Read train and test data
def read_data(file_dir):
    data=[]
    f = open(file_dir)
    for line in f.readlines():
        if len(line.split(','))==3:
            u, i, r = line.split(',')
            data.append([int(u), int(i), float(r)])
        else:
            data.append([int(i) for i in line.split(',')])
    
    return data


# Genarate instances from train dataset
def generate_list(train_mat, positive_size=1, list_length=5, sample_size=2):
    data = []
    users_num,items_num = train_mat.shape
    for u in range(users_num):
        rated_items = np.where(train_mat[u,:]>0)[0]
        unrated_items = np.where(train_mat[u,:]==0)[0]
        
        for item0 in rated_items:
            line0 = []
            line0.append([u,item0,1])
            
            positive_items = np.random.choice(rated_items, size=positive_size-1, replace=False)
            for item1 in positive_items:
                line0.append([u,item1,1])
           
            for _ in range(sample_size):
                line1 = []
                negtive_items = np.random.choice(unrated_items, size=list_length-positive_size, replace=False)
                for item1 in negtive_items:
                    line1.append([u,item1,0])
                line2 = line0 + line1
                np.random.shuffle(line2) 
                data.append(np.array(line2))
            
    return data


# Generate train and test from raw user-item interactions
def get_train_test(rating_mat):
    n,m = rating_mat.shape
    
    selected_items,rest_ratings,negtive_items = [],[],[]
    for user_line in rating_mat:
        rated_items = np.where(user_line>0)[0]
        rated_num = len(rated_items)
        random_ids = [i for i in range(rated_num)]
        np.random.shuffle(random_ids)
        selected_id = random_ids[0]
        selected_items.append(rated_items[selected_id])
        rest_ratings.append(rated_items[random_ids[1:]])
        
        unrated_items = np.where(user_line==0)[0]
        unrated_num = len(unrated_items)
        random_ids = [i for i in range(unrated_num)]
        np.random.shuffle(random_ids)
        negtive_items.append(unrated_items[random_ids[:99]])
        
    train = [[user, item, rating_mat[user,item]] for user in range(n) for item in rest_ratings[user]]   
    test = [[user, selected_items[user]] for user in range(n)]
    
    length = int(n*0.1)
    rated_size = np.sum(rating_mat>0,1)
    rated_order = np.argsort(rated_size)
    sparse_user = rated_order[:length]
    
    np.random.shuffle(train)  
    return train,test,negtive_items,sparse_user
