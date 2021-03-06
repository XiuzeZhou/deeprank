# DeepRank

DeepRank: Learning to Rank with Neural Networks for Recommendation

- **Run listwise DeepRank**:

$ python DeepRank.py --path datasets --data_name ml-100k/u.data --epoches 40 --batch_size 512 --user_factors 16 --item_factors 16 --layers [16,8] --reg 0.00001 --list_length 5 --num_positive 2 --sample_time 2 --top_n 10 --lr 0.01 --path_model model

- **Run pairwise DeepRank**:

$ python DeepRank.py --path datasets --data_name ml-100k/u.data --epoches 40 --batch_size 512 --user_factors 16 --item_factors 16 --layers [16,8] --reg 0.00001 --list_length 2 --num_positive 1 --sample_time 4 --top_n 10 --lr 0.01 --path_model model

## Parameter description：
- path：Input data path.
- data_name：Name of dataset
- epoches：Number of epoches.
- batch_size：Batch size.
- user_factors：Embedding size of users.
- item_factors: Embedding size of items.
- layers：Size of each layer. Note that the first hidden layer is the interaction layer.
- reg: Regularization for user and item embeddings.
- list_length: Length of list for training. In pairwise DeepRank list_length=2; in listwise DeepRank list_length>2.
- num_positive: Number of positive instances in training list. In pairwise DeepRank num_positive=1;
- sample_time: Time of sample from instances.
- top_n: Number of top_n list for recommendation.
- lr: Learning rate.
- path_model: Output path for saving pre_trained model.
