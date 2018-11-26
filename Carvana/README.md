# Carvana data challange  

For this assignment you will be working with the Apple Computers Twitter Sentiment dataset (URL: https://www.figure-eight.com/wp-content/uploads/2016/03/Apple-Twitter-Sentiment-DFE.csv). We’d like you to write a ML pipeline using Python and TensorFlow (feel free to use Keras) that: 
Reads the dataset into memory   
1-Computes or uses pre-trained word embeddings. Some available ones are:   
- ELMO word embeddings using TensorFlow hub: https://tfhub.dev/google/elmo/2  
- GloVe word embedding: there are a number of TensorFlow tutorials online for this  
2-Computes 2 engineered features. Your choice of what you’d like those features to be, but one should use a regular expression to compute the feature.  
3-Merges the embeddings and the 2 engineered features  
4-Trains any classifier to predict the 'sentiment' class . 
Freezes the graph 
