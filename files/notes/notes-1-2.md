In today's lesson we finish covering chapter 1 of the book, looking more at test/validation sets, avoiding machine learning project failures, and the foundations of transfer learning.

Then we move on to looking at the critical machine learning topic of evidence, including discussing confidence intervals, priors, and the use of visualization to better understand evidence.

Finally, we begin our look into productionization of models (chapter 2 of the book), including discussing the overall project plan for model development, and how to create your own datasets. Here's a helpful [forum post](https://forums.fast.ai/t/getting-the-bing-image-search-key/67417) explaining how to get the Bing API key you'll need for downloading images.

## Questionnaire

(If you're not sure of the answer to a question, try watching the next lesson and then coming back to this one, or read the first two chapters of the book.)

1. Can we always use a random sample for a validation set? Why or why not?  



1. What is overfitting? Provide an example.  

Overfitting is the ML pitfall that occurs when the model is trained too closely to the training data and performs significantly worse on validation and testing data. It's a result of the model not being generalizable enough to work on any data outside of its training data. This can occur due to training data not being diverse enough or representative of real-world data or if the training process is run for too many epochs. 

1. What is a metric? How does it differ from "loss"?  

The loss is a function (i.e. root mean-squared error) that takes in the prediction and label as input and produces a number that is fed into the learning algo (performs gradient descent and tunes the weights via backpropagation). A metric is used to measure the quality/effectiveness of a model (i.e. accuracy, error rate) on validation and testing datasets. 

1. How can pretrained models help?  

Pretrained models have been trained on extremely massive, diverse, and well-curated datasets, producing models with the weights already tuned to solve a specific task/problem. Thus, the features that the model has been trained to look for are already encoded in the model. Since there's a lot of overlap b/w these features and that of our own dataset, this allows us to fine-tune it for our own needs pretty effectively, saving us training data, money, and time. 

1. What is the "head" of a model?  



1. What kinds of features do the early layers of a CNN find? How about the later layers?  

The CNN's early layers find high-level features like colors and shapes. The deeper layers find more low-level and sophisticated features hard to notice, like texture, text in image, and gradients.

1. Are image models only useful for photos?  

No, CNNs can be used in many other applications like ASR and NLP. 

1. What is an "architecture"?  

The architecture of a model is its structure of weights/params and functions used to process them; in other words, how each weight of the neural network is arranged and connected to other layers.   

1. What is segmentation?  

Segmentation is the task of identifying and classifying multiple components/parts/segments of a single image.   

1. What is `y_range` used for? When do we need it?  



1. What are "hyperparameters"?  

Hyperparameters are values that drive the architecture of the model (structure, nonlinear and loss functions, inputs of these functions). These are different from parameters/weights, which are the values that determine how much weight to assign to encoded features in the dataset, used in making predictions.   

1. What's the best way to avoid failures when using AI in an organization?  

Proper version control of models  
Pipelines to standardize data cleaning, transformation, feature engineering  
Constantly monitor model metrics (i.e. accuracy, error rate) and if and how input data changes  
Write some test cases to ensure model is working properly and doesn't regress  
Be mindful of what and how data is collected and ensure that it is as free of biases as possible  

1. What is a p value?
 
Value used in determining if there is a significant relationship b/w independent and dependent variable given data (testing hypothesis)    
Process: Pick a null hypothesis (default state of the world) -> gather data of variables -> calculate % of time relationship is seen by chance (i.e. p-value)  
**Do not only** measure probability that hypothesis is true since it's also based on how much data is collected; thus even if a p-value threshold is met (p < .05), that just might mean that you collected a lot of data, not that the hypothesis is necessarily true
Having said that, a p-value > .05 does not mean that hypothesis is false and disproven; it just means that you don't have enough data to make a definitive conclusion on the validity of the hypothesis
The practical importance of the relationship/hypothesis is dependent on the slope of the graph of the independent and dependent variables. However, this is assuming that the model is accurate in the first place.   

1. What is a prior?

Previous assumptions that we know to be true that can be used in creating a practical model

1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
1. Where do text models currently have a major deficiency?

Terrible in using context in its predictions, especially when the context window is very big (i.e. info is located far away from words of focus)  

1. What are possible negative societal implications of text generation models?

Rapid propagation of misinformation and increase in spam and fraudulent bots  

1. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
1. What kind of tabular data is deep learning particularly good at?

Features with high cardinality (many possible values); often found in recommendation tasks with user actions  

1. What's a key downside of directly using a deep learning model for recommendation systems?
1. What are the steps of the Drivetrain Approach?
1. How do the steps of the Drivetrain Approach map to a recommendation system?
1. Create an image recognition model using data you curate, and deploy it on the web.  

See notebook-1-2.ipynb  

1. What is `DataLoaders`?  

Class to load, split, and batch data for training  

1. What four things do we need to tell fastai to create `DataLoaders`?  

Form of input data (e.g. image vs text), labels, percentage of input data to use as validation, data transformation  

1. What does the `splitter` parameter to `DataBlock` do?  

How to split input data into training and validation (percentage and how random selection of validation data should be)  

1. How do we ensure a random split always gives the same validation set?  

Set the seed to 42  
