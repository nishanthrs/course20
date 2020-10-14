**Note**: *you can collapse the sidebars on the left and right (i.e this notes pane, and the contents pane) by clicking the arrow icons at the top next to each of them.*

Welcome to Deep Learning for Coders! In this first lesson, we learn about what deep learning is, and how it's connected to machine learning, and regular computer programming. We get our GPU-powered deep learning server set up, and use it to train models across vision, NLP, tabular data, and collaborative filtering. We do this all in Jupyter Notebooks, using transfer learning from pretrained models for the vision and NLP training.

We discuss the important topics of test and validation sets, and how to create and use them to avoid over-fitting. We learn about some key jargon used in deep learning.

We also discuss how AI projects can fail, and techniques for avoiding failure.

## Questionnaire

(If you're not sure of the answer to a question, try watching the next lesson and then coming back to this one, or read the first chapter of the book.)

**Do you need these for deep learning?**  

   - Lots of math T / F: False  
   - Lots of data T / F: False  
   - Lots of expensive computers T / F: False  
   - A PhD T / F: False  
   
**Name five areas where deep learning is now the best in the world.**  

Natural language processing (e.g. sentiment analysis)  
Trading stocks (in the short term)  
Computer vision  (e.g. ImageNet)  
Content recommendations  (e.g. FB ads)    
Playing games (e.g. AlphaGo)  

**What was the name of the first device that was based on the principle of the artificial neuron?**  

Perceptron (mark I)  

**Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?**  

Multiple processing units -> activation -> output function -> connectivity -> propagation -> activation -> learning rule -> env  

**What were the two theoretical misunderstandings that held back the field of neural networks?**  

Perceptrons' lack of ability to model XOR logic, but these can be resolved with multiple hidden layers  
Believed that only two hidden layers was enough to model any mathematical function (more layers were needed, but this was infeasible back then anyways due to lack of data and hardware constraints)  

**What is a GPU?**  

Device/chip (graphics processing unit) that can perform many different computations in parallel  

1. Open a notebook and execute a cell containing: `1+1`. What happens?
1. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.
1. Complete the Jupyter Notebook online appendix.
1. Why is it hard to use a traditional computer program to recognize images in a photo?

Normal computer programs are simply input -> function/model -> output. However, computers are stupid and we have to specify each step involved in the function/model used in recognizing images. There are many criteria used in image recognition due to the immense variance of images, and it's impossible to explicitly specify, in detail, each of these rules in recognizing images. It's also impossible for us to specify robust enough rules.

**What did Samuel mean by "weight assignment"?**  

If we have weights or parameters whereas the permutations of values of each parameter = permutations of performing a task (e.g. playing a game), then we can slowly adjust these weights/params using a learning process. This learning process uses the expected/correct label and loss function to adjust the weights in such a way that the model gradually gets better at performing that task with each training example.  

**What term do we normally use in deep learning for what Samuel called "weights"?**  

Parameters  

**Draw a picture that summarizes Samuel's view of a machine learning model.**  

Inputs and Parameters -> Model -> Prediction and Label -> Loss Function and Calculation -> Update Parameters (repeat for another data point/epoch)  

**Why is it hard to understand why a deep learning model makes a particular prediction?**  

Neural networks hard to interpret b/c it's hard to tell which input features it uses to output predictions  
There is no neat mapping b/w a neural network's weights/parameters and an input feature  
Furthermore, two neural networks with the same architecture can have different params and still output the exact same results  

**What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?**  

Universal approximation theorem  

**What do you need in order to train a model?**  

Lots of diverse data, representative of real-world data (same features, similar variance in each feature, etc)  

**How could a feedback loop impact the rollout of a predictive policing model?**  

Model initially trained on arrests in poor neighborhoods with mostly minorities -> police use this same model to make arrests -> police continue to update this model with these new arrests -> reinforces model with biases inherent in both past AND current arrests, making law enforcement extremely targeted towards minorities  

**Do we always have to use 224×224-pixel images with the cat recognition model?**  

Yes, each pixel = input feature that is fed into model (all training examples in dataset must have same dimensions)  

**What is the difference between classification and regression?**  

Classification predicts discrete labels, regression predicts continuous labels (diff algos required for each kind of task)  

**What is a validation set? What is a test set? Why do we need them?**  

Validation set ensures that the model doesn't overfit to training data.  
However, there is a possibility of overfitting the model to the validation set since we can look at the model's metrics on the validation set and adjust our hyperparameters, gradually making the model too specialized for the training and validation data.  
Testing set ensures that the model doesn't overfit to training and validation data and works properly in production on real-world data. It's totally hidden from us before model deployment.   

**What will fastai do if you don't provide a validation set?**  

Don't know, banish me to the depths of hell? All I know is that we should always have a validation set to prevent overfitting (k-fold cross validation is even better since it increases our chances of identifying if our model has overfitted or not)  
