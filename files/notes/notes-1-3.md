Today we finish creating and deploying our own app. We discuss data augmentation, and look at the most important types of augmentation used in modern computer vision models. We also see how fastai helps you process your images to get them ready for your model.

We look at building GUIs, both for interactive apps inside notebooks, and also for standalone web applications. We discuss how to deploy web applications that incorporate deep learning models. In doing so, we look at the pros and cons of different approaches, such as server-based and edge-device deployment.

Our final area for productionization is looking at what can go wrong, and how to avoid problems, and keep your data product working effectively in practice.

Then we skip over to chapter 4 of the book, and learn about the underlying math and code of Stochastic Gradient Descent, which lies at the heart of neural network training.

## Questionnaire

**What letters are often used to signify the independent and dependent variables?**  

**What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?**  

**What is data augmentation? Why is it needed?**  

Data augmentation is the process of creating different variations of currently existing data. Examples include image rotation, color filters, and zooming in and out. Its purpose is to make the training data more diverse and representative of real-world messy data, making the model more robust, less prone to overfitting, and increasing its ability to generalize across variations of the similar images.   

**What is the difference between `item_tfms` and `batch_tfms`?**  
**What is a confusion matrix?**  

Matrix with two axes: predicted and expected  
For each label, it shows how many labels were predicted correctly (on the diagonal) and how many were inaccurately predicted  
For a binary classification example, it'll show the number of true positives, true negatives, false positives, and false negatives  

**What does `export` save?**  

`export` saves the model, usually into a pickle file (.pkl). Usually the model is exported to a pickle file after the training loop and then loaded again before performing inference.     

**What is it called when we use a model for getting predictions, instead of training?**  

Inference  

**What are IPython widgets?**  

GUI elements that can be rendered directly in jupyter notebooks  

**When might you want to use CPU for deployment? When might GPU be better?**  

CPUs might be better on a lower budget or when the quantity of inferences the model has to perform isn't as high. GPUs would be better with more inferences (data constantly being streamed to the server for inference) since the server can batch together multiple pieces of data and run it on the GPU, where the inferences can be performed much quicker than on a CPU where it executes sequentially. In many use cases however, GPUs' performance to price ratio isn't as high as CPUs for inference. GPUs see maximum benefit on training since training tasks involve much more data than inference tasks.  

**What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?**  

In mission-critical and latency-sensitive applications in environments of poor internet connection like self-driving or crop recognition where milliseconds can mean tens of thousands of dollars, edge inference is far better than sending and waiting for data from a server.  

**What are three examples of problems that could occur when rolling out a bear warning system in practice?**  

The model that we trained our bears on had very clean pictures from Bing. The reality is that the kind of data we feed into the bear warning model are not going to be as clean and curated. The frames captured from the video feed will most likely be a lot blurrier with less colors, the data may be taken in diff environment conditions like rain and nightime, and the bears may be obscured by other objects like bushes and trees. These situations would definitely confuse the model if we didn't train for it on images taken under these situations.  

**What is "out-of-domain data"?**  

An example of out-of-domain data is shown perfectly above, when real-world data does not align with the data used to train and validate the model. ML models rely heavily on correlation and patterns rather than causal reasoning, so if it's not trained on a lot of data that is similar to the data it's supposed to predict, it will fail in unpredictable ways.  

**What is "domain shift"?**  

Domain shift is when the real-world data that's fed into the model changes over time so significantly that the old data used to train and validate the model no longer proves effective in training the model to reliably predict this new real-world data. This old data, the foundation for your model, is now essentially useless due to this domain shift.  

**What are the three steps in the deployment process?**  

To ensure that these models are rolled out in the real-world reliably while mitigating potential risks of failure:  
1. Run the model in parallel with the old system (might be a human operator or normal statistical model or previous model version). Compare/check the model's performance with that of the old system or human. Feed the real-world data and its labels (if possible) into the model, continuously training and improving it. For example, the park ranger will still perform his/her job in detecting bears, but the model will be running concurrently, with them inputting the actual value into the model.  
2. Continue monitoring the model's performance with some manual or automatic supervision. Run in one or limited regions or areas (i.e. A/B testing). Develop ways of accurately monitoring the model's accuracy, latency, and performance. For example, the model will be rolled out to 2-3 national parks in Utah and park rangers will check the model's predictions before approving it.  
3. Gradually roll this out to more regions, all the while carefully monitoring it for unexpected failures. If the number of bear alerts halves or doubles with all other factors in bear appearance being consistent, that's a cause for worry.  

**How is a grayscale image represented on a computer? How about a color image?**  

Grayscale images are represented as 2D matrices of numbers, each number ([0,255]) representing the color of the pixel. Color images are represented as 3D matrices of numbers, each number representing the red or green or blue color value ([0,255]) of the pixel (in a RGB image).   

**How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?**  

There is a folder per label (in this case, 10 folders for 10 digits) and a series of corresponding images in each folder.  

**Explain how the "pixel similarity" approach to classifying digits works.**  

Pixel similarity is a simple baseline model which simply takes the average pixel value of all images for each label. Then for inference of a given image, we simply see which for which averaged image label the image is closest to, choosing a distance metric like L1 or L2 norm.  

**What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.**  

```
nums = [1,2,3]
double_nums = [num*2 for num in nums]
```  

**What is a "rank-3 tensor"?**  

Tensor with rank 3 has 3 dimensions or axes.  

**What is the difference between tensor rank and shape? How do you get the rank from the shape?**  

Tensor rank = # dimensions, tensor shape = size of each axis of a tensor  
Ex. 1000 MNIST images tensor is of size (1000, 28, 28). Rank = 3 and shape = (1000, 28, 28)  

**What are RMSE and L1 norm?**  

Loss functions that measure the "distance" b/w two pieces of data:  
```l1_norm = (a3 - predicted_a3).abs().mean()```  
```rmse = ((a3 - predicted_a3)**2).mean().sqrt()```  

**How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?**  

Vectorization, multiprocessing, list comprehension (not thousands of times faster)  

**Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.**  

```
import torch
tensor = torch.randint(low=1, high=10, size=(3, 3), dtype=torch.uint8)
tensor *= 2
bottom_right_four_nums = tensor[1:3,1:3]
```  

**What is broadcasting?**  

Broadcasting is Pytorch's ability to perform operations on two tensors of different dimensions by expanding the tensor with smaller rank to match the dimensions of the tensor with larger rank. For example, if you're subtracting x of shape (1000, 28, 28) and y of shape (28,28), Pytorch will automatically expand y to (1000, 28, 28) by "creating" 1000 copies of the original 28x28 tensor to create the third dimension. Broadcasting is awesome b/c it doesn't *actually create 1000 copies* in the above example. In fact, it doesn't allocate *any* additional memory when expanding y, making the tensor computation run super efficiently.  
```
# validation_set_of_3s: (1000, 28, 28)
# mean_3s: (28, 28)
dist_b/w_validation_and_mean_3s = validation_set_of_3s - mean_3s
# dist_b/w_validation_and_mean_3s: (1000, 28, 28)
```  

**Are metrics generally calculated using the training set, or the validation set? Why?**  

The model's metrics are generally calculated using the validation set. Since metrics indicate performance, it's not helpful to measure metrics based on the training set since the model's already been trained on the training set and is expected to perform well on it. The main question we're trying to answer when collecting a model's metrics is how well it performs on real data. Otherwise, we're just going to overfit on the training data, making the model more suspectible to failure on real-world data.  

**What is SGD?**  

SGD is an algorithm that performs a variant of gradient descent (stochastic gradient descent) in order to adjust the model weights during training.  

**Why does SGD use mini-batches?**  


**What are the seven steps in SGD for machine learning?**  

1. Initialize the neural network's weights at random.  
2. Begin training the model. Perform the forward step (dot products of input features, weights, and biases and non-linear activation functions) for a batch of training examples.  
3. Calculate the loss using whatever loss function is specified.  
4. Perform gradient descent by performing backpropagation.  
    Calculate the gradient:
    ```
    weights = torch.randn(num_rows, num_cols, requires_grad=True)
    ...
    weights_grad = weights.grad
    ```
    Update the weights via `new_weights = weights += (learning_rate * weights_grad)`.  
5. Repeat for each batch of training examples in the training set.  

**How do we initialize the weights in a model?**  

No clear or best way of doing it; random initialization usually works best.  

**What is "loss"?**  

Loss is calculated via a function that calculates the distance b/w a predicted and expected label (e.g. mean squared Euclidean distance, RMSE, mean absolute difference). This is then used to adjust the weights for the goal of reducing the network's loss as much as possible in the gradient descent algorithm (backpropagation).  

**Why can't we always use a high learning rate?**  

High learning rates mean that there's a greater chance of overshooting and missing the minimal point in the loss function. Thus, the loss gradient calculated after adjusting the weights can bounce back and forth b/w a negative and positive number, but never actually decrease in magnitude to a loss gradient of 0, which is what we're aiming for (not exactly 0 since that would be overfitting, but at least trend in that direction).  

**What is a "gradient"?**  

The gradient is the slope (derivative) of the function at a particular point with respect to inputs. In the machine learning application, it refers to the derivative of the loss function at a particular point with respect to the model's weights. In other words, it describes how changing the weights would change the loss.   