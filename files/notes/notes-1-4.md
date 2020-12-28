Today we continue our dive deep into the foundations of how a neural net really works, by training one from scratch.

We look at the sigmoid function, and see why it is that it's needed for classification models. We refactor our data input code to create batches, in the process learning about the `DataLoader` class. We also learn about a number of useful features of arrays and tensors in python, including `view` and the `@` operator.

Then we look more closely at how gradients are calculated and used in a PyTorch training loop. We go from a simple single-layer network, to create our first "deep" network from scratch, by adding non-linearities (with ReLU) to our network! We discuss why we need deep networks to get good results in practice.

Finally, we start looking at the `softmax` activation function, which is used in most non-binary classification models.

## Questionnaire

**Why can't we use accuracy as a loss function?**

Accuracy is useful as a human-readable metrics, but not as a loss function because a loss function change when the weights are updated. Accuracy is constant nearly everywhere even when weights are changed (in other words, dA/dw = 0 b/c changing the weights does not result in much of a change in accuracy). We need a function that changes when weights are updated in order to use it as a loss function.  

**Draw the sigmoid function. What is special about its shape?**  

The sigmoid function is nonlinear, S-curved, continuous, and normalizes any value b/w 0 and 1.  

**What is the difference between a loss function and a metric?**  

A loss function measures the distance b/w the predicted and actual label for a given training example. Its gradient is used to update weights in the objective of minimizing the loss function value as much as possible.  

**What is the function to calculate new weights using a learning rate?**

Backpropagation is the process by which the parameters (weights and biases) are updated via gradients and learning rates. For SGD for example, a forward pass and loss (`forward()`) is calculated for each mini-batch of training examples. Then for each mini-batch, the parameters' gradients are calculated (`backward()`); in other words, how much the loss changes given an update in parameters. Then the params are updated (`optimize()`) in order to reduce the loss. Repeated for each mini-batch of training examples and then repeated for each epoch.  

**What does the `DataLoader` class do?**  

The `DataLoader` class offers an abstraction (methods) to mini-batch a dataset.  

**Write pseudocode showing the basic steps taken in each epoch for SGD.**  

```
inputs  # tensor of shape (1000, 28, 28)
weights = torch.randn(28*28, 1)
biases = torch.randn(1)
params = (weights, biases)
training_set_mini_batches = DataLoader(training_set, batches=5, shuffle=True)
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for (xmb, ymb) in training_set_mini_batches:
        preds = model.forward(xmb)  # Get predictions (linear and nonlinear activation functions)
        loss = compute_l2_norm(preds, ymb)  # Compute loss b/w predicted and actual labels
        loss.backward()  # Compute gradient of loss with respect to model params
        optimizer.step()  # Update weights based on learning rate and gradient
        params.grad.zero_()  # Zero out gradients
```

**Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?**  

```
nums_list = [1,2,3,4]
labels = ['a', 'b', 'c', 'd']
dataset = list(zip(nums_list, labels))
x, y = dataset[0]
```  
When indexed, dataset outputs a tuple of form (training example, training label).  

**What does `view` do in PyTorch?**  

Resizes tensor (-1 can be used to infer a dimension size)  

**What are the "bias" parameters in a neural network? Why do we need them?**  

```
tensor = torch.randint(low=1, high=10, size=(3, 3), dtype=torch.uint8) 
resized_3d_tensor = tensor.view(-1, 3, 3)  # Outputs a 3-dimensional tensor of size (1, 3, 3)
resized_1d_tensor = tensor.view(-1)  # Outputs a 1-dimensional tensor of size 9
```  

**What does the `@` operator do in Python?**  

Matrix multiplication (series of dot products that can be performed in parallel on GPU)  

**What does the `backward` method do?**  

`loss.backward()` method calculates the loss gradient with respect to the model's params given a loss tensor.  

**Why do we have to zero the gradients?**  

If the gradients aren't zeroed out, then the existing gradient might point somewhere else besides the objective (minima of loss function) on the next training batch. To make sure weights update properly, we have to make sure the gradients are zeroed out before moving to the next training batch. Pytorch, by default, accumulates the gradient across training batches.  

**What information do we have to pass to `Learner`?**  



1. Show Python or pseudocode for the basic steps of a training loop.
1. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.
1. What is an "activation function"?
1. What's the difference between `F.relu` and `nn.ReLU`?
1. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?
1. If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.
1. What are the two ways in which data is most commonly provided, for most deep learning datasets?
1. Look up the documentation for `L` and try using a few of the new methods is that it adds.
1. Look up the documentation for the Python `pathlib` module and try using a few methods of the `Path` class.
1. Give two examples of ways that image transformations can degrade the quality of the data.
1. What method does fastai provide to view the data in a `DataLoaders`?
1. What method does fastai provide to help you debug a `DataBlock`?
1. Should you hold off on training a model until you have thoroughly cleaned your data?
1. What are the two pieces that are combined into cross-entropy loss in PyTorch?
1. What are the two properties of activations that softmax ensures? Why is this important?
1. When might you want your activations to not have these two properties?
1. Calculate the `exp` and `softmax` columns of <<bear_softmax>> yourself (i.e., in a spreadsheet, with a calculator, or in a notebook).
