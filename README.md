# Image-colorization-supervised-and-selfsupervised-ML

## Group members:
1. Ardavan Khalij
2. Xhulio Isufi
3. Mahsa Alirezaee

### Step 1: Download data
Our data is from Kaggle, and you can find them at this link: https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization

After downloading the data, you should make two folders, train and test. You should put 6900 color photos in the train folder and the rest of the color photos in the test. Please note that the photos should be in a folder called class inside of the train and test but you should'nt put the class in the path in the program.

### Step 2: Saved train models
The saved train models for all models are available in the submitted folder.

### Step 2: Training the models
There are trained models saved in the submitted project, but you can also train them yourself. The models are designed in a way that you can continue to learn the existing checkpoints even more.

To train the 3-layer model, you should run 3layer_model.py; for 5 layers, you should run 5layer_model.py; for 7 layers, you should run 7layer_model.py; for 13 layers, you should run 13layer_model.py; and for resnet model, you should run resnet_model.py.

### Step 3: Testing the results and see the results
We prepared several Jupiter notebooks so you can run them with the correct path of the data and see the results and accuracy of the models.

To run and test the 3-layer model, you should run 3layer_model.ipynb; for 5 layers, you should run 5layer_model.ipynb; for 7 layers, you should run 7layer_model.ipynb; for 13 layers, you should run 13layer_model.ipynb, and for resnet model, you should run resnet_model.ipynb.
