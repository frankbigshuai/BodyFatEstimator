# Body-Fat-Regression-from-Reddit-Image-Dataset
A deep learning project that aims to use images scraped using the reddit web scraper, to predict the body fat of a male from a front facing picture of their body.

## Background:

The goal of this Model is to be able to receive the customer's picture and give a prediction of body fat percentage based on reddit user perception. *The target consumers are adult males who may want a rough estimate of their bodyfat. *
Crucially, this project will not give a perfect estimate as it has been trained on the pereception of a person's bodyfat and not their exact bodyfat. Some of the images in the dataset have also been hand labelled and so it introduces further intrinsic bias and noise to the dataset.

## Product targets and specification:

We aim to produce a model that can predict a body fat percentage within +-2% 90% of the time.

## Dataset used:

The dataset used consists of 1022 self submitted reddit post images. 1/3 of the images come from r/guess_my_bf which were assigned labels from user made comments and 773 images were sourced from r/bulk_or_cut, these were hand labeled.

![newplot](https://user-images.githubusercontent.com/79870177/123670269-eeba0700-d834-11eb-8ee3-547615593435.png)

![8trcju](https://user-images.githubusercontent.com/79870177/123670492-2de85800-d835-11eb-9b12-9f330034a053.jpg)
![9cmsnv](https://user-images.githubusercontent.com/79870177/123670502-317bdf00-d835-11eb-9633-b202281c81a5.jpg)

All the pictures used are male since there were not enough female samples to accurately predict their bodyfats and their inclusion would have affected predicitons for males. Especially as women naturally have higher body fats. The body fat data follows a normal random distribution as one might expect.

## Procedure:

The dataset is all loaded into a dataset class and where the images and their labels (body fat percentage) are split into train and test sets. the __len__()and __getitem__() magic functions are defined to allow for the dataloader to load the data in batches.

An initial pretrained resnet model was used as a base the classifier for this was changed and the model altered iteratively optimising for minimum validation MSE loss.

Once a model had been chosen, in order for the maximum datahandling efficiency, the final model was retrained on a dataset containing both the train and validation sets. Finally the model was tested on it's evaluation of the separate test set.
Description of the Model used:

The base model used and was chosen as a good interpreter of image data, was the resnet model. The rational behind that is that being trained on many images prior the model would hold some base intuition about images in it's logic which we may exploit.

The dataset had to be transformed in order to use the pretrained model and hence all images were resized to a (224,224) dimension image. Additionally, the image dataset is normalised to ensure the pixel distribution is in a similar scale for faster conversions. Further augmentations were used to facilitate better training of the model. These augmentations were as follows:
- random horizontal flip
- random vertical flip
- random rotation
- gaussian blur

## Analysis of the results and conclusion:

results on the unseen test dataset show a mean squared error of 10.2457 this indicates that on average the result is on average 3.2% off and can be attributed to the lack of data to train from as deep learning models generally require a lot of data to train from to a good accuracy level.
