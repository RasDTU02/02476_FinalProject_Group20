# 02476_FinalProject_Group20

## Project Description
### Overall goal of the project
The overall goal of this project, is to facilitate the developyment and deployment of machine learning models. This is done by adhering MLOps best practices. The goal is to create a reliable, efficient and scalable process. This project aims to build a Rice Classifier, that is trained on a machine learning model. The structured format should support development, experimentation and deployment. Here, tools such as Docker and version control are used. A part of the best practices is to provide config and requirements files to ensure reproducability and changes to hyperparameters.
### What framework are you going to use, and you do you intend to include the framework into your project?
(Kun en ide)
In this project, a framework called FastAI is going to be used. This framework is built on PyTorch, and is used to make machine learning accessible, while maintaining flexibility and performance. It has a strong fundament in machine learning, and has built-in data handling, called DataBlock. To integrate this, the DataBlock is used to initialize and preprocess the data, pre-trained models are used and fine-tuned to our purpose. The model is evaluated and the model is deployed. All this is possible through the FastAI framework.

### What data are you going to run on
The data used for this project is called "Rice_Image_Dataset", and contains subfolders with different varities of rice. These rice are Arborio, Basmati, Ipsala, Jasmine, Karacadag. The data are images in RGB format of rice grains. Each subfolder acts as a class label, so the images are already labeled. The data is collected in 2021, and contains 75000 data samples. The data can be found here. [Murat Koklu's Datasets](https://www.muratkoklu.com/datasets/).

### What models do you expect to use
In this project, we expect to use pre-traied models, in order to successfully make the classification task. The models we are going to focus on will be the following:
* Resnet (Residual Networks)
  *  Good for image classification, because of it's deep architecture.
  *  Pre-traied and accessable through PyTorch
 
* EfficientNet
  * Is known to have a high accuracy, but a reduced amout of parameters compared to other models.
  * Ideal for testing a more simple approach.
 
* VGG
  * Simple and effective, but computationally expensive
  * Is good as a benchmark when compared to more modern architectures.
