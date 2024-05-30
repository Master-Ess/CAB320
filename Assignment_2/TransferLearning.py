# -*- coding: utf-8 -*-
'''

The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

Last modified 2024-05-07 by Anthony Vanderkop.
Hopefully without introducing new bugs.
'''


### LIBRARY IMPORTS HERE ###
import os
import keras.preprocessing
import numpy as np
import keras.applications as ka
import keras

import tensorflow as tf #why not already imported???

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, metrics, mixed_precision
from sklearn.metrics import recall_score, precision_score, f1_score #check that we can use this module
    
def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(10755012, "Kenzie", "Haigh"), (1, "Luke", "Whitton"), (2, "Emma", "Wu")]
    raise NotImplementedError
    
def load_model():
    '''
    Load in a model using the tf.keras.applications model and return it.
    MobileNetV2
    '''

    resp = keras.applications.MobileNetV2(weights='imagenet') #, include_top=True

    return resp

    raise NotImplementedError
    

def load_data(path):
    '''
    Load in the dataset from its home path. Path should be a string of the path
    to the home directory the dataset is found in. Should return a numpy array
    with paired images and class labels.
    
    Insert a more detailed description here.
    '''
    

    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, path)

    image_size = 224

    dataset = image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),  
        batch_size=32,
        label_mode='int'  #categorical --> one-hot
    )

    images = []
    labels = []

    for batch in dataset:
        batch_images, batch_labels = batch
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    return images, labels

    raise NotImplementedError
    
    
def split_data(X, Y, train_fraction, randomize=False, eval_set=True):
    """
    Split the data into training and testing sets. If eval_set is True, also create
    an evaluation dataset. There should be two outputs if eval_set there should
    be three outputs (train, test, eval), otherwise two outputs (train, test).
    
    To see what type train, test, and eval should be, refer to the inputs of 
    transfer_learning().
    
    Insert a more detailed description here.
    """

    #remove before submit or find reference if actually needed??? Why is the arg in the function?????
    if randomize: #not explictily referenced?
        # shufle
        ind = np.arange(X.shape[0])
        np.random.shuffle(ind)
        X = X[ind]
        Y = Y[ind]
    
    
    num_train = int(train_fraction * X.shape[0])
    X_train, Y_train = X[:num_train], Y[:num_train]
    X_test, Y_test = X[num_train:], Y[num_train:]

    if eval_set:
        num_test = X_test.shape[0]
        num_eval = num_test // 2
        X_eval, Y_eval = X_test[:num_eval], Y_test[:num_eval]
        X_test, Y_test = X_test[num_eval:], Y_test[num_eval:]
        
        return (X_train, Y_train, X_test, Y_test, X_eval, Y_eval)
    
    return (X_train, Y_train, X_test, Y_test)
    #raise NotImplementedError
    
    

def confusion_matrix(predictions, ground_truth, plot=False, all_classes=None):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the confusion matrix of the classifier's performance.

    Inputs:
        - predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        - ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
        - plot: boolean. If true, create a plot of the confusion matrix with
                either matplotlib or with sklearn.
        - classes: a set of all unique classes that are expected in the dataset.
                   If None is provided we assume all relevant classes are in 
                   the ground_truth instead.
    Outputs:
        - cm: type np.ndarray of shape (c,c) where c is the number of unique  
              classes in the ground_truth
              
              Each row corresponds to a unique class in the ground truth and
              each column to a prediction of a unique class by a classifier
    '''
    
    raise NotImplementedError
    return cm
    

def precision(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's precision
    
    Inputs: see confusion_matrix above
    Outputs:
        - precision: type np.ndarray of length c,
                     values are the precision for each class
    '''
    raise NotImplementedError
    return precision

def recall(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's recall
    
    Inputs: see confusion_matrix above
    Outputs:
        - recall: type np.ndarray of length c,
                     values are the recall for each class
    '''
    raise NotImplementedError
    return recall

def f1(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's f1 score
    Inputs:
        - see confusion_matrix above for predictions, ground_truth
    Outputs:
        - f1: type nd.ndarry of length c where c is the number of classes
    '''
    
    raise NotImplementedError
    return f1

def k_fold_validation(features, ground_truth, classifier, k=2):
    '''
    Inputs:
        - features: np.ndarray of features in the dataset
        - ground_truth: np.ndarray of class values associated with the features
        - fit_func: f
        - classifier: class object with both fit() and predict() methods which
        can be applied to subsets of the features and ground_truth inputs.
        - predict_func: function, calling predict_func(features) should return
        a numpy array of class predictions which can in turn be input to the 
        functions in this script to calculate performance metrics.
        - k: int, number of sub-sets to partition the data into. default is k=2
    Outputs:
        - avg_metrics: np.ndarray of shape (3, c) where c is the number of classes.
        The first row is the average precision for each class over the k
        validation steps. Second row is recall and third row is f1 score.
        - sigma_metrics: np.ndarray, each value is the standard deviation of 
        the performance metrics [precision, recall, f1_score]
    '''
    
    #split data
    ### YOUR CODE HERE ###
    
    #go through each partition and use it as a test set.
    for partition_no in range(k):
        #determine test and train sets
        ### YOUR CODE HERE###
        
        #fit model to training data and perform predictions on the test set
        classifier.fit(train_features, train_classes)
        predictions = classifier.predict(test_features)
        
        #calculate performance metrics
        ### YOUR CODE HERE###
    
    #perform statistical analyses on metrics
    ### YOUR CODE HERE###
    
    raise NotImplementedError
    return avg_metrics, sigma_metrics


##################### MAIN ASSIGNMENT CODE FROM HERE ######################

def transfer_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform standard transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)

    Outputs:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the 
            model on the test_set (list of np.ndarray)
    '''
    learning_rate, momentum, nesterov = parameters


    base_model = model

    
    base_model.trainable = False

    # remove a layer and add the 5 node dense later
    inputs = tf.keras.Input(shape=(224, 224, 3))
    number_of_outputs = len(set(train_set[1])) #should return 5

    shaved_base_model = base_model(inputs, training=False)[-2].output #this should get the base_model but also remove the last layer so we can add or own
    outputs = layers.Dense(number_of_outputs, activation='softmax', dtype='float32')(shaved_base_model) #add the additional layer as requested
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    X_train, Y_train = train_set
    X_eval, Y_eval = eval_set
    model.fit(
        X_train, Y_train,
        epochs=10, #check 
        validation_data=(X_eval, Y_eval)
    )

    # unfreeze
    base_model.trainable = True
    fine_tune_at = 100  # anything specific
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate / 10, momentum=momentum, nesterov=nesterov),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # more training with new variables
    model.fit(
        X_train, Y_train,
        epochs=10,  
        validation_data=(X_eval, Y_eval)
    )

    # Evaluate the model on the test set
    X_test, Y_test = test_set
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    # Calculate classwise recall, precision, and f1 scores
    recall = recall_score(Y_test, Y_pred_classes, average=None)
    precision = precision_score(Y_test, Y_pred_classes, average=None)
    f1 = f1_score(Y_test, Y_pred_classes, average=None)

    metrics = [recall, precision, f1]

    return model, metrics


    
def accelerated_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform accelerated transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)


    Outputs:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the 
            model on the test_set (list of np.ndarray)

    '''
    learning_rate, momentum, nesterov = parameters

    #makes stuff faster
    mixed_precision.set_global_policy('mixed_float16')

    base_model = model

    base_model.trainable = False

    # remove a layer and add the 5 node dense later
    inputs = tf.keras.Input(shape=(224, 224, 3))
    number_of_outputs = len(set(train_set[1])) #should return 5

    shaved_base_model = base_model(inputs, training=False)[-2].output #this should get the base_model but also remove the last layer so we can add or own
    outputs = layers.Dense(number_of_outputs, activation='softmax', dtype='float32')(shaved_base_model) #add the additional layer as requested in task sheet
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # data augementation - do we chuck anything specific here or not? Can i just have whatever variables i want? 
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    X_train, Y_train = train_set
    X_eval, Y_eval = eval_set

    # learning rate scheduler 
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    # train
    model.fit(
        datagen.flow(X_train, Y_train, batch_size=32),
        epochs=20,  # check number
        validation_data=(X_eval, Y_eval),
        callbacks=[callback]
    )

    # unfreeze layers
    base_model.trainable = True #correct?
    fine_tune_at = 100  # what to put here?
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate / 10, momentum=momentum, nesterov=nesterov),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Continue training with fine-tuning and data augmentation
    model.fit(
        datagen.flow(X_train, Y_train, batch_size=32),
        epochs=10,  # check number
        validation_data=(X_eval, Y_eval),
        callbacks=[callback]
    )

    # eval
    X_test, Y_test = test_set
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    recall = recall_score(Y_test, Y_pred_classes, average=None)
    precision = precision_score(Y_test, Y_pred_classes, average=None)
    f1 = f1_score(Y_test, Y_pred_classes, average=None)

    metrics = [recall, precision, f1]

    return model, metrics

if __name__ == "__main__":
    
    model = load_model()
    X, Y = load_data("small_flower_dataset")

    split = 0.8 #80% referenced in assignment
    X_train, Y_train, X_test, Y_test, X_eval, Y_eval = split_data(X, Y, split, eval_set=True) #
    
    train_set = (X_train, Y_train)
    eval_set = (X_eval, Y_eval)
    test_set = (X_test, Y_test)

    learning_rate = 0.001
    momentum = 0.0
    nesterov = False



    model, metrics = transfer_learning(train_set, eval_set, test_set, model,(learning_rate, momentum, nesterov))
    
    model, metrics = accelerated_learning(train_set, eval_set, test_set, model, (learning_rate, momentum, nesterov)) #be careful this is very slow without a gpu. Like my 3080 did all 20 epochs in the time my 10700 did one epoch


    
    
#########################  CODE GRAVEYARD  #############################
