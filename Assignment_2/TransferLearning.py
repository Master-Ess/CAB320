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
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, mixed_precision, Model
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix as sk_confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import itertools

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(10755012, "Kenzie", "Haigh"), (1, "Luke", "Whitton"), (2, "Emma", "Wu")]
    
def load_model():
    '''
    Load in a model using the tf.keras.applications model and return it.
    MobileNetV2
    '''
    model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model = model
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    number_of_outputs = 5

    x = base_model(inputs, training=False) # gets around the wrong type error
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(number_of_outputs, activation='softmax', dtype='float32')(x)
    model = models.Model(inputs, outputs)

    return model
    

def load_data(path):
    '''
    Load in the dataset from its home path. Path should be a string of the path
    to the home directory the dataset is found in. Should return a numpy array
    with paired images and class labels.
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, path)

    image_size = 224

    dataset = image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),  
        batch_size=32,
        label_mode='int'
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
    
def split_data(X, Y, train_fraction, randomize=False, eval_set=True):
    """
    Split the data into training and testing sets. If eval_set is True, also create
    an evaluation dataset. There should be two outputs if eval_set there should
    be three outputs (train, test, eval), otherwise two outputs (train, test).
    
    To see what type train, test, and eval should be, refer to the inputs of 
    transfer_learning().
    
    Insert a more detailed description here.
    """

    if randomize:
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

def confusion_matrix(predictions, ground_truth, plot=False, all_classes=None):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the confusion matrix of the classifier's performance.
    '''
    if all_classes is None:
        all_classes = np.unique(ground_truth)
    num_classes = len(all_classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(ground_truth, predictions):
        cm[true, pred] += 1
    return cm

def plot_confusion_matrix(cm, classes):
    '''
    Plot the confusion matrix.
    '''
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def precision(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's precision
    '''
    cm = confusion_matrix(predictions, ground_truth)
    precision_scores = np.diag(cm) / np.sum(cm, axis=0)
    return np.nan_to_num(precision_scores)

def recall(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's recall
    '''
    cm = confusion_matrix(predictions, ground_truth)
    recall_scores = np.diag(cm) / np.sum(cm, axis=1)
    return np.nan_to_num(recall_scores)

def f1(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's f1 score
    '''
    prec = precision(predictions, ground_truth)
    rec = recall(predictions, ground_truth)
    f1_scores = 2 * (prec * rec) / (prec + rec)
    return np.nan_to_num(f1_scores)

def k_fold_validation(features, ground_truth, classifier, k=2):
    '''
    Inputs:
        - features: np.ndarray of features in the dataset
        - ground_truth: np.ndarray of class values associated with the features
        - classifier: class object with both fit() and predict() methods which
        can be applied to subsets of the features and ground_truth inputs.
        - k: int, number of sub-sets to partition the data into. default is k=2
    Outputs:
        - avg_metrics: np.ndarray of shape (3, c) where c is the number of classes.
        The first row is the average precision for each class over the k
        validation steps. Second row is recall and third row is f1 score.
        - sigma_metrics: np.ndarray, each value is the standard deviation of 
        the performance metrics [precision, recall, f1_score]
    '''
    kf = KFold(n_splits=k, shuffle=True)
    all_precisions, all_recalls, all_f1s = [], [], []
    
    for train_index, test_index in kf.split(features):
        train_X, test_X = features[train_index], features[test_index]
        train_Y, test_Y = ground_truth[train_index], ground_truth[test_index]
        
        classifier.fit(train_X, train_Y)
        predictions = classifier.predict(test_X)
        
        precision_scores = precision(predictions, test_Y)
        recall_scores = recall(predictions, test_Y)
        f1_scores = f1(predictions, test_Y)
        
        all_precisions.append(precision_scores)
        all_recalls.append(recall_scores)
        all_f1s.append(f1_scores)
    
    avg_precision = np.mean(all_precisions, axis=0)
    avg_recall = np.mean(all_recalls, axis=0)
    avg_f1 = np.mean(all_f1s, axis=0)
    
    sigma_precision = np.std(all_precisions, axis=0)
    sigma_recall = np.std(all_recalls, axis=0)
    sigma_f1 = np.std(all_f1s, axis=0)
    
    avg_metrics = np.array([avg_precision, avg_recall, avg_f1])
    sigma_metrics = np.array([sigma_precision, sigma_recall, sigma_f1])
    
    return avg_metrics, sigma_metrics

def transfer_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform standard transfer learning here.
    '''
    learning_rate, momentum, nesterov = parameters

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True
    )

    # Train
    X_train, Y_train = train_set
    X_eval, Y_eval = eval_set
    history = model.fit(
        X_train, Y_train,
        epochs=10,
        validation_data=(X_eval, Y_eval)
    )

    # unfreeze
    model.trainable = True
    fine_tune_at = 100
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate / 10, momentum=momentum, nesterov=nesterov),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True
    )

    # more training with new variables
    history_fine = model.fit(
        X_train, Y_train,
        epochs=10,
        validation_data=(X_eval, Y_eval)
    )

    # Evaluate the model on the test set
    X_test, Y_test = test_set
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    # Calculate classwise recall, precision, and f1 scores
    recall_scores = recall(Y_pred_classes, Y_test)
    precision_scores = precision(Y_pred_classes, Y_test)
    f1_scores = f1(Y_pred_classes, Y_test)

    metrics = [recall_scores, precision_scores, f1_scores]

    return model, metrics, history, history_fine, Y_pred_classes

def accelerated_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform accelerated transfer learning here.
    '''
    learning_rate, momentum, nesterov = parameters

    mixed_precision.set_global_policy('mixed_float16')

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True
    )

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

    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    history = model.fit(
        datagen.flow(X_train, Y_train, batch_size=32),
        epochs=20,
        validation_data=(X_eval, Y_eval),
        callbacks=[callback]
    )

    model.trainable = True
    fine_tune_at = 100
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate / 10, momentum=momentum, nesterov=nesterov),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True
    )

    history_fine = model.fit(
        datagen.flow(X_train, Y_train, batch_size=32),
        epochs=10,
        validation_data=(X_eval, Y_eval),
        callbacks=[callback]
    )

    X_test, Y_test = test_set
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    recall_scores = recall(Y_pred_classes, Y_test)
    precision_scores = precision(Y_pred_classes, Y_test)
    f1_scores = f1(Y_pred_classes, Y_test)

    metrics = [recall_scores, precision_scores, f1_scores]

    return model, metrics, history, history_fine, Y_pred_classes

def plot_history(history, title="Training and Validation Metrics"):
    '''
    Plots the training and validation loss and accuracy from a Keras history object
    '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    print("code_start")

    model = load_model()
    X, Y = load_data("small_flower_dataset")

    split = 0.8
    X_train, Y_train, X_test, Y_test, X_eval, Y_eval = split_data(X, Y, split, eval_set=True)
    
    train_set = (X_train, Y_train)
    eval_set = (X_eval, Y_eval)
    test_set = (X_test, Y_test)

    learning_rates = [0.1, 0.01, 0.001]
    best_lr = 0.001
    best_metrics = None
    best_model = None
    best_history = None
    best_history_fine = None
    best_predictions = None

    for lr in learning_rates:
        model = load_model()
        model, metrics, history, history_fine, predictions = transfer_learning(train_set, eval_set, test_set, model, (lr, 0.0, False))
        print(f"Learning rate: {lr}, Transfer Learning Metrics:", metrics)
        plot_history(history, title=f"Transfer Learning - Initial Training with lr={lr}")
        plot_history(history_fine, title=f"Transfer Learning - Fine Tuning with lr={lr}")
        
        if best_metrics is None or metrics[2].mean() > best_metrics[2].mean(): # Choose the best model based on f1 score
            best_lr = lr
            best_metrics = metrics
            best_model = model
            best_history = history
            best_history_fine = history_fine
            best_predictions = predictions

    print(f"Best learning rate: {best_lr}")

    # Plot confusion matrix for the best model
    cm = sk_confusion_matrix(Y_test, best_predictions)
    plot_confusion_matrix(cm, classes=[0, 1, 2, 3, 4])

    # Precision, Recall, F1 scores for the best model
    precision_scores = precision(best_predictions, Y_test)
    recall_scores = recall(best_predictions, Y_test)
    f1_scores = f1(best_predictions, Y_test)

    print(f"Precision: {precision_scores}")
    print(f"Recall: {recall_scores}")
    print(f"F1 Score: {f1_scores}")

    plot_history(best_history, title=f"Transfer Learning - Initial Training with best lr={best_lr}")
    plot_history(best_history_fine, title=f"Transfer Learning - Fine Tuning with best lr={best_lr}")
