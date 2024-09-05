import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

#prepare data - load and preprocess
input_dir = 'clf-data'
categories = ['empty', 'dog']

data = []
labels = []

#we make INTEGER labels for categories instead of a string
#this approach used in ML
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)

        # Skip empty or corrupted images
        if img.size == 0:
            continue

        # Ensure image is in RGB format
        if img.ndim == 2:  # Grayscale image
            img = np.stack((img,) * 3, axis=-1)

        img = resize(img, (15, 15))
        data.append(img.flatten())  # Flatten image to array
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train the classifier
classifier = SVC()

#we are training many classifiers for each combination gamma,C -- 12 image classificators
parameters = [{'gamma':[0.01, 0.001, 0.0001], 'C':[0.1, 1, 10, 100, 1000]}]  # Added a lower C value for stronger regularization


grid_search = GridSearchCV(classifier,parameters)
grid_search.fit(x_train,y_train)

# test performance
best_estimator = grid_search.best_estimator_ #among the 12 classifiers we choose the best - thats our model

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score*100)))
# 99.34372436423298%
train_sizes, train_scores, test_scores = learning_curve(best_estimator, data, labels, cv=5)

plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.show()

pickle.dump(best_estimator, open('./modelWithDog.p', 'wb'))