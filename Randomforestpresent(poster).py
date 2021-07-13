import pandas
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np 

features = pandas.read_excel('dataset.xlsx')
features.head()
#print ('We have {} days of data with {} variables'.format(*features.shape))
print('The shape of our features is:', features.shape)
#features.describe()



# One Hot Encoding
#features = pandas.get_dummies(features)

transform = features['Class']
print(transform)
#Encode Label

le = preprocessing.LabelEncoder()
le.fit(["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy ","Very Unhealthy","Hazardous"])


LabelEncoder()
list(le.classes_)
labels = le.transform(transform)


list(le.inverse_transform([0,1,2,3,4,5]))

# Extract features and labels
#labels = features['Airquality']
features = features.drop('Class', axis = 1)




# List of features for later use
feature_list = list(features.columns)
features = np.array(features)
labels = np.array(labels)

# Convert to numpy arrays
import numpy as np

features = np.array(features)
labels = np.array(labels)

# Training and Testing Sets
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels,test_size = 0.2, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('PM2.5')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
predictions = predictions.astype(int)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
#print('Accuracy:', round(accuracy, 2), '%.')
print('Accuracy:',accuracy_score(test_labels,predictions))
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
plt.show()

# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
# Make a line graph
#plt.plot(x_values, cumulative_importances, 'g-')
# Draw line at 95% of importance retained
#plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
# Format x ticks and labels
#plt.xticks(x_values, sorted_features, rotation = 'vertical')
# Axis labels and title
#plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');
#plt.show()

# Find number of features for cumulative importance of 95%
# Add 1 because Python is zero-indexed
#print('Number of features for 95% importance:', np.where(cumulative_importances > 0.95)[0][0] + 1)

# Extract the names of the most important features
important_feature_names = [feature[0] for feature in feature_importances[0:4]]
# Find the columns of the most important features
important_indices = [feature_list.index(feature) for feature in important_feature_names]
# Create training and testing sets with only the important features
important_train_features = train_features[:, important_indices]
important_test_features = test_features[:, important_indices]
# Sanity check on operations
print('Important train features shape:', important_train_features.shape)
print('Important test features shape:', important_test_features.shape)

predictions = predictions.astype(int)
cm = confusion_matrix(test_labels,predictions)
print(cm)
#cre:https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#cre: https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd

recall = np.diag(cm) / np.sum(cm, axis = 1)
recall = pandas.DataFrame(recall)
recall = recall.replace(np.nan,0)
recall = np.array(recall)

precision = np.diag(cm) / np.sum(cm, axis = 0)
precision = pandas.DataFrame(precision)
precision = precision.replace(np.nan,0)
precision = np.array(precision)

AVG_precision = sum(precision)/len(precision)
AVG_precision = AVG_precision*100
AVG_precision = AVG_precision[0]

AVG_recall = sum(recall)/len(recall)
AVG_recall = AVG_recall*100
AVG_recall = AVG_recall[0]

AVG_f1 = 2 * ((AVG_precision * AVG_recall)/(AVG_precision + AVG_recall))

print("AVG_precision",AVG_precision)
print("AVG_recall",AVG_recall)
print("AVG_f1",AVG_f1)


cm_with_precision = cm/np.sum(cm, axis = 0)
cm_with_precision = np.nan_to_num(cm_with_precision)

cm_with_recall = (cm.T/np.sum(cm, axis = 1)).T
cm_with_recall = np.nan_to_num(cm_with_recall)
print(cm_with_precision)
print(cm_with_recall)
