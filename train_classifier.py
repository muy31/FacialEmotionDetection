import os
import argparse
import pickle
import numpy as np
from sklearn import tree, ensemble
from sklearn.decomposition import PCA
from PIL import Image
import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Parse arguments
par = argparse.ArgumentParser()
par.add_argument('--training_data', type = str, default = "./images/train", help = 'The root directory where all training images are stored')
par.add_argument('--validation_data', type = str, default = "./images/validation", help = 'The root directory where all validation images are stored')
par.add_argument('--pca_path', type = str, default = None, help = 'Path to pre-made PCA model to avoid regeneration')
par.add_argument('--use_model', type = str, default = None, help = 'Path to pre-made RF obj to avoid re-training')
par.add_argument('--feature_set_length', type = int, default = None, help = 'How extensive should the PCA feature set be')
par.add_argument('--forest_size', type = int, default = 10, help = "The number of trees to generate on data, more => better model, but longer time")
par.add_argument('--leaf_size', type = int, default = None, help = 'The number of datapoints in each leaf. Higher = smoother, less complexity')
par.add_argument('--max_leaves', type = int, default = None, help = 'The number of leaves in model. Higher = more complexity')
par.add_argument('--model_save', type = str, default = "./sample_results", help = 'The directory to save model and data visualizations')

args = par.parse_args()

#Other initializer global variables
debug = False
save_folder_path = args.model_save + "/" + datetime.datetime.now().strftime('%H_%M_%S-%Y-%m-%d') + "/"
os.makedirs(save_folder_path, exist_ok=True)
print("Made folder: " + save_folder_path)

def create_pca(data, num_components):

    if args.pca_path != None:
        with open(args.pca_path, "rb") as pca_file:
            pca = pickle.load(pca_file)
            print("Successfully loaded pca.")
        with open(save_folder_path + "pca_image.obj", "wb") as pca_file:
            pickle.dump(pca, pca_file)
            print("Successfully saved pca.")
    else:
        pca = PCA(n_components=num_components)
        pca.fit(data)
        #Save pca as pickle file
        with open(save_folder_path + "pca_image.obj", "wb") as pca_file:
            pickle.dump(pca, pca_file)
            print("Successfully saved pca.")

    return pca

def reconstruct_image(transformed_image, pca):
    # Inverse transform the transformed image to reconstruct the original shape
    reconstructed_image = pca.inverse_transform(transformed_image)

    # Reshape the reconstructed image to its original shape
    #reconstructed_image = reconstructed_image.reshape(pca.mean_.shape)
    return reconstructed_image

#Takes an image file and outputs an array of features (to be used as inputs to our model)
def encode_image(image, pca, debug = False):
    #Run principal component analysis
    transformed_image = pca.transform(image)
    return transformed_image.flatten().astype(float)

def visualize_data(pca, transformed_data, chosen_feature_index = (0, 1), labels = None, save = True, saveId = ''):
    #Convert data
    transformed_data = np.array(transformed_data)
    reduced_x = transformed_data[:, chosen_feature_index[0]]
    reduced_y = transformed_data[:, chosen_feature_index[1]]

    color_dict = {
        "angry" : "red",
        "fear": "green",
        "happy" : "yellow",
        "disgust" : "purple",
        "neutral" : "black",
        "sad" : "blue",
        "surprise" : "pink"
        }

    color_map = [color_dict[label] for label in labels]

    # Plot the datapoints in 2D
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_x, reduced_y, c=color_map, alpha = 0.7, marker = '.')

    plt.title('PCA Visualization of Image Datapoints')
    plt.xlabel('Principal Component ' + str(chosen_feature_index[0]))
    plt.ylabel('Principal Component ' + str(chosen_feature_index[1]))

    markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in color_dict.values()]
    plt.legend(markers, color_dict.keys(), numpoints = 1)

    if save:
        filename = 'PC' + str(chosen_feature_index[0]) + "vsPC" + str(chosen_feature_index[1])
        plt.savefig(save_folder_path + filename + str(saveId) + ".png", bbox_inches='tight')

    plt.show()

def generate_rf(n_estimators, max_leaves, leaf_size, train_data, train_labels):
    #Generate and save random forest model
    #Build RF
    if leaf_size == None:
        dtr = ensemble.RandomForestClassifier(n_estimators = n_estimators, max_leaf_nodes = max_leaves)
    else:
        dtr = ensemble.RandomForestClassifier(n_estimators = n_estimators, max_leaf_nodes = max_leaves, min_samples_leaf = leaf_size)

    #Train RF
    regr = dtr.fit(train_data, train_labels)

    #Save RF model
    with open(save_folder_path + "rf_model.obj", "wb") as filehandler:
        pickle.dump(regr, filehandler)
        print("Model trained and saved successfully")

    return regr


def shape_transformed_data(data, n_features):
    data = np.array(data)

    if data.shape[1] > n_features:
        #Our pca had too many features trim our data (take first n)
        data = data[:, 0:n_features]
        
    elif data.shape[1] < n_features:
        #We need to pad our data
        data = np.pad(data, [(0, 0), (0, n_features - data.shape[1])], mode = 'constant')

    print(data.shape[1])
    return data

#Load training data
label_names = os.listdir(args.training_data) #Should show angry, happy, etc.

raw_data = []
re_expressed_data = []
train_labels = []

for folder in label_names:
    for image_name in os.listdir(args.training_data + "/" + folder):
        image_name = args.training_data + "/" + folder + "/" + image_name

        #Unsure if we should flatten or not
        img_arr = np.array(Image.open(image_name).convert('L'))
        raw_data.append(img_arr.flatten())
        train_labels.append(folder)

#Prepare and generate PCA model
raw_data = np.array(raw_data)

print(raw_data)

pca = create_pca(raw_data, args.feature_set_length)

#Transform all images according to PCA
for image in raw_data:
    re_expressed_data.append(encode_image(image.reshape(1, -1), pca, False))

#Get random forest model
if args.use_model == None:
    print("Attempting to train a novel rf model")
    regr = generate_rf(args.forest_size, args.max_leaves, args.leaf_size, re_expressed_data, train_labels)
else:
    #Get rf model from file
    print("Attempting to read rf model from file.")
    with open(args.use_model, 'rb') as filehandler:
        regr = pickle.load(filehandler)
        print("Successfully loaded random forest model with " + str(regr.n_features_in_) + " input features.")

#Visualize random forest model
print(regr.feature_importances_)

#Get two best principal components
best_pcs = np.argpartition(regr.feature_importances_, -2)[-2:]
print(best_pcs, regr.feature_importances_[best_pcs])

#Trim re_expressed_data if necessary
re_expressed_data = shape_transformed_data(re_expressed_data, regr.n_features_in_)

visualize_data(pca, re_expressed_data, labels = train_labels, chosen_feature_index = best_pcs, save = True, saveId = "train_data")

#Visualize training error
pred_train = regr.predict(re_expressed_data)

accuracy_count = 0
for index in range(0, len(train_labels)):
    if pred_train[index] == train_labels[index]:
        accuracy_count += 1

print("Training Accuracy:")
print(accuracy_count, len(train_labels), accuracy_count/len(train_labels))

#Load validation data
test_labels = []
test_data = []

for folder in label_names:
    for image_name in os.listdir(args.validation_data + "/" + folder):
        image_name = args.validation_data + "/" + folder + "/" + image_name

        img_arr = np.array(Image.open(image_name).convert('L'))
        test_labels.append(folder)
        test_data.append(encode_image(img_arr.flatten().reshape(1, -1), pca, debug = False))


#Visualize the first two components of the data
visualize_data(pca, test_data, labels = test_labels, chosen_feature_index = best_pcs, save = True, saveId = "test_data")

#Shape data if necessary
test_data = shape_transformed_data(test_data, regr.n_features_in_)

#Visualize validation error
pred_test = regr.predict(test_data)

accuracy_count = 0
for index in range(0, len(test_labels)):
    if pred_test[index] == test_labels[index]:
        accuracy_count += 1

print("Testing Accuracy")
print(accuracy_count, len(test_labels), accuracy_count/len(test_labels))



