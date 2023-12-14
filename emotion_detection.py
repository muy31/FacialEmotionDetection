import os
import argparse
import pickle
import numpy as np
from sklearn import tree, ensemble
from sklearn.decomposition import PCA
from PIL import Image
import datetime

#Parse arguments
par = argparse.ArgumentParser()
par.add_argument('--training_data', type = str, default = "./images/train", help = 'The root directory where all training images are stored')
par.add_argument('--validation_data', type = str, default = "./images/validation", help = 'The root directory where all validation images are stored')
par.add_argument('--feature_set_length', type = int, default = None, help = 'How extensive should the PCA feature set be')
par.add_argument('--leaf_size', type = int, default = None, help = 'The number of datapoints in each leaf. Higher = smoother, less complexity')
par.add_argument('--max_leaves', type = int, default = None, help = 'The number of leaves in model. Higher = more complexity')
par.add_argument('--model_save', type = str, default = "./sample_results", help = 'The directory to save model and data visualizations')

args = par.parse_args()

#Other initializer global variables
debug = False
save_folder_path = args.model_save + "/" + datetime.datetime.now().strftime('%H_%M_%S-%Y-%m-%d') + "/"
os.makedirs(save_folder_path, exist_ok=True)

def create_pca(data, num_components):
    pca = PCA(n_components=num_components)
    pca.fit(data)

    #Save pca as pickle file
    with open(save_folder_path + "pca_image", "wb") as pca_file:
        pickle.dump(pca, pca_file)

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

    if debug:
        Image.fromarray(image.reshape(48, 48)).show()
        Image.fromarray(reconstruct_image(transformed_image, pca).reshape(48, 48)).show()

    return transformed_image.flatten().astype(float)

#Load training data
label_names = os.listdir(args.training_data) #Should show angry, happy, etc.
print(label_names)

dictionary = {}
data = []

for folder in label_names:
    for image_name in os.listdir(args.training_data + "/" + folder):
        image_name = args.training_data + "/" + folder + "/" + image_name

        #Unsure if we should flatten or not
        img_arr = np.array(Image.open(image_name).convert('L'))
        dictionary[image_name] = (img_arr, folder)
        data.append(img_arr.flatten())

data = np.array(data)
print(data)
pca = create_pca(data, args.feature_set_length)

re_expressed_data = []
train_labels = []

#Honestly probably don't need the dictionary, I'm just afraid of the preservation of order
for image_name in dictionary:
        if debug:
            debug = False
            print(image_name)
            re_expressed_data.append(encode_image(dictionary[image_name][0].flatten().reshape(1, -1), pca, True))
            train_labels.append(dictionary[image_name][1])
        else:
            re_expressed_data.append(encode_image(dictionary[image_name][0].flatten().reshape(1, -1), pca, False))
            train_labels.append(dictionary[image_name][1])


print(len(re_expressed_data))

#Generate and save random forest model
#Build RF
if args.leaf_size == None:
    dtr = ensemble.RandomForestClassifier(n_estimators = 1000, max_leaf_nodes = args.max_leaves)
else:
    dtr = ensemble.RandomForestClassifier(n_estimators = 1000, max_leaf_nodes = args.max_leaves, min_samples_leaf = args.leaf_size)

#Train RF
regr = dtr.fit(re_expressed_data, train_labels)

with open(save_folder_path + "rf_model.obj", "wb") as filehandler:
    pickle.dump(regr, filehandler)

#Training Error
pred_train = regr.predict(re_expressed_data)

accuracy_count = 0
for index in range(0, len(train_labels)):
    if pred_train[index] == train_labels[index]:
        accuracy_count += 1

print("Training Accuracy:")
print(accuracy_count, len(train_labels), accuracy_count/len(train_labels))


#Testing Error
#Load validation data
test_labels = []
test_data = []

for folder in label_names:
    for image_name in os.listdir(args.validation_data + "/" + folder):
        image_name = args.validation_data + "/" + folder + "/" + image_name

        img_arr = np.array(Image.open(image_name).convert('L'))
        test_labels.append(folder)
        test_data.append(encode_image(img_arr.flatten().reshape(1, -1), pca, debug = False))

pred_test = regr.predict(test_data)

accuracy_count = 0
for index in range(0, len(test_labels)):
    if pred_test[index] == test_labels[index]:
        accuracy_count += 1

print("Testing Accuracy")
print(accuracy_count, len(test_labels), accuracy_count/len(test_labels))



