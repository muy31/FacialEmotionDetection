import os
import argparse
import pickle
from PIL import Image, ImageOps
import numpy as np
from sklearn import tree, ensemble

#Parse arguments
par = argparse.ArgumentParser()
par.add_argument('--model', type = str, help = 'The path to the trained random forest model')
par.add_argument('--pca_path', type = str, help = 'The path to the trained PCA')
par.add_argument('--image_folder', type = str, default = "./sample_image/image4.jpg", help = 'The path to the image to run inference on')
par.add_argument('--save_path', type = str, default = "./sample_results", help = 'The directory to save inference and visualizations')

args = par.parse_args()

folder = args.image_folder
isImage = False
if folder.endswith(".jpg") or folder.endswith(".png"):
    isImage = True

#Open and unpickle model and pca
if args.model == None:
    print("No model given!")
    sys.exit(0)

if args.pca_path == None:
    print("No PCA given!")
    sys.exit(0)

with open(args.model, 'rb') as model_file:
    classifier_model = pickle.load(model_file)

with open(args.pca_path, 'rb') as pca_file:
    pca = pickle.load(pca_file)

def pad_to_size(img, length = 1280, height = 720):
    x_pad = length - img.shape[1]
    y_pad = height - img.shape[0]
    return np.pad(img,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

def crop_new(arr):

    mask = arr != 0
    n = mask.ndim
    dims = range(n)
    slices = [None]*n

    for i in dims:
        mask_i = mask.any(tuple([*dims[:i] , *dims[i+1:]]))
        slices[i] = (mask_i.argmax(), len(mask_i) - mask_i[::-1].argmax())

    return arr[tuple([slice(*s) for s in slices])]

def load_image(filename):
    #Image loader and transforms
    image = Image.open(filename).convert('L')
    image = ImageOps.exif_transpose(image)
    image = crop_new(np.array(image))
    image = Image.fromarray(image)
    image.thumbnail((48, 48), Image.Resampling.LANCZOS) #Keep aspect ratio
    image = np.array(image)
    image = pad_to_size(image, 48, 48) #Add black for everything else

    return image

def reconstruct_image(transformed_image, pca):
    # Inverse transform the transformed image to reconstruct the original shape
    reconstructed_image = pca.inverse_transform(transformed_image).reshape(48,48)
    return reconstructed_image

def encode_image(img, pca):
    img = img.flatten().reshape(1, -1)
    transformed_image = pca.transform(img)
    return transformed_image

def shape_transformed_data(data, n_features):
    if data.shape[1] > n_features:
        #Our pca had too many features trim our data (take first n)
        data = data[:, 0:n_features]
        
    elif data.shape[1] < n_features:
        #We need to pad our data with zeros
        data = np.pad(data, [(0, 0), (0, n_features - data.shape[1])], mode = 'constant')

    return data

def inferImage(img, pca, model):
    img = img.flatten().reshape(1, -1)
    expressed_img = pca.transform(img)
    expressed_img = shape_transformed_data(expressed_img, model.n_features_in_)
    return model.predict(expressed_img)

#Run inference on all images
results_dict = {}
reconstructions = []

if isImage:

    image = load_image(folder)

    #Inference
    results_dict[folder] = inferImage(image, pca, classifier_model)

    #Reconstruction
    encoded_image = encode_image(image, pca)
    reconstructed_image = reconstruct_image(encoded_image, pca)
    reconstructions.append([image, encoded_image, reconstructed_image])

else:
    for name in os.listdir(folder):
        if name.endswith(".jpg"):
            img_name = folder + "/" + name

            #Image loader
            image = load_image(img_name)

            #Inference
            results_dict[img_name] = inferImage(image, pca, classifier_model)[0]

            #Reconstruction
            encoded_image = encode_image(image, pca)
            reconstructed_image = reconstruct_image(encoded_image, pca)
            reconstructions.append([image, encoded_image, reconstructed_image])

#Print Inference
for thing in results_dict.items():
    print(thing[0], thing[1])

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

#See reconstruction in action
for triple in reconstructions:
    orig = Image.fromarray(triple[0])
    reexpressed = Image.fromarray(triple[1])
    reconst = Image.fromarray(triple[2])
    
    combined_image = get_concat_h(orig, reconst)
    combined_image.show()




