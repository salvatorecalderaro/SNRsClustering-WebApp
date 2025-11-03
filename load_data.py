from astropy.io import fits
import numpy as np
import scipy as sp
import os 
from PIL import Image

img_size = (256, 256)


def read_image(path):
    image_data=fits.getdata(path, ext=0)
    return image_data

def calculate_area(obj):
    row_slice, col_slice = obj
    row_length = row_slice.stop - row_slice.start
    col_length = col_slice.stop - col_slice.start
    return row_length * col_length

def resizing_xy(new_size_x, new_size_y, image_element):
    margin_y = np.abs(new_size_y - image_element.shape[0]) // 2
    margin_x = np.abs(new_size_x - image_element.shape[1]) // 2
    
    if (new_size_y - image_element.shape[0]) % 2 !=0 and (new_size_x - image_element.shape[1]) % 2 ==0:
        resized_image = np.pad(image_element, ((margin_y, margin_y + 1),(margin_x, margin_x )), mode='constant')
        
    if (new_size_y - image_element.shape[0]) % 2 ==0 and (new_size_x - image_element.shape[1]) % 2 !=0:
        resized_image = np.pad(image_element, ((margin_y, margin_y),(margin_x, margin_x + 1)), mode='constant')
        
    if (new_size_y - image_element.shape[0]) % 2 !=0 and (new_size_x - image_element.shape[1]) % 2 !=0:
        resized_image = np.pad(image_element, ((margin_y, margin_y +1),(margin_x, margin_x + 1)), mode='constant')
        
    if (new_size_y - image_element.shape[0]) % 2 ==0 and (new_size_x - image_element.shape[1]) % 2 ==0:
        resized_image = np.pad(image_element, ((margin_y, margin_y ),(margin_x, margin_x )), mode='constant')
    return resized_image



def countor_mask(path):
    image_mask = fits.getdata(path, ext=0)
    mask_image_data=image_mask.copy()
    mask_image_data[mask_image_data!=0]=1
    mask_image_data=mask_image_data.astype(mask_image_data.dtype.newbyteorder('<'))
    
    labels, nlabels=sp.ndimage.label(mask_image_data)
    objects=sp.ndimage.find_objects(labels, max_label=0)
    
    largest_object = max(objects, key=calculate_area)
    largest_index=max(range(len(objects)), key=lambda i: calculate_area(objects[i])) +1 
    return labels, largest_index



def crop_to_mask(image_data,labels_matrix,largest_index):
    x_min, x_max=np.min(np.where(labels_matrix==largest_index)[0]), np.max(np.where(labels_matrix==largest_index)[0])
    y_min, y_max=np.min(np.where(labels_matrix==largest_index)[1]), np.max(np.where(labels_matrix==largest_index)[1])
    image_data=image_data[x_min:x_max, y_min:y_max]
    shape_x, shape_y=image_data.shape
    new_size=int(np.max([shape_x, shape_y]))
    resized_img=resizing_xy(new_size, new_size, image_data)
    return image_data,resized_img


def load_data(star,image_type,clustering_method):
    mhd_bands = [
        "Fe L",
        "Mg tot",
        "Si tot",
        "S tot",
        "Ar tot",
        "Ca tot",
        "Cont 1",
        "Cont 2",
        "Fe XXV",
        "cont 3 / Ni XXVI"    
    ]
    
    sn_bands = {
        "CAS_A": [
            "Mg_XI",
            "Mg_XII",
            "Si_XIII",
            "Si_XIV",
            "Si_XIII_2",
            "S_XV",
            "S_XVI",
            "S_XV_2",
            "Ar_XVII",
            "Ca_XIX",
            "Ca_XX",
            "cont_1",
            "cont_2/Ti",
            "cont_3",
            "cont_4/Cr",
            "cont_5/Mg",
            "Fe_XXV",
            "cont_6",
            "Ni_XXVI"
        ],
        "G292":[
            "O VIII", 
            "Ne IX", 
            "Ne X",  
            "cont 1", 
            "Mg XI", 
            "Mg XII", 
            "cont 2", 
            "Si XIII", 
            "Si XIV", 
            "cont 3", 
            "S XV", 
            "cont 4", 
            "Ar XVII", 
            "Ar XVIII", 
            "cont 5", 
            "Ca XIX", 
            "cont 6", 
            "cont 7", 
            "Fe XXV", 
            "cont 8"
            ],
        "KEPLER":[
            "cont 1",
            "Si XIII",
            "cont 2",
            "Si XIII 2",
            "S XV",
            "cont 3",
            "Ar XVII",
            "cont 4",
            "cont 5",
            "cont/CrMg",
            "cont 6"],
        
        "TYCHO": [
            "Mg XI",
            "Si XIII 2",
            "S Total",
            "S XV 2",
            "cont 1",
            "cont/Cr",
            "cont/Mg",
            "Fe XXV",
            "cont 2"
        ],
        "W49B": [
            "Si XII",
            "Si XIV",
            "cont 1",
            "S XV",
            "S XVI",
            "S XV 2",
            "Ar XVII",
            "Ar XVIII",
            "cont 2",
            "Ca XIX",
            "Ca XX",
            "cont 3",
            "cont/Ti",
            "cont 4",
            "cont/Cr",
            "cont 5",
            "Fe XXV",
            "cont 6",
            "Ni XXVI"
        ]
    }
    
    if "ALL" not in star:
        if "MHD" in star:
            bands = mhd_bands
        else:
            bands = sn_bands[star]
        
        mask_path = f"data/{image_type}/{star}/mask_{star}.fits"
        folder = os.path.join("data", image_type, star, "images")
        images = []
        flattened_images = []
        for imname in sorted(os.listdir(folder)):
            img_path = os.path.join(folder, imname)
            image = read_image(img_path)
            labels,largest_index = countor_mask(mask_path)
            masked_image,masked_image_square = crop_to_mask(image, labels, largest_index) 
            im = Image.fromarray(masked_image_square).resize(img_size)
            images.append(np.array(im))
            flattened_images.append(masked_image_square.flatten())
    else:
        pass
    
    return images,flattened_images,bands