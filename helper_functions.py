import imutils
import cv2
import os
from sklearn.utils import shuffle as shuffle


def extract_patches(image, mask, class_type, rots, hwsize, dict_labels, patches_dataset_dir, basename):
    """
    Helper function used for patch extraction.

    Parameters:
    --------
    image: nparray
        The image to be processed

    mask: nparray
        The mask used for patch extraction

    class_type: str
        The class of the extracted patches ("class1/class0"), representing the positive/negative samples

    rots: int array
        The array of rotations that will be used in creating the patches

    hwsize: int
        The size used for creating the patches

    dict_labels: dict
        The dictionary that stores the labels of the patches

    patches_dataset_dir: str
        The location of the dataset folder

    basename: str
        The name of the original image (eg. 01_02)

    """

    # Get the size of the image (width, height)
    imsize_w = image.shape[0]
    imsize_h = image.shape[1]

    # Get the row and column coordinates of the positive pixels
    rows = mask[0]
    cols = mask[1]

    # Establish the condition based on which the positive pixels will be kept
    # ie. remove the pixels that are located at any of the borders of the image
    # because they cannot form valid patches
    # Note. [2*hwsize] is used because the initial patches will be double the size,
    # since we want to be able to rotate them
    condition_row = (rows + 2 * hwsize <= imsize_h) & (rows - 2 * hwsize >= 0)
    condition_col = (cols + 2 * hwsize <= imsize_w) & (cols - 2 * hwsize >= 0)

    # Remove the border pixels; keep the original rows, cols
    rows_filt = rows[condition_row & condition_col]
    cols_filt = cols[condition_col & condition_row]

    # Shuffle the array elements (but keep the mapping)
    # This is useful for the negative class, where not all pixels will be sampled
    rows_filt, cols_filt = shuffle(rows_filt, cols_filt, random_state=0)

    print(class_type + " number of pix after edge filtering= " + str(len(rows_filt)))

    # Decide which folder label to use, based on the input class type
    if class_type == 'class1':
        patch_folder = 'positive'
        # Calculate the number of samples to be extracted: total number of positive pixels
        dim = round(len(rows_filt))
    else:
        patch_folder = 'negative'
        # Calculate the number of samples to be extracted: 1/500 of the total number of negative pixels
        dim = round(len(rows_filt) / 500)

    if not os.path.exists(patches_dataset_dir + patch_folder):
        os.mkdir(patches_dataset_dir + patch_folder)

    print(" final number of pix to be extracted: ", str(dim))

    # For each center
    for idx in range(dim):
        try:
            # Extract the 64x64 patch 
            patch = image[rows_filt[idx] - hwsize:rows_filt[idx] + hwsize,
                          cols_filt[idx] - hwsize:cols_filt[idx] + hwsize]
        except:
            #             print("rows_filt["+str(idx)+"]="+str(rows_filt[idx]))
            continue

        # For each rotation
        for rot in rots:

            # Generate a patch
            rpatch = imutils.rotate(patch, angle=rot)

            patch_name = os.path.join(patches_dataset_dir,
                                      patch_folder,
                                      basename + "_" +
                                      class_type +
                                      "_center" + str(idx) +
                                      "_rot" + str(rot) +
                                      ".png")

            # Store the label in the dictionary
            dict_labels["patch_name"] = class_type

            # Resize the patch to 32x32 
            # TODO: must find a more elegant way for this
            rpatch_resized = rpatch[16:48, 16:48, :]

            status = cv2.imwrite(patch_name, rpatch_resized)

            if ~status:
                # print("Image " + os.path.split(patch_name)[1] + " was NOT written")
                continue

            # break "rot" loop
#             break

# break center "idx" loop
#         break
