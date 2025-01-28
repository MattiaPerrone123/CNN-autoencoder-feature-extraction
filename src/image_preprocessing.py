import numpy as np
import SimpleITK as sitk
import nibabel as nib
import scipy
import math
from skimage import measure
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.ndimage import rotate


def load_image_and_mask(directory_segm,directory_no_segm):
    """Load the image and mask"""
    image=nib.load(directory_no_segm)
    mask=nib.load(directory_segm)
    return image,mask


def process_array(array_main):
    """Separate vertebrae, discs, and spinal cord masks"""
    unique_values=np.unique(array_main)
    
    new_arrays={'array_vertebrae':np.zeros_like(array_main),'array_spinal':np.zeros_like(array_main),'array_disk':np.zeros_like(array_main)}
    
    for value in unique_values:
        if value!=0:
            if 1<=value<=99:
                new_arrays['array_vertebrae']+=np.where(array_main==value,array_main,0)
            elif value==100:
                new_arrays['array_spinal']=np.where(array_main==value,array_main,0)
            elif value>=201:
                new_arrays['array_disk']+=np.where(array_main==value,array_main,0)
    
    return new_arrays


def resample_array_to_target_shape(np_array,target_shape):
    """Resample a mask to a new specified spacing"""
    factors=[float(target)/float(original) for target,original in zip(target_shape,np_array.shape)]
    rescaled_array=scipy.ndimage.zoom(np_array,factors,order=1)
    
    return rescaled_array


def resample_sitk_image(sitk_image,new_spacing):
    """Resample a SimpleITK Image to a new specified spacing"""
    original_size=sitk_image.GetSize()
    original_spacing=sitk_image.GetSpacing()
    new_size=[int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size,original_spacing,new_spacing)]
    
    resampled_sitk_image=sitk.Resample(sitk_image,new_size,sitk.Transform(),sitk.sitkBSpline,
                                       sitk_image.GetOrigin(),new_spacing,sitk_image.GetDirection(),0,
                                       sitk_image.GetPixelID())
    
    return resampled_sitk_image


def expand_bounding_box(data,expansion_margin=10):
    """Identify the smallest 3D bounding box around non-zero elements and expand it"""
    if not isinstance(data,np.ndarray):
        raise ValueError("Data must be a 3D numpy array.")
    if data.ndim!=3:
        raise ValueError("Data must be a 3D numpy array.")
    
    non_zero_indices=np.argwhere(data)
    min_idx=non_zero_indices.min(axis=0)
    max_idx=non_zero_indices.max(axis=0)
    
    min_x,min_y,min_z=np.maximum(min_idx-expansion_margin,0)
    max_x,max_y,max_z=np.minimum(max_idx+expansion_margin,np.array(data.shape)-1)
    
    return (slice(min_x,max_x+1),slice(min_y,max_y+1),slice(min_z,max_z+1))


def resample_image(image,target_resolution):
    """Resample the given image to the target resolution"""
    return resample_sitk_image(image,target_resolution)


def process_image(img_file,target_resolution):
    """Resample a sitk file to a target resolution, converting it to NumPy array"""
    image=sitk.ReadImage(img_file)
    img_rescaled=resample_image(image,target_resolution)
    np_image=sitk.GetArrayFromImage(img_rescaled)
    
    if 2*np_image.shape[-1]>np_image.shape[1] or 2*np_image.shape[-1]>np_image.shape[0]:
        return None
    
    return np_image


def process_mask(msk_file,np_image_shape):
    """Read and process a mask file, categorizing it into arrays"""
    msk=sitk.ReadImage(msk_file)
    np_msk=sitk.GetArrayFromImage(msk)
    processed_masks=process_array(np_msk)
    
    np_msk_vert_rescaled=resample_array_to_target_shape(processed_masks["array_vertebrae"],np_image_shape)
    np_msk_disk_rescaled=resample_array_to_target_shape(processed_masks["array_disk"],np_image_shape)
    np_msk_spinal_rescaled=resample_array_to_target_shape(processed_masks["array_spinal"],np_image_shape)
    
    np_msk_vert_rescaled=np.clip(np_msk_vert_rescaled,0,1)
    np_msk_disk_rescaled=np.clip(np_msk_disk_rescaled,0,1)
    np_msk_spinal_rescaled=np.clip(np_msk_spinal_rescaled,0,1)
    
    return {"vert":np_msk_vert_rescaled,"disk":np_msk_disk_rescaled,"spinal":np_msk_spinal_rescaled}


def crop_images_masks(image_list,mask_list):
    """Crop a list of images and masks according to the bounding boxes"""
    cropped_images=[]
    cropped_masks=[]
    
    for img,msk in zip(image_list,mask_list):
        bbox_slices_msk=expand_bounding_box(msk)
        cropped_images.append(img[bbox_slices_msk])
        cropped_masks.append(msk[bbox_slices_msk])
    
    return cropped_images,cropped_masks


def normalize_slice(slice_img,epsilon=1e-7):
    """Normalize a slice using min-max scaling and z-score standardization"""
    min_value=np.min(slice_img)
    max_value=np.max(slice_img)
    scaled_slice=(slice_img-min_value)/(max_value-min_value+epsilon)
    mean_value=np.mean(scaled_slice)
    std_value=np.std(scaled_slice)
    standardized_slice=(scaled_slice-mean_value)/(std_value+epsilon)
    return sigmoid(standardized_slice)


def process_single_volume(volume,epsilon=1e-7):
    """Process a single 3D volume by normalizing each slice"""
    processed_slices=[normalize_slice(volume[:,:,i],epsilon) for i in range(volume.shape[2])]
    return np.stack(processed_slices,axis=0)  


def process_image_4d(volumes,mask):
    """Perform normalization and sigmoid transformation on 4D images"""
    if mask!=0:
        raise ValueError("Mask value must be 0 for processing volumes.")
    processed_volumes=[np.expand_dims(process_single_volume(volume),axis=0) for volume in volumes]
    result=np.concatenate(processed_volumes,axis=0)

    return np.moveaxis(result,[0,1,2,3],[0,3,1,2])


def pad_images_masks(image_list,mask_list,target_dims,pad_value_img,pad_value_msk):
    """Pad images and masks to target dimensions"""
    padded_images=[]
    padded_masks=[]
    
    for img,msk in zip(image_list,mask_list):
        if not all(current<=target for current,target in zip(img.shape,target_dims)):
            raise ValueError("Target dimensions must be greater than or equal to image dimensions")
        
        padding=[]
        for current,target in zip(img.shape,target_dims):
            total_padding=target-current
            padding_before=total_padding//2
            padding_after=total_padding-padding_before
            padding.append((padding_before,padding_after))
        
        padded_img=np.pad(img,padding,mode='constant',constant_values=pad_value_img)
        padded_msk=np.pad(msk,padding,mode='constant',constant_values=pad_value_msk)
        padded_images.append(padded_img)
        padded_masks.append(padded_msk)
    
    return padded_images,padded_masks
    

def rotate_and_swap_axes(image):
    """Rotate a 3D image by swapping axes and flipping along the second axis"""
    temp_img=np.moveaxis(image, [0,1,2], [1,2,0])
    return np.flip(temp_img, axis=1)


def next_multiple_of_32(number):
    """Calculate the smallest multiple of 32 greater than or equal to a number"""
    return math.ceil(number/32)*32


def find_max_dimensions(image_list):
    """Find the maximum size for each dimension across all images"""
    max_dim1=max_dim2=max_dim3=0
    
    for img in image_list:
        if img.ndim!=3:
            raise ValueError("All images must be 3D.")
        max_dim1=max(max_dim1,img.shape[0])
        max_dim2=max(max_dim2,img.shape[1])
        max_dim3=max(max_dim3,img.shape[2])
    
    return max_dim1,max_dim2,max_dim3


def sigmoid(x):
    """Apply sigmoid function to squash values"""
    return 1/(1+np.exp(-x))


def calculate_target_dimensions(cropped_masks):
    """Find the target dimensions for padding based on max dimensions of cropped masks"""
    max_dim1, max_dim2, max_dim3 = find_max_dimensions(cropped_masks)
    target_dims = (
        next_multiple_of_32(max_dim1),
        next_multiple_of_32(max_dim2),
        next_multiple_of_32(max_dim3),
    )
    return target_dims


def process_images_and_masks(padded_images, padded_masks):
    """Process and normalize images and masks"""
    stacked_images = np.stack(padded_images)
    stacked_masks = np.stack(padded_masks)
    processed_images = np.expand_dims(process_image_4d(stacked_images, mask=0), axis=1)
    processed_masks = np.expand_dims(stacked_masks, axis=1)
    return processed_images, processed_masks


def rotate_and_reformat_images_and_masks(processed_images, processed_masks):
    """Rotate and reformat images and masks to the desired shape"""
    rotated_images = [rotate_and_swap_axes(processed_images[i, 0]) for i in range(processed_images.shape[0])]
    rotated_masks = [rotate_and_swap_axes(processed_masks[i, 0]) for i in range(processed_masks.shape[0])]
    return np.stack(rotated_images), np.stack(rotated_masks)


def preprocess_and_rotate_images_and_masks(images, masks, pad_value_img=-1000, pad_value_msk=0):
    """Full preprocessing pipeline: crop, pad, normalize, and rotate images and masks"""

    cropped_images, cropped_masks = crop_images_masks(images, masks)
    target_dims = calculate_target_dimensions(cropped_masks)
    padded_images, padded_masks = pad_images_masks(cropped_images, cropped_masks, target_dims, pad_value_img, pad_value_msk)
    processed_images, processed_masks = process_images_and_masks(padded_images, padded_masks)
    rotated_images, rotated_masks = rotate_and_reformat_images_and_masks(processed_images, processed_masks)
    
    return rotated_images, rotated_masks



def keep_largest_blob(image):
    "Keeps only the largest labeled blob in a binary image"
    labeled_image,num_features=label(image)
    if num_features==0:
        return np.zeros_like(image, dtype=image.dtype)
    blob_sizes=np.bincount(labeled_image.ravel())[1:]
    largest_blob_label=np.argmax(blob_sizes)+1
    largest_blob_mask=np.where(labeled_image==largest_blob_label, 1, 0)
    return largest_blob_mask


def separate_blobs(image):
    "Separates distinct labeled regions (non-zero values) in an image into individual binary masks"
    unique_values=np.unique(image)
    unique_values=unique_values[unique_values != 0]
    output_images=[]
    for value in unique_values:
        mask=np.where(image==value, 1, 0)
        output_images.append(mask)

    return output_images


def process_test_set_masks(msk_t1_MRI_new_test):
    "Processes test set masks by separating blobs and concatenating all separated blobs"
    msk_t1_MRI_single_list=[]
    for i in range(len(msk_t1_MRI_new_test)):
        curr_image_multilabel=measure.label(msk_t1_MRI_new_test[i])
        curr_image_sep=separate_blobs(curr_image_multilabel)
        msk_t1_MRI_single_list.append(np.stack(curr_image_sep))
    msk_t1_MRI_single=np.concatenate(msk_t1_MRI_single_list, axis=0)
    return msk_t1_MRI_single


def crop_and_pad_mask(mask):
    "Crops the mask to its bounding box and pads it to a 64x64x64 shape"
    non_zero_indices=np.argwhere(mask)
    if non_zero_indices.size==0:
        return np.zeros((64, 64, 64), dtype=mask.dtype)
    min_coords=non_zero_indices.min(axis=0)
    max_coords=non_zero_indices.max(axis=0)
    cropped_mask=mask[min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1, min_coords[2]:max_coords[2]+1]
    current_shape=cropped_mask.shape
    pad_height=(64-current_shape[0])//2
    pad_width=(64-current_shape[1])//2
    pad_depth=(64-current_shape[2])//2
    pad_height_extra=(64-current_shape[0])%2
    pad_width_extra=(64-current_shape[1])%2
    pad_depth_extra=(64-current_shape[2])%2
    padded_mask=np.pad(cropped_mask,((pad_height,pad_height+pad_height_extra),(pad_width,pad_width+pad_width_extra),(pad_depth,pad_depth+pad_depth_extra)),mode='constant')
    return padded_mask


def process_and_stack_masks(msk_t1_MRI_single):
    "Applies crop and pad to each mask and stacks the results into a single array"
    msk_t1_MRI_single_list=[]
    for i in range(len(msk_t1_MRI_single)):
        msk_t1_MRI_single_list.append(crop_and_pad_mask(msk_t1_MRI_single[i]))
    msk_t1_MRI_single_def=np.stack(msk_t1_MRI_single_list)
    return msk_t1_MRI_single_def



class MaskAlignerAllAngles:
    def __init__(self, mask, slice_dim=0):
        """Initialize with a 3D mask and user-specified slicing dimension (0, 1, or 2)"""
        if not isinstance(mask, np.ndarray) or mask.ndim!=3:
            raise ValueError("Input mask must be a 3D NumPy array")
        if slice_dim not in [0, 1, 2]:
            raise ValueError("slice_dim must be 0, 1, or 2. Choose based on the desired slicing orientation")
        
        self.mask=mask
        self.slice_dim=slice_dim
        self.middle_idx=self._get_middle_index()
        self.middle_slice=self._get_middle_slice()
        self.angle_degrees=None

    def _get_middle_index(self):
        """Compute the middle index for slicing"""
        return self.mask.shape[self.slice_dim]//2

    def _get_middle_slice(self):
        """Extract the middle slice"""
        if self.slice_dim==0:
            return self.mask[self.middle_idx, :, :]
        elif self.slice_dim==1:
            return self.mask[:, self.middle_idx, :]
        elif self.slice_dim==2:
            return self.mask[:, :, self.middle_idx]

    def compute_rotation_angle(self):
        """Compute rotation angle to align the mask horizontally"""
        coords=np.array(np.nonzero(self.middle_slice)).T
        mean=np.mean(coords, axis=0)
        coords_centered=coords-mean
        _, _, vh=np.linalg.svd(coords_centered, full_matrices=False)
        principal_axis=vh[0]
        angle=np.arctan2(principal_axis[1], principal_axis[0])
        angle_degrees=np.degrees(angle)
        if angle_degrees>90:
            angle_degrees-=180
        elif angle_degrees<-90:
            angle_degrees+=180
        self.angle_degrees=angle_degrees

    def align_mask(self):
        """Rotate the mask based on the computed angle"""
        rotated_mask=np.zeros_like(self.mask)
        if self.slice_dim==0:
            if self.angle_degrees<0:
                self.angle_degrees+=90
            else:
                self.angle_degrees-=90
            for i in range(self.mask.shape[0]):
                rotated_mask[i, :, :]=rotate(self.mask[i, :, :], -self.angle_degrees, reshape=False, order=1)
        elif self.slice_dim==1:
            for i in range(self.mask.shape[1]):
                rotated_mask[:, i, :]=rotate(self.mask[:, i, :], -self.angle_degrees, reshape=False, order=1)
        elif self.slice_dim==2:
            for i in range(self.mask.shape[2]):
                rotated_mask[:, :, i]=rotate(self.mask[:, :, i], -self.angle_degrees, reshape=False, order=1)
        return rotated_mask, self.angle_degrees

    def visualize(self):
        """Visualize the original and rotated middle slices"""
        if self.angle_degrees is None:
            raise ValueError("Call compute_rotation_angle() and align_mask() before visualization")
        rotated_middle=rotate(self.middle_slice, -self.angle_degrees, reshape=False, order=1)
        plt.subplot(1, 2, 1)
        plt.title("Original Middle Slice")
        plt.imshow(self.middle_slice, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Rotated Middle Slice")
        plt.imshow(rotated_middle, cmap='gray')
        plt.show()

