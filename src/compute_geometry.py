import numpy as np
from .image_preprocessing import MaskAlignerAllAngles


def find_medial_mask_indices(mask):
    """Find the medial indices (centroid) of a 3D mask"""
    indices=np.argwhere(mask)
    z_centroid=int(np.median(indices[:,0]))
    y_centroid=int(np.median(indices[:,1]))
    x_centroid=int(np.median(indices[:,2]))
    return z_centroid, y_centroid, x_centroid


def compute_disc_height(volume):
    """Compute the disc height from an axial projection of the volume"""
    axial_projection=np.sum(volume, axis=0)
    positive_pixels=axial_projection[axial_projection>0]
    positive_mean=positive_pixels.mean() if positive_pixels.size>0 else 0
    return axial_projection, positive_mean


def calculate_centroid(mask):
    """Calculate the centroid coordinates of a binary 2D mask"""
    y_coords, x_coords=np.mgrid[:mask.shape[0], :mask.shape[1]]
    centroid_x=np.mean(x_coords[mask.astype(bool)])
    centroid_y=np.mean(y_coords[mask.astype(bool)])
    return centroid_x, centroid_y


def find_perimeter_ones(matrix):
    """Find all '1's on the perimeter of a 2D binary matrix"""
    perimeter_ones=[]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j]==1:
                if (i==0 or i==matrix.shape[0]-1 or
                    j==0 or j==matrix.shape[1]-1 or
                    matrix[i-1,j]==0 or matrix[i+1,j]==0 or
                    matrix[i,j-1]==0 or matrix[i,j+1]==0):
                    perimeter_ones.append((i,j))
    return perimeter_ones


def find_intersection_points_horiz(matrix, centroid_x, centroid_y):
    """Find horizontal intersection points for a 2D binary matrix"""
    perimeter_ones=find_perimeter_ones(matrix)
    intersection_points=[]
    for perimeter_point in perimeter_ones:
        x, y=perimeter_point
        if y==centroid_x:
            intersection_points.append((x,y))
    return intersection_points


def find_intersection_points_vert(matrix, centroid_x, centroid_y):
    """Find vertical intersection points for a 2D binary matrix"""
    perimeter_ones=find_perimeter_ones(matrix)
    intersection_points=[]
    for perimeter_point in perimeter_ones:
        x, y=perimeter_point
        if x==centroid_y:
            intersection_points.append((x,y))
    return intersection_points


def compute_lat_width(volume):
    """Compute the lateral width from the axial slice of a 3D volume"""
    mid_axial_slice=find_medial_mask_indices(volume)[0]
    centroid_x, centroid_y=calculate_centroid(volume[mid_axial_slice])
    intersection_points=find_intersection_points_horiz(volume[mid_axial_slice], int(centroid_x), centroid_y)
    lat_width=np.abs(intersection_points[0][0]-intersection_points[-1][0])
    return lat_width


def compute_ap_width(volume):
    """Compute the anterior-posterior width from the axial slice of a 3D volume"""
    mid_axial_slice=find_medial_mask_indices(volume)[0]
    centroid_x, centroid_y=calculate_centroid(volume[mid_axial_slice])
    intersection_points=find_intersection_points_vert(volume[mid_axial_slice], int(centroid_x), int(centroid_y))
    ap_width=np.abs(intersection_points[0][1]-intersection_points[-1][1])
    return ap_width


def compute_angles_for_masks(masks):
    """Compute sagittal, frontal, and transverse angles for a list of masks"""
    sag_angle=[]
    front_angle=[]
    trasv_angle=[]
    for slice_dim in range(3):
        for mask in masks:
            aligner=MaskAlignerAllAngles(mask, slice_dim=slice_dim)
            aligner.compute_rotation_angle()
            _, angle=aligner.align_mask()
            if slice_dim==0:
                sag_angle.append(angle)
            elif slice_dim==1:
                trasv_angle.append(angle)
            elif slice_dim==2:
                front_angle.append(angle)
    return sag_angle, trasv_angle, front_angle


def rotate_masks_sequentially(masks):
    """Rotate each mask sequentially across all three planes (sagittal -> transverse -> frontal)"""
    rotated_masks=[]
    for mask in masks:
        aligner_sag=MaskAlignerAllAngles(mask, slice_dim=0)
        aligner_sag.compute_rotation_angle()
        rotated_mask_sag, _=aligner_sag.align_mask()
        aligner_trans=MaskAlignerAllAngles(rotated_mask_sag, slice_dim=1)
        aligner_trans.compute_rotation_angle()
        rotated_mask_trans, _=aligner_trans.align_mask()
        aligner_front=MaskAlignerAllAngles(rotated_mask_trans, slice_dim=2)
        aligner_front.compute_rotation_angle()
        rotated_mask_final, _=aligner_front.align_mask()
        rotated_masks.append(rotated_mask_final)
    return np.stack(rotated_masks)


def compute_disc_dimensions(rotated_masks):
    """Compute disc height, AP width, and lateral width for a list of rotated masks"""
    disc_height_list=[]
    lat_width_list=[]
    ap_width_list=[]
    for mask in rotated_masks:
        _, curr_disc_height=compute_disc_height(mask)
        curr_ap_width=compute_ap_width(mask)
        curr_lat_width=compute_lat_width(mask)
        disc_height_list.append(curr_disc_height)
        ap_width_list.append(curr_ap_width)
        lat_width_list.append(curr_lat_width)
    return disc_height_list, ap_width_list, lat_width_list
