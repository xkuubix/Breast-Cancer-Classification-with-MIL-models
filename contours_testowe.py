#%%
from typing import List, Set
from pydicom.uid import UID
import pydicom
from pydicom import dcmread
# to dodać do końcowej funkcji

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage as ski

# DICOM tags for linking and identification
Referenced_SOP_Instance_UID = (0x8, 0x1155)  # Link to DICOM by ID
SOP_Instance_UID = (0x8, 0x18)               # DICOM ID

# DICOM tags for dicom image properties (pixels)
Imager_Pixel_Spacing = (0x18, 0x1164) # Scaling mm to px
Rows = (0x28, 0x10)                   # Image height in px
Columns = (0x28, 0x11)                # Image width

# DICOM tags for referencing and sequences
Referenced_Frame_of_Reference_Sequence = (0x3006, 0x10)
RT_Referenced_Study_Sequence = (0x3006, 0x12)
RT_Referenced_Series_Sequence = (0x3006, 0x14)
Contour_Image_Sequence = (0x3006, 0x16)

# DICOM tags for ROI and contour sequences
Structure_Set_ROI_Sequence = (0x3006, 0x20)
ROI_Number = (0x3006, 0x22)  # Link between ROI Contour Sequence and Structure Set ROI Sequence
ROI_Name = (0x3006, 0x26)    # User-defined name for ROI

ROI_Contour_Sequence = (0x3006, 0x39)  # Sequence of Contour Sequences defining ROIs
Contour_Sequence = (0x3006, 0x40)      # Sequence of Contours defining ROI
Contour_Data = (0x3006, 0x50)          # Sequence of (x, y, z) triplets defining a contour
Ref_ROI_Number = (0x3006, 0x84)

#%% 
# 
# 
# TUTAJ zrobić funkcję rtstruct2nii
# 
# 
os.chdir('/media/dysk_a/jr_buler/Mammografie/a_Dane_Hist_Path/UC6_8_test')
folders = os.listdir(os.getcwd())
for folder in folders:
    if folder != 'exp_ECI_GUM_S0080_189597':
        continue
    print('\n\n---------------------------------------')

    dcm_rts = get_rtstruct(folder)
    if dcm_rts != None:
        ref_sop_uid_mask_list = parse_ref_sop_uid(dcm_rts)
                            # break
    for root, _, files in os.walk(folder, topdown=False):
        for name in files:
            if name.endswith(".dcm"):
                dcm = dcmread(os.path.join(root, name))
                if dcm.Modality == 'RTSTRUCT':
                    continue
                for ref_sop_uid in ref_sop_uid_mask_list:
                    if ref_sop_uid == dcm[SOP_Instance_UID].value:
                        mask = create_mask(dcm, dcm_rts, ref_sop_uid)
                        print(os.path.join(root, name))
                        plot_mask(dcm, mask)
    # break # 1 subject
#%%
def get_rtstruct(folder: str) -> pydicom.FileDataset:
    """Returns dicom (FileDataset) with rtstruct data from given folder
    of single dicom series
    
    Parameters
    - folder: str or PathLike or file-like object
    """
    dcm = None
    for root, _, files in os.walk(folder, topdown=False):
        for name in files:
            if name.endswith(".dcm"):
                dcm = dcmread(os.path.join(root, name))
                if dcm.Modality == 'RTSTRUCT':
                    for i in dcm[(0x3006, 0x020)]:
                        print(i[ROI_Name].value)
                    return dcm
    if not dcm:
        raise FileNotFoundError

def parse_ref_sop_uid(dcm_rts: pydicom.FileDataset) -> Set[UID]:
    """Returns Set of pydicom.uid.UID objects, links between DCM and DCM_RT-struct
    ----------
    Parameters
    - dcm_rts: dicom (FileDataset) with rtstruct data
    """
    ref_sop_uid_mask_set: Set[UID] = set()
    for item in dcm_rts[ROI_Contour_Sequence]:
        for contours in item[Contour_Sequence].value:
            for contour in contours[Contour_Image_Sequence]:
                if contour[Referenced_SOP_Instance_UID].value not in ref_sop_uid_mask_set:
                    ref_sop_uid_mask_set.update([contour[Referenced_SOP_Instance_UID].value])
                    # print(contour[Referenced_SOP_Instance_UID].value) 
    return ref_sop_uid_mask_set
#%%
# 
# 
# TUTAJ zrobić: roi name; zapis do nii
# 
# 
def create_mask(dcm: pydicom.FileDataset, dcm_rts: pydicom.FileDataset,
                ref_sop_uid: pydicom.uid.UID, bitwise_operator="OR") -> np.ndarray:
    """Returns NumPy array mask
    
    Parameters
    - dcm: dicom (FileDataset) data
    - dcm_rts: dicom (FileDataset) with rtstruct data
    - ref_sop_uid: pydicom.uid.UID
    - bitwise_operator: str -> "OR" or "XOR" for joining contour sequences
    """
    image_spacing = dcm[Imager_Pixel_Spacing].value
    image_shape = (dcm[Rows].value, dcm[Columns].value)
    roi_contour_seq = dcm_rts[ROI_Contour_Sequence]
    roi_struct_set = dcm_rts[Structure_Set_ROI_Sequence]
    mask = np.zeros(image_shape, dtype=np.uint8)
    roi_dict = {"number": [], "name": [], "mask" :[]}
    
    for item in roi_struct_set:
        roi_dict['name'].append(item.ROIName)
        roi_dict['number'].append(item.ROINumber)
        roi_dict['mask'].append(np.zeros(image_shape, dtype=bool))
    for roi in roi_contour_seq:
        print('ref roi num:', roi[Ref_ROI_Number].value, '\t', end=' ')
        print('roi dict', roi_dict['number'], '\t', end=' ')
        print('roi dict', roi_dict['name'])
        for contour_sequence in roi[Contour_Sequence].value:
            for contour in contour_sequence[Contour_Image_Sequence]:
                if contour[Referenced_SOP_Instance_UID].value == ref_sop_uid:
                    points = np.array(contour_sequence[Contour_Data].value)
                    points = points.reshape(-1, 3)
                    points = np.delete(points, obj=2, axis=1) # del empty z axis xy[z] - 01[2] column
                    points = points / image_spacing
                    points = np.floor(points)
                    points = points[:, [1, 0]]
                    mask_ith = ski.draw.polygon2mask(image_shape, points)
                    mask_ith.astype(bool)
                    if bitwise_operator == "OR":
                        roi_dict['mask'][roi[Ref_ROI_Number].value - 1] = np.logical_or(roi_dict['mask'][roi[Ref_ROI_Number].value - 1], mask_ith)
                    elif bitwise_operator == "XOR":
                        roi_dict['mask'][roi[Ref_ROI_Number].value - 1] = np.logical_xor(roi_dict['mask'][roi[Ref_ROI_Number].value - 1], mask_ith)
                    else:
                        raise ValueError
                else:
                    continue
        for item in roi_dict['name']:
            if item[0] == 'L':
                item = np.uint8(item.split('L')[1])
                print(item)
                #tu coś
                mask += roi_dict['mask'][roi[Ref_ROI_Number].value - 1]*item
    print('unique mask', np.unique(mask))
    return mask
#%%
def plot_mask(dcm: pydicom.FileDataset, mask: np.ndarray):
    """Plots masks and images
    
    Parameters
    - dcm: dicom (FileDataset) data
    - mask: array with segmentation mask
    """
    image = dcm.pixel_array

    num_rows = 1
    num_cols = 3
    figsize = (dcm.Rows/100, dcm.Columns/100)
    fig, ax = plt.subplots(num_rows, num_cols)
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])

    masked = np.ma.masked_where(mask == 0, mask)
    ax[0].imshow(image, cmap='gray', interpolation='none')

    ax[0].tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)
    # contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Green color, thickness 2
    # for contour in contours:
    #     ax[0].imshow(contour[0])
    ax[1].imshow(image, cmap='gray', interpolation='none')
    ax[1].imshow(masked, cmap='autumn', interpolation='none', alpha=0.5)

    ax[1].tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)
    # cmap = plt.get_cmap('gray', len(np.unique(mask)))
    ax[2].imshow(mask, cmap='gray', interpolation='none')

    ax[2].tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)
    plt.show()
#%%

dir_rts = '/media/dysk_a/jr_buler/Mammografie/a_Dane_Hist_Path/UC6_8_test/exp_ECI_GUM_S0079_189592/2/annotations/event_4d4ea55c-6da8-4d59-a3b3-8c3bcefbc092/segmentation.dcm'
dcm_rts = dcmread(dir_rts)
dir_dcm = '/media/dysk_a/jr_buler/Mammografie/a_Dane_Hist_Path/UC6_8_test/exp_ECI_GUM_S0079_189592/2/DICOM/ECI_GUM_S0079_S01E001_1_0012.dcm'
dcm = dcmread(dir_dcm)
image = dcm.pixel_array
image_spacing = dcm[Imager_Pixel_Spacing].value
roi_contour_seq = dcm_rts[ROI_Contour_Sequence]
#%%
# image_dict = dcm.to_json_dict(None, None)
# image_spacing = image_dict['00181164']['Value']
# contour_seq = contour_seq.to_json_dict(None, None)
#contour_seq['Value'][1]['30060040']['Value'][od 0 do ok.1300]['30060050']['Value']
# print(len(contour_seq['Value'][0]['30060040']['Value'])) 
# print(len(contour_seq['Value'][1]['30060040']['Value']))
contour_dict = {}
contour_dict['seq'] = []
contour_dict['data'] = []
contour_dict['points'] = []

for i in range (len(roi_contour_seq.value)):
    contour_dict['seq'].append(roi_contour_seq.value[i][Contour_Sequence].value)

for i, sequence in enumerate(contour_dict['seq']):
    contour_dict['data'].append([])
    for v in range(len(sequence)):
        contour_dict['data'][i].extend(sequence[v][Contour_Data].value)

for sd in contour_dict['data']:
    contour_dict['points'].append(np.array([sd[0::3], sd[1::3]]).transpose() / image_spacing)


#%%
#---------------------------------------------XOR LUB OR
image_shape = (dcm[Rows].value, dcm[Columns].value)
mask = np.zeros(image_shape, dtype=bool)
for i in range(len(contour_dict['seq'][0])):
    points = np.array(contour_dict['seq'][0][i][Contour_Data].value)
    points = points.reshape(-1, 3)
    points = np.delete(points, obj=2, axis=1) # del empty z axis xy[z] - 01[2] column
    points = points / image_spacing
    points = np.floor(points)
    points = points[:, [1, 0]]
    mask_ith = ski.draw.polygon2mask(image_shape, points)
    mask_ith.astype(np.uint8)
    mask = np.bitwise_or(mask, mask_ith)
    # plt.scatter(points[:, 0], points[:,1], color='red', marker='.')
    # plt.fill(points[:, 0], points[:,1])
fig, ax = plt.subplots(1,3, figsize=(24, 8))

masked = np.ma.masked_where(mask == 0, mask)
ax[0].imshow(image, cmap='gray', interpolation='none')
ax[0].set_xlim(1200, 1700)
ax[0].set_ylim(2200, 1700)
ax[0].tick_params(left = False, right = False , labelleft = False ,
                  labelbottom = False, bottom = False)
contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Green color, thickness 2
for contour in contours:
    ax[0].imshow(contour[0])
ax[1].imshow(image, cmap='gray', interpolation='none')
ax[1].imshow(masked, cmap='viridis', interpolation='none', alpha=0.5)
ax[1].set_xlim(1200, 1700)
ax[1].set_ylim(2200, 1700)
ax[1].tick_params(left = False, right = False , labelleft = False ,
                  labelbottom = False, bottom = False)
ax[2].imshow(mask, cmap='gray', interpolation='none')
ax[2].set_xlim(1200, 1700)
ax[2].set_ylim(2200, 1700)
ax[2].tick_params(left = False, right = False , labelleft = False ,
                  labelbottom = False, bottom = False)
plt.show()
#---------------------------------------------KKK
#%%
# plt.plot(contour_dict['points'][1][:,0], contour_dict['points'][1][:,1], '.')
# plt.show()
plt.imshow(image, cmap='gray')
plt.scatter(contour_dict['points'][1][:,0] ,contour_dict['points'][1][:,1], color='red', marker='.')
plt.show()

plt.imshow(image, cmap='gray')
plt.scatter(contour_dict['points'][1][:,0] ,contour_dict['points'][1][:,1], color='red', marker='.')
plt.xlim(contour_dict['points'][1][:,0].min(), contour_dict['points'][1][:,0].max())
plt.ylim(contour_dict['points'][1][:,1].min(), contour_dict['points'][1][:,1].max())
plt.show()
# %%


dir_dcm = '/media/dysk_a/jr_buler/Mammografie/a_Dane_Hist_Path/UC6_8_test/exp_ECI_GUM_S0079_189592/2/DICOM/ECI_GUM_S0079_S01E001_1_0010.dcm'
dcm = dcmread(dir_dcm)
image = dcm.pixel_array

# plt.plot(contour_dict['points'][0][:,0], contour_dict['points'][0][:,1], '.')
# plt.show()
plt.imshow(image, cmap='gray')
plt.scatter(contour_dict['points'][0][:,0] ,contour_dict['points'][0][:,1], color='red', marker='.')
plt.show()

plt.imshow(image, cmap='gray')
plt.scatter(contour_dict['points'][0][:,0] ,contour_dict['points'][0][:,1], color='red', marker='.')
plt.xlim(contour_dict['points'][0][:,0].min(), contour_dict['points'][0][:,0].max())
plt.ylim(contour_dict['points'][0][:,1].min(), contour_dict['points'][0][:,1].max())
plt.show()

#%%
points_int = np.round(contour_dict['points'][0]).astype(int)


empty_img = np.zeros((3518, 2800), np.uint8)
empty_img[points_int[:,0], points_int[:,1]] = 255
img = empty_img.transpose(1, 0)
contours, _ = cv2.findContours(img,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


plt.imshow(img, cmap='gray')
plt.xlim(1200, 1700)
plt.ylim(1700, 2200)
plt.show()
for contour in contours:
    cv2.drawContours(empty_img, [contour], 0, 255, thickness=cv2.FILLED)


# img = cv2.drawContours(img,[points_int],0,(0,0,0),2)
# plt.show(img)



# %%
from matplotlib.animation import FuncAnimation

fig = plt.figure()
graph, = plt.plot([], '.')
plt.imshow(image, cmap='gray')
plt.xlim(1200, 1700)
plt.ylim(1700, 2200)

def animate(i):
    graph.set_data(contour_dict['points'][0][:i+1,0], contour_dict['points'][0][:i+1,1])
    return graph

# Set up the animation
num_frames = len(contour_dict['points'][0])
ani = FuncAnimation(fig, animate, frames=num_frames, interval=10)
ani.save('testowe_ani.mp4')
plt.show()
#%%