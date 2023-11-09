#%%
from pydicom import dcmread
import torch
import os
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from skimage import img_as_float32
from tile_maker import convert_img_to_bag
import random
from fnmatch import fnmatch

#%%
def gauss_noise(img, p):
    if p < torch.rand(1):
        return img
    img_shape = img.shape
    r = random.randint(1, 10) * 1e-4
    noise = (r**0.5)*torch.randn(img_shape)
    img = img + noise
    return img


class BreastCancerDataset(torch.utils.data.Dataset):
    '''
        Take path to folder of one class, set view to
        CC for cranio-cauadal or
        MLO for medio-lateral oblique
        to select dataset type.
        Call with index returns img, target view/labels
    '''
    def __init__(self, root, df, view: list, transforms,
                 conv_to_bag=False, bag_size=50, tiles=None,
                 img_size=[3518, 2800], is_multimodal=False):
        self.root = root
        self.view = view
        self.df = df
        self.img_size = img_size
        self.multimodal = is_multimodal
        self.views, self.dicoms, self.class_name = self.__select_view()
        self.transforms = transforms
        self.convert_to_bag = conv_to_bag
        self.bag_size = bag_size
        self.tiles = tiles
        # self.multimodal = True  # ##################

    def __getitem__(self, idx):
        os.chdir(os.path.join(self.root, self.class_name[idx]))

        if self.multimodal:
            dcm = dcmread(self.dicoms[idx][0])
            img_CC = dcm.pixel_array
            height, width = img_CC.shape
            img_CC = img_CC/4095
            img_CC = img_as_float32(img_CC)
            img_CC = torch.from_numpy(img_CC).unsqueeze(0).repeat(3, 1, 1)

            dcm = dcmread(self.dicoms[idx][1])
            img_MLO = dcm.pixel_array
            img_MLO = img_MLO/4095
            img_MLO = img_as_float32(img_MLO)
            img_MLO = torch.from_numpy(img_MLO).unsqueeze(0).repeat(3, 1, 1)

            img = torch.cat((img_MLO, img_CC), dim=1)
            _, height, width = img.shape
            #####
            if (height != self.img_size[0]) and (width != self.img_size[1]):
                t = T.Resize((self.img_size[0], self.img_size[1]), antialias=True)
                img = t(img)
            #####
        else:
            dcm = dcmread(self.dicoms[idx])
            img = dcm.pixel_array
            height, width = img.shape
            img = img/4095
            img = img_as_float32(img)
            img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)

            #####
            if (height != self.img_size[0]) and (width != self.img_size[1]):
                t = T.Resize((self.img_size[0], self.img_size[1]), antialias=True)
                img = t(img)
            #####
        # img = img/torch.max(img)

        target = {}
        # CCE long 0 1 2 3, BCE float 0. 1.
        if self.class_name[idx] == 'Normal':
            target["labels"] = torch.tensor(0)
            target["class"] = 'Normal'
        elif self.class_name[idx] == 'Benign':
            target["labels"] = torch.tensor(1)
            target["class"] = 'Benign'
        elif self.class_name[idx] == 'Malignant':
            target["labels"] = torch.tensor(2)
            target["class"] = 'Malignant'
        elif self.class_name[idx] == 'Lymph_nodes':
            target["labels"] = torch.tensor(3)
            target["class"] = 'Lymph_nodes'

        target["view"] = self.views[idx]
        target["file"] = self.dicoms[idx]
        target['patient_id'] = dcm.PatientID
        target["age"] = self.__get_age(dcm)
        target["laterality"] = self.__get_laterality(dcm)
        target["img_h"] = height
        target["img_w"] = width

        if target["laterality"] == 'R':
            t = T.RandomHorizontalFlip(p=1.0)
            img = t(img)
        # translation -px (white strips near image border)
        img = TF.affine(img, angle=0, translate=(-20, 0), scale=1, shear=0)

        if self.convert_to_bag:
            target['full_image'] = img
            # Multi scale bag instances
            if len(self.tiles) == 2:
                if target["laterality"] == 'R':
                    t = T.RandomHorizontalFlip(p=1.0)
                    img = t(img)

                instances_scale_1, t_id_scale_1, t_cord1 = convert_img_to_bag(
                    img, self.tiles[0], self.bag_size)
                instances_scale_2, t_id_scale_2, t_cord2 = convert_img_to_bag(
                    img, self.tiles[1], self.bag_size)
                t = T.Resize(224, antialias=True)
                instances_scale_2 = t(instances_scale_2)
                img = torch.cat((instances_scale_1, instances_scale_2),
                                dim=0)
                if self.transforms is not None:
                    for i, image in enumerate(img):
                        angle = random.choice([-90, 0, 90, 180])
                        img[i] = TF.rotate(img[i], angle)
                        # img[i] = self.transforms(img[i])
                    img = self.transforms(img)
                target['tiles_indices'] = [t_id_scale_1, t_id_scale_2]
            # Single scale bag instances
            else:
                img, t_id, t_cord = convert_img_to_bag(img, self.tiles,
                                                       self.bag_size)
                target['tiles_indices'] = t_id
                target['tile_cords'] = t_cord

                if self.transforms is not None:
                    prob_cj = random.choice([0, 1])
                    prob_g = random.choice([0, 1])
                    color_jitter = T.ColorJitter(0.25, 0.25, 0.25, 0.25)
                    gaussian_blur = T.GaussianBlur(kernel_size=23,
                                                   sigma=(0.1, 2.0))
                    # jtt_gss = T.Compose([color_jitter, gaussian_blur])
                    for i, _ in enumerate(img):
                        angle = random.choice([-90, 0, 90, 180])
                        img[i] = TF.rotate(img[i], angle)
                        if prob_cj < 0.5:
                            img[i] = color_jitter(img[i])
                        if prob_g < 0.5:
                            img[i] = gaussian_blur(img[i])
                    # img = gauss_noise(img, p=0.5)
                    img = self.transforms(img)

            # img = (img - img.mean())/img.std()
            # img = (img - 0.2327)/0.1592
            # img = (img - 0.221)/0.146
            # b_lims = torch.zeros(img.shape)
            # u_lims = torch.ones(img.shape)
            # img = torch.where(img < 0., b_lims, img)
            # img = torch.where(img > 1., u_lims, img)

        return img, target

    def __len__(self):
        return len(self.dicoms)

    def __select_view(self):
        '''Select only given view(s) and return list of filenames
           and class names(folder names)
        '''
        class_names_list = []
        filenames_list = []
        view_list = []
        patients = self.df.to_dict('records')

        if self.multimodal:
            for patient in patients:
                # take 2 first rows (sorted) if 2 or more rows are present
                if 'LCC' in patient['view'] and 'LMLO' in patient['view']:
                    # if patient['class'][0] in ['Malignant', 'Lymph_nodes']:
                    class_names_list.append(patient['class'][0])
                    filenames_list.append(patient['filename'][0:2])
                    view_list.append('Left')
                # take 2 last rows (sorted) if 2 or more rows are present
                if 'RCC' in patient['view'] and 'RMLO' in patient['view']:
                    # if patient['class'][-1] in ['Malignant', 'Lymph_nodes']:
                    class_names_list.append(patient['class'][-1])
                    filenames_list.append(patient['filename'][-2:])
                    view_list.append('Rigth')
        else:
            for patient in patients:
                for item in range(len(patient['class'])):
                    for v in self.view:
                        if patient['view'][item].__contains__(v):
                            class_names_list.append(patient['class'][item])
                            filenames_list.append(patient['filename'][item])
                            view_list.append(patient['view'][item])
            # if patient['class'][item].find('Malignant') is not -1:
            #     class_names_list.append(patient['class'][item])
            #     filenames_list.append(patient['filename'][item])
            #     view_list.append(patient['view'][item])
            # if patient['class'][item].find('Lymph_nodes') is not -1:
            #     class_names_list.append(patient['class'][item])
            #     filenames_list.append(patient['filename'][item])
            #     view_list.append(patient['view'][item])

        return view_list, filenames_list, class_names_list

    def __get_age(self, dcm):
        '''Read Patient's age from DICOM data'''
        dcm_tag = (0x0010, 0x1010)
        # 0x0010, 0x1010 - Patient's Age in form 'dddY'
        idx_end = str(dcm[dcm_tag]).find('Y')
        return int(str(dcm[dcm_tag])[idx_end-3:idx_end])

    def __get_laterality(self, dcm):
        '''
        Read Image Laterality from DICOM data
        Returns 'L' or 'R' as string type dependent on breast laterality
        '''
        return dcm.ImageLaterality


class CMMD_DS(torch.utils.data.Dataset):
    '''
       df = pd.read_csv('CMMD_clinicaldata_revision_CSV.csv', sep=';')
       root = '/media/dysk/student2/CMMD/TheChineseMammographyDatabase/CMMD/'
    '''
    def __init__(self, root, df, view: list, transforms,
                 conv_to_bag=False, bag_size=50, tiles=None,
                 img_size=[1914, 2294], is_multimodal=False):
        self.root = root
        self.view = view
        self.df = df
        self.img_size = img_size
        self.multimodal = is_multimodal
        self.views, self.dicoms, self.class_name = self.__make_df()
        self.transforms = transforms
        self.convert_to_bag = conv_to_bag
        self.bag_size = bag_size
        self.tiles = tiles
        # self.multimodal = True  # ##################

    def __getitem__(self, idx):
        # os.chdir(os.path.join(self.root, self.class_name[idx]))

        if self.multimodal:
            dcm = dcmread(self.dicoms[idx][0])
            img_CC = dcm.pixel_array
            height, width = img_CC.shape
            img_CC = img_CC/255
            img_CC = img_as_float32(img_CC)
            img_CC = torch.from_numpy(img_CC).unsqueeze(0).repeat(3, 1, 1)

            dcm = dcmread(self.dicoms[idx][1])
            img_MLO = dcm.pixel_array
            img_MLO = img_MLO/255
            img_MLO = img_as_float32(img_MLO)
            img_MLO = torch.from_numpy(img_MLO).unsqueeze(0).repeat(3, 1, 1)

            img = torch.cat((img_MLO, img_CC), dim=1)
            _, height, width = img.shape
            #####
            if (height != self.img_size[0]) and (width != self.img_size[1]):
                t = T.Resize((self.img_size[0], self.img_size[1]), antialias=True)
                img = t(img)
            #####
        else:
            raise NotImplementedError

        target = {}

        if self.class_name[idx] == 'Benign':
            target["labels"] = torch.tensor(0.)
            target["class"] = 'Benign'
        elif self.class_name[idx] == 'Malignant':
            target["labels"] = torch.tensor(1.)
            target["class"] = 'Malignant'

        target["view"] = self.views[idx]
        target["file"] = self.dicoms[idx]
        target["age"] = self.__get_age(dcm)
        target["laterality"] = self.__get_laterality(dcm)
        target["img_h"] = height
        target["img_w"] = width

        if target["laterality"] == 'R':
            t = T.RandomHorizontalFlip(p=1.0)
            img = t(img)

        if self.convert_to_bag:
            target['full_image'] = img
            # Multi scale bag instances
            if len(self.tiles) == 2:
                if target["laterality"] == 'R':
                    t = T.RandomHorizontalFlip(p=1.0)
                    img = t(img)

                instances_scale_1, t_id_scale_1, t_cord1 = convert_img_to_bag(
                    img, self.tiles[0], self.bag_size)
                instances_scale_2, t_id_scale_2, t_cord2 = convert_img_to_bag(
                    img, self.tiles[1], self.bag_size)
                t = T.Resize(224, antialias=True)
                instances_scale_2 = t(instances_scale_2)
                img = torch.cat((instances_scale_1, instances_scale_2),
                                dim=0)
                if self.transforms is not None:
                    for i, image in enumerate(img):
                        angle = random.choice([-90, 0, 90, 180])
                        img[i] = TF.rotate(img[i], angle)
                        # img[i] = self.transforms(img[i])
                    img = self.transforms(img)
                target['tiles_indices'] = [t_id_scale_1, t_id_scale_2]
            # Single scale bag instances
            else:
                img, t_id, t_cord = convert_img_to_bag(img, self.tiles,
                                                       self.bag_size)
                target['tiles_indices'] = t_id
                target['tile_cords'] = t_cord

                if self.transforms is not None:
                    # prob_cj = random.choice([0, 1])
                    # prob_g = random.choice([0, 1])
                    color_jitter = T.ColorJitter(0.25, 0.25, 0.25, 0.25)
                    gaussian_blur = T.GaussianBlur(kernel_size=23,
                                                   sigma=(0.1, 2.0))
                    # jtt_gss = T.Compose([color_jitter, gaussian_blur])
                    for i, _ in enumerate(img):
                        angle = random.choice([-90, 0, 90, 180])
                        img[i] = TF.rotate(img[i], angle)
                        if 0:
                            img[i] = color_jitter(img[i])
                        if 0:
                            img[i] = gaussian_blur(img[i])
                    # img = gauss_noise(img, p=0.5)
                    img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.dicoms)

    def __make_df(self):
        '''Select only given view(s) and return list of filenames
           and class names(folder names)
        '''
        class_names_list = []
        filenames_list = []
        view_list = []
        patients = self.df.to_dict('records')

        if self.multimodal:
            for patient in patients:
                pattern = '*.dcm'
                empty_list = []
                for path, subdirs, files in os.walk(
                        os.path.join(self.root, str(patient['ID1']))):
                    for name in files:
                        if fnmatch(name, pattern):
                            empty_list.append(os.path.join(path, name))
                if len(patient['classification']) == 1:
                    # filenames_list.append(empty_list)
                    to_append_CC = ''
                    to_append_MLO = ''
                    for f in empty_list:
                        dcm = dcmread(f)
                        if dcm.ImageLaterality != patient['LeftRight'][0]:
                            continue
                        else:
                            if str(dcm[(0x054,
                                        0x220)][0][(0x008,
                                                    0x104)]).__contains__(
                                                        'cranio'):
                                to_append_CC = f
                        # view_list.append(dcm.ImageLaterality + 'CC')
                            elif str(dcm[(0x054,
                                          0x220)][0][(0x008,
                                                      0x104)]).__contains__(
                                                          'medio'):
                                to_append_MLO = f
                    if (to_append_CC == '') or (to_append_MLO == ''):
                        continue
                    else:
                        class_names_list.append(patient['classification'][0])
                        filenames_list.append([to_append_CC, to_append_MLO])
                        view_list.append(patient['LeftRight'])
                elif len(patient['classification']) == 2:
                    for i in range(len(patient['classification'])):
                        to_append_CC = ''
                        to_append_MLO = ''
                        for f in empty_list:
                            dcm = dcmread(f)
                            if dcm.ImageLaterality != patient['LeftRight'][i]:
                                continue
                            else:
                                if str(dcm[(0x054,
                                            0x220)][0][(0x008,
                                                        0x104)]).__contains__(
                                                            'cranio'):
                                    to_append_CC = f
                            # view_list.append(dcm.ImageLaterality + 'CC')
                                elif str(dcm[(0x054,
                                              0x220)][0][
                                                  (0x008,
                                                   0x104)]).__contains__(
                                                              'medio'):
                                    to_append_MLO = f
                    if (to_append_CC == '') or (to_append_MLO == ''):
                        continue
                    else:
                        class_names_list.append(patient['classification'][i])
                        filenames_list.append([to_append_CC, to_append_MLO])
                        view_list.append(patient['LeftRight'][i])
        else:
            raise NotImplementedError

        return view_list, filenames_list, class_names_list

    def __get_age(self, dcm):
        '''Read Patient's age from DICOM data'''
        dcm_tag = (0x0010, 0x1010)
        # 0x0010, 0x1010 - Patient's Age in form 'dddY'
        idx_end = str(dcm[dcm_tag]).find('Y')
        return int(str(dcm[dcm_tag])[idx_end-3:idx_end])

    def __get_laterality(self, dcm):
        '''
        Read Image Laterality from DICOM data
        Returns 'L' or 'R' as string type dependent on breast laterality
        '''
        return dcm.ImageLaterality

# %%
