from pydicom import dcmread
import torch
import os
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from skimage import img_as_float32
from tile_maker import convert_img_to_bag
import random


class BreastCancerDataset(torch.utils.data.Dataset):
    '''
        Take path to folder of one class, set view to
        CC for cranio-cauadal or
        MLO for medio-lateral oblique
        to select dataset type.
        Call with index returns img, target view/labels
    '''
    def __init__(self, root, df, view: list, transforms,
                 conv_to_bag=False, bag_size=50, tiles=None):
        self.root = root
        self.view = view
        self.df = df
        self.views, self.dicoms, self.class_name = self.__select_view()
        self.transforms = transforms
        self.convert_to_bag = conv_to_bag
        self.bag_size = bag_size
        self.tiles = tiles

    def __getitem__(self, idx):
        os.chdir(os.path.join(self.root, self.class_name[idx]))
        dcm = dcmread(self.dicoms[idx])

        img = dcm.pixel_array
        height, width = img.shape
        img = img/4095
        img = img_as_float32(img)
        img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        # img = img/torch.max(img)

        target = {}
        # CCE long 0 1 2 3, BCE float 0. 1.
        if self.class_name[idx] == 'Normal':
            target["labels"] = torch.tensor(0.)
            target["class"] = 'Normal'
        elif self.class_name[idx] == 'Benign':
            target["labels"] = torch.tensor(0.)
            target["class"] = 'Benign'
        elif self.class_name[idx] == 'Malignant':
            target["labels"] = torch.tensor(1.)
            target["class"] = 'Malignant'
        elif self.class_name[idx] == 'Lymph_nodes':
            target["labels"] = torch.tensor(1.)
            target["class"] = 'Lymph_nodes'

        target["view"] = self.views[idx]
        target["file"] = self.dicoms[idx]
        target['patient_id'] = dcm.PatientID
        target["age"] = self.__get_age(dcm)
        target["laterality"] = self.__get_laterality(dcm)
        target["img_h"] = height
        target["img_w"] = width

        # if target["laterality"] == 'R':
        #     t = T.RandomHorizontalFlip(p=1.0)
        #     img = t(img)

        if self.convert_to_bag:
            target['full_image'] = img
            # Multi scale bag instances
            if len(self.tiles) == 2:
                if target["laterality"] == 'R':
                    t = T.RandomHorizontalFlip(p=1.0)
                    img = t(img)

                instances_scale_1, t_id_scale_1 = convert_img_to_bag(
                    img, self.tiles[0], self.bag_size)
                instances_scale_2, t_id_scale_2 = convert_img_to_bag(
                    img, self.tiles[1], self.bag_size)
                t = T.Resize(224)
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
                img, t_id = convert_img_to_bag(img, self.tiles, self.bag_size)
                target['tiles_indices'] = t_id
                if self.transforms is not None:
                    for i, image in enumerate(img):
                        angle = random.choice([-90, 0, 90, 180])
                        img[i] = TF.rotate(img[i], angle)
                        # img[i] = self.transforms(img[i])
                    img = self.transforms(img)

            # mean = [0.2347, 0.2347, 0.2347]
            # std = [0.1602, 0.1602, 0.1602]
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            # t = T.Normalize(mean, std)
            # img = t(img)

        return img, target

    def __len__(self):
        return len(self.dicoms)

    def __select_view(self):
        '''Select only given view(s) and return list of filenames
           and class names(folder names)
        '''
        # 0x0018, 0x5101 - View Position
        class_names_list = []
        filenames_list = []
        view_list = []
        patients = self.df.to_dict('records')
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
