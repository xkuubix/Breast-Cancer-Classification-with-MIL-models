from pydicom import dcmread
import torch
import os
from torchvision import transforms
from skimage import img_as_float32


class BreastCancerDataset(torch.utils.data.Dataset):
    '''
        Take path to folder of one class, set view to
        CC for cranio-cauadal or
        MLO for medio-lateral oblique
        to select dataset type.
        Call with index returns img, target view/labels
    '''
    def __init__(self, root, view: list, transforms):
        self.root = root
        self.view = view
        self.dicoms = self.__select_view(filter_img_size=True)
        self.transforms = transforms

    def __getitem__(self, idx):
        os.chdir(self.root)
        dcm = dcmread(self.dicoms[idx])

        img = dcm.pixel_array
        height, width = img.shape

        img = img_as_float32(img)
        img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        img = img/torch.max(img)

        if self.transforms is not None:
            img = self.transforms(img)
        target = {}
        # CCE long 0 1 2 3, BCE float 0. 1.
        if os.path.basename(os.getcwd()) == 'Normal':
            target["labels"] = torch.tensor(0.)
            target["class"] = 'Normal'
        elif os.path.basename(os.getcwd()) == 'Benign':
            target["labels"] = torch.tensor(0.)
            target["class"] = 'Benign'
        elif os.path.basename(os.getcwd()) == 'Malignant':
            target["labels"] = torch.tensor(1.)
            target["class"] = 'Malignant'
        elif os.path.basename(os.getcwd()) == 'Lymph_nodes':
            target["labels"] = torch.tensor(1.)
            target["class"] = 'Lymph_nodes'

        target["view"] = dcm.ViewPosition
        target["file"] = self.dicoms[idx]
        target['patient_id'] = dcm.PatientID
        target["age"] = self.__get_age(dcm)
        target["laterality"] = self.__get_laterality(dcm)
        target["img_h"] = height
        target["img_w"] = width

        if target["laterality"] == 'R':
            t = transforms.RandomHorizontalFlip(p=1.0)
            img = t(img)

        return img, target

    def __len__(self):
        return len(self.dicoms)

    def __select_view(self, filter_img_size):
        '''Select only one view and return list of filenames'''
        # 0x0018, 0x5101 - View Position
        os.chdir(self.root)
        dicoms = [file for file in os.listdir(os.getcwd())
                  if str(dcmread(file).ViewPosition).find(str(self.view[0]))
                  is not -1
                  or str(dcmread(file).ViewPosition).find(str(self.view[1]))
                  is not -1
                  and filter_img_size
                  and dcmread(file).Rows == 3518
                  and dcmread(file).Columns == 2800
                  or str(dcmread(file).ViewPosition).find(str(self.view[0]))
                  is not -1
                  or str(dcmread(file).ViewPosition).find(str(self.view[1]))
                  is not -1
                  and not filter_img_size
                  ]
        return dicoms

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
