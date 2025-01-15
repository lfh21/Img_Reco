import os
import PIL.Image as Image
import numpy as np
import torch
from torchvision import transforms


class Dataset:

    def __init__(self, datapath, downsample_rate=1.0, gray=False, crop=False,dataset='SVM',datatypes_select=[],labelmap={}):
        self.datapath = datapath
        self.downsample_rate = downsample_rate
        self.gray = gray
        self.crop = crop
        self.dataset = dataset
        self.datatypes_select = datatypes_select
        self.labelmap = labelmap
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor()         # 将图像转换为张量，并将像素值归一化到[0, 1]
])

    
    def load(self):
        """Load dataset according to dataset."""
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        if self.dataset == 'SVM':
            for datatype_select in self.datatypes_select:
                for filename in os.listdir(os.path.join(self.datapath, 'train', datatype_select)):
                    if  filename.endswith('pt'):
                        train_data.append(torch.load(os.path.join(self.datapath, 'train', datatype_select, filename)).numpy().reshape(-1))
                        train_label.append(datatype_select)
                for filename in os.listdir(os.path.join(self.datapath, 'val',datatype_select)):
                    if  filename.endswith('pt'):
                        test_data.append(torch.load(os.path.join(self.datapath, 'val', datatype_select, filename)).numpy().reshape(-1))
                        test_label.append(datatype_select)
        elif self.dataset == 'CNN':
            # Load train data
            index_train = 0
            for datatype in os.listdir(os.path.join(self.datapath, 'train')):
                datatype_path = os.path.join(self.datapath, 'train', datatype)
                if os.path.isdir(datatype_path):
                    self.labelmap[datatype] = index_train
                    index_train += 1
                    for filename in os.listdir(datatype_path):
                        # load img
                        if filename.endswith('JPEG'):
                            img = Image.open(os.path.join(datatype_path, filename))
                            # img preprocessing
                            if self.gray:
                                img = img.convert('L')
                            if self.crop:
                                original_size = img.size
                                img = img.crop([48, 71, 128, 192])
                                img = img.resize(original_size)
                            if self.downsample_rate != 0:
                                img = img.resize([int(s * self.downsample_rate) for s in img.size])
                            # 将图像转换为张量
                            img_tensor = self.transform(img)
                            train_data.append(img_tensor)
                            train_label.append(datatype)
            # Load test data
            for datatype in os.listdir(os.path.join(self.datapath, 'val')):
                datatype_path = os.path.join(self.datapath, 'val', datatype)
                if os.path.isdir(datatype_path):
                    for filename in os.listdir(datatype_path):
                        # load img
                        if filename.endswith('JPEG'):
                            img = Image.open(os.path.join(datatype_path, filename))
                            # img preprocessing
                            if self.gray:
                                img = img.convert('L')
                            if self.crop:
                                original_size = img.size
                                img = img.crop([48, 71, 128, 192])
                                img = img.resize(original_size)
                            if self.downsample_rate != 0:
                                img = img.resize([int(s * self.downsample_rate) for s in img.size])
                            # 将图像转换为张量
                            img_tensor = self.transform(img)
                            test_data.append(img_tensor)
                            test_label.append(datatype)
        
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label


    def get_labels(self):    
        #获取identity
        if self.dataset == 'SVM':
            for index in range(len(self.train_label)):
                if self.labelmap[self.train_label[index]]<0:
                    print("out of bound")
            
                self.train_label[index] = int(self.labelmap[self.train_label[index]])
        elif self.dataset == 'CNN':
            for index in range(len(self.train_label)):
                if self.labelmap[self.train_label[index]]<0:
                    print("out of bound")
                one_hot = np.zeros(len(self.labelmap))
                one_hot[self.labelmap[self.train_label[index]]] = 1
                self.train_label[index] = torch.tensor(one_hot)

                #self.train_label = np.stack(self.train_label, axis=0)
        for index in range(len(self.test_label)):
            if self.labelmap[self.test_label[index]]<0:
                print("out of bound")
            
            self.test_label[index] = int(self.labelmap[self.test_label[index]])
            #self.test_label = np.stack(self.test_label, axis=0)

    def get_data(self):
        return self.train_data, self.train_label,self.test_data,self.test_label 






    
if __name__ == "__main__":

    ## test dataset
    # datatypes_select = ["n01514668","n01531178","n01534433","n01622779","n01664065","n01682714","n01694178","n01748264","n01774384","n01824575"]
    # labelmap= {"n01514668":0,"n01531178":1,"n01534433":2,"n01622779":3,"n01664065":4,"n01682714":5,"n01694178":6,"n01748264":7,"n01774384":8,"n01824575":9}
    # data = Dataset(r"C:\Users\liang\Desktop\imagenet_mini_with_feature\imagenet_mini",dataset='SVM',datatypes_select=datatypes_select,labelmap=labelmap)
    # data.load()
    # data.get_labels()
    # train_data, train_labels, test_data, test_labels = data.get_data()
    # train_data = np.array(train_data)
    # train_labels = np.array(train_labels)
    # test_data = np.array(test_data)
    # test_labels = np.array(test_labels)
    data = Dataset(r"imagenet_mini",
                   dataset='CNN',downsample_rate=0,gray=True)
    data.load()
    data.get_labels()
    train_data, train_labels, test_data, test_labels = data.get_data()
    # 将张量列表转换为一个整体张量
    train_data_tensor = torch.stack(train_data)
    train_labels_tensor = torch.stack(train_labels)

    # 对测试数据进行相同操作
    test_data_tensor = torch.stack(test_data)
    test_labels_tensor = torch.tensor(test_labels,dtype=torch.long)
    breakpoint()