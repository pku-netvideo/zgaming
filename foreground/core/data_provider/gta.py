__author__ = 'gaozhifeng, jkw'
import numpy as np
import os
import cv2
from PIL import Image
import logging
import random
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class InputHandle(Dataset):
    def __init__(self, datas, indices, input_param):
        tmp_indices = []
        frames_id = []
        for tmp in indices:
            tmp = tmp.split('_')
            tmp_indices.append(int(tmp[0]))
            seq_name = tmp[3].split('.')[0]
            frames_id.append('{}_{}'.format(seq_name, tmp[2]))
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.stride = input_param['stride']
        self.indices = tmp_indices
        self.frames_id = frames_id
        self.current_position = 0
        self.current_batch_indices = []
        self.current_batch_frames_id = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]
        self.current_batch_frames_id = self.frames_id[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]
        self.current_batch_frames_id = self.frames_id[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, 3)).astype(
            self.input_data_type)
        frames_id_batch = []
        index_stride = (self.current_input_length - 1) * self.stride
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + index_stride + 1
            frames_path = self.datas[begin:end:self.stride]
            data_slice = []
            for frame_path in frames_path:
                frame_im = Image.open(frame_path)
                frame_np = np.array(frame_im)/255  # (1000, 1000) numpy array
                frame_np = frame_np[:, :, :] #
                frame_np = cv2.resize(frame_np, [self.image_width, self.image_width])
                data_slice.append(frame_np)
            data_slice = np.array(data_slice)
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            frames_id_batch.append(self.current_batch_frames_id[i])

        input_batch = input_batch.astype(self.input_data_type)
        return [input_batch,frames_id_batch]

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))

    def __len__(self):
        return self.minibatch_size * (self.total() // self.minibatch_size)

class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.category_1 = ['train_1024']
        #self.category_1 = ['train_256']
        # self.category_1 = []
        # self.category_2 = ['test']
        self.category_2 = ['test_1024']
        # self.category_2 = ['qp42_short']
        # self.category_2 = ['jogging']
        self.category = self.category_1 + self.category_2
        self.image_width = input_param['image_width']

        self.train_person = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21']
        #self.train_person = ['21']
        self.test_person = ['00']
        #self.test_person = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']

        self.input_param = input_param
        self.seq_len = input_param['seq_length']
        self.stride = input_param['stride']
        self.test_step = input_param['test_step']

    def load_data(self, paths, mode='train'):
        '''
        frame -- action -- person_seq(a dir)
        :param paths: action_path list
        :return:
        '''

        path = paths[0]
        if mode == 'train':
            person_id = self.train_person
        elif mode == 'test':
            person_id = self.test_person
        else:
            print("ERROR!")
        print('begin load data' + str(path))

        frames_np = []
        frames_file_name = []
        frames_person_mark = []
        frames_category = []
        person_mark = 0

        c_dir_list = self.category
        frame_category_flag = -1
        for c_dir in c_dir_list: # handwaving
            if c_dir in self.category_1:
                frame_category_flag = 1 # 20 step
            elif c_dir in self.category_2:
                frame_category_flag = 2 # 3 step
            else:
                print("category error!!!")

            c_dir_path = os.path.join(path, c_dir)
            p_c_dir_list = os.listdir(c_dir_path)

            for p_c_dir in p_c_dir_list:
                if p_c_dir[6:8] not in person_id:
                    continue
                person_mark += 1
                dir_path = os.path.join(c_dir_path, p_c_dir)
                filelist = os.listdir(dir_path)
                filelist.sort()
                filelist_len = len(filelist)
                if filelist_len >= (filelist_len // 10) * 10 + 1:
                    filelist_len = (filelist_len // 10) * 10 + 1
                else:
                    filelist_len = (filelist_len // 10 - 1) * 10 + 1
                filelist = filelist[0 : filelist_len]
                for file in filelist:
                    if file.startswith('image') == False:
                        continue
                    # print(file)
                    # print(os.path.join(dir_path, file))
                    '''
                    frame_im = Image.open(os.path.join(dir_path, file))
                    frame_np = np.array(frame_im)  # (1000, 1000) numpy array
                    # print(frame_np.shape)
                    frame_np = frame_np[:, :, :] #
                    frames_np.append(frame_np)
                    '''
                    frames_np.append(os.path.join(dir_path, file))

                    seq_name = p_c_dir.split('_')[3]
                    file = file.replace('.','_{}.'.format(seq_name))
                    frames_file_name.append(file)
                    frames_person_mark.append(person_mark)
                    frames_category.append(frame_category_flag)
        # is it a begin index of sequence
        indices = []
        for step in range(self.stride):
            '''
            if len(indices) > 600:
                break
            '''
            index = len(frames_person_mark) - 1 - step
            if frames_category[index] == 2 and step > 0:
                break
            index_stride = (self.seq_len - 1) * self.stride
            while index >= index_stride:
                if frames_person_mark[index] == frames_person_mark[index - index_stride]:
                    end = int(frames_file_name[index][6:10])
                    start = int(frames_file_name[index - index_stride][6:10])
                    # TODO: mode == 'test'
                    if end - start == index_stride:
                        indice = index - index_stride
                        indice = '{}_{}'.format(indice, frames_file_name[indice])
                        indices.append(indice)
                        '''
                        if len(indices) > 600:
                            break
                        '''
                        if frames_category[index] == 1:
                            index -= index_stride
                        elif frames_category[index] == 2:
                            index -= self.test_step - 1
                        else:
                            print("category error 2 !!!")
                index -= 1
        '''
        frames_np = np.asarray(frames_np)
        data = np.zeros((frames_np.shape[0], self.image_width, self.image_width , 3), dtype=np.float32)
        for i in range(len(frames_np)):
            temp = np.float32(frames_np[i, :, :])
            data[i,:,:,:]=cv2.resize(temp,(self.image_width,self.image_width))/255
        '''
        data = frames_np
        #print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(data)) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)

