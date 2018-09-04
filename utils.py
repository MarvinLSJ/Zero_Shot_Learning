from keras.preprocessing import image
import pandas as pd
import numpy as np
import pickle
import os

def load_data(save_dir):

    # """
    # Load Data
    # """
    # if os.path.exists(os.path.join(save_dir, 'data.pkl')):
    #     with open(os.path.join(save_dir, 'data.pkl'), 'rb') as data_file:
    #         data = pickle.load(data_file)
    #     return data
    #
    # """
    # Make Data
    # """

    train_root_path = "data/0813/DatasetA_train_20180813"
    test_root_path = "data/0813/DatasetA_test_20180813/DatasetA_test"

    """ Train """

    Y_all_train = pd.read_csv(os.path.join(train_root_path, 'train.txt'), sep='\t', header=None)
    Y_all_train.columns = ['filename', 'tag']

    Y_train = pd.read_csv(os.path.join(train_root_path, 'train_local.txt'), sep='\t', header=None)
    Y_train.columns = ['filename', 'tag']

    Y_valid = pd.read_csv(os.path.join(train_root_path, 'dev_local.txt'), sep='\t', header=None)
    Y_valid.columns = ['filename', 'tag']

    rootpath = os.path.join(train_root_path, "train")  # 文件夹目录
    X_train = []
    X_valid = []

    # total_train = []
    # for filename in Y_all_train['filename']:  # 遍历文件夹
    #     pic = image.load_img(rootpath + "/" + filename, target_size=(100, 100))
    #     pic = image.img_to_array(pic)
    #     total_train.append(pic)
    # total_train = np.array(total_train)
    # print('Total train set has {} color images'.format(total_train.shape[0]))


    for filename in Y_train['filename']:  # 遍历文件夹
        pic = image.load_img(rootpath + "/" + filename, target_size=(100, 100))
        pic = image.img_to_array(pic)
        X_train.append(pic)

    for filename in Y_valid['filename']:  # 遍历文件夹
        pic = image.load_img(rootpath + "/" + filename, target_size=(100, 100))
        pic = image.img_to_array(pic)
        X_valid.append(pic)

    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    print('Train set has {} color images'.format(X_train.shape[0]))
    print('Validation set has {} color images.'.format(X_valid.shape[0]))

    print('Image shape:', X_train[0].shape)

    """ Test """

    test_files = pd.read_csv(os.path.join(test_root_path, 'image.txt'), sep='\t', header=None)
    test_files.columns = ['filename']

    rootpath = os.path.join(test_root_path, "test")  # 文件夹目录

    X_test = []

    for filename in test_files['filename']:  # 遍历文件夹
        pic = image.load_img(rootpath + "/" + filename, target_size=(100, 100))
        pic = image.img_to_array(pic)
        X_test.append(pic)

    X_test = np.array(X_test)
    print('Test set has {} color images'.format(X_test.shape[0]))
    print('Image shape:', X_test[0].shape)


    """ Attributes """

    tag2attr = pd.read_csv(os.path.join(train_root_path, 'attributes_per_class.txt'), sep='\t', header=None)
    attr2name = pd.read_csv(os.path.join(train_root_path, 'attribute_list.txt'), sep='\t', header=None)
    tag2label = pd.read_csv(os.path.join(train_root_path, 'label_list.txt'), sep='\t', header=None)
    tag2label.columns = ['tag', 'label']
    attr2name = attr2name.drop([0], axis=1)
    attr2name.columns = ['name']
    col_name = ['tag'] + attr2name['name'].tolist()
    tag2attr.columns = col_name

    """ Inspect Tags """
    train_tags = Y_all_train['tag'].unique()
    total_tags = tag2attr['tag'].unique()
    zero_shot_tags = list(set(total_tags) - set(train_tags))
    print("{} unique tags in total (Including Test Data). \n{} unique tags in Training Data. \n{} Zero-shot tags.".format(
            len(total_tags), len(train_tags), len(zero_shot_tags)))

    # """ Associate tags and attributes """
    #
    # # total_train_attr = img_label_attr(Y_all_train, tag2label, tag2attr, attr2name)
    # Y_train_attr = img_label_attr(Y_train, tag2label, tag2attr, attr2name)
    # Y_valid_attr = img_label_attr(Y_valid, tag2label, tag2attr, attr2name)


    """ Word Embedding """

    word_dict = {}
    embed_path = os.path.join(train_root_path, 'class_wordembeddings.txt')
    with open(embed_path, 'r') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word_dict[word] = np.array(list(map(float, vec.split())))


    data = {}

    data['X_train'] = X_train
    data['Y_train'] = Y_train
    data['X_valid'] = X_valid
    data['Y_valid'] = Y_valid
    data['X_test'] = X_test
    data['test_files'] = test_files

    data['tag2attr'] = tag2attr
    data['tag2label'] = tag2label
    data['total_tags'] = total_tags
    data['zero_shot_tags'] = zero_shot_tags

    data['word_dict'] = word_dict

    print("Data Loaded!")

    # data['total_attr'] = total_train_attr
    # data['train_attr'] = Y_train_attr
    # data['valid_attr'] = Y_valid_attr

    # """ Save Data for second use """
    #
    # with open(os.path.join(save_dir, 'data.pkl'), 'wb') as data_file:
    #     pickle.dump(data, data_file, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # print("Data File Saved to {}".format(os.path.join(save_dir, 'data.pkl')))

    return data


def img_label_attr(df, tag2label, tag2attr, attr2name):
    Y_cate = []
    for tag in df['tag']:
        arr = [tag2label[tag2label['tag'] == tag].iloc[0, :].tolist()[1]] \
                + tag2attr[tag2attr['tag'] == tag].iloc[0, :].tolist()
        Y_cate.append(arr)

    train_attr = pd.DataFrame(Y_cate)
    col_name = ['label'] + ['tag'] + attr2name['name'].tolist()
    train_attr.columns = col_name
    print('Converted {} image tags into categories'.format(train_attr.shape[0]))
    return train_attr

def cos_dis(v1, v2):
    return np.linalg.norm(v1-v2)