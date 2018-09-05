import torch
from keras.layers import concatenate
from keras.applications.xception import Xception
from keras.layers import Input, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model

from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #         self.height = img.shape[1]
        #         self.width = img.shape[2]
        #         self.pool_size = pool_size

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        # 50*50

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # 25*25

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # 12*12

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # 6*6

        # conv output shape:  out_channels * height/pool_size * width/pool_size
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(512 * 6 * 6, 5000),
            torch.nn.ReLU(),
            torch.nn.Linear(5000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 1)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        #         print('conv1 output shape:', conv1_out.shape)
        conv2_out = self.conv2(conv1_out)
        #         print('conv2 output shape:', conv2_out.shape)
        conv3_out = self.conv3(conv2_out)
        #         print('conv3 output shape:', conv3_out.shape)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        conv7_out = self.conv7(conv6_out)
        res = conv7_out.view(conv7_out.size(0), -1)

        # output sigmoid to [0,1]
        out = torch.sigmoid(self.dense(res))
        #         # then round to first decimal digit to be the prediction
        #         out = torchround(out)

        return out


def Combine_Categorical_Model():
    """
    Image Feature Extraction
    """

    # Pre-trained Xception net
    base_model = Xception(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Out to match semantic size
    img_out = Dense(300, activation='relu')(x)

    """
    Semantic Feature Extraction
    """
    word_input = Input(shape=(300,), name='word_input', dtype='float32')

    """
    Combine
    """
    x = concatenate([img_out, word_input])

    """
    Categorical Prediction
    """
    prediction = Dense(230, activation='softmax')(x)

    model = Model(inputs=[base_model.input, word_input],
                  outputs=prediction)

    return model


def img_feature_model():
    """
    Image Feature Extraction
    """

    # Pre-trained Xception net
    base_model = Xception(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = GlobalMaxPooling2D()(x)
    # Out to match semantic size
    #x = Dense(1024, activation='relu')(x)
    #x = Dense(512, activation='relu')(x)
    img_out = Dense(300)(x)

    model = Model(inputs=base_model.input,
                  outputs=img_out)

    return model
