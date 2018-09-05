import numpy as np
import pandas as pd
from keras.models import load_model


from utils import load_data, cos_dis
from models import Combine_Categorical_Model, img_feature_model

inference = False

if not inference:

    """
    Read Data
    """
    data = load_data("data/0813")

    X_train = data['X_train']
    X_valid = data['X_valid']
    Y_train = data['Y_train']
    Y_valid = data['Y_valid']
    tag2label = data['tag2label']
    word_dict = data['word_dict']


    """
    Pre-process
    """

# get label
    train_word_input = [tag2label[tag2label['tag'] == x]['label'].tolist()[0] for x in Y_train['tag']]
    valid_word_input = [tag2label[tag2label['tag'] == x]['label'].tolist()[0] for x in Y_valid['tag']]

# Embedding Labels
    train_word_input = [word_dict[w] for w in train_word_input]
    valid_word_input = [word_dict[w] for w in valid_word_input]


    """
    Model Train
    """
# Y_train_list = list(map(lambda x: tag2int[x], Y_train['tag'].tolist()))
# Y_valid_list = list(map(lambda x: tag2int[x], Y_valid['tag'].tolist()))
#
# # one-hot encode tag types for loss calculation
# y_train = to_categorical(Y_train_list, num_classes=230)
# y_valid = to_categorical(Y_valid_list, num_classes=230)
#
# ccm = Combine_Categorical_Model()
# ccm.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# history = ccm.fit([X_train, np.array(train_word_input)], y_train,
#         validation_data=([X_valid, np.array(valid_word_input)], y_valid), epochs=20, verbose=1)

    img2sematic = img_feature_model()
    img2sematic.compile(optimizer='rmsprop', loss='MSE')
    history = img2sematic.fit(X_train, np.array(train_word_input),
            validation_data=(X_valid, np.array(valid_word_input)), epochs=20, verbose=1, batch_size=32)

    img2sematic.save('img2sematic.h5')


    keys = []
    values = []

    for key, value in word_dict.items():
        keys.append(key)
        values.append(value)

# Compare with every label, find the one with nearest cosine distance

# Valid

    Pred_valid = img2sematic.predict(X_valid)

    res_valid = []
    for v1 in Pred_valid:
        score = []
        for v2 in values:
            score.append(cos_dis(v1, v2))
        label = keys[score.index(min(score))]
        res_valid.append(tag2label[tag2label['label'] == label]['tag'].tolist())

    pred_valid = [s[0] for s in res_valid]
    ground_truth = Y_valid['tag'].tolist()
    count = .0
    for i in range(len(ground_truth)):
        if ground_truth[i] == pred_valid[i]:
            count += 1
    print("Valid Accuracy:", round(count/len(ground_truth), 4))

else:
# Test Result
    print("Start Inference...")
    # Read Data
    data = load_data("data/0813")
    X_test = data['X_test']
    test_files = data['test_files']
    word_dict = data['word_dict']
    tag2label = data['tag2label']
    # Predict
    model = load_model('img2sematic.h5')
    Y_test = model.predict(X_test)

    keys = []
    values = []

    for key, value in word_dict.items():
        keys.append(key)
        values.append(value)

    res = []

    for v1 in Y_test:
        score = []
        for v2 in values:
            score.append(cos_dis(v1, v2))

        label = keys[score.index(min(score))]
        res.append(tag2label[tag2label['label'] == label]['tag'].tolist())

    result = pd.concat([test_files, pd.DataFrame(res)], axis=1)
    result.to_csv('submit.txt', sep='\t', header=None, index=None)


    """
    Result Analysis
    """

    zero_shot_tags = data['zero_shot_tags']
    count = 0
    for s in res:
        if s[0] in zero_shot_tags:
            count+=1
    print('Prediction contains {} zero-shot tags'.format(count))
