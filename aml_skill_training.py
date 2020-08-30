'''
LSTM based text classification model.
Used GloVe vector as Embeddings.
Used azureml-sdk to train model in Azure Machine Learning
'''

import configparser
from azureml.core import Workspace, Datastore, Dataset, Run
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.layers.convolutional import Conv1D, MaxPooling1D
import pickle


np.random.seed(7)

# Read parameters from ini file
config = configparser.ConfigParser()
config.read('model_train.ini')

# Get Azure Machine Learning Run environment
run = Run.get_context()
workspace = run.experiment.workspace

def main():

    # Parameter loading
    trainin_data_path = config['TRAINING_PATH']['training_data_path']
    glove_vector = config['TRAINING_PATH']['glove_vector']
    skill_tokenizer = config['MODEL']['skill_tokenizer']
    skill_classifier = config['MODEL']['skill_classifier']
    test_size = config['PARAMETERS']['test_size']
    pad_len = config['PARAMETERS']['pad_len']
    dropout_size = config['PARAMETERS']['dropout_size']
    epochs = config['PARAMETERS']['epochs']
    max_nb_words = config['PARAMETERS']['max_nb_words']
    embedding_dim = config['PARAMETERS']['embedding_dim']

    # Read training dataset from AML registered dataset
    ds_training = trainin_data_path
    df = Dataset.get_by_name(workspace=workspace, name=ds_training)
    training_df = df.to_pandas_dataframe()

    skill_word = training_df['skill'].value
    label = training_df['label'].value

    # Split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        skill_word,
        label,
        stratify=label,
        test_size=float(test_size),
        random_state=123
    )

    # Tokenizing words
    tokenizer = Tokenizer(num_words=int(max_nb_words))
    tokenizer.fit_on_texts(skill_word)
    xtrain = tokenizer.texts_to_sequences(x_train)
    print("Train size: ", len(xtrain))
    xtest = tokenizer.texts_to_sequences((x_test))
    print("Test size: ", len(xtest))

    word_index = tokenizer.word_index
    vocabulary_size = len(word_index) + 1
    print("Vocabulary size: ", len(word_index))

    xtrain = pad_sequences(xtrain, padding='post', maxlen=int(pad_len))
    xtest = pad_sequences(xtest, padding='post', maxlen=int(pad_len))

    # Extract word embedding from GloVe vector. Used Stanford GloVe vector(https://nlp.stanford.edu/projects/glove/) glove.6B.100.txt
    embedding_index = dict()

    # Read GloVe vector from AML registered dataset
    glove = glove_vector
    glove_text = Dataset.get_by_name(workspace=workspace, name=glove)
    glove_df = glove_text.to_pandas_dataframe()
    glove_df.columns = ['word','col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16','col17','col18','col19','col20',
                        'col21', 'col22', 'col23', 'col24', 'col25', 'col26', 'col27', 'col28', 'col29', 'col30', 'col31', 'col32', 'col33', 'col34', 'col35', 'col36', 'col37', 'col38', 'col39', 'col40',
                        'col41', 'col42', 'col43', 'col44', 'col45', 'col46', 'col47', 'col48', 'col49', 'col50', 'col51', 'col52', 'col53', 'col54', 'col55', 'col56', 'col57', 'col58', 'col59', 'col60',
                        'col61', 'col62', 'col63', 'col64', 'col65', 'col66', 'col67', 'col68', 'col69', 'col70', 'col71', 'col72', 'col73', 'col74', 'co75', 'col76', 'col77', 'col78', 'col79', 'col80',
                        'col81', 'col82', 'col83', 'col84', 'col85', 'col86', 'col87', 'col88', 'col89', 'col90', 'col91', 'col92', 'col93', 'col94', 'co95', 'col96', 'col97', 'col98', 'col99', 'col100'
                        ]

    embedding_index = glove_df.set_index('word').T.to_dict('list')

    # Create weighted matrix
    embedding_matrix = np.zeros((vocabulary_size, int(embedding_dim)))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size:
            break
        else:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    # Model building
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocabulary_size, output_dim=int(embedding_dim),
                               input_length=int(pad_len),
                               weights=[embedding_matrix], trainable=False))

    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(layers.LSTM(units=50, return_sequences=True))
    model.add(layers.LSTM(units=10))
    model.add(layers.Dropout(float(dropout_size)))
    model.add(layers.Dense(8))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', matrices=['accuracy'])
    model.summary()

    # Model training
    model.fit(xtrain, y_train, epochs=int(epochs), batch_size=16)

    loss, acc = model.evaluate(xtrain, y_train)
    print("Training Accuracy: ", acc)

    # Add Training Accuracy matrices into AML Experiment workspace
    run.log("Training Accuracy: ", acc)

    loss, acc = model.evaluate(xtest, y_test)
    print("Test Accuracy: ", acc)

    # Add Test Accuracy matrices into AML Experiment workspace
    run.log("Test Accuracy: ", acc)

    ypred = model.predict(xtest)

    result = zip(x_test, y_test, ypred)
    for i in result:
        print(i)

    # Saving Tokenizer and Classification model in outputs dir in AML experiment workspace
    with open(skill_tokenizer, 'wb') as handle:
        pickle.dump(tokenizer, protocol=pickle.HIGHEST_PROTOCOL)

    model.save(skill_classifier)


if __name__ == '__main__':
    main()
