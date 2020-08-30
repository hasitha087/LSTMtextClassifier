import configparser

config = configparser.ConfigParser()

config['TRAINING_PATH'] = {
    'training_data_path': 'skill-training',
    'glove_vector': 'glove_vector'
}

config['MODEL'] = {
    'skill_tokenizer': 'outputs/skill_tokenizer.pkl',
    'skill_classifier': 'outputs/skill_classifier'
}

config['PARAMETERS'] = {
    'test_size': '0.1',
    'pad_len': '20',
    'max_nb_words': '1000',
    'embedding_dim': '100',
    'droupout_size': '0.2',
    'epochs': '20'
}


with open('model_train.ini', 'w') as configfile:
    config.write(configfile)