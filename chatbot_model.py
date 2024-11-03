import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Path to the file
PATH_TO_FILE = 'human_chat.txt'

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self.load_data()
        self.vocab = sorted(set(self.text))
        self.char2idx, self.idx2char = self.create_lookup_tables()

    def load_data(self):
        # Read binary file and decode using utf-8
        with open(self.file_path, 'rb') as f:
            text = f.read().decode(encoding='utf-8')
        # Remove label of annotators
        text = re.sub(r'Human: ', '', text)
        print("The length of characters in the text of the database:", len(text))
        return text

    def create_lookup_tables(self):
        # Create character to index and index to character mappings
        char2idx = {u: i for i, u in enumerate(self.vocab)}
        idx2char = np.array(self.vocab)
        return char2idx, idx2char

    def text_to_int(self):
        # Convert the text to integers
        return np.array([self.char2idx[c] for c in self.text])

class TextDataset:
    def __init__(self, text_as_int, seq_length=100, batch_size=64, buffer_size=10000):
        self.text_as_int = text_as_int
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dataset = self.create_dataset()

    def create_dataset(self):
        char_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)
        sequences = char_dataset.batch(self.seq_length + 1, drop_remainder=True)

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)
        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)
        return dataset

class RNNModel:
    def __init__(self, vocab_size, embedding_dim=256, rnn_units=1024, batch_size=64):
        self.model = self.build_model(vocab_size, embedding_dim, rnn_units, batch_size)

    def build_model(self, vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=(None,)),  # Modified line
            tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model


    def compile_model(self):
        # Loss function
        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        self.model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    def fit(self, dataset, epochs=50, checkpoint_dir='./training_checkpoints'):
        # Set up checkpointing
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

        history = self.model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
        return history

    def evaluate(self, dataset):
        results = self.model.evaluate(dataset)
        print("Evaluation results: Loss: {}, Accuracy: {}".format(results[0], results[1]))
        
    def save_model(self, file_name):
        self.model.save(file_name)


def main():
    # Data Processing
    data_processor = DataProcessor(PATH_TO_FILE)
    text_as_int = data_processor.text_to_int()

    # Create Dataset
    text_dataset = TextDataset(text_as_int)
    
    # Model Building
    model = RNNModel(vocab_size=len(data_processor.vocab))
    model.compile_model()

    # Fit the model
    history = model.fit(text_dataset.dataset)

    # Evaluate the model
    model.evaluate(text_dataset.dataset)
    
    # Save the model 
    model.save_model("human_chat_model.h5")
    print("Model saved as human_chat_model.h5")

if __name__ == "__main__":
    main()
