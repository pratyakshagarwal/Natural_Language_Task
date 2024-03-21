import tensorflow as tf
from keras import Sequential
from keras.layers import LayerNormalization, Dropout, Embedding, Dense, Input, LSTM ,Bidirectional
from keras.models import Model
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.models import load_model

configuration = {
   'vocab_size': 283,
   'max_len': 57,
   'output_dim': 283,
   'emd_dim': 100,
   'dropout_rate': 0.0,
   'rnn_dim1': 150,
   'rnn_dim2': 150,
   'leanring_rate':0.01
}

class GRU_Model():
    def __init__(self, optimizer=SGD(learning_rate=configuration['leanring_rate']), loss_fn=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()]):
        # X_Input = Input(shape=[None])  # Change input shape
        # X_Embed = Embedding(input_dim=configuration['vocab_size'], output_dim=configuration['emd_dim'], input_length=configuration['max_len'])(X_Input)

        # X = Bidirectional(GRU(units=configuration['rnn_dim1'], kernel_initializer='he_uniform', activation='relu', recurrent_dropout=configuration['dropout_rate'], return_sequences=True))(X_Embed)  # Set return_sequences=True

        # X = LayerNormalization()(X)
        # # X = Bidirectional(GRU(units=configuration['rnn_dim2'], kernel_initializer='he_uniform', activation='relu', recurrent_dropout=configuration['dropout_rate']))(X)

        # # X = LayerNormalization()(X)
        # X_Out = Dense(configuration['output_dim'], activation='softmax', kernel_initializer='he_uniform')(X)  # Correct output_dim name

        # self.model = Model(inputs=X_Input, outputs=X_Out)        
        # self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metrics])

        self.model = Sequential()
        self.model.add(Embedding(configuration['vocab_size'], configuration['emd_dim']))
        self.model.add(LSTM(configuration['rnn_dim1'], return_sequences=True))
        self.model.add(LSTM(configuration['rnn_dim2']))
        self.model.add(Dense(configuration['output_dim'], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    def train(self,train_data=None, train_label=None, val_data=None, val_labels=None ,train_dataset=None, val_dataset=None, epochs=10, verbose=1, callbacks=[EarlyStopping()]):
        if train_dataset  is not None:
            if val_dataset is None:
                history = self.model.fit(train_dataset, validation_split=0.2, epochs=epochs, verbose=verbose, callbacks=callbacks)
            else:
                history = self.model.fit(train_dataset, validation_data=(val_dataset), epochs=epochs, verbose=verbose, callbacks=callbacks)

        else :
            if val_data is None:
                history = self.model.fit(train_data, train_label, epochs=epochs, verbose=verbose, callbacks=callbacks)
            else:
                history = self.model.fit(train_data, train_label, validation_data=(val_data, val_labels), epochs=epochs, verbose=verbose, callbacks=callbacks)

        return history


    def get_summary(self):
        return self.model.summary()

    def save_model(self, filename):
        self.model.save(filename)
        print('Model saved successfully')
   
    def evaluate(self, test_dataset=None, test_data=None, test_labels=None):
        if test_data is None:
            print(self.model.evaluate(test_dataset))
        else :
            print(self.model.evaluate(test_data, test_labels))
    
    def load_model(self, modelpath):
        self.model = load_model(modelpath)


if __name__ == '__main__':
    model = GRU_Model()
    print(model.get_summary())