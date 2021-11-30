import tensorflow as tf

def get_model(args):
    # Stacked LSTMs
    seq = []
    seq.append(
        tf.keras.layers.LSTM(
            128,
            recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0.01),
            return_sequences=True,
            activation='tanh',
            input_shape=(args.past_history,4))
        )
    for i in range(args.layers-2):
        seq.append(
            tf.keras.layers.LSTM(
                128,
                recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0.01),
                return_sequences=True,
                activation='tanh')
            )
    
    seq.append(
        tf.keras.layers.LSTM(
            128,
            recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0.01),
            return_sequences=False,
            activation='tanh')
        )
    
    seq.append(tf.keras.layers.Dense(3))
    
    model = tf.keras.models.Sequential(seq)
    model.compile(optimizer=tf.keras.optimizers.Adam(
                                 learning_rate=args.initial_lr, beta_1=0.9, beta_2=0.999,
                                 epsilon=1e-07, amsgrad=False, name='Adam'
                                 ),
                             loss = "mse"
                             )
    return model