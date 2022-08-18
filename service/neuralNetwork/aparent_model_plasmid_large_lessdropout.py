from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate

from neuralNetwork.aparent_losses import *

def load_aparent_model_Plasmid(seq_length):
    #APARENT parameters
    seq_input_shape = (seq_length, 4, 1)
    lib_input_shape = (13,)
    distal_pas_shape = (1,)
    num_outputs_iso = 1

    
    #Shared model definition
    layer_1 = Conv2D(96, (8, 4), padding='valid', activation='relu')
    layer_1_pool = MaxPooling2D(pool_size=(2, 1))
    layer_2 = Conv2D(128, (6, 1), padding='valid', activation='relu')
    layer_dense = Dense(512, activation='relu')#(Concatenate()([Flatten()(layer_2), distal_pas_input]))
    layer_drop = Dropout(0.1)
    layer_dense2 = Dense(256, activation='relu')
    layer_drop2 = Dropout(0.1)

    def shared_model(seq_input, distal_pas_input) :
        return layer_drop2(
                    layer_dense2(
                        layer_drop(
                            layer_dense(
                                Concatenate()([
                                    Flatten()(
                                        layer_2(
                                            layer_1_pool(
                                                layer_1(
                                                    seq_input
                                                )
                                            )
                                        )
                                    ),
                                    distal_pas_input
                                ])
                            )
                        )
                    )
                )

    
    #Plasmid model definition

    #Inputs
    seq_input = Input(shape=seq_input_shape)
    lib_input = Input(shape=lib_input_shape)
    distal_pas_input = Input(shape=distal_pas_shape)

    plasmid_out_shared = Concatenate()([shared_model(seq_input, distal_pas_input), lib_input])

    plasmid_out_iso = Dense(num_outputs_iso, activation='sigmoid', kernel_initializer='zeros')(plasmid_out_shared)

    plasmid_model = Model(
        inputs=[
            seq_input,
            lib_input,
            distal_pas_input
        ],
        outputs=[
            plasmid_out_iso
        ]
    )


    return ('plasmid_iso_cut_distalpas_large_lessdropout', plasmid_model)

def load_aparent_model_Plasmid_withoutLibrary(seq_length):

    #APARENT parameters
    seq_input_shape = (seq_length, 4, 1)
    distal_pas_shape = (1,)
    num_outputs_iso = 1

    
    #Shared model definition
    layer_1 = Conv2D(96, (8, 4), padding='valid', activation='relu')
    layer_1_pool = MaxPooling2D(pool_size=(2, 1))
    layer_2 = Conv2D(128, (6, 1), padding='valid', activation='relu')
    layer_dense = Dense(512, activation='relu')#(Concatenate()([Flatten()(layer_2), distal_pas_input]))
    layer_drop = Dropout(0.1)
    layer_dense2 = Dense(256, activation='relu')
    layer_drop2 = Dropout(0.1)

    def shared_model(seq_input, distal_pas_input) :
        return layer_drop2(
                    layer_dense2(
                        layer_drop(
                            layer_dense(
                                Concatenate()([
                                    Flatten()(
                                        layer_2(
                                            layer_1_pool(
                                                layer_1(
                                                    seq_input
                                                )
                                            )
                                        )
                                    ),
                                    distal_pas_input
                                ])
                            )
                        )
                    )
                )

    
    #Plasmid model definition

    #Inputs
    seq_input = Input(shape=seq_input_shape)
    distal_pas_input = Input(shape=distal_pas_shape)

    plasmid_out_shared = shared_model(seq_input, distal_pas_input)

    plasmid_out_iso = Dense(num_outputs_iso, activation='sigmoid', kernel_initializer='zeros')(plasmid_out_shared)

    plasmid_model = Model(
        inputs=[
            seq_input,
            distal_pas_input
        ],
        outputs=[
            plasmid_out_iso
        ]
    )

    return ('plasmid_iso_cut_distalpas_large_lessdropout', plasmid_model)