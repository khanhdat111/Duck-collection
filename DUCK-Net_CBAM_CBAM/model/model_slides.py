from keras.layers import Conv2D, UpSampling2D , Conv2DTranspose
from keras.models import Model
from keras.layers import add
from Layers.conv_block2d import conv_block_2D
from Layers.cbam import cbam_block
import tensorflow as tf

kernel_initializer = 'he_uniform'
interpolation = "nearest"



def create_model(img_height, img_width, input_chanels, out_classes, starting_filters):
    input_layer = tf.keras.layers.Input((img_height, img_width, input_chanels))

    print('Starting DUCK-Net')

    p1 = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(input_layer)
    p1cb = cbam_block(p1)
    p2 = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(p1)
    p2cb = cbam_block(p2)
    p3 = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(p2)
    p3cb = cbam_block(p3)
    p4 = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(p3)
    p4cb = cbam_block(p4)
    p5 = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(p4)
    p5cb = cbam_block(p5)

    t0 = conv_block_2D(input_layer, starting_filters , 'double_convolution', repeat=1)
    t0cb = cbam_block(t0)

    l1i = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(t0cb)
    l1icb = cbam_block(l1i)
    s1 = add([l1icb, p1cb])
    t1 = conv_block_2D(s1, starting_filters * 2, 'double_convolution', repeat=1)

    l2i = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(t1)
    l2icb = cbam_block(l2i)
    s2 = add([l2icb, p2cb])
    t2 = conv_block_2D(s2, starting_filters * 4, 'double_convolution', repeat=1)

    l3i = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(t2)
    l3icb = cbam_block(l3i)
    s3 = add([l3icb, p3cb])
    t3 = conv_block_2D(s3, starting_filters * 8, 'double_convolution', repeat=1)

    l4i = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(t3)
    l4icb = cbam_block(l4i)
    s4 = add([l4icb, p4cb])
    t4 = conv_block_2D(s4, starting_filters * 16, 'double_convolution', repeat=1)

    l5i = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(t4)
    s5 = add([l5i, p5cb])
    t51 = conv_block_2D(s5, starting_filters * 32, 'resnet', repeat=2)
    t53 = conv_block_2D(t51, starting_filters * 16, 'resnet', repeat=2)

    #----------------------------------------------------------------------------------#

    t53 = cbam_block(t53)
    t53cb = Conv2DTranspose( starting_filters * 8, 2, strides=2, padding='same')(t53)
    
    t54 = cbam_block(t53cb)
    t54cb = Conv2DTranspose( starting_filters * 4, 2, strides=2, padding='same')(t54)
    
    t55 = cbam_block(t54cb)
    t55cb =Conv2DTranspose( starting_filters * 2, 2, strides=2, padding='same')(t55)

    t56 = cbam_block(t55cb)
    t56cb =  Conv2DTranspose( starting_filters , 2, strides=2, padding='same')(t56)
    
    #----------------------------------------------------------------------------------#

    l5o = UpSampling2D((2, 2), interpolation=interpolation)(t53)
    c4 = add([l5o, l4icb])
    q4 = conv_block_2D(c4, starting_filters * 8, 'double_convolution', repeat=1)
    
    q4 = cbam_block(q4)
    v4 = add([q4,t53cb])
    
    l4o = UpSampling2D((2, 2), interpolation=interpolation)(v4)
    c3 = add([l4o, l3icb])
    q3 = conv_block_2D(c3, starting_filters * 4, 'double_convolution', repeat=1)

    q3 = cbam_block(q3)
    v3 = add([q3, t54cb])

    l3o = UpSampling2D((2, 2), interpolation=interpolation)(v3)
    c2 = add([l3o, l2icb])
    q6 = conv_block_2D(c2, starting_filters * 2, 'double_convolution', repeat=1)
    #print(q6.shape)
    
    q6 = cbam_block(q6)
    v3 = add([q6,t55cb])
    
    l2o = UpSampling2D((2, 2), interpolation=interpolation)(v3)
    c1 = add([l2o, l1icb])
    q1 = conv_block_2D(c1, starting_filters, 'double_convolution', repeat=1)
    #print(q1.shape)

    q1 = cbam_block(q1)
    v2 = add([q1 ,t56cb])

    l1o = UpSampling2D((2, 2), interpolation=interpolation)(v2)
    c0 = add([l1o, t0cb])
    z1 = conv_block_2D(c0, starting_filters, 'double_convolution', repeat=1)
    #print(z1.shape)

    output = Conv2D(out_classes, (1, 1), activation='sigmoid')(z1)

    model = Model(inputs=input_layer, outputs=output)

    return model
