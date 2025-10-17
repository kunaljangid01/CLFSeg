import tensorflow as tf
from keras.layers import BatchNormalizationV2, Add, Concatenate
from keras.layers import Conv2D


kernel_initializer = 'he_uniform'

def conv_block_2D(x, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
    result = x

    for i in range(0, repeat):

        if block_type == 'separated':
            result = separated_conv2D_block(result, filters, size=size, padding=padding)
        elif block_type == 'duckv2':
            result = duckv2_conv2D_block(result, filters, size=size)
        elif block_type == 'midscope':
            result = midscope_conv2D_block(result, filters)
        elif block_type == 'widescope':
            result = widescope_conv2D_block(result, filters)
        elif block_type == 'resnet':
            result = resnet_conv2D_block(result, filters, dilation_rate)
        elif block_type == 'conv':
            result = Conv2D(filters, (size, size),
                            activation='relu', kernel_initializer=kernel_initializer, padding=padding)(result)
        elif block_type == 'double_convolution':
            result = double_convolution_with_batch_normalization(result, filters, dilation_rate)

        else:
            return None

    return result


class FAG(tf.keras.layers.Layer):
    def __init__(self, B, H, W, channels, num_fuzzy=20, filters=None):
        super(FAG, self).__init__()
        self.channels = channels
        self.num_fuzzy = num_fuzzy
        self.filters = filters
    
        self.leakyrelu = tf.keras.layers.LeakyReLU()
        self.norm = tf.keras.layers.LayerNormalization()
        self.relu = tf.keras.layers.ReLU()

        trainable = True

        # self.we = tf.Variable(initial_value=tf.random.normal(shape=(H, W, channels)), trainable=trainable)
        # self.be = tf.Variable(initial_value=tf.zeros(shape=(H, W, channels)), trainable=trainable)

        self.wd = tf.Variable(initial_value=tf.random.normal(shape=(H, W, channels)), trainable=trainable)
        self.bd = tf.Variable(initial_value=tf.zeros(shape=(H, W, channels)), trainable=trainable)

        self.mu = tf.Variable(initial_value=tf.random.normal(shape=(self.num_fuzzy, 1)), trainable=trainable)
        self.sig = tf.Variable(initial_value=tf.random.normal(shape=(self.num_fuzzy, 1)), trainable=trainable)

        self.out = tf.keras.layers.Conv2D(
            self.filters, 
            kernel_size=1, 
            padding="same",
            activation=self.relu 
        )

    def call(self, d):
        # x1 = self.leakyrelu(self.leakyrelu(self.norm((e * self.we) + self.be)))
        x2 = self.leakyrelu(self.norm((d * self.wd) + self.bd))
        x = x2
        
        x_expanded = tf.expand_dims(x, axis=-2)
        mu_expanded = tf.reshape(self.mu, [1, 1, 1, self.num_fuzzy, 1])
        sig_expanded = tf.reshape(self.sig, [1, 1, 1, self.num_fuzzy, 1])
        
        fuzzy_sets = tf.exp(-tf.square(x_expanded - mu_expanded) / (2 * tf.square(sig_expanded)))
        
        fuzzy_sets_mean = tf.reduce_sum(fuzzy_sets, axis=-2)

        fuzzy_sets_mean = self.out(fuzzy_sets_mean)
        
        return fuzzy_sets_mean


class DWConv(tf.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(3, padding="same")
        self.pointwise = tf.keras.layers.Conv2D(dim, 1, padding="same")

    def __call__(self, x, H, W):
        _, N, C = x.shape
        x = tf.reshape(x, (-1, H, W, C))
        x = self.pointwise(self.dwconv(x))
        x = tf.reshape(x, (-1, N, x.shape[-1]))
        return x
        
class ConvolutionalGLU(tf.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features=None, 
            out_features=None, 
            act_layer=tf.keras.activations.gelu, 
            drop=0.):
        super().__init__()
        self.out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = tf.keras.layers.Dense(hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer
        self.fc2 = tf.keras.layers.Dense(self.out_features)
        self.drop = tf.keras.layers.Dropout(drop)

    def __call__(self, x, H, W):
        x = tf.reshape(x, (-1, H*W, x.shape[-1]))
        x, v = tf.split(self.fc1(x), 2, axis=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = tf.reshape(x, (-1, H, W, self.out_features))
        return x



def duckv2_conv2D_block(x, filters, size):
    x = BatchNormalizationV2(axis=-1)(x)
    x1 = widescope_conv2D_block(x, filters)

    x2 = midscope_conv2D_block(x, filters)

    x3 = conv_block_2D(x, filters, 'resnet', repeat=1)

    x4 = separated_conv2D_block(x, filters, size=6, padding='same')
    
    B, H, W, C = x.shape

    if C > 3:  
        B, H, W, C = x.shape
        fuzz = FAG(B, H, W, C, 20, filters=filters)(x)

        glu = ConvolutionalGLU(in_features=C, out_features=filters)
        x5 = glu(fuzz, H, W)

        x = Add()([x1, x2, x3, x4, x5])

    else:
        x = Add()([x1, x2, x3, x4])
    
    x = BatchNormalizationV2(axis=-1)(x)

    return x


def separated_conv2D_block(x, filters, size=3, padding='same'):
    x = Conv2D(filters, (1, size), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (size, 1), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    return x


def midscope_conv2D_block(x, filters):
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=1)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=2)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    return x


def widescope_conv2D_block(x, filters):
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=1)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=2)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=3)(x)

    x = BatchNormalizationV2(axis=-1)(x)

    return x


def resnet_conv2D_block(x, filters, dilation_rate=1):
    x1 = Conv2D(filters, (1, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same',
                dilation_rate=dilation_rate)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalizationV2(axis=-1)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalizationV2(axis=-1)(x)
    x_final = Add()([x, x1])

    x_final = BatchNormalizationV2(axis=-1)(x_final)

    return x_final


def double_convolution_with_batch_normalization(x, filters, dilation_rate=1):
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalizationV2(axis=-1)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalizationV2(axis=-1)(x)

    return x