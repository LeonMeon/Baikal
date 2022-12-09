import tensorflow as tf
import numpy as np
import h5py as h5
import random as rd

log_dir = "/home/ivkhar/Baikal/logs/graph_like/"
#log_dir = "/home3/ivkhar/Baikal/logs/graph_like/"

max_len = 110
batch_size = 32

h5f = '/home/ivkhar/Baikal/data/mc_baikal_norm_cut-0_ordered_equal_big.h5'
#h5f = '/home3/ivkhar/Baikal/data/mc_baikal_norm_cut-0_ordered_equal_big.h5'

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


### encode nodes, will be used to prepare messages
# returns transformed encoding plus initial
# pass empty filters to avoid new encs
# in case k!=1 neighbours effect the final encoding 
class NodesEncoder(tf.keras.layers.Layer):
    
    # filters, kernels - sequances
    # kernels should be 1, as the initial idea   
    def __init__(self, filters, kernels, activation):
        super(NodesEncoder, self).__init__()
        assert len(filters)==len(kernels)
        self.num_layers = len(filters)
        self.conv_layers = [ tf.keras.layers.Conv1D(f, k, padding='same') for (f,k) in zip(filters,kernels) ]
        self.activations = [ activation() for f in filters ]
        self.bn_layers = [ tf.keras.layers.BatchNormalization() for f in filters ]
        self.filters = filters
        
    def build(self, input_shape):
        if self.num_layers==0:
            self.out_encs_length = input_shape[-1]
        else:
            self.out_encs_length = self.filters[-1]+input_shape[-1]

    def call(self, x, training=False):
        init_x = x
        for (conv,bn,ac) in zip(self.conv_layers,self.bn_layers,self.activations):
            x = conv(x)
            x = bn(x, training=training)
            x = ac(x)
            x = tf.concat((x,init_x),axis=-1)
        return x

### message passer
# conv-based
class MessagePasser(tf.keras.layers.Layer):
  
    # filters, kernels, strides - sequances
    def __init__(self, filters, kernels, strides, units, activation):
        super(MessagePasser, self).__init__()
        assert len(filters)==len(kernels)
        assert len(filters)==len(strides)
        self.num_layers = len(filters)
        self.conv_layers = [ tf.keras.layers.Conv1D(f, k, strides=s, padding='same') for (f,k,s) in zip(filters,kernels,strides) ]
        self.dence_layer = tf.keras.layers.Dense(units)
        self.activations = [ activation() for f in filters ]
        self.last_act = activation()
        self.bn_layers = [ tf.keras.layers.BatchNormalization() for f in filters ]
        self.max_poolings = [  tf.keras.layers.MaxPooling1D() for f in filters ]
        self.flattener = tf.keras.layers.Flatten()

    def call(self, x, training=False):
        for (conv,bn,mp,ac) in zip(self.conv_layers,self.bn_layers,self.max_poolings,self.activations):
            x = conv(x)
            x = bn(x, training=training)
            x = ac(x)
            x = mp(x)
        x = self.flattener(x)
        x = self.dence_layer(x)
        x = self.last_act(x)
        return x

# usual, dense based
class MessagePasserDense(tf.keras.layers.Layer):
  
    # filters, kernels, strides - sequances
    def __init__(self, filters, kernels, strides, units, activation):
        super(MessagePasserDense, self).__init__()
        self.dence_layer = tf.keras.layers.Dense(units)
        self.activation = activation
        self.bn_layer = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        # take mean over OMs
        x = tf.math.reduce_mean(x, axis=1)
        # normilize
        x = self.bn_layer(x)
        # transform
        #x = self.dence_layer(x)
        #x = self.activation(x)
        return x

### attention calculator
# takes pairs of channels info and calculates relevance
class AttentionEstablisher(tf.keras.layers.Layer):
  
    # units - sequance
    def __init__(self, hiden_units, activation=None):
        super(AttentionEstablisher, self).__init__()
        self.num_layers = len(hiden_units)
        self.sigm_layer = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
        self.dense_layers = [ tf.keras.layers.Dense(u) for u in hiden_units ]
        self.activations = [ activation() for u in hiden_units ]
        self.bn_layers = [ tf.keras.layers.BatchNormalization() for u in hiden_units ]

    def call(self, x, training=False):
        for (dense,bn,ac) in zip(self.dense_layers,self.bn_layers,self.activations):
            x = dense(x)
            x = bn(x, training=training)
            x = ac(x)
        # last - sigmoid
        x = self.sigm_layer(x)
        return x

### state updater
class StateUpdater(tf.keras.layers.Layer):
  
    # units - sequance
    def __init__(self, units, activation):
        super(StateUpdater, self).__init__()
        self.num_layers = len(units)
        self.dense_layers = [ tf.keras.layers.Dense(u) for u in units ]
        self.activations = [ activation() for u in units ]
        self.bn_layers = [ tf.keras.layers.BatchNormalization() for u in units ]

    def call(self, x, training=False):
        for (dense,bn,ac) in zip(self.dense_layers,self.bn_layers,self.activations):
            x = dense(x)
            x = bn(x, training=training)
            x = ac(x)
        return x

class GraphStepLayer(tf.keras.layers.Layer):
    
    def __init__(self, nodes_encoder, attention_layer, messege_passer, state_updater, data_length, batch_size):
        super(GraphStepLayer, self).__init__()
        self.nodes_encoder = nodes_encoder
        self.attention_layer = attention_layer
        self.message_passer = messege_passer
        self.state_updater = state_updater
        self.pos_channel = np.expand_dims(np.expand_dims(np.arange(-1,1,2/data_length),axis=-1),axis=0)
        self.data_length = data_length
        #self.nodes_encs_expanded_1 = tf.Variable(tf.zeros((batch_size,self.data_length,2*nodes_encoder.out_encs_length+2)),trainable=False)
        #self.nodes_encs_expanded_2 = tf.Variable(tf.zeros((batch_size,self.data_length,2*nodes_encoder.out_encs_length+2)),trainable=False)
        self.bs = batch_size

    def build(self, input_shape):
        self.nodes_encoder.build(input_shape)
        self.nodes_encs_expanded_1 = tf.Variable(tf.zeros((batch_size,self.data_length,2*self.nodes_encoder.out_encs_length+2)),trainable=False)
        self.nodes_encs_expanded_2 = tf.Variable(tf.zeros((batch_size,self.data_length,2*self.nodes_encoder.out_encs_length+2)),trainable=False)
        
    def call(self, inputs):

        ### encode nodes for message passing
        node_encs = self.nodes_encoder(inputs) # (batch,len,channels)

        ### calculate attentions
        # add position channels
        pos_encs = np.repeat( self.pos_channel, self.bs, axis=0 )
        node_encs_att = tf.concat( (node_encs,pos_encs), axis=-1 ) # (batch,len,channels)
        # make proper data: (b,om,om',c+c')
        chs = node_encs_att.shape[-1]
        
        self.nodes_encs_expanded_1[:,:,:chs].assign(node_encs_att)
        self.nodes_encs_expanded_2[:,:,chs:].assign(node_encs_att)
        
        nodes_encs_exp_broad_1 = tf.expand_dims(self.nodes_encs_expanded_1, axis=2)
        nodes_encs_exp_broad_2 = tf.expand_dims(self.nodes_encs_expanded_2, axis=1)
        nodes_encs_for_attention = nodes_encs_exp_broad_1 + nodes_encs_exp_broad_2
        
        # reshape and feed
        nodes_encs_for_attention = tf.reshape( nodes_encs_for_attention, (-1,2*chs) ) # (b x om x om, chs)
        attentions = self.attention_layer( nodes_encs_for_attention ) # (b x om x om, 1)
        attentions = tf.reshape( attentions, (self.bs,self.data_length,self.data_length) ) # (b, om, om)

        ### prepare messages
        atts = tf.expand_dims(attentions,axis=-1) # (b, om, om, 1)
        node_encs_ext = tf.expand_dims(node_encs, axis=1) # (b, 1, om, ch)
        node_encs_ext = tf.repeat( node_encs_ext, self.data_length, axis=1 ) # (b, om, om, ch)
        nodes_relevant_message = node_encs_ext*atts # (b, om, om, ch)
        nodes_relevant_message_re = tf.reshape( nodes_relevant_message, (-1,self.data_length,nodes_relevant_message.shape[-1])) # (b x om, om, ch)
        messages = self.message_passer(nodes_relevant_message_re) # (b x om, ch')
        messages = tf.reshape(messages, (-1,self.data_length,messages.shape[-1])) # (b, om, ch)

        ### update states
        # CHANGED TO INIT
        update_from = tf.concat((node_encs,messages), axis=-1) # (b, om, ch')
        out_encs = self.state_updater( update_from ) # (b, om, ch)

#         # add softmax for testing
#         res = self.dense_soft(out_encs)
        
        return out_encs

class GraphLikeModel(tf.keras.Model):
    
    def __init__(self, nodes_encs_length_s, attention_units_s, message_length_s, mess_convs, new_state_length_s, data_length, batch_size, activation, kernels_node_encoder):
        super(GraphLikeModel, self).__init__()
        self.data_length = data_length
        self.bs = batch_size
        self.graph_layers = []
        (mess_filters,mess_kernels,mess_strides) = mess_convs
        for i,(nodes,atts,mess,news,k_ne) in enumerate(zip(nodes_encs_length_s, attention_units_s, message_length_s, new_state_length_s, kernels_node_encoder)):
            nodes_encoder = NodesEncoder(nodes, k_ne, activation)
            attention_layer = AttentionEstablisher(atts, activation)
            message_passer = MessagePasser(mess_filters, mess_kernels, mess_strides, mess, activation)
            state_updater = StateUpdater(news, activation)
            graph_layer = GraphStepLayer(nodes_encoder, attention_layer, message_passer, state_updater, data_length, batch_size)
            self.graph_layers.append(graph_layer)
        self.dense_soft = tf.keras.layers.Dense(2, activation='softmax')
            
    def call(self, x):
        for gr_layer in self.graph_layers:
            x = gr_layer(x)   
        # last layer - softmax
        res = self.dense_soft(x)
        return res

def compile_model(model, lr):
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# generator without shuffling
class generator_no_shuffle:
    
    def __init__(self, file, regime, batch_size, return_reminder):
        self.file = file
        self.regime = regime
        self.batch_size = batch_size
        self.return_reminder = return_reminder
        with h5.File(self.file,'r') as hf:
            self.num = hf[self.regime+'/data'].shape[0]     
        self.batch_num = self.num // self.batch_size
        if return_reminder:
            self.gen_num = self.num
        else:
            self.gen_num = self.batch_num*self.batch_size

    def __call__(self):
        start = 0
        stop = self.batch_size
        with h5.File(self.file, 'r') as hf:
            for i in range(self.batch_num):
                mask = hf[self.regime+'/mask'][start:stop]
                mask_channel = np.expand_dims( mask, axis=-1 )
                labels = hf[self.regime+'/labels_signal_noise'][start:stop]
                yield ( np.concatenate((hf[self.regime+'/data'][start:stop],mask_channel), axis=-1), 
                  labels )
                start += self.batch_size
                stop += self.batch_size
            if self.return_reminder:
                mask = hf[self.regime+'/mask'][start:stop]
                mask_channel = np.expand_dims( mask, axis=-1 )
                labels = hf[self.regime+'/labels_signal_noise'][start:stop]
                yield ( np.concatenate((hf[self.regime+'/data'][start:stop],mask_channel), axis=-1), 
                  labels )

# generator with shuffling
class generator_with_shuffle:
    
    def __init__(self, file, regime, batch_size, buffer_size, return_reminder):
        self.file = file
        self.regime = regime
        self.batch_size = batch_size
        self.return_reminder = return_reminder
        self.buffer_size = buffer_size
        with h5.File(self.file,'r') as hf:
            self.num = hf[self.regime+'/data/'].shape[0] 
        self.batch_num = (self.num-self.buffer_size) // self.batch_size
        self.last_batches_num = self.buffer_size // self.batch_size
        if return_reminder:
            self.gen_num = self.num
        else:
            self.gen_num = (self.batch_num+self.last_batches_num)*self.batch_size

    def __call__(self):
        start = self.buffer_size
        stop = self.buffer_size + self.batch_size
        with h5.File(self.file, 'r') as hf:
            mask = hf[self.regime+'/mask'][:self.buffer_size]
            mask_channel = np.expand_dims( mask, axis=-1 )
            buffer_data = np.concatenate((hf[self.regime+'/data'][:self.buffer_size],mask_channel), axis=-1)
            buffer_labels = hf[self.regime+'/labels_signal_noise'][:self.buffer_size]
            for i in range(self.batch_num):
                idxs = rd.sample( range(self.buffer_size), k=self.batch_size )
                yield ( buffer_data[idxs] , buffer_labels[idxs] )
                mask = hf[self.regime+'/mask'][start:stop]
                mask_channel = np.expand_dims( mask, axis=-1 )
                buffer_data[idxs] = np.concatenate((hf[self.regime+'/data'][start:stop],mask_channel),axis=-1)
                labels = hf[self.regime+'/labels_signal_noise'][start:stop]
                buffer_labels[idxs] = labels
                start += self.batch_size
                stop += self.batch_size
            # fill the buffer with left data, if any
            # this a bit increases buffer size and MIGT BE COSTLY
            mask = hf[self.regime+'/mask'][start:stop]
            mask_channel = np.expand_dims( mask, axis=-1 )
            buffer_data = np.concatenate( (buffer_data,np.concatenate((hf[self.regime+'/data'][start:stop],mask_channel),axis=-1)), axis=0 )
            labels = hf[self.regime+'/labels_signal_noise'][start:stop]
            buffer_labels  = np.concatenate( (buffer_labels,labels), axis=0 )
            sh_idxs = rd.sample( range(buffer_labels.shape[0]), k=buffer_labels.shape[0] )
            start = 0
            stop = self.batch_size
            for i in range(self.last_batches_num):
                idxs = sh_idxs[start:stop]
                yield ( buffer_data[idxs], buffer_labels[idxs] )
                start += self.batch_size
                stop += self.batch_size
            if self.return_reminder:
                idxs = sh_idxs[start:stop]
                yield ( buffer_data[idxs], buffer_labels[idxs])

### MADE SO BATCH SIZE IS CONSTANT AND NO REMINDER
def make_datasets(h5f,make_generator_shuffle,return_batch_reminder,train_batch_size,train_buffer_size,test_batch_size):
    # generator for training data
    if make_generator_shuffle:
        tr_generator = generator_with_shuffle(h5f,'train',train_batch_size,train_buffer_size,return_batch_reminder)
    else:
        tr_generator = generator_no_shuffle(h5f,'train',train_batch_size,return_batch_reminder)
    if return_batch_reminder:
        # size of the last batch is unknown
        tr_batch_size = None
    else:
        tr_batch_size = train_batch_size

    train_dataset = tf.data.Dataset.from_generator( tr_generator, 
                        output_signature=( tf.TensorSpec(shape=(tr_batch_size,max_len,6)), tf.TensorSpec(shape=(tr_batch_size,max_len,2))
                                         ) )

    if make_generator_shuffle:
        train_dataset = train_dataset.repeat(-1).prefetch(tf.data.AUTOTUNE)
    else:
        train_dataset = train_dataset.repeat(-1).shuffle(num_batch_shuffle)

    # generator for validation data
    te_generator = generator_no_shuffle(h5f,'test',test_batch_size,False)
    te_batch_size = tr_batch_size
    test_dataset = tf.data.Dataset.from_generator( te_generator, 
                        output_signature=( tf.TensorSpec(shape=(tr_batch_size,max_len,6)), tf.TensorSpec(shape=(tr_batch_size,max_len,2))
                                          ) )
    
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset

train_dataset, test_dataset = make_datasets(h5f,True,False,batch_size,500*batch_size,batch_size)

val_batch_size = batch_size
val_generator = generator_no_shuffle(h5f,'val',val_batch_size,False)
val_dataset = tf.data.Dataset.from_generator( val_generator, 
                    output_signature=( tf.TensorSpec(shape=(val_batch_size,max_len,6)), tf.TensorSpec(shape=(val_batch_size,max_len,2))
                                        ) )

val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

### DEFINE MODEL

depth = 5
try_params = [8,16,32,64,128]
#try_params = [3,5]

nodes_encs_length_s = [[16],[16],[16],[16],[16]]
attention_units_s = [[16],[16],[16],[16],[16]]
message_length_s = [16,16,16,16,16]
new_state_length_s = [[32],[32],[32],[32],[32]]
kernels_node_encoder = [[1],[1],[1],[1],[1]]

mess_filters = (12,)
mess_kernels = (5,)
mess_strides = (2,)
mess_convs = (mess_filters,mess_kernels,mess_strides)

#activation = tf.keras.layers.PReLU
activation = tf.keras.layers.LeakyReLU

### vary nodes encs length
for tr in try_params:
	postfix = 'attention_units_'+str(tr)
	loging_dir = log_dir+postfix
	attention_units_s = [ [tr] for _ in range(depth) ]
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=loging_dir,  
                                                    write_graph=False,
                                                    update_freq=2500, embeddings_freq=0 )
	callbacks = [tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=5 ),tensorboard_callback]
	graph_model = GraphLikeModel(nodes_encs_length_s, attention_units_s, message_length_s, mess_convs, new_state_length_s, max_len, batch_size, activation, kernels_node_encoder)
	model = compile_model(graph_model, 0.0008)
	model.fit(train_dataset, steps_per_epoch=2500, validation_steps=500, epochs=100, validation_data=test_dataset, callbacks=callbacks, verbose=0)

