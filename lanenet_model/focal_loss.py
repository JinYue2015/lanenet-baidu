import tensorflow as tf
from tensorflow.python.ops import array_ops

def focal_loss(prediction_tensor, target_tensor,depth, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.

        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.

    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    softmax_p = tf.nn.softmax(prediction_tensor,axis=-1)
    zeros = array_ops.zeros_like(softmax_p, dtype=softmax_p.dtype)
    target_tensor = tf.reshape(
                target_tensor,
                shape=[target_tensor.get_shape().as_list()[0] ,
                       target_tensor.get_shape().as_list()[1] ,
                       target_tensor.get_shape().as_list()[2]])
    target_tensor = tf.one_hot (target_tensor,depth = depth,on_value = 1, off_value = 0)
    target_tensor = tf.cast(target_tensor,tf.float32)
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - softmax_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, softmax_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(softmax_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - softmax_p, 1e-8, 1.0))

    sum_entry_cross_ent = tf.reduce_sum(per_entry_cross_ent,-1)

    return tf.reduce_mean(sum_entry_cross_ent,axis=[1,2])

def main():
    pred = tf.placeholder(dtype='float32',shape=(2,10,10,8))
    targ = tf.placeholder(dtype='int32' ,shape=(2,10,10))
    #fcloss = focal_loss(pred,targ,8)
    #print(fcloss.shape)
    a= tf.constant([0,0,0,1,5,8,7,3,9,10,2,12,5,14,1,16,10,18,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,10,11,12,13,14,15,16,17,18,1,2,3,4,5,6,7,8,9],dtype='float32',shape=[2,3,3,3])
    a_reshape = tf.reshape(a,shape=[2,9,3])
    b = tf.nn.softmax(a)
    c = tf.nn.softmax(a,axis=-1)
    sess = tf.Session()
    a1,a_res,b1,c1 =  sess.run(fetches=[a,a_reshape,b,c])
    print('a=',a1,'\n a_reshape=',a_res,'\n b=',b1,'\n c=',c1)
if __name__ == "__main__":
    main()