import tensorflow as tf
from tensorflow.keras import backend as K 

def CategoricalFocalLoss(name='categorical_focal_loss', gamma=2.0):
    """
    Focal loss for multi-classification problem.
    
    This might look gross but it allows us to generate custom versions of
    focal loss using specified gamma values with ease, without creating a 
    loss functioin that keras will reject.
    
    :param gamma: float, the focusing parameter gamma.
    :return: A loss function object that can be used with TensorFlow model.
    """
    
    def focal_loss(y_true, y_pred):
        """
        Compute the focal loss given the ground truth labels (y_true) and predicted labels (y_pred).
        
        :param y_true: tensor of true labels.
        :param y_pred: tensor of predicted labels.
        :return: scalar tensor representing the focal loss value.
        """
        
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Compute cross entropy
        cross_entropy = -float(y_true) * K.log(y_pred)
        
        # Calculate the class weights required to balance the output
        #this is the equivalent to alpha
        ClassRepresentation = tf.math.reduce_sum(y_true, axis=0, keepdims=True)
        
        alpha = tf.math.divide_no_nan(tf.cast(tf.math.reduce_max(ClassRepresentation),dtype=tf.float32), 
                                      tf.cast(ClassRepresentation, dtype=tf.float32)) 
        
        # Compute weights based on class balance and gamma
        weights = alpha * tf.pow(1 - y_pred, gamma)
        
        # Compute final categorical focal loss
        loss = K.sum(weights * cross_entropy, axis=-1)
        
        return loss
    
    return focal_loss