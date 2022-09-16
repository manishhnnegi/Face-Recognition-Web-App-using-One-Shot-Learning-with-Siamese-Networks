import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall







class Classify:
    def __init__(self,filename):
        self.filename =filename


    def recognition(self):
        # load model
        #model = load_model('model.h5')
        # Reload model 
       
        # summarize model
        #model.summary()
        def access_imgfiles_and_preprocess(file_path):
    
            # Read in image from file path( retrun the content of file as tensor)
            byte_img = tf.io.read_file(file_path)
            
            # Load in the image (Decode a JPEG-encoded image to a uint8 tensor.)
            img = tf.io.decode_jpeg(byte_img)
            
            # Preprocessing steps - resizing the image to be 105x105x3
            img = tf.image.resize(img, (105,105))
            # Scale image to be between 0 and 1 
            img = img / 255.0
            
            # Return image
            return img

        # Siamese L1 Distance class
        class DistanceLayer(Layer):
            
            # Init method - inheritance
            def __init__(self, **kwargs):
                super().__init__()
            
            # Magic happens here - similarity calculation
            def call(self, input_embedding, validation_embedding):
                return tf.math.abs(input_embedding - validation_embedding)
                
        
        class CustomModel(Model):
            def train_step(self, batch):
                # Unpack the data. Its structure depends on your model and
                # on what you pass to `fit()`.
                #x, y = data
                x = batch[:2]
                # Get label
                y = batch[2]

                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)  # Forward pass
                    # Compute the loss value
                    # (the loss function is configured in `compile()`)
                    loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

                # Compute gradients
                trainable_vars = self.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                # Update weights
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                # Update metrics (includes the metric that tracks the loss)
                self.compiled_metrics.update_state(y, y_pred)
                # Return a dict mapping metric names to current value
                return {m.name: m.result() for m in self.metrics}
                
        model = tf.keras.models.load_model('final2000siamesemodel.h5',custom_objects = {"CustomModel": CustomModel, "DistanceLayer": DistanceLayer})
        def verify_image(model, detection_threshold, verification_threshold):
            # Build results array
            results = []
            for image in os.listdir(os.path.join('application_data', 'verification_images')):
                input_img = access_imgfiles_and_preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
                validation_img = access_imgfiles_and_preprocess(os.path.join('application_data', 'verification_images', image))
                
                # Make Predictions 
                result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
                results.append(result)
            
            # Detection Threshold: Metric above which a prediciton is considered positive 
            detection = np.sum(np.array(results) > detection_threshold)
            
            # Verification Threshold: Proportion of positive predictions / total positive samples 
            verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
            verified = verification > verification_threshold
            
            return results, verified
        results, verified = verify_image(model, 0.5, 0.5)
        print(verified)


            #imagename = self.filename
            #test_image = image.load_img(imagename, target_size = (64, 64))
            #test_image = image.img_to_array(test_image)
            #test_image = np.expand_dims(test_image, axis = 0)
            #result = model.predict(test_image)

       
        if verified:
            prediction = 'IMAGE VARIFIED!!!!'
            return [prediction]
        else:
            prediction = 'NOT VARIFIED!!'
            return [prediction]
     