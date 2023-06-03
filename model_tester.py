# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 03:45:55 2023

@author: Jack Cribbin - 19328253
"""

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pygame

import cv2

def findEyes(frame):
    '''
    Function to determine the location of a pair of eyes in a given image

    Parameters
    ----------
    frame : 3D uint8 array
        The 480 x 640 x 3 color image taken from the webcam

    Returns
    -------
    eyes : 2D int32 array
        The array of coordinates for the 2 eyes, if they were found 

    '''
    # Convert the frame to grayscale for the haar classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load the first haar classifier
    directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(directory, 'haarcascade_eye_tree_eyeglasses.xml')
    eye_detector=cv2.CascadeClassifier(path)
    
    # Try to detect eyes in the frame
    eyes = eye_detector.detectMultiScale(gray,
            scaleFactor=1.05,
            minNeighbors=12,
            minSize=(80,80),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
    
    # If the first eye detector didn't find both eyes
    if(len(eyes) != 2):
        print('Eye:',eyes)
        print('Using alt classifier')
    
        # Set the second haar classifier
        path = os.path.join(directory, 'haarcascade_frontalface_alt.xml')
        eye_detector=cv2.CascadeClassifier(path)
        
        # Try to detect eyes in the frame
        eyes = eye_detector.detectMultiScale(gray,
                scaleFactor=1.05,
                minNeighbors=12,
                minSize=(80,80),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
    
    # If the first and second eye detector didn't find both eyes
    if(len(eyes) != 2):
        print('Eye:',eyes)
        print('Using alt classifier 2')
    
        # Set the third haar classifier
        path = os.path.join(directory, 'haarcascade_eye.xml')
        eye_detector=cv2.CascadeClassifier(path)
        
        # Try to detect eyes in the frame
        eyes = eye_detector.detectMultiScale(gray,
                scaleFactor=1.05,
                minNeighbors=12,
                minSize=(80,80),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
    
    print('\n\n\n\nEyes:',eyes)
    return eyes

def cutoutEyes(frame, eyes):
    '''
    Function to cut-out and combine two images of the eyes found in an image

    Parameters
    ----------
    frame : 3D uint8 array
        The 480 x 640 x 3 color image taken from the webcam
    eyes : 2D int32 array
        The array of coordinates for the 2 eyes, if they were found 

    Returns
    -------
    image : 3D uint8 array
        The 200 x 100 x 3 color image of both eyes

    '''
    # Make the eye cut-outs square, 100 x 100 pixels
    for i in range(2):
        for j in range(2,4):
            eyes[i][j] = 100
    
    # Convert the dimensions of the first eye cut-out
    # to coordinates for the square
    x1 = eyes[0][0]
    y1 = eyes[0][1]
    x2 = eyes[0][2] + x1
    y2 = eyes[0][3] + y1
    
    # Cut the eye section out of the frame
    roi1 = frame[y1:y2, x1:x2]
    
    
    #if(eyes[0][0] > eyes[1][0]):
     #   side = 'left'
    #else:
     #   side = 'right'
    
    # Convert the dimensions of the second eye cut-out
    # to coordinates for the square 
    x1 = eyes[1][0]
    y1 = eyes[1][1]
    x2 = eyes[1][2] + x1
    y2 = eyes[1][3] + y1
    
    # Cut the eye section out of the frame
    roi2 = frame[y1:y2, x1:x2]
    
    #if(eyes[0][0] > eyes[1][0]):
     #   side = 'right'
    #else:
     #   side = 'left' 
    
    # Combine the two cutouts into one image
    if(eyes[0][0] > eyes[1][0]):
        image = np.concatenate((roi2,roi1),axis=1)
    else:
        image = np.concatenate((roi1,roi2),axis=1)
    
    return image
    
def displayGuess(size, score, res, scoreLR=[], scoreUD=[]):
    '''
    Function to display a dot at where the model predicts the user 
    is looking

    Parameters
    ----------
    size : list
        A list containing the dimensions of the screen
    score : list
        A list containing the confidence levels for all 
        possible screen sections

    Returns
    -------
    None.

    '''
    
    offset = 0.9
    if(res == 2):
        # Using the prediction confidence levels and screen size, compute x and y
        # coordinates for a dot at the position of highest confidence
        x = size[0]/2 + (size[0]/2 * scoreLR[1]) - (size[0]/2 * scoreLR[0])
        y = size[1]/2 - (size[1]/2 * scoreUD[1]) + (size[1]/2 * scoreUD[0])
        
    elif(res == 4):
        # Using the prediction confidence levels and screen size, compute x and y
        # coordinates for a dot at the position of highest confidence
        x = (size[0]/2 - (size[0]/2 * score[0]) + (size[0]/2 * score[1]) - 
             (size[0]/2 * score[2]) + (size[0]/2 * score[3]))
        y = (size[1]/2 + (size[1]/2 * score[0]) + (size[1]/2 * score[1]) - 
             (size[1]/2 * score[2]) - (size[1]/2 * score[3]))
        
    elif(res == 5):
        # Using the prediction confidence levels and screen size, compute x and y
        # coordinates for a dot at the position of highest confidence
        x = (size[0]/2 - (size[0]/2 * score[0]) + (size[0]/2 * score[1]) - 
             (size[0]/2 * score[3]) + (size[0]/2 * score[4]))
        y = (size[1]/2 + (size[1]/2 * score[0]) + (size[1]/2 * score[1]) - 
             (size[1]/2 * score[3]) - (size[1]/2 * score[4]))
        
    elif(res == 6):
        # Using the prediction confidence levels and screen size, compute x and y
        # coordinates for a dot at the position of highest confidence
        x = size[0]/2 - (size[0]/2 * scoreLR[0] * offset) + (size[0]/2 * scoreLR[2] * offset) 
        y = size[1]/2 + (size[1]/2 * scoreUD[0] * offset) - (size[1]/2 * scoreUD[2] * offset) 

        
    # If the coordinates are too close to the edge of the screen
    # Move it out from the edge
    if(x < 20):
        x = 20
    if(y < 20):
        y = 20
    if(x > size[0]-20):
        x = size[0]-20
    if(y > size[1]-20):
        y = size[1]-20
        
    
    
    # Initialize Pygame
    pygame.init()
    
    # Create a Pygame display surface
    screen = pygame.display.set_mode((size[0], size[1]), pygame.FULLSCREEN)
                    
    # Set the size of the red dot
    dot_size = 100
    
    # Set the color of the red dot
    dot_color = (255, 0, 0)  # red
    
    # Fill the screen with white
    screen.fill((255, 255, 255))  # white
    
    # Draw the red dot on the screen
    pygame.draw.circle(screen, dot_color, (x, y), dot_size)
    
    # Update the display
    pygame.display.flip()
    

# Turn off unneccesary error warnings 
tf.get_logger().setLevel('ERROR')

# Load the saved model 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#model = keras.models.load_model('models/model_res=2')

res = 6

#modelLR = keras.models.load_model('models/model_res=2_LvsR')
#modelUD = keras.models.load_model('models/model_res=2')

modelLR = keras.models.load_model('models/model_res=6H_best')
modelUD = keras.models.load_model('models/model_res=6V_best')

#modelLR = keras.models.load_model('models/model_res=16H')
#modelUD = keras.models.load_model('models/model_res=6V_best')

# Set the class names that'll be used by the classifier 
#class_namesLR = ['Left','Right'] 
#class_namesUD = ['Bottom','Top'] 

'''
talk about the image recognition technique used in facial recognition 
in phone cameras
'''

class_namesUD = ['Down', 'MiddleV', 'Up']
class_namesLR = ['Left', 'MiddleH', 'Right']

#class_namesUD = ['Down', 'MiddleV', 'Up']
#class_namesLR = ['H0','H1','H2','H3']

#class_names = ['BottomLeft', 'BottomRight', 
 #              'TopLeft', 'TopRight']

#class_names = ['BottomLeft', 'BottomRight', 'Middle',
 #              'TopLeft', 'TopRight']


# Initialize the webcam 
cap = cv2.VideoCapture(0)

try:
    # Loop until the loop is broken from within 
    loop = True
    while loop:
        
        # Display the webcam until a frame is captured
        while True:
            
            # Capture frame-by-frame and display it
            ret, frame = cap.read()
            key = cv2.waitKey(1) & 0xFF
            cv2.rectangle(frame,(100,340),(540,440),(0,255,0),3)
            cv2.imshow("Webcam", frame)
            
            # Break the loop if q is pressed, saving the last frame
            if key == ord("q"):
                break
            
            # Break the loop and the outer loop if 1 is pressed
            if key == ord("1"):
                loop = False
                break
            
        # Turn the webcam off
        cv2.destroyAllWindows()
        
        # If the outer loop hasn't been broken
        if loop:
            
            # Call the findEyes() function on the image to find the 
            # location of eyes in the image
            eyes = findEyes(frame)
            
            # If both eyes were found
            if(len(eyes) == 2):
                
                check = True
                if(eyes[0][1] <= 300):
                    check = False
                    print('Please move camera higher or head lower')
                elif(eyes[0][1] >= 400):
                    check = False
                    print('Please move camera lower or head higher')
                    
                if(eyes[0][1] - eyes[1][1] > 100 or eyes[0][1] - eyes[1][1] < -100):
                    check = False
                    print('Please level your camera or head out')
                
                if(frame.shape[0] - eyes[0][1]  <= 100):
                    print('Please move camera lower or head higher')
                    check = False
                if(frame.shape[0] - eyes[1][1]  <= 100):
                    print('Please move camera lower or head higher')
                    check = False
                    
                    
                if(check):
                    
                    display = frame
                    #cv2.imshow('Eyes found',frame) 
                    #key = cv2.waitKey(1) & 0xFF
                    
                    # Call the cutoutEyes() function to cut-out, combine and 
                    # save an image of both eyes
                    image = cutoutEyes(frame, eyes)
                    
                    
                    
                    ##### Single Model #####
                    
                    '''
                    # Convert the image into a keras tensor array
                    img_array = tf.keras.utils.img_to_array(image)
                    img_array = tf.expand_dims(img_array, 0)
                     
                    # Make a prediction for the label of the image
                    predictions = model.predict(img_array)
                    
                    # Compute the confidence level of the prediction
                    score = tf.nn.softmax(predictions[0])
                    
                    
                    # Print the prediction and the confidence level 
                    print(
                        "This image most likely belongs to {} with a {:.2f} percent confidence."
                        .format(class_names[np.argmax(score)], 100 * np.max(score))
                    )
                    
                    # Convert the score Tensor to an array
                    score = np.array(score)
                    print(score)
                    
                    # Store the dimensions of the screen
                    size = [1920, 1080]
                    
                    #displayGuess(size, score, res)
                    '''
                    
                    ##### Single Model #####
                    
                    
                    
                    
                    
                    
                    
                    ##### Double Model #####
                    
                    # Convert the image into a keras tensor array
                    img_array = tf.keras.utils.img_to_array(image)
                    img_array = tf.expand_dims(img_array, 0)
                     
                    # Make a prediction for the label of the image
                    predictions = modelLR.predict(img_array)
                    
                    
                    
                    # Compute the confidence level of the prediction
                    score = tf.nn.softmax(predictions[0])
                    
                    # Print the prediction and the confidence level 
                    print(
                        "\nThis image most likely belongs to {} with a {:.2f} percent confidence."
                        .format(class_namesLR[np.argmax(score)], 100 * np.max(score))
                    )
                    
                    # Convert the score Tensor to an array
                    scoreLR = np.array(score)
                    print(scoreLR)
                    
                    # Make a prediction for the label of the image
                    predictions = modelUD.predict(img_array)
                    
                    # Compute the confidence level of the prediction
                    score = tf.nn.softmax(predictions[0])
                    
                    
                    # Print the prediction and the confidence level 
                    print(
                        "\nThis image most likely belongs to {} with a {:.2f} percent confidence."
                        .format(class_namesUD[np.argmax(score)], 100 * np.max(score))
                    )
                    
                    # Convert the score Tensor to an array
                    scoreUD = np.array(score)
                    print(scoreUD)
                    
                    # Store the dimensions of the screen
                    size = [1920, 1080]
                    
                    #displayGuess(size, score, res, scoreLR, scoreUD)
                    
                    
                    ##### Double Model #####
                    
                    
                                    
                    while True:
                        # Display the camera view with the region highlighted
                        cv2.imshow('Center eyes in green region',display) 
                        key = cv2.waitKey(1) & 0xFF
                        
                        # Break the loop if q is pressed, saving the last frame
                        if key == ord("q"):
                            # Turn the webcam off
                            cv2.destroyAllWindows()
                            break
                        
                        # Break the loop and the outer loop if 1 is pressed
                        if key == ord("1"):
                            loop = False
                            break
                    
                    # Close the pygame display window
                    pygame.quit()
            
            else:
                print('Eyes not found!')
        
except:
    print('###################################')
    print('\n\nThere was an error\n\n')
    print('###################################')
    
    cv2.destroyAllWindows()
    cap.release() 
    pygame.quit()
        
    
# Destroy the webcam
cv2.destroyAllWindows()
cap.release() 

print('done')



