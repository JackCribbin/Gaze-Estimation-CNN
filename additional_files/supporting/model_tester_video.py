# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:33:57 2023

@author: Jack Cribbin - 19328253

This file uses the computer's webcam to collect live images of a user's face.
It then loads a pretrained CNN model and estimates what section of the screen
the user is looking in, and displays this guess to the screen
"""

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pygame
import sys
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
    
    # Convert the dimensions of the second eye cut-out
    # to coordinates for the square 
    x1 = eyes[1][0]
    y1 = eyes[1][1]
    x2 = eyes[1][2] + x1
    y2 = eyes[1][3] + y1
    
    # Cut the eye section out of the frame
    roi2 = frame[y1:y2, x1:x2]
    
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
    res : String
        The resolution model that is being used
    scoreLR : list
        The confidence levels in the horizontal direction
    scoreUD : list
        The confidence levels in the vertical direction

    Returns
    -------
    None.

    '''
    
    # Set the value by which the x and y coordinates should be multiplied by
    offset = 0.9
    
    if(res == '2H'):
        # Using the prediction confidence levels and screen size, compute x and y
        # coordinates for a dot at the position of highest confidence
        x = size[0]/2 + (size[0]/2 * scoreLR[1] * offset) - (size[0]/2 * scoreLR[0] * offset)
        y = 0
        
    elif(res == '2V'):
        # Using the prediction confidence levels and screen size, compute x and y
        # coordinates for a dot at the position of highest confidence
        x = 0
        y = size[1]/2 + (size[1]/2 * scoreUD[0] * offset) - (size[1]/2 * scoreUD[1] * offset)
        
    elif(res == '4'):
        # Using the prediction confidence levels and screen size, compute x and y
        # coordinates for a dot at the position of highest confidence
        x = size[0]/2 - (size[0]/2 * scoreLR[0] * offset) + (size[0]/2 * scoreLR[1] * offset) 
        y = size[1]/2 + (size[1]/2 * scoreUD[0] * offset) - (size[1]/2 * scoreUD[1] * offset) 
                
    elif(res == '9'):
        # Using the prediction confidence levels and screen size, compute x and y
        # coordinates for a dot at the position of highest confidence
        x = size[0]/2 - (size[0]/2 * scoreLR[0] * offset) + (size[0]/2 * scoreLR[2] * offset) 
        y = size[1]/2 + (size[1]/2 * scoreUD[0] * offset) - (size[1]/2 * scoreUD[2] * offset) 

    if(res == '9' or res == '4'):
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
        
        if(res == '9'):
            # Change the colour of the 9th of the where the confidence 
            # is highest
            xsection = np.argmax(scoreLR)
            ysection = np.argmax(scoreUD)
            
            if(xsection == 0 and ysection == 0):
                pygame.draw.rect(screen, (0,0,255), (0, size[1]*2/3, size[0]/3, size[1]/3), 1000)
            elif(xsection == 0 and ysection == 1):
                pygame.draw.rect(screen, (0,0,255), (0, size[1]*1/3, size[0]/3, size[1]/3), 1000)
            elif(xsection == 0 and ysection == 2):
                pygame.draw.rect(screen, (0,0,255), (0, 0, size[0]/3, size[1]/3), 1000)
            elif(xsection == 1 and ysection == 0):
                pygame.draw.rect(screen, (0,0,255), (size[0]/3, size[1]*2/3, size[0]/3, size[1]/3), 1000)
            elif(xsection == 1 and ysection == 1):
                pygame.draw.rect(screen, (0,0,255), (size[0]/3, size[1]*1/3, size[0]/3, size[1]/3), 1000)
            elif(xsection == 1 and ysection == 2):
                pygame.draw.rect(screen, (0,0,255), (size[0]/3, 0, size[0]/3, size[1]/3), 1000)
            elif(xsection == 2 and ysection == 0):
                pygame.draw.rect(screen, (0,0,255), (size[0]*2/3, size[1]*2/3, size[0]/3, size[1]/3), 1000)
            elif(xsection == 2 and ysection == 1):
                pygame.draw.rect(screen, (0,0,255), (size[0]*2/3, size[1]*1/3, size[0]/3, size[1]/3), 1000)
            elif(xsection == 2 and ysection == 2):
                pygame.draw.rect(screen, (0,0,255), (size[0]*2/3, 0, size[0]/3, size[1]/3), 1000)
        
        # Draw the red dot on the screen
        pygame.draw.circle(screen, dot_color, (x, y), dot_size)
        
        # Update the display
        pygame.display.flip()
    
    else:
        
        # Initialize Pygame
        pygame.init()
        
        # Create a Pygame display surface
        screen = pygame.display.set_mode((size[0], size[1]), pygame.FULLSCREEN)
                        
        # Set the size of the red dot
        line_width = 10
        
        # Set the color of the red dot
        line_color = (255, 0, 0)  # red
        
        # Fill the screen with white
        screen.fill((255, 255, 255))  # white
        
        # Depending on if the horizontal or vertical model is being used
        # Highlight the correct half of the screen
        if(res == '2H'):
            if(x >= size[0]/2):
                pygame.draw.rect(screen, line_color, (size[0]/2,0,size[0]/2,size[1]), line_width*100)
            elif(x < size[0]/2):
                pygame.draw.rect(screen, line_color, (0,0,size[0]/2,size[1]), line_width*100)
                  
        elif(res == '2V'):
            if(y >= size[1]/2):
                pygame.draw.rect(screen, line_color, (0,size[1]/2,size[0],size[1]/2), line_width*100)
            elif(y < size[1]/2):
                pygame.draw.rect(screen, line_color, (0,0,size[0],size[1]/2), line_width*100)
            
        # Update the display
        pygame.display.flip()
    
# Turn off unneccesary error warnings
tf.get_logger().setLevel('ERROR')

# Move to the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Query the user for what resolution model they would like to use
res = input('\n\nWhat resolution model would you like to load? (2H, 2V, 4, 9): ').upper()

if(res == '9'):
    # Load the two models for horizontal and vertical position approximation
    modelLR = keras.models.load_model('models/model_res=9H')
    modelUD = keras.models.load_model('models/model_res=9V')
    
    # Store the 3 possible class names for horizontal and vetical guessing
    class_namesLR = ['Left', 'MiddleH', 'Right']
    class_namesUD = ['Down', 'MiddleV', 'Up']
    
elif(res == '4'):
    # Load the two models for horizontal and vertical position approximation
    modelLR = keras.models.load_model('models/model_res=2H')
    modelUD = keras.models.load_model('models/model_res=2V')
    
    # Store the 2 possible class names for horizontal and vertical guessing
    class_namesLR = ['Left', 'Right']
    class_namesUD = ['Down', 'Up']
    
elif(res == '2H'):
    # Load the two models for horizontal and vertical position approximation
    modelLR = keras.models.load_model('models/model_res=2H')
    
    # Store the 2 possible class names 
    class_namesLR = ['Left', 'Right']
    
elif(res == '2V'):
    # Load the two models for horizontal and vertical position approximation
    modelUD = keras.models.load_model('models/model_res=2V')
    
    # Store the 2 possible class names 
    class_namesUD = ['Down', 'Up']

# Initialize the webcam
cap = cv2.VideoCapture(0)
check = False
count = 0

try:
    loop = True
    while loop:
        
        # Skip running the program every second loop to reduce choppy-ness 
        count  = count + 1
        if(count % 1 == 0):
            
            # Capture frame-by-frame
            ret, frame = cap.read()
            key = cv2.waitKey(1) & 0xFF
            
            # Call the findEyes() function on the image to find the 
            # location of eyes in the image
            eyes = findEyes(frame)
            
            # If the user's eyes aren't in the correct position
            if(check == False):
                
                # Display the camera view with the region highlighted
                cv2.rectangle(frame,(100,340),(540,440),(0,255,0),3)
                cv2.imshow('Center eyes in green region',frame) 
                pygame.quit()
                
            # If both eyes were found
            if(len(eyes) == 2):
                
                # Checks to ensure that the user's eyes are in the correct
                # region of view
                check = True
                
                # Checks if the user's eyes are too high
                if(eyes[0][1] <= 300):
                    check = False
                    print('\nPlease move camera higher or head lower')
                # Checks if the user's eyes are too low
                elif(eyes[0][1] >= 400):
                    check = False
                    print('\nPlease move camera lower or head higher')
                # Checks if the user's eyes are not level
                elif(eyes[0][1] - eyes[1][1] > 100 or eyes[0][1] - eyes[1][1] < -100):
                    check = False
                    print('\nPlease level your camera or head out')
                # Checks if the user's eyes are too low
                elif(frame.shape[0] - eyes[0][1]  <= 100):
                    print('\nPlease move camera lower or head higher')
                    check = False
                # Checks if the user's eyes are too high
                elif(frame.shape[0] - eyes[1][1]  <= 100):
                    print('\nPlease move camera lower or head higher')
                    check = False
                    
                # If the user's eyes are in roughly the correct position
                if(check):
                    
                    # Turn off the display of the webcam
                    cv2.destroyAllWindows()
                    
                    # Call the cutoutEyes() function to cut-out, combine and 
                    # save an image of both eyes
                    image = cutoutEyes(frame, eyes)
                    
                    # Convert the image into a keras tensor array
                    img_array = tf.keras.utils.img_to_array(image)
                    img_array = tf.expand_dims(img_array, 0)
                     
                    if(res != '2V'):
                        # Make a prediction for the label of the image
                        # in regards to its horizontal positioning
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
                    else:
                        scoreLR = []
                    
                    if(res != '2H'):
                        # Make a prediction for the label of the image
                        # in regards to its vertical positioning
                        predictions = modelUD.predict(img_array)
                        
                        # Compute the confidence level of the prediction
                        score = tf.nn.softmax(predictions[0])
                        
                        # Print the prediction and the confidence level 
                        print(
                            "This image most likely belongs to {} with a {:.2f} percent confidence."
                            .format(class_namesUD[np.argmax(score)], 100 * np.max(score))
                        )
                        
                        # Convert the score Tensor to an array
                        scoreUD = np.array(score)
                    else:
                        scoreUD = []
                    
                    # Set the dimensions of the screen
                    size = [1920, 1080]
                    
                    # Call the displayGuess() function to display a dot
                    # on the screen that is an approximation of where the
                    # user is looking
                    displayGuess(size, score, res, scoreLR, scoreUD)
                    
                    # Checks for key press while the guess screen is up
                    for event in pygame.event.get():
                        
                        # End the program if the window is quit
                        if event.type == pygame.QUIT:
                            running = False
                            loop = False
                        
                        # Check for when a key is pressed
                        if event.type == pygame.KEYDOWN:
                            # If escape is pressed, end the loop and the program
                            if event.key == pygame.K_ESCAPE:
                                loop = False
                                running = False
            
            # If no eyes are found, inform the user and repeat the loop
            else:
                check = False
                print('No eyes found!')
        
        
            # Break the loop if 1 is pressed
            if key == ord("1"):
                loop = False
                break
        
    # Destroy the webcam
    cv2.destroyAllWindows()
    cap.release() 
    
# If an of some kind occurs, close all windows and inform the user
except Exception as e:
    print('###################################')
    print('\n\nThere was an error\n\n')
    print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
    print(repr(e))
    print('###################################')
    
    cv2.destroyAllWindows()
    cap.release() 
    pygame.quit()

# Close all windows 
cv2.destroyAllWindows()
cap.release() 
pygame.quit()


print('\nDone')







