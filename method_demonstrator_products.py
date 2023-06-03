# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 12:48:22 2023

@author: jackp
"""

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pygame
import sys
import cv2
from time import sleep


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
  
def checkAnswer(size, pic1, pic2, scoreLR, scoreUD):
    '''
    Function to check if what product the user is looking at

    Parameters
    ----------
    size : list
        A list containing the dimensions of the screen
    pic1 : String
        The name of the first image file that is to be loaded
    pic1 : String
        The name of the second image file that is to be loaded
    scoreLR : list
        The confidence levels in the horizontal direction
    scoreUD : list
        The confidence levels in the vertical direction

    Returns
    -------
    int
        Either a zero or a one, to indicate what the user's answer was

    '''
    
    # Set the value by which the x and y coordinates should be multiplied by
    offset = 0.9
    
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
    pygame.display.set_caption('Product Design Test')
    
    # Define colors
    white = (255, 255, 255)
    
    # Fill the screen with white
    screen.fill(white)
    
    pic1 = pygame.image.load(directory+'\\'+pic1).convert()
    pic2 = pygame.image.load(directory+'\\'+pic2).convert()
    
    pic1 = pygame.transform.scale(pic1, (775,775))
    pic2 = pygame.transform.scale(pic2, (775,775))
    
    screen.blit(pic1, (0,size[1]/6))
    screen.blit(pic2, (size[0]*3/5, size[1]/6))
    
    # Plot a dot where the user is looking
    # Set the size of the red dot
    dot_size = 15
    
    # Set the color of the red dot
    dot_color = (0, 0, 255)  # red
        
    # Draw the red dot on the screen
    pygame.draw.circle(screen, dot_color, (x, y), dot_size)
    
    # Update the display
    pygame.display.flip() 






# Turn off unneccesary error warnings
tf.get_logger().setLevel('ERROR')

# Move to the correct directory
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

# Load the two models for horizontal and vertical position approximation
modelLR = keras.models.load_model('models/model_res=9H')
modelUD = keras.models.load_model('models/model_res=9V')

# Store the 3 possible class names for horizontal and vetical guessing
class_namesLR = ['Left', 'MiddleH', 'Right']
class_namesUD = ['Down', 'MiddleV', 'Up']
    

# Initialize the webcam
cap = cv2.VideoCapture(1)
check = False
count = 0

pic1 = 'pasta1.jpg'
pic2 = 'pasta2.jpg'

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
                    
                    # Set the dimensions of the screen
                    size = [1920, 1080]
                    
                    # Call the checkAnswer() function to display a dot
                    # on the screen that is an approximation of where the
                    # user is looking
                    ans = checkAnswer(size, pic1, pic2, scoreLR, scoreUD)

                    
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
                                
                            # If space is pressed, save the answer given
                            if event.key == pygame.K_SPACE:
                                sleep(1)
                                answer = ans
                                
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
    
# If an error of some kind occurs, close all windows and inform the user
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



print('done')







