# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:38:52 2023

@author: Jack Cribbin - 19328253

This file is a short demonstration of a possible use for the results of this 
project. It prompts the user to answer questions, and the is able to respond 
using only their gaze and pressing space
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
  
def checkAnswer(size, question, answers, scoreLR, scoreUD):
    '''
    Function to check if a user is choosing one of the two answers to a question

    Parameters
    ----------
    size : list
        A list containing the dimensions of the screen
    question : String
        The question posed to the user
    answers : list
        The possible answers to the question
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
    pygame.display.set_caption('Display Answer')
    
    # Set the font and size of the text
    font = pygame.font.Font(None, 80)
    
    # Define colors
    white = (255, 255, 255)
    black = (0, 0, 0)
    
    # Fill the screen with white
    screen.fill(white)
    
    # Print the question to the screen
    for i in range(len(question)):
        # Create the text surface and rect
        text_surface = font.render(question[i], True, black)
        text_rect = text_surface.get_rect()
    
        # Set the position of the text on the screen
        text_rect.centerx = screen.get_rect().centerx
        text_rect.centery = (screen.get_rect().height / 7) + (60 * i)
        
        # Blit the text surface onto the screen
        screen.blit(text_surface, text_rect)


    # Plot two rectangles in the bottom corners of the screen
    line_color = (0, 255, 0)
    
    # Left rectangle corners
    left_x1 = 0
    left_x2 = size[0] * 1/3
    left_y1 = size[1] * 2/3
    left_y2 = size[1]
    
    # Right rectangle corners
    right_x1 = size[0] * 2/3
    right_x2 = size[0] 
    right_y1 = size[1] * 2/3
    right_y2 = size[1]
    
    # Draw both corners
    pygame.draw.rect(screen, line_color, (left_x1, left_y1, left_x2, left_y2), 1000)
    pygame.draw.rect(screen, line_color, (right_x1, right_y1, right_x2, right_y2), 1000)
    
    # Plot the first answer on the screen
    # Create the text surface and rect
    text_surface1 = font.render(answers[0], True, black)
    text_rect1 = text_surface1.get_rect()

    # Set the position of the text on the screen
    text_rect1.centerx = screen.get_rect().width * 1/6
    text_rect1.centery = (screen.get_rect().height * 6/7)
    
    # Blit the text surface onto the screen
    screen.blit(text_surface1, text_rect1)
    
    # Plot the second answer on the screen
    # Create the text surface and rect
    text_surface2 = font.render(answers[1], True, black)
    text_rect2 = text_surface2.get_rect()

    # Set the position of the text on the screen
    text_rect2.centerx = screen.get_rect().width * 5/6
    text_rect2.centery = (screen.get_rect().height * 6/7)
    
    # Blit the text surface onto the screen
    screen.blit(text_surface2, text_rect2)
    
    # Plot a dot where the user is looking
    # Set the size of the red dot
    dot_size = 15
    
    # Set the color of the red dot
    dot_color = (255, 0, 0)  # red
        
    # Draw the red dot on the screen
    pygame.draw.circle(screen, dot_color, (x, y), dot_size)
    
    # If the dot is inside the left rectangle
    if(x < left_x2 and y > left_y1):
        # Change the colour
        pygame.draw.rect(screen, (0,0,255), (left_x1, left_y1, left_x2, left_y2), 1000)
        screen.blit(text_surface1, text_rect1)
        
        # Draw the red dot on the screen
        pygame.draw.circle(screen, dot_color, (x, y), dot_size)
        
        # Update the display
        pygame.display.flip()
        
        return 0
    
    # If the dot is inside the right rectangle
    elif(x > right_x1  and y > right_y1):
        # Change the colour
        pygame.draw.rect(screen, (0,0,255), (right_x1, right_y1, right_x2, right_y2), 1000)
        screen.blit(text_surface2, text_rect2)
        
        # Draw the red dot on the screen
        pygame.draw.circle(screen, dot_color, (x, y), dot_size)
        
        # Update the display
        pygame.display.flip()
        
        return 1
    
    
    # Update the display
    pygame.display.flip()  

def finalScreen(size, text):
    '''
    Function to display the final screen after all questions have
    been answered

    Parameters
    ----------
    size : list
        The screen dimensions
    text : String
        The text to display on screen

    Returns
    -------
    None.

    '''
    # Initialize Pygame
    pygame.init()
    
    # Create a Pygame display surface
    screen = pygame.display.set_mode((size[0], size[1]), pygame.FULLSCREEN)
    pygame.display.set_caption('Display Answer')
    
    # Set the font and size of the text
    font = pygame.font.Font(None, 80)
    
    # Define colors
    white = (255, 255, 255)
    black = (0, 0, 0)
    
    # Fill the screen with white
    screen.fill(white)
    
    # Create the text surface and rect
    text_surface1 = font.render(text, True, black)
    text_rect1 = text_surface1.get_rect()

    # Set the position of the text on the screen
    text_rect1.centerx = screen.get_rect().width * 1/2
    text_rect1.centery = (screen.get_rect().height * 1/2)
    
    # Blit the text surface onto the screen
    screen.blit(text_surface1, text_rect1)
    
    # Update the display
    pygame.display.flip()
    
    
# Turn off unneccesary error warnings
tf.get_logger().setLevel('ERROR')

# Load the saved model
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the two models for horizontal and vertical position approximation
modelLR = keras.models.load_model('models/model_res=9H_test')
modelUD = keras.models.load_model('models/model_res=9V_test')

# Store the 3 possible class names for horizontal and vetical guessing
class_namesLR = ['Left', 'MiddleH', 'Right']
class_namesUD = ['Down', 'MiddleV', 'Up']
    

# Initialize the webcam
cap = cv2.VideoCapture(0)
check = False
count = 0
step = 0
mod = 0
answer = 'empty'

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
                    
                    # Change which step to use depending on the current
                    # step and the user's answer
                    if(step == 0 and answer == 0):
                        step = 1
                        answer = None
                    elif(step == 0 and answer == 1):
                        step = 2
                        answer = None
                        
                    if(step == 1 and answer == 0):
                        step = 3
                        answer = None
                    elif(step == 1 and answer == 1):
                        step = 4
                        answer = None
                        
                    if(step == 2 and answer == 0):
                        step = 5
                    elif(step == 2 and answer == 1):
                        step = 6
                        
                    
                    # For each different step, call the checkAnswer() function
                    # with the corresponding question and answers
                    if(step == 0):
                        
                        question = ['Would you like coffee, or tea?',
                                'Look in the corresponding section and press 1',
                                'to lock in your choice']
                        answers = 'Tea', 'Coffee'
                        
                        # Call the checkAnswer() function to display a dot
                        # on the screen that is an approximation of where the
                        # user is looking
                        ans = checkAnswer(size, question, answers, scoreLR, scoreUD)
                    
                    elif(step == 1):
                        
                            
                        question = ['Would you like milk with your tea?',
                                    'Look in the corresponding section and press 2',
                                    'to lock in your choice']
                        answers = 'Yes', 'No'
                        
                        # Call the checkAnswer() function to display a dot
                        # on the screen that is an approximation of where the
                        # user is looking
                        ans = checkAnswer(size, question, answers, scoreLR, scoreUD)
                        
                    elif(step == 2):
                        
                            
                        question = ['Would you like milk with your coffee?']
                        answers = 'Yes', 'No'
                        
                        # Call the checkAnswer() function to display a dot
                        # on the screen that is an approximation of where the
                        # user is looking
                        ans = checkAnswer(size, question, answers, scoreLR, scoreUD)
                    
                    # When all questions have been answered, display the
                    # final screen using the finalScreen() function
                    elif(step == 3):
                        text = 'Enjoy your tea with milk!'

                        # Call the checkAnswer() function to display a dot
                        # on the screen that is an approximation of where the
                        # user is looking
                        ans = finalScreen(size, text)
                    elif(step == 4):
                        text = 'Enjoy your tea with no milk!'

                        # Call the checkAnswer() function to display a dot
                        # on the screen that is an approximation of where the
                        # user is looking
                        ans = finalScreen(size, text)
                        
                    elif(step == 5):
                        text = 'Enjoy your coffee with milk!'

                        # Call the checkAnswer() function to display a dot
                        # on the screen that is an approximation of where the
                        # user is looking
                        ans = finalScreen(size, text)
                    elif(step == 6):
                        text = 'Enjoy your coffee with no milk!'

                        # Call the checkAnswer() function to display a dot
                        # on the screen that is an approximation of where the
                        # user is looking
                        ans = finalScreen(size, text)
                    
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



print('done')







