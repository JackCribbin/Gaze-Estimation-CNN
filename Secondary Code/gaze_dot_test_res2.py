# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 03:19:22 2023

@author: jackp
"""

import numpy as np
import pygame

class_names1 = ['Bottom','Top']
class_names2 = ['Left','Right']

score1 = [1,0]
score2 = [0,1]

size = [1920, 1080]

x = size[0]/2 + (size[0]/2 * score2[1]) - (size[0]/2 * score2[0])
y = size[1]/2 - (size[1]/2 * score1[1]) + (size[1]/2 * score1[0])



print(x,y)

if(x < 20):
    x = 20
if(y < 20):
    y = 20
if(x > size[0]-20):
    x = size[0]-20
if(y > size[1]-20):
    y = size[1]-20


# initialize Pygame
pygame.init()



# create a Pygame display surface
screen = pygame.display.set_mode((size[0], size[1]), pygame.FULLSCREEN)

# set the pixel values for the red dot
dot_x = x
dot_y = y

# set the size of the red dot
dot_size = 10

# set the color of the red dot
dot_color = (255, 0, 0)  # red

# fill the screen with white
screen.fill((255, 255, 255))  # white

# draw the red dot on the screen
pygame.draw.circle(screen, dot_color, (dot_x, dot_y), dot_size)

# update the display
pygame.display.flip()



print(x,y)

running = True
while running:
    for event in pygame.event.get():
        
        # End the program if the window is quit
        if event.type == pygame.QUIT:
            running = False
            loop = False
        
        # Check for when a key is pressed
        if event.type == pygame.KEYDOWN:
            
            # If q is pressed, save the current frame from 
            # the camera and end the loop                
            if event.key == pygame.K_q:                
                running = False
            
            # If escape is pressed, end the loop and the program
            if event.key == pygame.K_ESCAPE:
                loop = False
                running = False
    
pygame.quit()
