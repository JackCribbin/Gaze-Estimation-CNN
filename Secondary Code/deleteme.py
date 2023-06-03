# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 22:24:24 2023

@author: jackp
"""
import pygame 
import os

def checkAnswer2(size, question, ans1, ans2, scoreLR, scoreUD):
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
    
    for i in range(len(question)):
        # Create the text surface and rect
        text_surface = font.render(question[i], True, black)
        text_rect = text_surface.get_rect()
    
        # Set the position of the text on the screen
        text_rect.centerx = screen.get_rect().centerx
        text_rect.centery = (screen.get_rect().height / 7) + (60 * i)
        
        # Blit the text surface onto the screen
        screen.blit(text_surface, text_rect)


    
    left_x1 = 0
    left_x2 = size[0] * 1/3
    left_y1 = size[1] * 2/3
    left_y2 = size[1]
    
    right_x1 = size[0] * 2/3
    right_x2 = size[0] 
    right_y1 = size[1] * 2/3
    right_y2 = size[1]

    line_color = (0, 255, 0)
    
    pygame.draw.rect(screen, line_color, (left_x1, left_y1, left_x2, left_y2), 1000)
    
    pygame.draw.rect(screen, line_color, (right_x1, right_y1, right_x2, right_y2), 1000)
    
    
    
    # Create the text surface and rect
    text_surface1 = font.render(ans1, True, black)
    text_rect1 = text_surface1.get_rect()

    # Set the position of the text on the screen
    text_rect1.centerx = screen.get_rect().width * 1/6
    text_rect1.centery = (screen.get_rect().height * 6/7)
    
    # Blit the text surface onto the screen
    screen.blit(text_surface1, text_rect1)
    
    
    # Create the text surface and rect
    text_surface2 = font.render(ans2, True, black)
    text_rect2 = text_surface2.get_rect()

    # Set the position of the text on the screen
    text_rect2.centerx = screen.get_rect().width * 5/6
    text_rect2.centery = (screen.get_rect().height * 6/7)
    
    # Blit the text surface onto the screen
    screen.blit(text_surface2, text_rect2)
    
    
    
    
    
    
    
    # Set the size of the red dot
    dot_size = 15
    
    # Set the color of the red dot
    dot_color = (255, 0, 0)  # red
    
    # Fill the screen with white
    #screen.fill((255, 255, 255))  # white
    
    # Draw the red dot on the screen
    pygame.draw.circle(screen, dot_color, (size[0]*2/3, size[1]*2/3), dot_size)
    
    if(x < left_x2 and y > left_y1):
        pygame.draw.rect(screen, (0,0,255), (left_x1, left_y1, left_x2, left_y2), 1000)
        
        screen.blit(text_surface1, text_rect1)
        
        # Draw the red dot on the screen
        pygame.draw.circle(screen, dot_color, (x, y), dot_size)
        
        # Update the display
        pygame.display.flip()
        
        return 0
        
    elif(x > right_x1  and y > right_y1):
        pygame.draw.rect(screen, (0,0,255), (right_x1, right_y1, right_x2, right_y2), 1000)
        
        screen.blit(text_surface2, text_rect2)
        
        # Draw the red dot on the screen
        pygame.draw.circle(screen, dot_color, (x, y), dot_size)
        
        # Update the display
        pygame.display.flip()
        
        return 1
    
    # Update the display
    pygame.display.flip()  
    
    
    
    
    
    
    
    
    
    
    
 
def checkAnswer(size, pic1, pic2, scoreLR, scoreUD):
    '''
    Function to check if what product the user is looking at

    Parameters
    ----------
    size : list
        A list containing the dimensions of the screen
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
    black = (0, 0, 0)
    
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
    dot_color = (255, 0, 0)  # red
        
    # Draw the red dot on the screen
    pygame.draw.circle(screen, dot_color, (x, y), dot_size)
    
    # Update the display
    pygame.display.flip()  



    
    





# Move to the correct directory
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

# Set the dimensions of the screen
size = [1920, 1080]

scoreLR = [0.5,0, 0.5]
scoreUD = [0.5,0, 0.5]

pygame.quit()

pic1 = 'pasta1.jpg'
pic2 = 'pasta2.jpg'

# Call the checkAnswer() function to display a dot
# on the screen that is an approximation of where the
# user is looking
ans = checkAnswer(size, pic1, pic2, scoreLR, scoreUD)

loop = True
while loop:
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
                
            # If space is pressed, end the loop and the program
            if event.key == pygame.K_a:
                answer = ans
                print('\nAnswer is:',answer)
                
                loop = False
                running = False
    
    
    
pygame.quit()
    
    
    
