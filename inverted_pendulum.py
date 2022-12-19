# Abstract Syntax Trees Module
from ast import Global

# Pygame and system libraries
import pygame, sys

# Numpy for arrays, matrices, and high-level mathematical functions
import numpy as np

# Pygame constants
from pygame.locals import *

# Math library for functions
import math

# Python file containing the PID controller
import Kalman_Filter_Object

# Parser for command line
import argparse

# Module to get time
import time

# Interfacing cameras
import pygame.camera

# Used to represent a PIL image
from PIL import Image

# Matlab functions
import matplotlib.pyplot as plt

# Global variable to hold the previous time step
global previous_timestamp

# Class 1: Inverted Pendulum = 5 functions
class InvertedPendulum(object):

    # Function 1.1: Initialize with the window dimensions, cart dimensions, pendulum dimensions, gravity, noise, range
    def __init__(
                # Inverted Pendulum Object
                self,
                
                # Dimensions of the screen window
                windowdims,

                # Dimensions of the cart
                cartdims,

                # Dimensions of the pendulum
                penddims,

                # Gravity
                gravity,

                # Boolean to add noise to the gravity and mass
                add_noise_to_gravity_and_mass,

                # Range of action
                action_range=[-1,1]):

        # Range
        self.action_range = action_range

        # Window width
        self.window_width = windowdims[0]
        
        # Window height
        self.window_height = windowdims[1]

        # Cart dimensions
        self.cart_width = cartdims[0]
        
        # Cart hight
        self.cart_height = cartdims[1]

        # Pendulum width
        self.pendulum_width = penddims[0]
        
        # Pendulum height
        self.pendulum_length = penddims[1]

        # Y cart
        self.Y_CART = 3 * self.window_height / 4
        
        # Function to reset the state
        self.reset_state()
        
        # If noise is added
        if add_noise_to_gravity_and_mass:

            # Add noise to the gravity, the mass of the cart, and the mass of the pole
            self.gravity = gravity + np.random.uniform(-5, 5)
            
            # Change the mass of the cart = 1.0
            self.masscart = get_args().mass_cart + np.random.uniform(-0.5, 0.5)
            
            # Change the mass of the pole = 0.1
            self.masspole = get_args().mass_pole + np.random.uniform(-0.05, 0.2)

        # If noise is not added
        else:

            # Set the gravity, mass of the cart, and the mass of the pole to their default values
            self.gravity = gravity
            
            # Mass of the cart
            self.masscart = get_args().mass_cart
            
            # Mass of the pole
            self.masspole = get_args().mass_pole

        # Total mass = mass of the pole + mass of the cart
        self.total_mass = self.masspole + self.masscart
        
        # Length
        self.length = 0.5  # actually half the pole's length
        
        # Length of the pole
        self.polemass_length = self.masspole * self.length
        
        # Magnitude of the force
        self.force_mag = 10.0
        
        # Time step = seconds between state updates
        self.dt = 0.005
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * math.pi/360
        
        # Threshold
        self.x_threshold = 2.4

        # Conversion of x
        self.x_conversion = self.window_width/2/self.x_threshold

    # Function 1.2: reset the state of the pendulum to the initial position ~ """initializes pendulum in upright state with small perturbation"""
    def reset_state(self):

        # Set the terminal to false and the time step back to 0
        self.terminal = False
        
        # Reset the Time step
        self.timestep = 0.1

        # Reset the velocity
        self.x_dot = np.random.uniform(-0.03, 0.03)
        
        # Reset the  position
        self.x = np.random.uniform(-0.01, 0.01)

        # Reset the theta angle
        self.theta = np.random.uniform(-0.03, 0.03)
        
        # theta angular velocity
        self.theta_dot = np.random.uniform(-0.01, 0.01)
        
        # Reset the score and reward values back to 0
        self.score = 0
        
        # Reset the reward
        self.reward = 0
        
    # Function 1.3: Get the current state of the Inverted Pendulum object
    def get_state(self):
        
        # Return the terminal, time step, x value, x dot value, theta angle, theta dot angle, and reward
        return (self.terminal,self.timestep,self.x,self.x_dot,self.theta,self.theta_dot,self.reward)
        
    # Function 1.4: Set the state of the Inverted Pendulum Object
    def set_state(self,state):

        # Store the requested terminal, time step, x, x dot, theta angle, theta dot angle
        terminal, timestep, x, x_dot, theta, theta_dot = state

        # Set the terminal, time step, x
        self.terminal = terminal
        
        # Set the time step
        self.timestep = timestep
        
        # Set the position
        self.x = x
        
        # Set the acceleration
        self.x_dot = x_dot
        
        # Set the theta in radians
        self.theta = theta
        
        # Set the angular velocity in radians
        self.theta_dot = theta_dot
        
    # Function 1.5:
    def step(self, action): 
        
        # Increment the time step
        self.timestep += 1

        # Limit the values in an array
        action = np.clip(action, -1, 1) #Max action -1, 1

        # Force is equal the action
        force = action * 10 #multiply action by 10 to scale
        
        # Cos of the theta
        costheta = math.cos(self.theta)
        
        # Sin of the theta
        sintheta = math.sin(self.theta)
                
        # set a temporary variable
        temp = (force + self.polemass_length * self.theta_dot ** 2 * sintheta) / self.total_mass

        # set a variable to the acceleration of the theta
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        
        # set a variable to the acceleration of the x
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # update the value of x
        self.x = self.x + self.dt * self.x_dot
        
        # update the value of the acceleration of x
        self.x_dot = self.x_dot + self.dt * xacc
        
        # update the value of the theta
        self.theta = self.theta + self.dt * self.theta_dot
        
        # update the value of the acceleration of the theta
        self.theta_dot = self.theta_dot + self.dt * thetaacc

        # set the terminal variable
        self.terminal = bool(
            
            # true if the current position is less than the negative threshold
            self.x < -self.x_threshold
            
            # true if the current position is greater than the positive threshold
            or self.x > self.x_threshold
            
            # true if the current theta is less than the negative theta threshold in radians
            or self.theta < -self.theta_threshold_radians
            
            # true if the current theta is greater than the positive theta threshold in radians
            or self.theta > self.theta_threshold_radians
        )

        # if the theta in radians to degrees is within -+ 15
        if (self.theta * 57.2958) < 15 and (self.theta * 57.2958) > -15:
            
            # Increment the score
            self.score += 1
            
            # Increment the reward
            self.reward = 1

        # else
        else:
            
            # reset the reward to 0
            self.reward = 0

        # return the current state
        return self.get_state()
    
# Class 2: Game = 10 functions
class InvertedPendulumGame(object):

    # Function 2.1: 
    def __init__(

                # Inverted Pendulum Game Object
                self,

                # path
                figure_path,
                
                # Dimensions of the game window = width x height
                windowdims=(800,400),

                # Dimensions of the cart = width x height
                cartdims=(50,10), 
                
                # Dimensions of the pendulum = width x height
                penddims=(6.0,150.0), 

                # Frequency of refreshing in 
                refreshfreq=1000, 

                # Gravity
                gravity=9.81,

                # magnitude
                manual_action_magnitude=1,

                # boolean to hold if random controller is on or off
                random_controller=False, 

                # Maximum time step
                max_timestep=1000, 

                # if nose is allowed
                noisy_actions=False, 

                # Mode
                mode=None,

                # Noise 
                add_noise_to_gravity_and_mass=False
                ):

        # mode of the pid controller
        self.PID_controller = mode
        
        # maximum time step
        self.max_timestep = max_timestep
        
        # inverted pendulum object
        self.pendulum = InvertedPendulum(windowdims,cartdims,penddims,gravity,add_noise_to_gravity_and_mass)
        
        # figure
        self.performance_figure_path = figure_path

        # width of the screen window
        self.window_width = windowdims[0]
        
        # height of the screen window
        self.window_height = windowdims[1]

        # width of the cart
        self.cart_width = cartdims[0]
        
        # height of the cart
        self.cart_height = cartdims[1]
        
        # width of the pendulum
        self.pendulum_width = penddims[0]
        
        # length of the pendulum
        self.pendulum_length = penddims[1]
        
        # magnitude
        self.manual_action_magnitude = manual_action_magnitude
        
        # random controller
        self.random_controller = random_controller
        
        # noise
        self.noisy_actions = noisy_actions

        # list of scores
        self.score_list = []
        
        # cart
        self.Y_CART = self.pendulum.Y_CART
        
        # self.time gives time in frames
        self.timestep = 0

        # initialize a new game
        pygame.init()
        
        # clock
        self.clock = pygame.time.Clock()
        
        # specify number of frames / state updates per second
        self.REFRESHFREQ = refreshfreq
        
        # surface
        self.surface = pygame.display.set_mode(windowdims, 0, 32)
        
        # caption
        pygame.display.set_caption('Inverted Pendulum Game')
        
        # array specifying corners of pendulum to be drawn
        self.static_pendulum_array = np.array([[-self.pendulum_width/2,0],[self.pendulum_width/2,0],[self.pendulum_width/2,-self.pendulum_length],[-self.pendulum_width/2,-self.pendulum_length]]).T
        
        # black colour
        self.BLACK = (0, 0, 0)
        
        # blue colour
        self.BLUE = (0, 0, 255)
        
        # red colour
        self.RED = (255, 0, 0)
        
        # white colour
        self.WHITE = (255, 255, 255)

    # Function 2.2
    def draw_cart(self, x, theta):

        # rectangle
        cart = pygame.Rect(self.pendulum.x * self.pendulum.x_conversion + self.pendulum.window_width/2 - self.cart_width // 2, self.Y_CART, self.cart_width, self.cart_height)
        
        # draw the rectangle
        pygame.draw.rect(self.surface, self.BLUE, cart)
        
        # pendulum array variable
        pendulum_array = np.dot(self.rotation_matrix(-theta), self.static_pendulum_array)
        
        # increment the pendulum array variable
        pendulum_array += np.array([[x*self.pendulum.x_conversion + self.pendulum.window_width/2], [self.Y_CART]])
        
        # set the pendulum to the drawn polygon
        pendulum = pygame.draw.polygon(self.surface, self.RED,((pendulum_array[0,0], pendulum_array[1,0]),(pendulum_array[0,1], pendulum_array[1,1]),(pendulum_array[0,2], pendulum_array[1,2]),(pendulum_array[0,3], pendulum_array[1,3])))

    # Function 2.3
    @staticmethod

    # function 2.3.1
    def rotation_matrix(theta):
        
        # rotation matrix with the theta angle
        return np.array([[np.cos(theta), np.sin(theta)],[-1 * np.sin(theta), np.cos(theta)]])

    # Function 2.4
    def render_text(self, text, point, position="center", fontsize=48):
        
        # set the font
        font = pygame.font.SysFont(None, fontsize)
        
        # sent the text render
        text_render = font.render(text, True, self.BLACK, self.WHITE)
        
        # set the text rectangles
        text_rect = text_render.get_rect()
        
        # if in the center
        if position == "center":
            
            # set the point
            text_rect.center = point
        
        # if in the top left
        elif position == "topleft":
            
            # set the point
            text_rect.topleft = point
        
        # build the surface
        self.surface.blit(text_render, text_rect)

    # Function 2.5
    def time_seconds(self):
        
        # refreshing time
        return self.timestep / float(self.REFRESHFREQ)

    # Function 2.6
    def starting_page(self):

        # Set the background color
        self.surface.fill(self.WHITE)
        
        # Add the text for the starting page
        self.render_text("Inverted Pendulum Control with a PID Controller",(0.5*self.window_width,0.4*self.window_height))

        # Add text
        self.render_text("COMP 417 Assignment 4 by Mohamed Mahmoud",(0.5*self.window_width,0.5*self.window_height),fontsize=30)
        
        # Add text
        self.render_text("Press enter to begin the game",(0.5 * self.window_width,0.7*self.window_height),fontsize=30)

        # Update the display for the game
        pygame.display.update()
    
    # Function 2.7
    def save_current_state_as_image(self,path):
        
        # set the image
        im = Image.fromarray(self.surface_array)
        
        # save the image
        im.save(path + "current_state.png")

    # Function 2.8
    def game_round(self):
        
        # Reset the state of the pendulum
        self.pendulum.reset_state()

        # List to store the theta differences
        theta_diff_list = []

        # Set action to 0
        action = 0

        # Iterate from 0 to the max timestep
        for i in range(self.max_timestep):
            
            # Set the surface array
            self.surface_array = pygame.surfarray.array3d(self.surface)

            # Set the surface array
            self.surface_array = np.transpose(self.surface_array, [1, 0, 2])
            
            # If the manual mode was chosen
            if self.PID_controller is None:
                
                # Iterate over every event in the events
                for event in pygame.event.get():
                    
                    # If the event is quit
                    if event.type == QUIT:

                        # Quit the game
                        pygame.quit()

                        # Exit the system
                        sys.exit()
                    
                    # If the event is key down
                    if event.type == KEYDOWN:
                       
                        # If the event key is left
                        if event.key == K_LEFT:

                            # Set the action to left
                            action = -self.manual_action_magnitude
                    
                        # If the event key is right
                        if event.key == K_RIGHT:

                            # Set the action to right
                            action = self.manual_action_magnitude
                    
                    # If the event is key up
                    if event.type == KEYUP:
                        
                        # If the event key is left
                        if event.key == K_LEFT:

                            # Set the action to 0
                            action = 0
                        
                        # If the event key is right
                        if event.key == K_RIGHT:

                            # Set the action to 0
                            action = 0
                        
                        # If the event key is escape
                        if event.key == K_ESCAPE:

                            # Quit the game
                            pygame.quit()

                            # Exit the system
                            sys.exit()

            # If PID controller mode was chosen
            else:
                
                args = get_args()

                # Compute the required force from the PID controller and set action to it
                action = self.PID_controller.get_action(self.pendulum.get_state(),args.disturbance,self.surface_array,random_controller=self.random_controller)
                
                # Iterate over every event in the events
                for event in pygame.event.get():
                    
                    # If the event is quit
                    if event.type == QUIT:

                        # Quit the game
                        pygame.quit()

                        # Exit the system
                        sys.exit()
                    
                    # If the event is keydown
                    if event.type == KEYDOWN:
                        
                        # If the event key is escape
                        if event.key == K_ESCAPE:

                            # Quit the game
                            pygame.quit()

                            # Exit the system
                            sys.exit()

            # If the noise is allowed and 
            if self.noisy_actions and Kalman_Filter_Object is None:
                action = action + np.random.uniform(-0.1, 0.1)

            # Set the terminal, time step, x, and theta variables
            terminal, timestep, x, _, theta, _, _ = self.pendulum.step(action)

            # Add the absolute value of the theta angle to the list of theta
            theta_diff_list.append(np.abs(theta))

            # set the time step
            self.timestep = timestep
            
            # fill the surface
            self.surface.fill(self.WHITE)
            
            # draw the cart
            self.draw_cart(x, theta)

            # set the time text
            time_text = "t = {}".format(self.pendulum.score)
            
            # render the text
            self.render_text(time_text, (0.1*self.window_width, 0.1*self.window_height),position="topleft", fontsize=40)

            # display the update
            pygame.display.update()
            
            # call the clock to tick
            self.clock.tick(self.REFRESHFREQ)

            # if the terminal has been reached
            if terminal:
                print("ENTERED HERE")
                break
        
        # If plotting turned on
        if args.plot == True:

            # Plot the theta vs time
            plt.plot(np.arange(len(theta_diff_list)), theta_diff_list)
            
            # Set the plot's horizontol label
            plt.xlabel('Time')
            
            # Set the plot's vertical label
            plt.ylabel('Theta(radians)')
            
            # plot the title
            plt.title("Theta vs Time")
            
            # Plot the grid
            plt.grid()
            
            # Save the figure
            plt.savefig(self.performance_figure_path + "_run_" + str(len(self.score_list)) + ".png")
            
            # Close the plot
            plt.close()

        # Append the score to the list of scores
        self.score_list.append(self.pendulum.score)

    # Function 2.9
    def end_of_round(self):
        
        # Set the background color to white
        self.surface.fill(self.WHITE)
        
        # Draw the cart
        self.draw_cart(self.pendulum.x, self.pendulum.theta)
        
        # Display the score
        self.render_text("Score: {}".format(self.pendulum.score),(0.5 * self.window_width, 0.3 * self.window_height))
        
        # Display the average score
        self.render_text("Average Score : {}".format(np.around(np.mean(self.score_list), 3)),(0.5 * self.window_width, 0.4 * self.window_height))
        
        # Display the standard deviation of the score
        self.render_text("Standard Deviation Score : {}".format(np.around(np.std(self.score_list), 3)),(0.5 * self.window_width, 0.5 * self.window_height))
        
        # Displat the number of runs
        self.render_text("Runs : {}".format(len(self.score_list)),(0.5 * self.window_width, 0.6 * self.window_height))
        
        # if the game has ended
        if self.PID_controller is None:

            # render the text
            self.render_text("(Enter to play again, ESC to exit)",(0.5 * self.window_width, 0.85 * self.window_height), fontsize=30)

        # Update the pygame display
        pygame.display.update()
        
        # Wait for 2 seconds
        time.sleep(2.0)

    # Function 2.10
    def game(self):

        # Set the starting page
        self.starting_page()
        
        # Runs until game is ended
        while True:
            
            # Time in Milliseconds            
            previous_timestamp = pygame.time.get_ticks()

            # Manual Mode
            if self.PID_controller is None:
                
                # Iterate over 
                for event in pygame.event.get():
                    
                    # Check if quit has been selected
                    if event.type == QUIT:

                        # Quit the game
                        pygame.quit()

                        # Exit the program
                        sys.exit()
                    
                    # Check if the keydown has been selected
                    if event.type == KEYDOWN:
                        
                        # Check if return has been selected
                        if event.key == K_RETURN:

                            # call the next round
                            self.game_round()

                            # end this round
                            self.end_of_round()
                        
                        # check if the escape has been selected
                        if event.key == K_ESCAPE:
                            
                            # quit the game
                            pygame.quit()
                            
                            # exit the program
                            sys.exit()

            # PID Controller
            else:

                # Run the game
                self.game_round()

                # End of the round
                self.end_of_round()
                
                # Reset the state
                self.pendulum.reset_state()

# Function 16 = retrieves the arguements from the command line
def get_args():

    # Arguement parser object
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default="manual")
    
    # Random Controller
    parser.add_argument('--random_controller', type=bool, default=False)

    # Noise
    parser.add_argument('--add_noise_to_gravity_and_mass', type=bool, default=False)
    
    # Maximum time step
    parser.add_argument('--max_timestep', type=int, default=1000)

    # Gravity
    parser.add_argument('--gravity', type=float, default=9.81)
    
    # Magnitude
    parser.add_argument('--manual_action_magnitude', type=float, default=1)
    
    # Seed
    parser.add_argument('--seed', type=int, default=0)
    
    # Noisy actions
    parser.add_argument('--noisy_actions', type=bool, default=False)

    # Mass of the cart
    parser.add_argument('--mass_cart',type=float,default=1)

    # Mass of the pole
    parser.add_argument('--mass_pole',type=float,default=0.1)
    
    # Path
    parser.add_argument('--performance_figure_path', type=str, default="performance_figure")

    # Disturbance
    parser.add_argument('--disturbance',type=bool,default=False)

    #  Plotting
    parser.add_argument('--plot',type=bool,default=False)

    # Store the arguments in a variable
    args = parser.parse_args()

    # Return the arguements
    return args

# Function 18
def main():

    # Store the command line arguements in a variable
    args = get_args()

    # Seed the generator
    np.random.seed(args.seed)

    # Check if the mode variable is manual
    if args.mode == "manual":
        
        # Mode is None
        inv = InvertedPendulumGame(
                                   # performance figure
                                   args.performance_figure_path,
                                   
                                   # mode
                                   mode=None,
                                   
                                   # gravity
                                   gravity=args.gravity,
                                   
                                   # magnitude
                                   manual_action_magnitude=args.manual_action_magnitude,
                                   
                                   # random controller
                                   random_controller=args.random_controller,
                                   
                                   # maximum time step
                                   max_timestep=args.max_timestep,
                                   
                                   # noise
                                   noisy_actions=args.noisy_actions,
                                   
                                   # allowing noise for the gravity and the mass
                                   add_noise_to_gravity_and_mass=args.add_noise_to_gravity_and_mass
                                   )
    
    # Mode is PID Controller
    else:

        # Store the game in a variable
        inv = InvertedPendulumGame(

                                   # performance figure
                                   args.performance_figure_path, 
                                   
                                   # mode
                                   mode=Kalman_Filter_Object.PID_controller(), 
                                   
                                   # gravity
                                   gravity=args.gravity, 
                                   
                                   # magnitude
                                   manual_action_magnitude=args.manual_action_magnitude,
                                   
                                   # random controller
                                   random_controller=args.random_controller,  
                                   
                                   # maximum time step
                                   max_timestep=args.max_timestep, 
                                   
                                   # noise
                                   noisy_actions=args.noisy_actions, 
                                   
                                   # allowing noise for the gravity and mass
                                   add_noise_to_gravity_and_mass=args.add_noise_to_gravity_and_mass
                                   )

    # Run the game    
    inv.game()

# Start of the program
if __name__ == '__main__':

    # Run the main script
    main()