# Library for the array, matrices, and high level mathematical calculations
import numpy as np

# To compute multiplicative inverse of a matrix
from numpy.linalg import inv

# OpenCV module
import cv2

# Math module
import math

# Matlab functions
import matplotlib.pyplot as plt

# Class to represent the PID controller
class PID_controller:

    # Initialize the Proportional-Integral-Derivative Controller
    def __init__(self):

        # List of estimated thetas
        self.estimated_list = []

        # List of the observations
        self.observations_list = []

        # List of the true values
        self.true_list = [] 

        # performance figure path
        self.performance_figure_path = "Performance_figure"
        
        # alpha
        self.alpha = 1
        
        # reset
        self.reset_state()
        
        # cart mass
        cart_mass = 1.0

        # pole mass
        pole_mass = 0.1

        # pole length
        pole_length = 1.0

        # Inertia
        inertia  = 1/3 * pole_mass * pole_length**2
    
        # denominator
        p = inertia *(cart_mass+pole_mass)+cart_mass*pole_mass*pole_length**2

        # time-step
        dt = 0.005

        # state-transition model = 2x2
        self.F = np.matrix([[1,dt],[0,1]])
        
        # control-input model = 2x1
        self.B = np.matrix([[0],[pole_mass*pole_length/p]])
        
        # observation model = 1x2
        self.H = np.matrix([[1,0]])
        
        # hyperparamters for Q
        hyp1 = 0.001**2
        hyp2 = 0.001**2

        # covariance of process noise = 2x2
        self.Q = np.matrix([[hyp1,0],[0,hyp2]])
        
        # hyperparameters for R
        hyp3 = 0.001

        # covariance of observation noise = 1x1 ~ detection error
        self.R = np.matrix([hyp3])

        # Previous action in torque
        self.prev_action = 0

        # Current value for integral
        self.integral = 0

        # Previous error
        self.previousError = 0

        # List to store all errors
        self.errorHistory = []

        # List to store all the integrals
        self.integralHistory = []

    # Computer Vision
    def theta_by_vision(self,image_state,plotting):
        
        # Crop the image to include only the inverted pendulum system
        croppedImage = image_state[100:len(image_state)]

        # Greyscale the cropped image
        greyyedImage = cv2.cvtColor(croppedImage,cv2.COLOR_BGR2GRAY)

        # Detect the edges
        edgedImage = cv2.Canny(greyyedImage,40,60,None,3)

        # Use the Hough line transformation = Input image, minVal, maxVal, aperture size, l2 gradient
        linedImage = cv2.HoughLines(edgedImage,1,np.pi/180,100)

        # Theta to be returned
        theta=0

        # If not empty
        if linedImage is not None:
            
            # Iterate over every line in the image
            for i in range(0, len(linedImage)):
                
                # Resolution of parameter r in pixels ~ distance from the origin to closest point on straight line
                rho = linedImage[i][0][0]

                # Resolution of theta in radians ~ angle between x axis and line connecting origin with closest point
                theta = linedImage[i][0][1]

                # Calculate cosine and sine of the theta
                a = math.cos(theta)
                b = math.sin(theta)

                # Hesse Normal Form
                x0 = a*rho
                y0 = b*rho
                
                # Computer point 1 and point 2
                pt1 = (int(x0+1000*(-b)), int(y0+1000*(a)))
                pt2 = (int(x0-1000*(-b)), int(y0-1000*(a)))
                
                # Draw the line
                cv2.line(greyyedImage, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

                # Recalculate theta if greater than
                if theta > np.pi/2:
                    theta = -1 * (np.pi-theta)

                # Store the 6 consecutive frames
                if self.timestep>49 and self.timestep<55 and plotting:
                    cv2.imwrite("theta"+str(self.timestep)+": value:"+str(theta)+".jpg",edgedImage)
                
        # Return the theta
        return theta

    # Kalmann filter
    def kalmann_filter(self,theta):
        
        # U = x10 to convert to newtons
        u = np.clip(self.prev_action, -1, 1) * 10

        # Theta
        z = theta

        # Predict Step
        
        # x_k|k-1 = F*x_k-1|k-1 + B*u
        x_prior = np.matmul(self.F, self.X) + self.B * u

        # P_k|k-1 = F*P_k-1|k-1*F' + Q
        P_prior = np.matmul(np.matmul(self.F, self.P), np.transpose(self.F)) + self.Q

        # Update Step
        
        # y_k = z_k - H*x_k|k-1
        error_calculation = z - np.matmul(self.H, x_prior)

        # S = H*P_k|k-1*H' + R
        S = np.matmul(np.matmul(self.H, P_prior), np.transpose(self.H)) + self.R

        # K = P_k|k-1*H'*inv(S)
        K = np.matmul(np.matmul(P_prior, np.transpose(self.H)), inv(S))

        # x_k|k = x_k|k-1 + K*y_k
        x_posterior = x_prior + np.matmul(K, error_calculation)

        # P_k|k = (I - K*H)*P_k|k-1
        P_posterior = np.matmul((np.eye(2) - np.matmul(K, self.H)), P_prior)

        # Store for next step
        self.x_last = x_posterior
        self.P_last = P_posterior

        # return the theta
        return x_posterior[0,0]

    # Function to reset the PID controller
    def reset_state(self):
        
        # Reset the error history
        self.errorHistory=[]

        # Reset the integral history
        self.integralHistory=[]

        # Reset the X
        self.X = np.matrix([[0],[0]])

        # Reset the P = 2x2 identity matrix
        self.P = self.alpha * np.eye(2)

    # Function to return the average error
    def averageError(self):
        return sum(self.errorHistory)/len(self.errorHistory)

    # Function to calculate the error given the theta
    def errorCalculator(self,theta):
        
        # Compute the previous error as theta mod 2 * pi
        previous_error = (theta%(2*math.pi))-0

        # Check if the previous error is greater than PI
        if previous_error > math.pi:

            # Set the previous error to the previous error minus 2 * pi
            previous_error = previous_error - (2*math.pi)

        # Return the previous error        
        return previous_error

    # PID Controller
    def pidController(self,time_delta,error,previous_error,integral):

        # Average error
        averageError = self.averageError()

        # p = 2,0,0
        # d = 0,1,0
        # i = 0,0,0.1
        
        # p,d = 5,484,0
        # p,i = 2,0,0.1
        # i,d =  0,0.1,0.1

        # p,i,d = -0.1, -0.1, 0.1

        Kp = 5
        Kd = 484
        Ki = 0

        # settling time = bound

        # Calculate the proportional error
        proportional = (Kp*error)

        # Calculate the derivative error
        derivative = Kd*((error-previous_error)/time_delta)

        # Calculate the integral error
        integral += error * time_delta

        # Compute the force using the equation
        F = proportional + derivative + (Ki*integral)

        # Return the force calculated
        return F

    def plot_check(self):

        if self.timestep>200:

            # Plot the theta vs time
            plt.plot(self.observations_list, self.true_list)
            
            # Set the plot's horizontol label
            plt.xlabel('observation values')
            
            # Set the plot's vertical label
            plt.ylabel('true values')
            
            # plot the title
            plt.title("Observation vs True (Time Range: 25 seconds)")
            
            # Plot the grid
            plt.grid()
            
            # Save the figure
            plt.savefig(self.performance_figure_path + "_run_" + str(self.timestep) + ".png")
            
            # Close the plot
            plt.close()

    # Function to get the required force to be applied to the cart ~  image state is a (800, 400, 3) numpy image array
    def get_action(self,state,disturbance,image_state,random_controller=False):

        # boolean / int / float [-2.4,2.4] / float [-inf,inf] / float [-pi/2,pi/2] radians / float [-inf,inf] / int [0,1] 
        terminal,timestep,x,x_dot,actualTheta,theta_dot,reward = state

        # Time-step
        self.timestep = timestep

        # If the random controller is selected
        if random_controller:

            # return a random force between -1 and 1 multiplied by a factor of 10
            return np.random.uniform(-1,1) * 10

        # If the PID controller is selected
        else:      
            
            # plotting flag
            plotting = False

            # Step 1: Compute theta using computer vision
            observed_theta = self.theta_by_vision(image_state,plotting)

            # Step 2: kalman filter
            estimated_theta = self.kalmann_filter(observed_theta)

            # Add the 3 values of theta to the history

            # Actual theta
            self.true_list.append(np.abs(actualTheta))

            # Observed theta by computer vision
            self.observations_list.append(np.abs(observed_theta))

            # Estimated theta by the Kalman filter
            self.estimated_list.append(np.abs(estimated_theta))

            # Plot
            if plotting:
                self.plot_check()
            
            # Step 3: Compute the error using the theta angle
            error = self.errorCalculator(estimated_theta)

            # Step 4: Add the error to the history of errors
            self.errorHistory.append(error)

            # Step 5: Force calculated by the PID Controller
            force = self.pidController(timestep,error,self.previousError,self.integral)

            # Step 6: Update the previous error
            self.previousError = self.errorHistory[-1]

            # Step 7: Random Disturbance 1: Less than 0.1% of the time
            if disturbance and np.random.rand()>0.999:

                # Return a random force multiplied by a factor of 100 + the force calculated by the PID controller
                return np.random.rand()*100 + force
            
            # Step 8: Return the force required to push the cart
            return force