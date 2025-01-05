#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:20:57 2024

@author: janot.vilaro & Alejandro Samaniego 
"""
import random 
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import time
import os
from sklearn.preprocessing import StandardScaler
#import cv2
#from joblib import dump, load

#################### for canvas display
#import tkinter as tk
#from PIL import Image, ImageDraw, ImageOps


class Neural_Network:
    def __init__(self, network_param): # network_param is a row vector with length equal to the number of layers and its components being the number of neurons of that layer.
        self.n_layers = len(network_param)
        self.network_param = network_param
        # we define empty objects for the biases and weights. For the bias on every element there will be a vector, and forthe weights, every element will be a matrix
        self.weights = []
        self.bias = []
        for ii in range(0,self.n_layers-1):
            row = self.network_param[ii+1]
            col = self.network_param[ii]
            
            ##RANDOM INITIALIZATION OF WEIGHTS
            self.weights.append(np.random.randn(row,col))
            # This is an object of indexed vectors. Index 0 contains the vectors of the 2nd layer (since the 1st layer are inputs and we do not add bias on those)
            #self.bias.append(np.random.randn(row))
            
            ##BETTER INITIALIZATION OF WEIGHTS (Gaussians, mean 0 and std=1/sqrt(·inputs _of_neuron))
            
            # This is an object of indexed matrices. Index 0 contains a matrix that describes the connection of 1st layer neurons with 2nd layer of neurons. This matrix has, in row one, the weights that will multiply the inputs to neuron one of layer 2 
            #self.weights.append(1/(np.sqrt(col))*np.random.randn(row,col))
            
            # This is an object of indexed vectors. Index 0 contains the vectors of the 2nd layer (since the 1st layer are inputs and we do not add bias on those)
            self.bias.append(np.random.randn(row))

    
    def SGD_quad(self, training_data, training_solutions, epochs, mini_batch_size, eta, test_data=None, test_solutions=None): #stochastic gradient descent.
        #training_data is de data we will use to train the model. A % of the EMNIST dataset, which is defined outside, in the run.py, by the user.   
        it = 0
        num_mini_batch = int(len(training_data)/mini_batch_size) #number of mini batches
        ittot = epochs*num_mini_batch*mini_batch_size*(self.n_layers-1) #total number of iterations, used only to print the percetage of the training that has been done
        
        for ii in range(0, epochs): #for many epochs
            for jj in range(0, num_mini_batch): #for all the mini batches
                rands = np.random.choice(len(training_data), mini_batch_size, replace=False) #We choose as many pictures as mini batch size, without repetition, to perform one iteration of our network training. 
                mini_batch = []
                mini_batch_solutions = []
                weights_correct = self.weights #we initialize a copy of the weights, and the actual weigths will be updated after all the pictures of a mini batch have been used
                bias_correct = self.bias  #idem as weigths
                for hh in range(0, len(rands)): #we create the minibatch with random pictures and their corresponding solutions
                    mini_batch.append(training_data[rands[hh]])
                    mini_batch_solutions.append(training_solutions[rands[hh]])
                for kk in range(0, mini_batch_size): #we loop for all the pictures in the mini batch
                    avec, zvec = self.feedforward(mini_batch[kk]) #output will be a matrix of vectors, every vector is the output of one layer
                    sol = self.num2vec(mini_batch_solutions[kk]) 
                    backput = self.backprop(zvec,sol) #backput is also a matrix. Every vector is the backproagated error of a given layer
                    for pp in range(0,self.n_layers-1): #for all the layers, we save biases and weigths with the estimated gradient of the cost function
                        weights_correct[pp] = weights_correct[pp] - eta/mini_batch_size*np.outer(backput[pp],avec[pp])
                        bias_correct[pp] = bias_correct[pp] - eta/mini_batch_size*backput[pp]
                        it = it + 1
                        
                        percent = it/ittot*100
                        if percent%5 == 0: #We print the progress of the training 
                            print(f"You are on it = {it}, out of {ittot}: , percentage = {percent}")
                   
                for ww in range(0,self.n_layers-1): #we update the real weights
                    self.weights[ww] = weights_correct[ww]
                    self.bias[ww] =  bias_correct[ww]
            if test_data is None or test_solutions is None:
                print("No test data inputed.")
            else: #if the iser inputs data to test, we do so
                good = 0
                predictv = []
                vecv = []
                for ll in range(0,len(test_data)): #we go over all the test pictures
                    predict, vec = self.prediction(test_data[ll])
                    predictv.append(predict)
                    vecv.append(vec)
                    if predict == test_solutions[ll]:
                        good = good +1
                        sol = self.num2vec(test_solutions[ll])
                percentage = good/len(test_data)*100
                random = np.random.randint(0,len(test_data),1)
                print(f"Percentage of success of epoch {ii}: {percentage} %") #we print the succes of that epoch
        

        # Set NumPy to save the full array. This is done to save the weigths and biases after training the NN.
        np.set_printoptions(threshold=np.inf)
        with open('final_weights_quad.txt', 'w') as file:
            self.convert_and_write(file, "Weights", self.weights)
            self.convert_and_write(file, "Bias", self.bias)
    
    #The comments of the above funtion apply directly to this funtion. The only differneces are highlighted       
    def SGD_entropy(self, training_data, training_solutions, epochs, mini_batch_size, eta, lambda_learn,test_data=None, test_solutions=None):
        start = time.time()
        it = 0
        num_mini_batch = int(len(training_data)/mini_batch_size) 
        ittot = epochs*num_mini_batch*mini_batch_size*(self.n_layers-1)
        
        for ii in range(0, epochs):
            for jj in range(0, num_mini_batch):
                rands = np.random.choice(len(training_data), mini_batch_size, replace=False) 
                mini_batch = []
                mini_batch_solutions = []
                weights_correct = self.weights
                bias_correct = self.bias
                for hh in range(0, len(rands)):
                    mini_batch.append(training_data[rands[hh]])
                    mini_batch_solutions.append(training_solutions[rands[hh]])
                for kk in range(0, mini_batch_size):
                    avec, zvec = self.feedforward(mini_batch[kk]) 
                    sol = self.num2vec(mini_batch_solutions[kk])
                    backput = self.backprop_entropy(zvec,sol) #This is the difference. We perform backpropagation but with the cross entropy cost function
                    for pp in range(0,self.n_layers-1):
                        weights_correct[pp] = weights_correct[pp] - eta/mini_batch_size*np.outer(backput[pp],avec[pp]) 
                        bias_correct[pp] = bias_correct[pp] - eta/mini_batch_size*backput[pp]
                        it = it + 1
                        
                        percent = it/ittot*100
                        if percent%5 == 0:
                            print(f"You are on it = {it}, out of {ittot}: , percentage = {percent}")
                for ww in range(0,self.n_layers-1):
                    self.weights[ww] = weights_correct[ww] - eta*lambda_learn/len(training_data)*self.weights[ww] #Notice we add this last term for renormalization. This makes the weights not change so much when the NN is performing quite good. This prevents overfitting.
                    self.bias[ww] =  bias_correct[ww]
            if test_data is None or test_solutions is None:
                print("No test data inputed.")
            else:
                good = 0
                predictv = []
                vecv = []
                for ll in range(0,len(test_data)):
                    predict, vec = self.prediction(test_data[ll])
                    predictv.append(predict)
                    vecv.append(vec)
                    if predict == test_solutions[ll]:
                        good = good +1
                        sol = self.num2vec(test_solutions[ll])
                percentage = good/len(test_data)*100
                random = np.random.randint(0,len(test_data),1)
                print(f"Percentage of success of epoch {ii}: {percentage} %")

 
        np.set_printoptions(threshold=np.inf)
        with open('final_weights_entropy.txt', 'w') as file:
            self.convert_and_write(file, "Weights", self.weights)
            self.convert_and_write(file, "Bias", self.bias)
        # Calculate the end time and time taken
        end = time.time()
        length = end - start    
        print("The code spent", length, "seconds.")
    
    def feedforward(self, entry_image):
        avec = [] # Here we will save many vectors, indexed. Right after aplying the sigmoid
        zvec = [] # Right before the sigmoid 
        a = entry_image #We make all the pixels of the picture "enter" the net
        avec.append(a)
       
        
        for ii in range(0,self.n_layers-1): #we go over all the layers
            z = np.matmul(self.weights[ii],a) + self.bias[ii] #we apply the weights and biases
            a = self.sigmoid(z) 
            
            if ii < self.n_layers - 2:
                avec.append(a)
            zvec.append(z)
        return avec, zvec
    
    def backprop(self, zvec, solution): 
        zL = zvec[-1]
        aL = self.sigmoid(zL)
        output_error = self.cost_derivative(solution,aL)*self.deriv_sigmoid(zL) #we compute the error of the output 
        output_error_vec = []
        output_error_vec.append(output_error) #we save the error
        for ii in range(self.n_layers-2,0,-1):
            weightsii_plus1_T = np.transpose(self.weights[ii])
            output_error = (np.matmul(weightsii_plus1_T,output_error))*self.deriv_sigmoid(zvec[ii-1]) #We calculate the error on all the layers
            output_error_vec.append(output_error)
        output_error_vec = output_error_vec[::-1]#we do this to reverse the list. start:stop:step, and since start is not specified it is 0, and stop not specified it gpoes t the end
        return output_error_vec
    
    #The idea is the same as the function above, but we use cross-entropy as cost function    
    def backprop_entropy(self, zvec, solution):
        zL = zvec[-1]
        aL = self.sigmoid(zL)
        output_error = aL - solution #The deriative of the cost function is so easy we don't create a function
        output_error_vec = []
        output_error_vec.append(output_error)
        for ii in range(self.n_layers-2,0,-1):
            weightsii_plus1_T = np.transpose(self.weights[ii])
            output_error = (np.matmul(weightsii_plus1_T,output_error))*self.deriv_sigmoid(zvec[ii-1]) #this term last term???? when using cross entropy loss function
            output_error_vec.append(output_error)
        output_error_vec = output_error_vec[::-1]
        return output_error_vec
    
    #Function that applies the sigmoid
    def sigmoid(self, z): 
        out = 1.0/(1.0+np.exp(-z))
        return out
        
    #self-explanatory
    def deriv_sigmoid(self, z):
        # out = np.exp(-z)/((1.0+np.exp(-z))**2)
        out = self.sigmoid(z)*(1-self.sigmoid(z))
        return out
    
    #Derivative of the cuadratic cost funtion
    def cost_derivative(self, y, output):
        out = output-y
        return out
    
    #Function that predicts the output. We decided to take as output the neuron that had a maximal value on the last layer.    
    def prediction(self,image):
        avec, zvec = self.feedforward(image)
        zL = zvec[-1]
        aL = self.sigmoid(zL)
        predict = np.argmax(aL)
        return predict, aL
     
    #Used to convert a from a number to a vector that can be mapped to the last layer of the NN (which has 10 neurons)  
    #For example, number 3 would be [0 0 0 1 0 0 0 0 0 0]
    def num2vec(self,num):
        long = self.network_param[-1] #size of the vector
        empty_vec = np.zeros(long)
        empty_vec[num] = 1
        vec = empty_vec
        return vec
        
############# HERE END THE NN FUNCTIONS. NOW EXTRA STUFF FOR IMAGE PROCESSING, MENU DISPLAY ETC. ###############
        
    def menu(self, X_future, y_future, scaler):
    # Predict the digit from the user-provided image
        print(
        colored('W', 'red'), colored('E', 'yellow'), colored('L', 'magenta'), colored('C', 'green'),
        colored('O', 'blue'), colored('M', 'magenta'), colored('E', 'red'), colored('T', 'yellow'),
        colored('O', 'green'), colored('T', 'blue'), colored('H', 'magenta'), colored('E', 'red'),
        colored('N', 'yellow'), colored('U', 'green'), colored('M', 'blue'), colored('B', 'magenta'),
        colored('E', 'red'), colored('R', 'yellow'), colored('P', 'green'), colored('R', 'blue'),
        colored('E', 'magenta'), colored('D', 'red'), colored('I', 'yellow'), colored('C', 'green'),
        colored('T', 'blue'), colored('O', 'magenta'), colored('R', 'red')
	)
	
    # Define a menu and get user's input for the image path
        directory_path = "/home/janot/our_NN/fotos"
        opt = "y"
        while opt.lower() == "y":
            mnist_o_no = int(input("Would you like to predict your own image (type 0) or a digit from MNIST (type 1)? "))
    
            if mnist_o_no == 0:
                print("Available images: ", os.listdir(directory_path))  # List all that is in the specified directory
                image_filename = input("Please provide the image filename: ")
                image_path = os.path.join(directory_path, image_filename)
        
                if not os.path.isfile(image_path):
                    print(colored('Please input a valid file name, consider adding .jpg', 'red'))
                else:
                    digit = self.predict_digit(image_path, scaler)  # Call the function to predict digits, which will call the function to preprocess the input image
                    print(colored('The predicted digit is:', 'green'), f" {digit}")
            
                # Display the image
                    img = Image.open(image_path)
                    plt.imshow(img, cmap='gray')
                    plt.title(f"Predicted Digit: {digit}")
                    plt.show()
     
            elif mnist_o_no == 1:
                index = int(input(f"You have available {len(X_future)} images. Introduce a number between 0 and {len(X_future)-1}: "))
        
                if 0 <= index < len(X_future):
                # Predict the digit for the chosen image
                    y_future_predicted, avec = self.prediction(X_future[index]) ###X_future.iloc?????????
              
                    y_future_real = y_future[index]
                # Display the image
                    plt.imshow(X_future[index].reshape(28, 28), cmap='gray')  # Fix
                    plt.title(f"Predicted Digit: {y_future_predicted}. Real Digit: {y_future_real}")
                    plt.show()
                else:
                    print("Please input a valid index.")
    
            else:
                print(colored('Please input a valid option', 'red'))
    
            opt = input("Would you like to predict another digit? (y/n): ")
    
     
    # This function calls the prediction function from the NN but ONLY when we are predcting images from the user, hence when they need preprocessing   
    def predict_digit(self, image_path, scaler):
        img_input_preprocessed = self.preprocess_image(image_path, scaler)  # Preprocess the image input by the user
        prediction, aL = self.prediction(img_input_preprocessed[0])  # Use SVM to predict the already preprocessed input image
        return prediction
        

    # Function to preprocess the user-provided image
    def preprocess_image(self, image_path, scaler):
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = np.array(img)  # Convert image to array
      
        #gives no. of rows along x-axis
        rows = np.size(img, 0) 
        #gives no. of columns along y-axis
        columns = np.size(img, 1)
        if rows > 28 and columns > 28:
        
            xmax = []
            xmin = []
            for ii in range(0,rows):
                xmax.append(max(img[ii]))
                xmin.append(min(img[ii]))
            xmax = max(xmax)
            xmin = min(xmin)
        
            img = 255/(xmax-xmin)*(img-xmin*np.ones((rows,columns)))
            for ii in range(0,rows):
                for jj in range(0, columns):
                    img[ii,jj] = int(img[ii,jj])
            plt.imshow(img, cmap='gray')
            plt.title(f"shade")
            plt.show()
            img_cut = []
            row_crit = []
            column_crit = []
      
            for ii in range(0,rows):
                for jj in range(0,columns-1):
                    change = abs(img[ii,jj+1] - img[ii,jj])
                    if change > 40:
                        row_crit.append(ii)
                        column_crit.append(jj+1)
            #row_max = max(row_crit) + int(rows*0.10)
            #row_min = min(row_crit) - int(rows*0.10)
            #column_max = max(column_crit) + int(columns*0.10)
            #column_min = min(column_crit) - int(columns*0.10)
            row_max = max(row_crit) 
            row_min = min(row_crit)
            column_max = max(column_crit) 
            column_min = min(column_crit) 
       
            for ii in range(row_min,row_max+1):
                counter = 0
                row_aux = np.zeros(column_max-column_min+1)
                for jj in range(column_min, column_max+1):
                    row_aux[counter] = img[ii,jj]
                    counter = counter +1
                img_cut.append(row_aux)
            rows = np.size(img_cut, 0) 
            columns = np.size(img_cut, 1)    
            step_row = int(rows/28) 
            step_col = int(columns/28)     
            rows = step_row*28
            columns = step_col*28 
            img_resize = []
            auxiliar_pixel = []
            for ii in range(0,rows,step_row):
                counter = 0
                for jj in range(0,columns,step_col):
               
                    for kk in range(0,step_row):
                        auxiliar_pixel.append(img_cut[ii + kk][jj:(jj+step_col-1)])
                    auxiliar_pixel = np.reshape(auxiliar_pixel,(1,-1))
                    img_resize.append(np.sum(auxiliar_pixel)/(step_row*step_col))
                
                    auxiliar_pixel = []
                #if (ii+1)%28 == 0:
                    #img_resize.append([np.sum(auxiliar_pixel[[np.arange(counter,len(auxiliar_pixel)-1),28]])])
                    #img_resize.append([np.sum(auxiliar_pixel[slice(counter,(len(auxiliar_pixel)),28)])])
       
                    
                   
       
        else:
            img_resize = np.reshape(img, (1,784))
            img_cut = np.reshape(img, (1,784))
        img_resize = abs(img_resize-255*np.ones(784))
        plt.imshow(img_cut, cmap='gray')
        plt.title(f"cut")
        plt.show()
        img_resize = np.reshape(img_resize,(28,28))
        plt.imshow(img_resize, cmap='gray')
        plt.title(f"resize")
        plt.show()   
        img_resize = np.reshape(img_resize,(1,-1))  
        xmax = max(img_resize[0])
        xmin = min(img_resize[0])
        

        img_resize = 255/(xmax-xmin)*(img_resize-xmin*np.ones(784))
         
        img_resize = img_resize / 255.0  # Normalize pixel values to [0, 1]
        
        return scaler.transform(img_resize)  # Standardize based on training data

    
    # This functions allows us to write the weights after training, so that we don't have t train thr NN every time e want to predict a digiit
    # It converts each item to a string representation
    def convert_and_write(self,file, name, data):
        file.write(f"{name}:\n")
        for item in data:
        # Handle both NumPy arrays and nested lists
            if isinstance(item, (list, np.ndarray)):
                item = np.array(item)  # Ensure it's a NumPy array
                file.write(np.array2string(item) + "\n")
            else:
                file.write(str(item) + "\n")
                
                
 ####### HERE ALL THE FUNCTIONS THAT HELP CREATE THE POP UP WINDOW FOR THE USER TO DRAW HUS NUMBER #######
    

    def setup_gui(self):
        """Sets up the GUI for drawing digits and predicting."""
    # Canvas dimensions
        self.canvas_size = 280  # Scaled canvas size
        
        self.image_size = 28    # Target size for prediction

    # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Draw a Digit")

    # Create canvas for drawing digits
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack()

    # Initialize a PIL image to hold the drawing
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)

    # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)

    # Add buttons and labels
        self.clear_button = tk.Button(self.root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack()

    # Canvas for displaying 10 squares
        self.prediction_canvas = tk.Canvas(self.root, width=600, height=100, bg="white")
        self.prediction_canvas.pack()

    # Add label below the squares
        self.prediction_label = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.prediction_label.pack()

    # Start automatic prediction timer
        self.running = True
        self.start_prediction_timer()

    # Start the Tkinter main loop
        self.root.mainloop()

 
    def paint(self, event):
        """Handles drawing on the canvas with square shapes and light gray for adjacent pixels."""
        x, y = event.x, event.y
        square_size = 10 # Size of the square (brush size)

    # Function to check if a pixel is already painted white
        def is_pixel_white(px, py):
            bbox = [px - 1, py - 1, px + 1, py + 1]
            cropped = self.image.crop(bbox)
            array = np.array(cropped)
            return (np.max(array.flatten) == 255)  # Check if all pixels in the square are white

        # Draw the main square in white if not already white
        if not is_pixel_white(x, y):
            self.canvas.create_rectangle(
                x - square_size // 2, y - square_size // 2,
                x + square_size // 2, y + square_size // 2,
                fill="white", outline="white"
            )
            self.draw.rectangle(
                [x - square_size // 2, y - square_size // 2, x + square_size // 2, y + square_size // 2],
                fill="white"
            )

    # Light gray for adjacent pixels if not already white
        adjacent_offset = square_size  # Offset for adjacent pixels
        light_gray = 150  # Light gray intensity

    # List of adjacent positions (up, down, left, right)
        adjacent_positions = [
            (x, y - adjacent_offset),  # Above
            (x, y + adjacent_offset),  # Below
            (x - adjacent_offset, y),  # Left
            (x + adjacent_offset, y)   # Right
        ]

        for adj_x, adj_y in adjacent_positions:
            if not is_pixel_white(adj_x, adj_y):
                self.canvas.create_rectangle(
                    adj_x - square_size // 2, adj_y - square_size // 2,
                    adj_x + square_size // 2, adj_y + square_size // 2,
                    fill=f"#{light_gray:02x}{light_gray:02x}{light_gray:02x}",
                    outline=f"#{light_gray:02x}{light_gray:02x}{light_gray:02x}"
                )
                self.draw.rectangle(
                    [adj_x - square_size // 2, adj_y - square_size // 2, adj_x + square_size // 2, adj_y + square_size // 2],
                    fill=light_gray
                )

 
 
    def preprocess_image_c(self):
        """Prepares the image for prediction."""
    # Resize to 28x28 (MNIST format)
        small_image = self.image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

    # Invert colors to match MNIST format
        #small_image = ImageOps.invert(small_image)

    # Normalize and flatten
        array = np.array(small_image) / 255.0
        return array.flatten()

    def predict_digit_popup(self):
        """Predicts the digit and updates the GUI with 10 squares."""
    # Preprocess the image
        input_array = self.preprocess_image_c()

    # Get prediction
        predicted_digit, aL = self.prediction(input_array)

    # Clear the previous squares
        self.prediction_canvas.delete("all")

    # Dimensions and spacing for the squares
        square_size = 45
        spacing = 12
        x_start = 20
        y_start = 20

        for ii in range(0,10):
        # Calculate the height of the filled portion based on value in aL (0 to 1 scale)
            fill_height = int(aL[ii] * square_size)

        # Coordinates for the square
            x0 = x_start + ii * (square_size + spacing)
            y0 = y_start
            x1 = x0 + square_size
            y1 = y0 + square_size

        # Draw the square outline
            self.prediction_canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")

        # Draw the filled portion
            self.prediction_canvas.create_rectangle(x0, y1 - fill_height, x1, y1, fill="black", outline="")

    # Update label below the squares
        self.prediction_label.config(text=f"0         1         2         3         4         5         6         7         8         9")

    def start_prediction_timer(self):
        """Runs the prediction every 0.1 seconds."""
        if self.running:
            self.predict_digit_popup()
            self.root.after(100, self.start_prediction_timer)

    def clear_canvas(self):
        """Clears the drawing canvas."""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)

