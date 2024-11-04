"""Temp file for dataset generation and encoding

TODO(Ruhao Tian): Refactor this file.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from PIL import Image
import pandas as pd

class grid_dataset:
    
    def __init__(self, input_file, coordinate_precision=8, color_precision=8):
        self.input_file = input_file
        self.df_input = pd.read_csv(input_file, header=0)
        self.coordinate_precision = coordinate_precision
        self.color_precision = color_precision
        self.tensors = None
        
        # print a sample of the input file
        print(self.df_input.head())
        
    def convert_to_binary(self):
        self.tensors = np.empty((0,self.coordinate_precision*2 + self.color_precision*3),dtype=int)
        # iterate all the rows in the input file
        for row in self.df_input.iterrows():
            # convert float to binary np array
            x_binary = np.binary_repr(int(row['x']*pow(2,self.coordinate_precision - 1)), width=self.coordinate_precision)
            y_binary = np.binary_repr(int(row['y']*pow(2,self.coordinate_precision - 1)), width=self.coordinate_precision)
            r_binary = np.binary_repr(int(row['r']), width=self.color_precision)
            g_binary = np.binary_repr(int(row['g']), width=self.color_precision)
            b_binary = np.binary_repr(int(row['b']), width=self.color_precision)
            # concatenate the binary arrays
            binary_array = np.array(list(map(int,(x_binary + y_binary + r_binary + g_binary + b_binary))))
            binary_array = binary_array > 0
            self.tensors = np.vstack((self.tensors,binary_array))
        
        print("Binary conversion finished")
        
        return self.tensors
    
    def convert_to_float(self, input_data):
        result = np.empty((0,5))
        for row in input_data:
            # print(row)
            x = int(''.join(map(str,row[:self.coordinate_precision])),2) / pow(2,self.coordinate_precision - 1)
            y = int(''.join(map(str,row[self.coordinate_precision:self.coordinate_precision*2])),2) / pow(2,self.coordinate_precision - 1)
            r = int(''.join(map(str,row[self.coordinate_precision*2:self.coordinate_precision*2 + self.color_precision])),2)
            g = int(''.join(map(str,row[self.coordinate_precision*2 + self.color_precision:self.coordinate_precision*2 + self.color_precision*2])),2)
            b = int(''.join(map(str,row[self.coordinate_precision*2 + self.color_precision*2:self.coordinate_precision*2 + self.color_precision*3])),2)
            result = np.vstack((result,np.array([x,y,r,g,b])))
            
        # print a sample of the result
        print(result[:5])
        return result
        
    def sparseness(self):
        # calculate the sparseness of the binary array
        total_ones = np.sum(self.tensors)
        total_elements = self.tensors.size
        
        return total_ones / total_elements

def catmull_rom_spline(control_points, num_points=100):
    # Extract x and y coordinates from control points
    control_points = np.array(control_points)
    x = control_points[:, 0]
    y = control_points[:, 1]
    
    # Generate parameter t
    t = np.linspace(0, 1, len(control_points))
    
    # Create cubic spline for x and y coordinates
    cs_x = CubicSpline(t, x, bc_type='clamped')
    cs_y = CubicSpline(t, y, bc_type='clamped')
    
    # Generate uniformly spaced points
    t_fine = np.linspace(0, 1, num_points)
    x_fine = cs_x(t_fine)
    y_fine = cs_y(t_fine)
    
    # Combine x and y coordinates
    spline_points = np.vstack((x_fine, y_fine)).T
    return spline_points

if __name__ == "__main__":

    # Define the 2D grid limits
    grid_width = 5
    grid_height = 5
    
    # Define control points for the Catmull-Rom spline
    control_points = [(0, 0), (1, 2), (3, 3), (4, 0), (5, 2)]

    # Generate the Catmull-Rom spline
    num_points = 100
    spline_points = catmull_rom_spline(control_points, num_points)

    # Load and resize the texture
    texture_path = 'abstract-beautiful-gradient-background-vector.jpg'  # Update with your texture image path
    texture = Image.open(texture_path)
    texture_array = np.array(texture)

    def sample_colors(spline_points, texture_array, grid_width, grid_height):
        # obtain the height and width of the texture
        texture_height, texture_width, _ = texture_array.shape
        print(texture_height, texture_width)
        # scale the spline points to the texture size
        scaled_spline_points = spline_points * np.array([texture_width / grid_width, texture_height / grid_height])
        colors = []
        for point in scaled_spline_points:
            x, y = point
            x = int(np.clip(x, 0, texture_width - 1))
            y = int(np.clip(y, 0, texture_height - 1))
            color = texture_array[y, x]  # Note: (y, x) indexing for image
            colors.append(color)
        return np.array(colors)

    # sample colors from the texture
    colors = sample_colors(spline_points, texture_array, grid_width, grid_height)
    print(colors.shape)

    # finalizing the dataset
    dataset = np.hstack((spline_points, colors))
    # scale the positions to 0-1
    dataset[:, :2] /= np.array([grid_width, grid_height])

    # save the dataset to csv
    with open('dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'r', 'g', 'b'])
        writer.writerows(dataset)

    #print("Sampled dataset:")
    print(dataset[:5])  # Print the first 5 rows of the dataset

    # Optional: Plot the spline (for visualization purposes)
    plt.figure()
    for i, point in enumerate(spline_points):
        plt.plot(point[0], point[1], 'o', color=(float(colors[i][0] / 255), float(colors[i][1] / 255), float(colors[i][2] / 255)))
    plt.plot([point[0] for point in control_points], [point[1] for point in control_points], 'ro--', label='Control Points')
    plt.legend()
    plt.show()
