# The main simulation code 
import os
import torch
import numpy as np  
from matplotlib import pyplot as plt 
from matplotlib.colors import Normalize

# starting with a simpler version of the simulation.



class simultation():
    def __init__(self, sim_length : int, n : int, L_value : float = 2.0, V_max_value : float = 5.0, D_critical_value : float = 20, tau_value : float = 0.1, b_value : float = 0.0):
        # Don't record gradient
        torch.set_grad_enabled(False)
        device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        self.device = torch.device(device)
        print(f"Using {device} device for the computation.")
        # hyper variables theta = [V_max, D_critial, L, tau, b]
        self.L_value = L_value # Length of the cars
        self.V_max_value = V_max_value # maximum velocity m/s
        self.D_critical_value = D_critical_value #10 # critical distance in meters
        self.tau_value = tau_value # time delay in reactoin
        self.b_value = b_value # distance esstimation bias in meters

        # other variables 
        self.epsilon = 0.0001 # the small value 
        self.delta_t = 0.01 # second in simulation per compute step
        self.sim_length = sim_length # length of simulation

        self.n = n # number of carss
        self.V_max = torch.ones(n)*self.V_max_value # maximum velocity of individual cars
        self.D_critical = torch.ones(n)*self.D_critical_value # critical distance of individual cars
        self.L = torch.ones(n)*self.L_value # length of individual cars
        self.tau = torch.ones(n)*self.tau_value # time delay of reaction for individual cars
        self.b = torch.ones(n)*self.b_value # distance estimation bias for individual cars
        self.C = self.V_max / (torch.log(self.D_critical/self.L)+self.epsilon) # the constant coefficient 
        self.last_V = torch.ones(n)*self.V_max_value # place holder 


        # initializing simulation variables
        self.X = (self.epsilon+self.D_critical+self.L)*torch.tensor([i for i in range(n)])
        self.V = torch.ones(n)*self.V_max_value # cars start at 0 speed
        self.D = torch.cat((self.X[1:] - (self.X[:1] + self.L[:1]), torch.tensor([self.D_critical[-1] + self.epsilon])))
        self.T = 0
        self.Iteration = 0

        # storage variables, initialize the history with current states
        self.V_data = torch.tensor([self.V.tolist() for i in range(self.sim_length)])
        self.X_data = torch.tensor([self.X.tolist() for i in range(self.sim_length)])
        self.D_data = torch.tensor([self.D.tolist() for i in range(self.sim_length)])
        self.T_data = torch.tensor([self.T+self.delta_t*i for i in range(self.sim_length)]) # time stamp


    def step(self):
        # compute index for delayed operations
        future_index = torch.clamp(self.Iteration + (self.tau) / self.delta_t, min = 0, max = self.sim_length-1).long()
        current_index = self.Iteration#int(self.T/self.delta_t)

        batch_indices = torch.arange(self.n) # index for each car 

        # compute V: the future velocity according to the current D
        future_V = self.C * torch.log((self.D-self.b)/(self.L))
        future_V = torch.where(self.D < self.L + self.epsilon, torch.tensor(0.0, device=self.D.device), future_V)  # Set to 0 if D < lower_bound
        future_V = torch.where(self.D > self.D_critical - self.epsilon, self.V_max, future_V)  # Set to 1 if D > upper_bound
        future_V[-1] = self.last_V[future_index[-1]] # speed of the last car is free to choose from.
        # Scatter future_V into V_data at the appropriate locations specified by future_index
        self.V_data[future_index, batch_indices] = future_V

        
        # grab the current V
        self.V = self.V_data[current_index]
        # compute X: the new position according to the current velocity V
        self.X = self.X + self.V * self.delta_t # x = x + dx

        # compute D: the gap distance between cars
        self.D[:-1] = self.X[1:] - (self.X[:-1] + self.L[:-1]) # X_i+1 - X_i
        self.D[-1] = self.D_critical[-1] + self.epsilon # fill up the last guy

        # increment time (Wow, so much power)
        self.T += self.delta_t
        self.Iteration += 1

        # store this step's data
        self.V_data[current_index] = self.V
        self.X_data[current_index] = self.X
        self.D_data[current_index] = self.D
        self.T_data[current_index] = self.T
        

    def get_data(self, verbose = False):
        if verbose:
            print('-'*10)
            print(f"T: {self.T}")
            print(f"X: {self.X}")
            print(f"V: {self.V}")
            print(f"D: {self.D}")
        return self.X_data, self.V_data, self.D_data, self.T_data
    
    def write_data(self,path):
        """
        Parameters:
        - path: The directory path where the data will be saved.
        """
        os.makedirs(path, exist_ok=True)  # Ensure directory exists
        
        # Save each data array as a .npy file
        np.save(os.path.join(path, "X_data.npy"), self.X_data)
        np.save(os.path.join(path, "V_data.npy"), self.V_data)
        np.save(os.path.join(path, "D_data.npy"), self.D_data)
        np.save(os.path.join(path, "T_data.npy"), self.T_data)

        
def read_data(path):
    """
    Parameters:
    - path: The directory path where the data is stored.
    Returns:
    - X_data, V_data, D_data, T_data: The data arrays loaded from .npy files.
    """
    # Load each .npy file
    X_data = np.load(os.path.join(path, "X_data.npy"))
    V_data = np.load(os.path.join(path, "V_data.npy"))
    D_data = np.load(os.path.join(path, "D_data.npy"))
    T_data = np.load(os.path.join(path, "T_data.npy"))
    
    return X_data, V_data, D_data, T_data

def plot_lines(data, time, x_label='', y_label='', title='', line_labels=None, cmap='tab10'):
    """
    Plots each car's data as a separate line.
    
    Parameters:
    - data: 2D numpy array, shape (n_timesteps, n_cars), the data to be plotted.
    - time: 1D numpy array, shape (n_timesteps,), the time points corresponding to the rows of data.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title for the plot.
    - line_labels: List of labels for each car (default: use index numbers).
    - cmap: Colormap for the lines (default: 'tab10').
    """
    n_timesteps, n_cars = data.shape
    if line_labels is None:
        line_labels = [f"Car {i}" for i in range(n_cars)]

    # Set up the colormap
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_cars))
    
    plt.figure(figsize=(16, 9))

    # Plot each car's data as a line
    for i in range(n_cars):
        plt.plot(time, data[:, i], label=line_labels[i], color=colors[i])

    # Add labels, title, and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Position legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cars")
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust layout to fit legend
    
    # Show plot
    plt.show()



def plot_heatmap(M, x_label='x', y_label='y', title='heatmap', cmap='binary'):
    """
    Plots a heatmap with a slim color bar on the right and proportional aspect ratio.
    
    Parameters:
    - M: 2D array-like data to plot as a heatmap.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the heatmap.
    - cmap: Colormap for the heatmap (default: 'binary').
    """
    # Dimensions of the data
    n_rows, n_cols = M.shape
    
    # Set up figure with fixed aspect ratio for the data
    aspect_ratio = n_cols  # Proportional aspect ratio
    figsize = (16, 9)  # Dynamically scale figure size
    
    plt.figure(figsize=figsize)
    
    # Plot the heatmap
    im = plt.imshow(M, cmap=cmap, aspect='auto')  # Aspect ratio = n_rows
    
    # Add a slim color bar
    cbar = plt.colorbar(im, fraction=0.02, pad=0.04)  # Slim color bar
    cbar.set_label("Value", rotation=270, labelpad=15)
    
    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Adjust layout to fit color bar and labels
    plt.tight_layout()
    
    # Show plot
    plt.show()

def gaussian_dip(t,Max,Min,Mu,Sigma):
    # Mu for where it dips, Max is where its flats, Min is how low the Dip is, Sigma makes it flat.
    # Min between 0 and Max
    return Max - (Max - Min)*torch.exp(-((t - Mu)**2)/(2*Sigma**2))

def experiment():
    N = 15000 # 100 simulation steps
    N_cars = 10
    sim = simultation(N, N_cars, tau_value = 0.5,D_critical_value = 5, b_value = 0.00)

    # initialize last car's behavior
    sim_time_length = int(N * sim.delta_t)
    t = torch.linspace(0, sim_time_length, N)
    last_v = gaussian_dip(t, sim.V_max_value, 0.5*sim.V_max_value, 10, 3)



    # manipulating internal variables
    sim.last_V = last_v # control the speed of the last car (MUST HAVE)

    # simulation loop
    for i in range(N):
        sim.step()
    sim.write_data("data1")

def visualization():
    # read the data
    X_data, V_data, D_data, T_data = read_data("data1")
    plot_lines(X_data, T_data, x_label = 'time', y_label = 'position (m)', title = 'Position over time')
    plot_lines(V_data, T_data, x_label = 'time', y_label = 'speed (m)', title = 'Speed over time')
    plot_heatmap(V_data, x_label = 'Car index', y_label = 'computation step', title = 'Speed')
    

if __name__ == '__main__':
    experiment()
    visualization()



        

