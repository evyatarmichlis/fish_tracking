# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def two_d_animate_scatters(iteration, data, scatters):
    labels = ["eye", "up_fin", "low_fin", "up_tail", "low_tail", "base_tail", "thorax"]

    for i in range(data[0].shape[0]):
        label = labels[i]
        offset = data[iteration][i*2:i*2+2]
        scatters[label].set_offsets(offset)
    return scatters.values()

def two_d_main(data,path = "output_2d.mp4" ):
    """
    Creates the 2D figure and animates it with the input data.
    Args:
        data (list): List of the data positions at each iteration.
        save (bool): Whether to save the recording of the animation. (Default to False).
    """

    # Attaching 2D axis to the figure
    fig, ax = plt.subplots()

    labels = ["eye", "up_fin", "low_fin", "up_tail", "low_tail", "base_tail", "thorax"]
    # Initialize scatters
    scatters = {}
    for i in range(data[0].shape[0]):
        label = labels[i]
        if label not in scatters:
            scatters[label] = ax.scatter(data[0][i, 0], data[0][i, 1], label=label)
        else:
            scatters[label].set_offsets(data[0][i])

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim([0, 1920])
    ax.set_xlabel('X')

    ax.set_ylim([0, 1080])
    ax.set_ylabel('Y')

    ax.set_title('2D Fish Coordinates')

    def two_d_animate_scatters(frame, data, scatters):
        for i in range(data[frame].shape[0]):
            label = labels[i]
            if label in scatters:
                scatters[label].set_offsets(data[frame][i])
            else:
                scatters[label] = ax.scatter(data[frame][i, 0], data[frame][i, 1], label=label)

    ani = animation.FuncAnimation(fig, two_d_animate_scatters, iterations, fargs=(data, scatters),
                                  interval=50, blit=False, repeat=True)

    ax.legend()


    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    ani.save('2d-scattered-animated-l.mp4', writer=writer)

    plt.show()


def generate_data(nbr_iterations, nbr_elements):
    """
    Generates dummy data.
    The elements will be assigned random initial positions and speed.
    Args:
        nbr_iterations (int): Number of iterations data needs to be generated for.
        nbr_elements (int): Number of elements (or points) that will move.
    Returns:
        list: list of positions of elements. (Iterations x (# Elements x Dimensions))
    """
    dims = (3,1)

    # Random initial positions.
    gaussian_mean = np.zeros(dims)
    gaussian_std = np.ones(dims)
    start_positions = np.array(list(map(np.random.normal, gaussian_mean, gaussian_std, [nbr_elements] * dims[0]))).T

    # Random speed
    start_speed = np.array(list(map(np.random.normal, gaussian_mean, gaussian_std, [nbr_elements] * dims[0]))).T

    # Computing trajectory
    data = [start_positions]
    for iteration in range(nbr_iterations):
        previous_positions = data[-1]
        new_positions = previous_positions + start_speed
        data.append(new_positions)

    return data

def animate_scatters(iteration, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
    return scatters

def main(data, save=False):
    """
    Creates the 3D figure and animates it with the input data.
    Args:
        data (list): List of the data positions at each iteration.
        save (bool): Whether to save the recording of the animation. (Default to False).
    """

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    labels = ["eye", "up_fin", "low_fin", "up_tail", "low_tail", "base_tail", "thorax"]
    # Initialize scatters
    scatters = [ax.scatter(data[0][i, 0:1], data[0][i, 1:2], data[0][i, 2:], label=labels[i]) for i in
                range(data[0].shape[0])]

    # scatters = [ ax.scatter(data[0][i,0:1], data[0][i,0:1], data[0][i,1:2]) for i in range(data[0].shape[0]) ]

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim3d([-15, 15])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 11])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0, 5])
    ax.set_zlabel('Z')
    ax.set_title('3D  Fish Coordinates')
    ax.view_init(elev=100, azim=-90)
    ax.legend()

    # Provide starting angle for the view.

    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                       interval=50, blit=False, repeat=True)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save('3d-scatted-animated-eye.mp4', writer=writer)

    plt.show()


# data = generate_data(100,7)
# main(data, save=True)