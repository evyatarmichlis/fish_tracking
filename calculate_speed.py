import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt


def plot_sevgol(x, sev_x, k=31):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    # Choose a Seaborn color palette
    palette = sns.color_palette("Set2")  # Using the 'husl' palette with 2 colors
    plt.plot(x[:200], label='x value before sevgol', color=palette[1])
    plt.plot(sev_x[:200], label=f'x value after sevgol with window = {k}', color=palette[0])
    plt.xlabel("Frames")
    plt.ylabel("x value")
    plt.title("Sevgol Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_mad(x, mad_x):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    # Choose a Seaborn color palette
    palette = sns.color_palette("Set2")  # Using the 'husl' palette with 2 colors
    plt.plot(x[:100], label='x locations before MAD and cubic interpolation', color=palette[0])
    plt.plot(mad_x[:100], label=f'x locations after MAD and cubic interpolation', color=palette[1])
    plt.xlabel("Frames")
    plt.ylabel("x value")
    plt.title("MAD and cubic interpolation")
    plt.legend()
    plt.tight_layout()
    plt.show()

def replace_anomalies(data, anomalies):
    anomaly_indices = np.nonzero(anomalies)[0]

    # Find the indices of the first and last elements in each run of consecutive anomalies
    run_starts, = np.where(np.diff(anomaly_indices) > 1)
    run_ends = np.concatenate([anomaly_indices[run_starts[1:]] - 1, [anomaly_indices[-1]]])
    run_starts = np.concatenate([[anomaly_indices[0]], anomaly_indices[run_starts + 1]])

    # Replace each run of consecutive anomalies with linearly interpolated values
    for start, end in zip(run_starts, run_ends):
        if start == 0:
            start_value = data[start]
        else:
            start_value = (data[start - 1] + data[start]) / 2

        if end == len(data) - 1:
            end_value = data[end]
        else:
            end_value = (data[end] + data[end + 1]) / 2

        num_values = end - start + 1
        new_values = np.linspace(start_value, end_value, num=num_values)

        # Use np.where to generate a boolean mask of the indices where the anomalies occur
        anomaly_mask = np.zeros_like(data, dtype=bool)
        anomaly_mask[start:end + 1] = True

        # Use the mask to assign the new values to the appropriate locations in the original data array
        data = np.where(anomaly_mask, new_values, data)

    return data


def replace_outliers(data, window_percentage = 10,is_array = True):
    if not is_array:
        data = data.to_numpy()
    def get_bands(data):
        MAD = np.median(np.abs(data - np.median(data)))
        return (np.median(data) + 3 * MAD, np.median(data) - 3 * MAD)
    k = int(len(data) * (window_percentage / 2 / 100))
    N = len(data)
    bands = [get_bands(data[range(0 if i - k < 0 else i - k, i + k if i + k < N else N)]) for i in range(0, N)]
    upper, lower = zip(*bands)
    anomalies = (data > upper) | (data < lower)

    x = np.arange(len(data))
    f = interp1d(x[~anomalies], data[~anomalies], kind='linear')
    try:
        data = np.where((anomalies & (data != 0)), f(x), data)
    except:
        data = data

    return anomalies, data


def polynomial_interpolation_on_zeros(data):
    from scipy.interpolate import CubicSpline, interp1d

    zero_indices = np.where(data == 0)[0]
    non_zero_indices = np.nonzero(data)[0]
    interpolated_data = np.copy(data)

    for index in zero_indices:
        nearest_indices = np.abs(non_zero_indices - index).argsort()[:2]
        nearest_x = non_zero_indices[nearest_indices]
        nearest_y = data[nearest_x]
        sorted_indices = np.argsort(nearest_x)
        sorted_x = nearest_x[sorted_indices]
        sorted_y = nearest_y[sorted_indices]
        interp_func = CubicSpline(sorted_x, sorted_y)
        value = interp_func(index)
        #
        # if not (sorted_y[0] - 50 < value.tolist() < sorted_y[1] + 50):
        #     alternative_interp_func = interp1d(sorted_x, sorted_y, kind='linear', fill_value="extrapolate")
        #     value = alternative_interp_func(index)

        interpolated_data[index] = value

    return interpolated_data


def plot_magnitude_of_movement(x, y, speed):
    sns.set(style="whitegrid")

    x, y, speed = x[:1000], y[:1000], speed[:1000]
    fig = plt.figure(figsize=(15, 6))
    vmin = 0
    vmax = 10

    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, c=speed, s=4, cmap="vlag", vmin=vmin, vmax=vmax)  # Use a Seaborn colormap

    ax.set_xlim(500, 1920)
    ax.set_ylim(500, 1080)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Eye Tracks Colored by Magnitude of Movement Speed')
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    cbar = plt.colorbar(sc)
    cbar.set_label('Magnitude of Movement Speed')
    plt.tight_layout()
    plt.show()



def movement_features_calc(tracking_data,frame_rate,plot = False):

    anomalies_x,x_remove_outlier = replace_outliers(tracking_data["x"].values)
    anomalies_y,y_remove_outlier = replace_outliers(tracking_data["y"].values)
    x_inter = polynomial_interpolation_on_zeros(x_remove_outlier)
    y_inter = polynomial_interpolation_on_zeros(y_remove_outlier)
    valid_indices = np.logical_and(x_inter >= 0, y_inter >= 0)
    x_inter_filtered = x_inter[valid_indices]
    y_inter_filtered = y_inter[valid_indices]    # Calculate the Euclidean distance between consecutive frames
    x_smooth,dx = savgol_filter(x_inter_filtered, window_length=31, polyorder=3),savgol_filter(x_inter_filtered, window_length=31, polyorder=3,deriv=1) #ad #add div = 1
    y_smooth,dy = savgol_filter(y_inter_filtered, window_length=31, polyorder=3),savgol_filter(y_inter_filtered, window_length=31, polyorder=3,deriv=1) #ad #add div = 1
    distance = np.sqrt(dx**2 + dy**2)
    time_elapsed = np.ones_like(distance) * (1/frame_rate)

    # Calculate the speed of the fish by dividing the distance by the time elapsed
    speed = distance / time_elapsed
    non_movement = len(speed[speed<50])
    # t =  71.06
    non_movement_percentile = len(speed[speed<50])
    percentile_10 = np.percentile(speed, 10)
    print(f"percentile_10  {percentile_10}")
    # Count the number of frames below the 10th percentile
    non_movement = len(speed[speed < percentile_10])
    dx_2 = savgol_filter(x_inter_filtered, window_length=31, polyorder=3,deriv=2)
    dy_2 = savgol_filter(y_inter_filtered, window_length=31, polyorder=3,deriv=2)
    ax = dx_2 / time_elapsed ** 2
    ay = dy_2 / time_elapsed ** 2
    acceleration = np.sqrt(ax**2 + ay**2)
    # Calculate the mean and median speed of the fish
    mean_speed = np.mean(speed)
    if plot:
        plot_magnitude_of_movement(x_smooth,y_smooth,distance)
        plot_sevgol(x_inter_filtered,x_smooth,31)
        plot_mad(tracking_data["x"].values,x_inter)

    return speed, mean_speed,non_movement




def movement_features_calc_3d(tracking_data,frame_rate=30):
    # Add outliers
    anomalies_x, x_remove_outlier = replace_outliers(tracking_data['x'],is_array=False)
    anomalies_y, y_remove_outlier = replace_outliers(tracking_data['y'],is_array=False)
    anomalies_z, z_remove_outlier = replace_outliers(tracking_data['z'],is_array=False)

    # Calculate the Euclidean distance between consecutive frames
    x_smooth, dx = savgol_filter(x_remove_outlier, window_length=51, polyorder=3), savgol_filter(x_remove_outlier,
                                                                                                 window_length=51,
                                                                                                 polyorder=3, deriv=1)
    y_smooth, dy = savgol_filter(y_remove_outlier, window_length=51, polyorder=3), savgol_filter(y_remove_outlier,
                                                                                                 window_length=51,
                                                                                                 polyorder=3, deriv=1)
    z_smooth, dz = savgol_filter(z_remove_outlier, window_length=51, polyorder=3), savgol_filter(z_remove_outlier,
                                                                                                 window_length=51,
                                                                                                 polyorder=3, deriv=1)
    distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    time_elapsed = np.ones_like(distance) * (1 / frame_rate)

    # Calculate the speed by dividing the distance by the time elapsed
    speed = distance / time_elapsed
    speed = speed[np.where(x_remove_outlier != 0)]

    dx_2 = savgol_filter(x_remove_outlier, window_length=31, polyorder=3, deriv=2)
    dy_2 = savgol_filter(y_remove_outlier, window_length=31, polyorder=3, deriv=2)
    dz_2 = savgol_filter(z_remove_outlier, window_length=31, polyorder=3, deriv=2)
    ax = dx_2 / time_elapsed ** 2
    ay = dy_2 / time_elapsed ** 2
    az = dz_2 / time_elapsed ** 2
    ax = ax[np.where(x_remove_outlier != 0)]
    ay = ay[np.where(x_remove_outlier != 0)]
    az = az[np.where(x_remove_outlier != 0)]
    acceleration = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    # Calculate the mean and median speed
    mean_speed = np.mean(speed)
    median_speed = np.median(speed)

    # Calculate the mean and median acceleration
    mean_acceleration = np.mean(acceleration)

    # Calculate the mean and median direction of movement over time
    dx = tracking_data['x'].values - tracking_data['x'].values[0]
    dy = tracking_data['y'].values - tracking_data['y'].values[0]
    dz = tracking_data['z'].values - tracking_data['z'].values[0]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    angle = angle % 360
    mean_angle = np.mean(angle)
    median_angle = np.median(angle)

    # Calculate the number of frames with no movement
    no_movement_frames = np.size(np.where(x_remove_outlier == 0))

    return speed, acceleration, angle

#
#
# import pandas as pd
# import numpy as np
#
#
# def movement_features_calc(tracking_data):
#     # Calculate the Euclidean distance between consecutive frames
#     tracking_data['distance'] = ((tracking_data['x'] - tracking_data['x'].shift(1)) ** 2 +
#                                  (tracking_data['y'] - tracking_data['y'].shift(1)) ** 2) ** 0.5
#
#     # Calculate the time elapsed between consecutive frames (assuming a frame rate of 30 frames per second)
#     tracking_data['time_elapsed'] = 1 / 30
#
#     # Calculate the speed of the fish by dividing the distance by the time elapsed
#     tracking_data['speed'] = tracking_data['distance'] / tracking_data['time_elapsed']
#
#     # Calculate the change in speed between consecutive frames
#     tracking_data['speed_diff'] = tracking_data['speed'] - tracking_data['speed'].shift(1)
#
#     # Calculate the acceleration of the fish by dividing the change in speed by the time elapsed
#     tracking_data['acceleration'] = tracking_data['speed_diff'] / tracking_data['time_elapsed']
#
#     # Calculate the mean and median speed of the fish
#     mean_speed = tracking_data['speed'].mean()
#     median_speed = tracking_data['speed'].median()
#
#     # Calculate the mean and median acceleration of the fish
#     mean_acceleration = tracking_data['acceleration'].mean()
#     median_acceleration = tracking_data['acceleration'].median()
#
#     # Calculate the angle between the fish's current position and its position in the previous frame
#     dx = tracking_data['x'] - tracking_data['x'].shift(1)
#     dy = tracking_data['y'] - tracking_data['y'].shift(1)
#     angle = np.arctan2(dy, dx) * 180 / np.pi
#     tracking_data['angle'] = angle % 360
#
#     # Calculate the mean and median angle of the fish
#     mean_angle = tracking_data['angle'].mean()
#     median_angle = tracking_data['angle'].median()
#
#     return tracking_data['speed'], tracking_data['acceleration'], tracking_data[
#         'angle'], median_speed, mean_speed, median_acceleration, mean_acceleration, mean_angle, median_angle
#
