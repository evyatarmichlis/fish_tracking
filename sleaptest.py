import os
import matplotlib.animation as animation

import h5py
import numpy as np
import pandas as pd

import calculate_speed
import fish_heatmap
import matlab_
import three_d_plots
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import another_try_4
diff = 3
base_line = 8



part_dict= {0 :"eye",1:"up_fin",2:"low_fin",3:"up_tail",4:"low_tail",5:"base_tail",6:"thorax"}






# def calculate_cords(x_left, x_right, y_left):
#     disparity = abs(x_left - x_right)
#     x = base_line * (x_left - cx) / disparity
#     y = base_line * fx * (y_left - fx) / (fx * disparity)
#     z = base_line * fx / disparity
#     return x, abs(y), z

def replace_zeros(arr):
    new_arr = arr.copy()
    for i in range(arr.shape[0]):
        if arr[i, 0] == 0 or arr[i, 1] == 0:
            # Find the indices of the closest non-zero neighbors of the corresponding coordinate
            non_zero_indices = np.where((arr[:, 0] != 0) & (arr[:, 1] != 0))[0]
            j,k = i ,i
            while j < len(arr)-1 and k >=0:
                if  not np.array_equal(arr[j], np.array([0, 0])) and  not np.array_equal(arr[k], np.array([0, 0])) :
                    new_arr[i] = (arr[j] + arr[k]) / 2
                    break
                if np.array_equal(arr[j], np.array([0, 0])):
                    j+=1

                if np.array_equal(arr[k], np.array([0, 0])):
                    k -= 1

    return new_arr


def find_closest_index_from_array(point, coordinates):
    coordinates = np.array(coordinates)
    distances = np.linalg.norm(coordinates.T - point, axis=1)

    closest_index = np.argmin(distances)

    return closest_index

def find_closest_value_index(i, j, locations,node):
    point = np.where(~np.isnan(locations[j,node,0]))[0]
    coordinates = np.where(~np.isnan(locations[i,node,0]))[0]
    x_y = (locations[i, node, 0, point],locations[i, node, 1, point])
    array_x_y = [(locations[i, node, 0, coord], locations[i, node, 1, coord]) for coord in coordinates]
    return coordinates[find_closest_index_from_array(x_y, array_x_y)]
def find_closest_index(i,node,cord ,locations):
    j = i
    k = i
    num_frames = locations.shape[0]
    while True:

        if j >= 0 and j < num_frames:
            not_nan_indices = np.where(~np.isnan(locations[j,node,cord]))[0]
            if not_nan_indices.size == 1:
                return j

        if k >= 0 and k < num_frames:
            not_nan_indices = np.where(~np.isnan(locations[k,node,cord]))[0]
            if not_nan_indices.size == 1:
                return k

        j += 1
        if j < num_frames:
            not_nan_indices = np.where(~np.isnan(locations[j,node,cord]))[0]
            if not_nan_indices.size == 1:
                return j

        k -= 1
        if k >= 0:
            not_nan_indices = np.where(~np.isnan(locations[k]))[0]
            if not_nan_indices.size == 1:
                return k
        if j >= num_frames and k < 0:
            return i


def get_x_y_from_sleap(filename):
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]

        print("===nodes===")
        for i, name in enumerate(node_names):
            print(f"{i}: {name}")
        print()

    frame_count, node_count, _, instance_count = locations.shape

    print("frame count:", frame_count)
    print("node count:", node_count)
    print("instance count:", instance_count)
    print("===filename===")
    print(filename)
    print()

    print("===HDF5 datasets===")
    print(dset_names)
    print()

    print("===locations data shape===")
    # new_locations = np.nan_to_num(new_locations, nan=0)
    new_locations = np.zeros((locations.shape[0], locations.shape[1], locations.shape[2]))
    indices_array = []
    for i in range(locations.shape[0]):
        for j in range(locations.shape[1]):
            for k in range(locations.shape[2]):
                not_nan_indices = np.where(~np.isnan(locations[i, j, k]))[0]
                if not_nan_indices.size == 1:
                    new_locations[i, j, k] = locations[i, j, k, not_nan_indices[0]]
                if len(not_nan_indices)>1 :
                    closest = find_closest_index(i,j,k,locations)
                    new_index = find_closest_value_index(i,closest,locations,j)
                    new_locations[i, j, k] = locations[i, j, k,new_index]

    new_locations = np.nan_to_num(new_locations, nan=-1)
    # new_locations = np.sum(np.nan_to_num(locations, nan=0), axis=3)
    print(new_locations.shape)
    eye = 0
    up_fin = 1
    low_fin = 2
    up_tail = 3
    low_tail = 4
    base_tail = 5
    thorax = 6
    eye_loc = new_locations[:, eye, :]

    up_fin_loc = new_locations[:, up_fin, :]
    low_fin_loc = new_locations[:, low_fin, :]
    up_tail_loc = new_locations[:, up_tail, :]
    low_tail_loc = new_locations[:, low_tail, :]
    base_tail_loc = new_locations[:, base_tail, :]
    thorax_loc = new_locations[:, thorax, :]
    parts = [eye_loc, up_fin_loc, low_fin_loc, up_tail_loc, low_tail_loc, base_tail_loc, thorax_loc]
    # parts = [replace_zeros(p) for p in parts]
    # for p in parts:
    #     fig, ax = plt.subplots()
    #     ax.plot(p[:, 0], p[:, 1], linewidth=2.0)
    #     plt.show()

    return parts


def df_from_part(x_y_rect_vid1, x_y_rect_vid2):
    results = matlab_.triangulate_points(x_y_rect_vid1, x_y_rect_vid2)

    filter_arr = results
    df = pd.DataFrame(filter_arr, columns=["x", "y", "z"])

    x_y_rect_vid1 = np.array(x_y_rect_vid1)
    x_y_rect_vid2 = np.array(x_y_rect_vid2)

    non_zero_rows = (x_y_rect_vid1[:, 0] != 0) & (x_y_rect_vid2[:, 0] != 0)
    filtered_df = df[non_zero_rows]

    return filtered_df


def remove_outliners(results):
    arr = results
    # Create a numpy array with some outliers
    # Calculate the quartiles and IQR
    q1, q3 = np.percentile(arr, [30, 70])
    iqr = q3 - q1
    # Calculate the lower and upper bounds for outliers
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    # Replace outliers with the average of their nearest neighbors
    arr_ = arr.copy()
    for i in range(len(arr)):
        j = i+1
        if j >= len(arr) -1 :
            break
        k = i -1
        if arr[i] < lower_bound or arr[i] > upper_bound:
            if i == 0:
                while arr_[j] < lower_bound or arr_[j] > upper_bound:
                    j+=1
                arr[i] = arr_[j]
            elif i == len(arr) - 1:
                while arr_[k] < lower_bound or arr_[k] > upper_bound:
                    k -= 1
                arr[i] = arr_[k]
            else:
                while  arr_[j] < lower_bound or arr_[j] > upper_bound:
                    if j >= len(arr) - 1:
                        break
                    j += 1
                while arr_[k] < lower_bound or arr_[k] > upper_bound:
                    k -= 1
                arr[i] = np.mean([arr[k], arr[j]])
    return arr


def main(filename_l):
    file_name = filename_l.split("\\")[-1]
    parts_r = get_x_y_from_sleap(filename_l)
    frame_rate = len(parts_r[0])/1200


    # data_transposed = np.transpose(parts_r, (1, 0, 2))

    # Convert the array to a list of numpy arrays
    # data_list = data_transposed.tolist()
    # three_d_plots.two_d_main(   [np.array(d) for d in data_list],True)

    eyes_df = pd.DataFrame(parts_r[0], columns=['x', 'y'])
    res = {"file_name":[],"mean_speed":[],"non_movement":[]}
    heat_maps = []
    for p in np.array_split(eyes_df, 6):
        heat_map = fish_heatmap.create_heatmap(p)
        heat_maps.append(heat_map)
        speed, mean_speed, non_movement = calculate_speed.movement_features_calc(p,frame_rate)
        print(f"mean_speed: {mean_speed}")
        print(f"non_movement: {non_movement}")
        res["file_name"].append(file_name)
        res["mean_speed"].append(mean_speed)
        res["non_movement"].append(non_movement)
    return res,heat_maps



    #
    # # dfs = [df_from_part(pl,pr) for pl,pr in zip(parts_l,parts_r)]
    # for i in range(len(dfs)):
    #     speed, acceleration, angle = calculate_speed.movement_features_calc(dfs[i], frame_rate)
    #     dfs[i]['speed'] = speed
    #     dfs[i]['acceleration'] = acceleration
    #     dfs[i]['angle'] = angle
    #     dfs[i].to_csv(f'results_csvs/{file_name}_{part_dict[i]}.csv', index=False)



def decision_tree_classifier(df):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    X = df[['mean_speed', 'non_movement']]

    label_encoder = LabelEncoder()

    # Fit the LabelEncoder to the label column and transform the labels
    df['Age'] = label_encoder.fit_transform(df['Age'])
    y = df['Age']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    plt.scatter(X_test[y_test == 0]['mean_speed'], X_test[y_test == 0]['non_movement'], color='blue',
                label='young (test)')
    plt.scatter(X_test[y_test == 1]['mean_speed'], X_test[y_test == 1]['non_movement'], color='red', label='old (test)')

    # Plot the decision boundary
    x_min, x_max = df['mean_speed'].min() - 1, df['mean_speed'].max() + 1
    y_min, y_max = df['non_movement'].min() - 1, df['non_movement'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
    # Add labels and legend
    plt.xlabel('Test Data On The Fitted Decision Tree')
    plt.xlabel('mean_speed')
    plt.ylabel('non_movement')
    plt.legend()
    # Show the plot
    plt.show()

def plot_heap_map(cubes,name):
    cubes = cubes[:, ~np.all(cubes == 0, axis=0)]
    cubes = cubes[~np.all(cubes == 0, axis=1), :]
    plt.imshow(cubes.T, origin='upper', cmap='coolwarm', extent=[0, 10, 0, 10])  # Use origin='upper'
    plt.colorbar(label='Count')
    plt.xlabel('Cube X')
    plt.ylabel('Cube Y')
    plt.title(f'Fish Movement Heatmap {name}')
    plt.show()

if __name__ == '__main__':

    path = r"C:\Users\Evyatar\PycharmProjects\pythonProject14\h5_files\young"
    file_list =os.listdir(path)
    results = []
    heat_maps_array_y = []
    for f in file_list:
        res,heat_maps = main(path+"\\"+f)
        results.append(res)
        heat_maps_array_y.append(heat_maps)
    df_y = pd.concat([pd.DataFrame(d) for d in results], ignore_index=True)
    df_y["Age"] = "young"
    average_cube = np.mean(np.mean(heat_maps_array_y, axis=0), axis=0)
    plot_heap_map(average_cube,"Young")
    path = r"C:\Users\Evyatar\PycharmProjects\pythonProject14\h5_files\old"
    file_list =os.listdir(path)
    results = []
    heat_maps_array_o = []
    for f in file_list:
       print(f)
       res,heat_maps = main(path+"\\"+f)
       results.append(res)
       heat_maps_array_o.append(heat_maps)

    df_o = pd.concat([pd.DataFrame(d) for d in results], ignore_index=True)
    df_o["Age"] = "old"

    average_cube = np.mean(np.mean(heat_maps_array_o, axis=0), axis=0)
    plot_heap_map(average_cube,"Old")

    df = pd.concat([df_y,df_o])


    df.to_csv("results.csv")
    from scipy import stats

    t_statistic_1, p_value_1 = stats.ttest_ind(df_y["mean_speed"], df_o["mean_speed"])
    t_statistic_2, p_value_2 = stats.ttest_ind(df_y["non_movement"], df_o["non_movement"])

    custom_reversed_palette = sns.color_palette("Set1")

    ax = sns.violinplot(x='Age', y='mean_speed', data=df, order=["old", "young"], palette=custom_reversed_palette)
    plt.setp(ax.collections, alpha=.7)
    plt.title('Mean Speed Comparison')
    plt.xlabel('Age')
    plt.ylabel('Mean speed')

    plt.show()

    ax = sns.violinplot(x='Age', y='non_movement', data=df, order=["old", "young"], palette=custom_reversed_palette)
    plt.setp(ax.collections, alpha=.7)
    plt.title('Non Movement Comparison')
    plt.xlabel('Age')
    plt.ylabel('Non Movement frames')

    plt.show()


    decision_tree_classifier(df)



    #
    # path = r"C:\Users\Evyatar\PycharmProjects\pythonProject14\h5_files\pairs"
    # file_list =os.listdir(path)
    # pairs =[]
    # for i in range(0, len(file_list), 2):
    #     # Get the current element and the next element
    #     current = file_list[i]
    #     next_element = file_list[i + 1]
    #
    #     # Create a tuple with the pair and add it to the list of pairs
    #     pair = (current, next_element)
    #     pairs.append(pair)
    # for pair in pairs[-1:]:
    #     main(path+"\\"+pair[0],path+"\\"+pair[1])
    #
    #
    #
    #
