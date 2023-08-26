import numpy as np

def create_heatmap(df):
    frame_size = (1920, 1080)
    cube_size = (frame_size[0] // 10, frame_size[1] // 10)
    cubes = np.zeros((10, 10))
    for _, row in df.iterrows():
        x, y = row['x'], row['y']
        if x!=0 and y!=0 :
            cube_x = min(int(x // cube_size[0]),9)
            cube_y = min(int(y // cube_size[1]),9)
            cubes[cube_x, cube_y] += 1
    cubes = cubes[:-1, :]

    return cubes
