import cv2
import os

path = r"C:\Users\Evyatar\PycharmProjects\pythonProject14\samples"
# file_list = os.listdir(path)
file_list = ["old-2023052302-l.mp4"]
for f in file_list:
    print(f)

    # cap = cv2.VideoCapture( path +"\\"+ f)
    cap = cv2.VideoCapture( r"C:\Users\Evyatar\PycharmProjects\pythonProject14\samples\old-2023052302-l.mp4")

    file_name = os.path.splitext(os.path.basename( f))[0]

    # Get the frame rate and size of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize variables to store the total sum and number of frames
    total_sum = None
    num_frames = 0

    # Define the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_name = r'C:\Users\Evyatar\PycharmProjects\pythonProject14\background\background_removal_' + file_name + '.mp4'
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height), isColor=True)

    # Loop through all the frames in the video
    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()

        # Stop the loop if we have reached the end of the video
        if not ret:
            break

        # Add the frame to the total sum
        if total_sum is None:
            total_sum = frame.astype('float')
        else:
            total_sum += frame.astype('float')

        # Increment the frame counter
        num_frames += 1
        # Calculate the average frame
        avg_frame = (total_sum / num_frames).astype('uint8')

        # Subtract the average frame from the current frame to obtain the background-subtracted frame
        # bg_subtracted = cv2.absdiff(frame, avg_frame)

        output_frame = frame.copy()

        bg_subtracted = cv2.absdiff(output_frame, avg_frame, output_frame)

        hsv = cv2.cvtColor(bg_subtracted, cv2.COLOR_BGR2HSV)

        hsv[:, :, 2] = hsv[:, :, 2] * 2

        output_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Display the background-subtracted frame
        cv2.imshow('Background Subtracted', output_frame)

        # Write the background-subtracted output frame to the output video file
        out.write(output_frame)

        # Wait for a key press and exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the input video and output video writer
    cap.release()
    out.release()

    # Close any open windows
    cv2.destroyAllWindows()