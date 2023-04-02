"""Basic Video Processing methods."""
import os
import cv2


# Replace ID1 and ID2 with your IDs.
ID1 = '206299463'
ID2 = '312497084'

INPUT_VIDEO = 'atrium.avi'
GRAYSCALE_VIDEO = f'{ID1}_{ID2}_atrium_grayscale.avi'
BLACK_AND_WHITE_VIDEO = f'{ID1}_{ID2}_atrium_black_and_white.avi'
SOBEL_VIDEO = f'{ID1}_{ID2}_atrium_sobel.avi'


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.
    Args:
        capture: cv2.VideoCapture object. The input video's VideoCapture.
    Returns:
        parameters: dict. A dictionary of parameters names to their values.
    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    parameters = {"fourcc": fourcc, "fps": fps, "height": height, "width": width}
    return parameters


def convert_video_to_grayscale(input_video_path: str,
                               output_video_path: str) -> None:
    """Convert the video in the input path to grayscale.

    Use VideoCapture from OpenCV to open the video and read its
    parameters using the capture's get method.
    Open an output video using OpenCV's VideoWriter.
    Iterate over the frames. For each frame, convert it to gray scale,
    and save the frame to the new video.
    Make sure to close all relevant captures and to destroy all windows.

    Args:
        input_video_path: str. Path to input video.
        output_video_path: str. Path to output video.

    Additional References:
    (1) What are fourcc parameters:
    https://docs.microsoft.com/en-us/windows/win32/medfound/video-fourccs
    """
    # open input video
    vc = cv2.VideoCapture(input_video_path)
    parameters = get_video_parameters(vc)

    # open output video as grayscale
    vw = cv2.VideoWriter(output_video_path, 
                         fourcc=parameters['fourcc'], 
                         fps=parameters['fps'], 
                         frameSize=(parameters['width'], parameters['height']),
                         isColor=False)

    # iterate over frames, convert each frame to grayscale and write it to the output video
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vw.write(frame)
    
    # close all captures and windows
    vc.release()
    vw.release()
    cv2.destroyAllWindows()

def convert_video_to_black_and_white(input_video_path: str,
                                     output_video_path: str) -> None:
    """Convert the video in the input path to black and white.

    Use VideoCapture from OpenCV to open the video and read its
    parameters using the capture's get method.
    Open an output video using OpenCV's VideoWriter.
    Iterate over the frames. For each frame, first convert it to gray scale,
    then use OpenCV's THRESH_OTSU to slice the gray color values to
    black (0) and white (1) and finally convert the frame format back to RGB.
    Save the frame to the new video.
    Make sure to close all relevant captures and to destroy all windows.

    Args:
        input_video_path: str. Path to input video.
        output_video_path: str. Path to output video.

    Additional References:
    (1) What are fourcc parameters:
    https://docs.microsoft.com/en-us/windows/win32/medfound/video-fourccs

    """
    # open input video
    vc = cv2.VideoCapture(input_video_path)
    parameters = get_video_parameters(vc)

    # open output video
    vw = cv2.VideoWriter(output_video_path,
                            fourcc=parameters['fourcc'],
                            fps=parameters['fps'],
                            frameSize=(parameters['width'], parameters['height']),
                            isColor=True)
    
    # iterate over frames, convert each frame to black and white and write it to the output video
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        vw.write(frame)

    # close all captures and windows
    vc.release()
    vw.release()
    cv2.destroyAllWindows()


def convert_video_to_sobel(input_video_path: str,
                           output_video_path: str) -> None:
    """Convert the video in the input path to sobel map.

    Use VideoCapture from OpenCV to open the video and read its
    parameters using the capture's get method.
    Open an output video using OpenCV's VideoWriter.
    Iterate over the frames. For each frame, first convert it to gray scale,
    then use OpenCV's THRESH_OTSU to slice the gray color values to
    black (0) and white (1) and finally convert the frame format back to RGB.
    Save the frame to the new video.
    Make sure to close all relevant captures and to destroy all windows.

    Args:
        input_video_path: str. Path to input video.
        output_video_path: str. Path to output video.

    Additional References:
    (1) What are fourcc parameters:
    https://docs.microsoft.com/en-us/windows/win32/medfound/video-fourccs

    """
    # open input video
    vc = cv2.VideoCapture(input_video_path)
    parameters = get_video_parameters(vc)

    # open output video
    vw = cv2.VideoWriter(output_video_path,
                            fourcc=parameters['fourcc'],
                            fps=parameters['fps'],
                            frameSize=(parameters['width'], parameters['height']),
                            isColor=True)
    
    # iterate over frames, convert each frame to sobel map and write it to the output video
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.Sobel(frame, ddepth=-1, dx=1, dy=1, ksize=5)
        frame = cv2.convertScaleAbs(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        vw.write(frame)

    # close all captures and windows
    vc.release()
    vw.release()
    cv2.destroyAllWindows()


def main():
    convert_video_to_grayscale(INPUT_VIDEO, GRAYSCALE_VIDEO)
    convert_video_to_black_and_white(INPUT_VIDEO, BLACK_AND_WHITE_VIDEO)
    convert_video_to_sobel(INPUT_VIDEO, SOBEL_VIDEO)


if __name__ == "__main__":
    main()
