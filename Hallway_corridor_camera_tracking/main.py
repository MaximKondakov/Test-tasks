import math
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from _collections import deque
import argparse


def get_args():
    """
    command for usage console: python main.py --video_name video/004.avi --n_frames 10 --get_coord no
    :return: Parsing input arguments
    """
    parser = argparse.ArgumentParser(description='Parser input arguments')
    parser.add_argument('-v', '--video_name', default='video/004.avi', type=str, action="store",
                        help="Path to video")
    parser.add_argument('-n', '--n_frames', default=10, type=int, action="store",
                        help="Number of tracer frames")
    parser.add_argument('-g', '--get_coord', default='no', type=str, action="store",
                        help="Select new coordinates from image")

    args = parser.parse_args()
    return args


def global_position(x1, y1, x2, y2, homo_matrix):
    """
    based on the homography matrix return new coordinates target
    :param x1: coordinate bounding box[0]
    :param y1: coordinate bounding box[1]
    :param x2: coordinate bounding box[2]
    :param y2: coordinate bounding box[3]
    :param homo_matrix: homography matrix to get new coordinates
    :return: list of [x_new, y_new] position
    """
    x_position = x2 / 2 + x1 / 2
    y_position = y2
    image_position = [x_position, y_position, float(1)]
    gp_temp = np.matmul(homo_matrix, image_position)
    gp_position = [int(gp_temp[0] / gp_temp[2]), int(gp_temp[1] / gp_temp[2])]
    return gp_position


def get_homography_matrix(input):
    """
    perform new output coordinates and compute homography matrix(input, output)
    :param input: list of points of floor: [TopLeft, TopRight, BottomRight, BottomLeft]
    :return: homography matrix for selected points
    """
    # get euclidian norm selected points floor
    width_0_1 = round(math.hypot(input[0, 0] - input[1, 0], input[0, 1] - input[1, 1]))
    height_0_3 = round(math.hypot(input[0, 0] - input[3, 0], input[0, 1] - input[3, 1]))
    width_3_2 = round(math.hypot(input[3, 0] - input[2, 0], input[3, 1] - input[2, 1]))
    height_1_2 = round(math.hypot(input[1, 0] - input[2, 0], input[1, 1] - input[2, 1]))
    width = np.minimum(width_0_1, width_3_2)
    height = np.minimum(height_0_3, height_1_2)
    # set upper left coordinates for output bird eye floor
    x = input[0, 0]
    y = input[0, 1]
    # specify output coordinates  in order TopLeft, TopRight, BottomRight, BottomLeft as x,
    output = np.float32([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])
    # compute Homography matrix
    h, status = cv2.findHomography(input, output)
    return h, status


def my_video_tracker(n_frames, video_name, selected_points):
    model = YOLO('yolov8n.pt')  # load an detection model
    object_tracker = DeepSort(max_age=5,  # Number of missed before a track is deleted
                              n_init=2,  # Number of frames that a track remains in initialization phase
                              nms_max_overlap=1.0,  # Suppression threshold
                              max_cosine_distance=0.3,  # Threshold for cosine distance
                              nn_budget=None,  # Maximum size of the appearance descriptors
                              override_track_class=None,  # Giving this will override default Track class
                              embedder="mobilenet",  # Feature extractor
                              half=True,  # Half precision for deep embedder
                              bgr=True,  # Expected to be BGR or not
                              embedder_gpu=True,  # Embedder uses gpu
                              embedder_model_name=None,  # Only used when embedder=='torchreid'
                              embedder_wts=None,  # Specification of path to embedded model weights
                              polygon=False,  # Detections are polygons
                              today=None  # Today's date, for naming of tracks
                              )

    # incorrect homography matrix for video_4
    # homo_matrix = np.array([[-3.69794989e-03, -1.84238981e-03,  1.74863914e+00],
    #                         [-7.49713432e-04,  1.04485645e-02, -4.18152922e+00],
    #                         [-1.15416774e-05,  5.49002935e-04, -8.96759665e-03]])
    # read video
    cap = cv2.VideoCapture(video_name)
    # detection threshold
    detection_threshold = 0.5
    color = (0, 0, 255)
    thickness = int(3)
    # deque for N frames tracer
    pts = [deque(maxlen=n_frames) for _ in range(100)]
    # list for all paths
    bird_eye_pts = [[] for _ in range(100)]

    # calculate new homography matrix for points of floor
    h, status = get_homography_matrix(selected_points)

    while cap.isOpened():
        ret, img = cap.read()
        # compute bird_eye image
        img_warp = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))
        # model(YOLO) predict
        results = model(img)
        # list for sample which detections_confidence > threshold_confidence
        detections = []
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold:
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], score, class_id))

        # expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        tracks = object_tracker.update_tracks(detections, frame=img)
        # track
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = ltrb

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

            coord_track = [int(((bbox[0]) + (bbox[2])) / 2), int((bbox[3]))]
            # for each track id append in all path and n_frames path
            track_id = int(track_id)
            pts[track_id].append(coord_track)
            bird_eye_pts[track_id].append(global_position(bbox[0], bbox[1], bbox[2], bbox[3], h))

            # all frames track in the bird_eye image
            for i in range(1, len(bird_eye_pts[track_id])):
                start_point = bird_eye_pts[track_id][i - 1]
                end_point = bird_eye_pts[track_id][i]
                cv2.line(img_warp, start_point, end_point, color, thickness)

            # n_frames track  in the image
            for j in range(1, len(pts[track_id])):
                start_point = pts[track_id][j - 1]
                end_point = pts[track_id][j]
                cv2.line(img, start_point, end_point, color, thickness)

        # concatenate image Horizontally
        horizontal_img = np.concatenate((img, img_warp), axis=1)
        cv2.imshow('img', horizontal_img)
        # close by Key == q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release and destroy all windows before termination
    cap.release()
    cv2.destroyAllWindows()


def click_event(event, x, y, flags, params):
    """click event for select points on the image"""
    # points left mouse clicks for the floor: TopLeft, TopRight, BottomRight, BottomLeft as x,
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        print(f'({x},{y})')
        # draw coordinates on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (0, 0, 255), 2)
        # draw point on the image
        cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
        cv2.imshow('image', img)
        # list of coordinates
        points_list.append([x, y])
        # select 4 point and close the window
        if len(points_list) == 4:
            # close the window
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # argparse
    args = get_args()
    if args.get_coord == 'yes':
        # list of selected coordinates
        points_list = []
        # read video
        cap = cv2.VideoCapture(args.video_name)
        ret, img = cap.read(0)
        # displaying the image
        cv2.imshow('image', img)
        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', click_event)
        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        print(f'selected points == {points_list}')
        selected_points = np.float32(points_list)

    elif args.get_coord == 'no':
        # points of floor: TopLeft, TopRight, BottomRight, BottomLeft as x,
        # my new points for the video/004.avi
        selected_points = np.float32([[390, 92], [458, 95], [492, 475], [0, 475]])

    my_video_tracker(args.n_frames, args.video_name, selected_points)
