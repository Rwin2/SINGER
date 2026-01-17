import os
import numpy as np

from PIL import Image
from io import BytesIO
from typing import Dict, List, Union

import sousvide.synthesize.synthesize_helper as sh

import cv2

def save_images_as_video(frames, video_path, fps=20):
    """ Writes 'frames' to a single video file. """
    if not frames:
        return
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for f in frames:
        bgr_frame = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    writer.release()

def load_video_frames(video_path,image_data_set) -> List[np.ndarray]:
    """
    Returns frames [start_idx ... end_idx] from 'video_path'.
    """

    start_idx = image_data_set["start_id"]
    end_idx = image_data_set["end_id"]

    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Move to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    current_idx = start_idx
    while current_idx <= end_idx:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames or error
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        current_idx += 1

    cap.release()

    frames_np = np.stack(frames, axis=0)# Stack into a single array of shape (N, H, W, 3)    
    return frames_np

def decompress_data(image_dict:Dict[str,Union[str,np.ndarray]]) -> Dict[str,Union[str,np.ndarray]]:
    """
    We apply a compression if the images are too large to be saved in the .pt file. This function
    decompresses the images back to their original form.
    """
    assert 'images' in image_dict, "No images found in data dictionary"
    # Check if the image are processed or not. We do this by checking the array
    # order. If the order is (N, C, H, W) then the images are processed. If the
    # order is (N, H, W, C) then the images are unprocessed. We only compress if
    # the images are processed.
    if type(image_dict['images'][0]) == bytes:
        prc_imgs = image_dict['images']
        raw_imgs = []
        for frame_idx in range(len(prc_imgs)):
            # Process each image here
            imgpil = Image.open(BytesIO(prc_imgs[frame_idx]))
            raw_imgs.append(np.array(imgpil))

        image_dict['images'] = np.stack(raw_imgs,axis=0)
    else:
        pass

    return image_dict

def compress_data(Images):
    """
    We apply a compression if the images are too large to be saved in the .pt file. This function
    compresses the images to a smaller size.
    """
    assert 'images' in Images[0], "No images found in data dictionary"

    for image_dict in Images:
        # Check if the image are processed or not. We do this by checking the array
        # order. If the order is (N, C, H, W) then the images are processed. If the
        # order is (N, H, W, C) then the images are unprocessed. We only compress if
        # the images are processed.

        if image_dict['images'].shape[-1] == 3:
            raw_imgs = image_dict['images']
            prc_imgs = []
            for frame_idx in range(raw_imgs.shape[0]):
                img_arr = raw_imgs[frame_idx]
                imgpil = Image.fromarray(img_arr)

                buffer = BytesIO()
                imgpil.save(buffer, format='PNG')   
                buffer.seek(0)
                prc_imgs.append(buffer.getvalue())

            image_dict['images'] = prc_imgs
        else:
            pass
    
    return Images


# def flight2rollout_data(cohort_name:str,drone_name:str):
#     # Some useful path(s)
#     workspace_path = os.path.dirname(
#         os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

#     # Load drone and method configs
#     cohort_path = os.path.join(workspace_path,"cohorts",cohort_name)
#     flight_data_path = os.path.join(cohort_path,"flight_data")
#     drone_path  = os.path.join(workspace_path,"configs","drones",drone_name+".json")

#     with open(drone_path) as json_file:
#         drone_config = json.load(json_file)

#     # Gather all the flight data files
#     file_list = os.listdir(flight_data_path)
#     file_list = [os.path.join(flight_data_path, f) for f in file_list if f.endswith('.pt')]

#     # Assume the drone is Carl
#     drone = qc.generate_preset_config(drone_config)

#     # Convert flight data to training data
#     data_names = {}
#     for idx,file in enumerate(file_list):
#         # Load the flight data
#         data = torch.load(file)

#         # Check if the data is in the old format
#         data = flightdata_check(data)

#         # Extract Some Useful Intermediate Variables
#         parts = file.split("/")[-1].split("_")
#         file_name_components = [part for part in parts if not part[0].isdigit()]
#         prefix = file_name_components[0]
#         course_name = "_".join(file_name_components[1:-1])

#         # Check if course is in the dictionary
#         data_name = "_".join((prefix,course_name))
#         if data_name not in data_names:
#             data_names[data_name] = 0
#         else:
#             data_names[data_name] += 1

#         # Generate ideal trajectory
#         course_path = os.path.join(workspace_path,"configs","courses",course_name+".json")
#         with open(course_path) as json_file:
#             course_config = json.load(json_file)

#         Tpi,CPi = ms.solve(course_config)
#         tXUi = th.ts_to_tXU(Tpi,CPi,drone)
        
#         # Create save directory
#         save_path = os.path.join(cohort_path,"rollout_data",course_name)
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)

#         # Pack into a dictionary
#         trajectory = {
#             "Tro":data["Tact"],"Xro":data["Xest"],"Uro":data["Uact"],
#             "Xid":None,"obj":data["obj"],"Ndata":data["Tact"].shape[0],"Tsol":data["Tsol"],"Adv":data["Adv"],
#             "rollout_id":str(data_names[data_name]).zfill(5),
#             "course":course_name,
#             "drone":drone
#             }

#         images = {
#             "images": data["Imgs"].numpy(),  # Reshape to (w, h, c)
#             "rollout_id":str(data_names[data_name]).zfill(5),"course":course_name
#         }

#         # Save the training data.
#         save_name = "_".join(("",prefix,course_name,str(data_names[data_name]+1).zfill(3)))
#         save_rollouts(cohort_name,course_name,[trajectory],[images],tXUi,save_name)

#         # # Print some diagnostics
#         # print("Generated ",data["Tact"].shape[0]," points of data for",course_name)

def flightdata_check(data:dict):
    # Keys reference
    new_keys = ['Imgs', 'Tact', 'Uact', 'Xref', 'Uref', 'Xest', 'Xext', 'Adv', 'Tsol', 'obj', 'n_im']
    old_keys = ['Tact', 'Xref', 'Uref', 'Xact', 'Uact', 'Adv', 'Tsol', 'Imgs', 'tXds', 'n_im']

    # Load the flight data
    data_keys = list(data.keys())

    if data_keys == new_keys:
        return data
    elif data_keys == old_keys:
        new_data = {
            'Imgs': data['Imgs'],
            'Tact': data['Tact'],
            'Uact': data['Uact'],
            'Xref': data['Xref'],
            'Uref': data['Uref'],
            'Xest': data['Xact'],
            'Xext': data['Xact'],
            'Adv': data['Adv'],
            'Tsol': data['Tsol'],
            'obj': sh.tXU_to_obj(data['tXds']),
            'n_im': data['n_im']
        }
        return new_data
    else:
        print("Data keys do not match expected keys")
        return None