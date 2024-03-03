import os
import sys
import json

from Library import lib

from dotenv import load_dotenv
load_dotenv()

API_TOKEN = os.getenv['API_TOKEN']
API_URL = 'https://api-inference.huggingface.co/models/facebook/detr-resnet-50'

# Folder containing images, select bikes, cars, motorbikes here
folder_path = 'data/bike/1'

# set up variables
total_images = 0
vehicle_count = 0
not_vehicle_count = 0
# specify type of vehicle here (car, bicycle, motorbike)
vehicle = "bicycle"

# Go over each image in the folder
for filename in os.listdir(folder_path):
    # Check filetype
    if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
        # Create image path
        image_path = os.path.join(folder_path, filename)

        # Classify the image
        try:
            # receive image data and store reponse
            response = lib.query(image_path, API_URL, API_TOKEN)

            # print image data to console
            print(json.dumps(lib.query(image_path, API_URL, API_TOKEN), indent=2))

            # print amount of specified vehicle to console (cars, bicycles, motorbikes)
            print(f"number of {vehicle} : ", lib.count_objects(image_path, API_URL, API_TOKEN, label=vehicle))

            # draw bounding boxes around specified vehicle
            lib.draw_rect(image_path, API_URL, API_TOKEN, label=vehicle)

            # Increment total_images counter
            total_images += 1

            # Check if the image was classified as specified vehicle and increment counters accordingly
            if any(vehicle in prediction['label'].lower() for prediction in response):
                vehicle_count += 1
            else:
                not_vehicle_count += 1

        except Exception as e:
            print(f"An error occurred while processing {image_path}: {e}")

# Print the final counts
print(f"Total images: {total_images}")
print(f"Classified as containing {vehicle}: {vehicle_count}")
print(f"Not classified as containing {vehicle}: {not_vehicle_count}")