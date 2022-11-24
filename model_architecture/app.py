import cv2
import numpy as np
from PIL import Image
import streamlit as st
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

# most of this code has been obtained from Datature's prediction script
# https://github.com/datature/resources/blob/main/scripts/bounding_box/prediction.py

st.set_option('deprecation.showfileUploaderEncoding', False)

def args_parser():
    parser = argparse.ArgumentParser(
        description="Datature Open Source Prediction Script")
    parser.add_argument(
        "--input",
        help="Path to folder that contains input images",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Path to folder to store predicted images",
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Path to tensorflow pb model",
        required=True,
		default='../saved_model'
    )
    parser.add_argument(
        "--label",
        help="Path to tensorflow label map",
        required=True,
		default="../label_map.pbtxt"
    )
    parser.add_argument("--width",
                        help="Width of image to load into model",
                        default=640)
    parser.add_argument("--height",
                        help="Height of image to load into model",
                        default=640)
    parser.add_argument("--threshold",
                        help="Prediction confidence threshold",
                        default=0.7)

    return parser.parse_args()

def load_label_map(label_map_path):
    """
    Reads label map in the format of .pbtxt and parse into dictionary
    Args:
      label_map_path: the file path to the label_map
    Returns:
      dictionary with the format of {label_index: {'id': label_index, 'name': label_name}}
    """
    label_map = {}

    with open(label_map_path, "r") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip('"')
                label_map[label_index] = {"id": label_index, "name": label_name}
    return label_map
	
def predict_class(image, model):
	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [640, 640])
	image = np.expand_dims(image, axis = 0)
	return model.predict(image)

def plot_boxes_on_img(color_map, classes, bboxes, image_origi, origi_shape):
	for idx, each_bbox in enumerate(bboxes):
		color = color_map[classes[idx]]

		## Draw bounding box
		cv2.rectangle(
			image_origi,
			(int(each_bbox[1] * origi_shape[1]),
			 int(each_bbox[0] * origi_shape[0]),),
			(int(each_bbox[3] * origi_shape[1]),
			 int(each_bbox[2] * origi_shape[0]),),
			color,
			2,
		)
		## Draw label background
		cv2.rectangle(
			image_origi,
			(int(each_bbox[1] * origi_shape[1]),
			 int(each_bbox[2] * origi_shape[0]),),
			(int(each_bbox[3] * origi_shape[1]),
			 int(each_bbox[2] * origi_shape[0] + 15),),
			color,
			-1,
		)
		## Insert label class & score
		cv2.putText(
			image_origi,
			"Class: {}, Score: {}".format(
				str(category_index[classes[idx]]["name"]),
				str(round(scores[idx], 2)),
			),
			(int(each_bbox[1] * origi_shape[1]),
			 int(each_bbox[2] * origi_shape[0] + 10),),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.3,
			(0, 0, 0),
			1,
			cv2.LINE_AA,
		)
	return image_origi

@st.cache(allow_output_mutation=True)
def load_model(model_path):
	return tf.saved_model.load(model_path)

st.title('Web App to detect Breast Cancer')

file = st.sidebar.file_uploader("Choose image to evaluate model", type=["jpg", "png"])

button = st.sidebar.button('Detect Breast Cancer!')

args = args_parser()

model = load_model(args.model)

if  button and file: 

	st.text('Running inference...')
	# open image
	test_image = Image.open(file).convert("RGB")
	origi_shape = np.asarray(test_image).shape
	# resize image to default shape
	image_resized = np.array(test_image.resize((args.width, args.height)))

	## Load color map
	category_index = load_label_map(args.label)

	# TODO Add more colors if there are more classes
  # color of each label. check label_map.pbtxt to check the index for each class
	color_map = {
		1: [255, 0, 0], # bad -> red
		2: [0, 255, 0] # good -> green
	}

	## The model input needs to be a tensor
	input_tensor = tf.convert_to_tensor(image_resized)
	## The model expects a batch of images, so add an axis with `tf.newaxis`.
	input_tensor = input_tensor[tf.newaxis, ...]

	## Feed image into model and obtain output
	detections_output = model(input_tensor)
	num_detections = int(detections_output.pop("num_detections"))
	detections = {key: value[0, :num_detections].numpy() for key, value in detections_output.items()}
	detections["num_detections"] = num_detections

	## Filter out predictions below threshold
	# if threshold is higher, there will be fewer predictions
	# TODO change this number to see how the predictions change
	indexes = np.where(detections["detection_scores"] > args.threshold)

	## Extract predicted bounding boxes
	bboxes = detections["detection_boxes"][indexes]
	# there are no predicted boxes
	if len(bboxes) == 0:
		st.error('No boxes predicted')
	# there are predicted boxes
	else:
		st.success('Boxes predicted')
		classes = detections["detection_classes"][indexes].astype(np.int64)
		scores = detections["detection_scores"][indexes]

		# plot boxes and labels on image
		image_origi = np.array(Image.fromarray(image_resized).resize((origi_shape[1], origi_shape[0])))
		image_origi = plot_boxes_on_img(color_map, classes, bboxes, image_origi, origi_shape)

		# show image in web page
		st.image(Image.fromarray(image_origi), caption="Image with predictions", width=400)
		st.markdown("### Predicted boxes")
		for idx in range(len((bboxes))):
			st.markdown(f"* Class: {str(category_index[classes[idx]]['name'])}, confidence score: {str(round(scores[idx], 2))}")