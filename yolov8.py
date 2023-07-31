import os
import shutil
from glob import glob

from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys

LABELSTUDIO_PATH = '/data/upload/20'
DATASET_ROOT_DIR = '/home/kieu/apps/label-studio/media/upload/20'
STORAGE_DIR = '/home/kieu/Downloads/LabelStudio-Yolov8-Detection-Backend/data'
MODEL_NAME = 'yolov8n_custom'
MODEL_INIT = '/home/kieu/Downloads/LabelStudio-Yolov8-Detection-Backend/model/yolov8n.pt'
MODEL_LATEST = f'/home/kieu/Downloads/LabelStudio-Yolov8-Detection-Backend/runs/detect/{MODEL_NAME}/weights/best.pt'
DETECTION_YAML = '/home/kieu/Downloads/LabelStudio-Yolov8-Detection-Backend/data/detection.yaml'


class SignatureDetectionAPI(LabelStudioMLBase):
	"""This simple Label Studio ML backend demonstrates training & inference steps with a simple scenario:
	on training: it gets the latest created annotation and stores it as "prediction model" artifact
	on inference: it returns the latest annotation as a pre-annotation for every incoming task

	When connected to Label Studio, this is a simple repeater model that repeats your last action on a new task
	"""
	def __init__(self, **kwargs):
		super(SignatureDetectionAPI, self).__init__(**kwargs)
		self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
			self.parsed_label_config, 'RectangleLabels', 'Image')
		self.labels_in_config = list(self.labels_in_config)

		if self.train_output:
			self.model = YOLO(self.train_output['checkpoint'])
		else:
			self.model = YOLO(MODEL_INIT)

	def predict(self, tasks, **kwargs):
		""" This is where inference happens:
			model returns the list of predictions based on input list of tasks

			:param tasks: Label Studio tasks in JSON format
		"""
		image_urls = [task['data'][self.value] for task in tasks]
		# image_url = image_urls[0].replace('/data/local-files/?d=', '/media/kieu/caithungrac/workspace/uni/')
		image_url = image_urls[0].replace(LABELSTUDIO_PATH, DATASET_ROOT_DIR)
		results = self.model.predict(image_url)

		output_prediction = []
		img_h, img_w = results[0].orig_shape[0], results[0].orig_shape[1]
		model_label = results[0].names
		for result in results[0]:
			boxes = result.boxes.data.tolist()[0]
			output_label = model_label[int(boxes[-1])]

			if output_label not in self.labels_in_config:
				print(output_label + ' label not found in project config.')
				continue

			output_prediction.append(
				{
					"to_name": self.to_name,
					"from_name": self.from_name,
					"type": "rectanglelabels",
					"value": {
						"x": boxes[0] / img_w * 100,
						"y": boxes[1] / img_h * 100,
						"width": (boxes[2] - boxes[0]) / img_w * 100,
						"height": (boxes[3] - boxes[1]) / img_w * 100,
						"rotation": 0,
						"rectanglelabels": [
							output_label
						]
					}
				}
			)
		return [{
			'result': output_prediction
		}]
	
	def _create_dataset(self, completions: list):
		# prepare storage folder
		label_folder = os.path.join(STORAGE_DIR, 'labels')
		image_folder = os.path.join(STORAGE_DIR, 'images')
		[os.makedirs(subfolder, exist_ok=True) for subfolder in [label_folder, image_folder]]
		train_images = glob(image_folder + '**.*')

		for i in completions:
			# save image path for list of image path
			data_image = i['data']['image']
			filename = os.path.join(image_folder, os.path.basename(data_image))
			fileext = os.path.splitext(filename)[-1]
			if filename not in train_images:
				shutil.copy(data_image.replace(LABELSTUDIO_PATH, DATASET_ROOT_DIR), filename)


			# remove old label file
			labelname = filename.replace('/images/', '/labels/').replace(fileext, '.txt')
			if os.path.exists(labelname):
				os.remove(labelname)

			# save annotation for label file
			annotations = i['annotations'][0]['result']
			for result in annotations:
				value = result['value']
				rectanglelabels = self.labels_in_config.index(value['rectanglelabels'][0])
				
				x = value['x']
				y = value['y']
				width = value['width']
				height = value['height']

				x_center = (x * 2 + width) / 200
				y_center = (y * 2 + height) / 200
				width_box = width / 100
				height_box = height / 100

				with open(labelname, "a") as f:
					f.write(f'{rectanglelabels} {x_center} {y_center} {width_box} {height_box}\n')
		

	def fit(self, completions, workdir=None, 
            batch_size=32, num_epochs=50, imgsz = 640, **kwargs):
		"""
		This method is called each time an annotation is created or updated
		:param kwargs: contains "data" and "event" key, that could be used to retrieve project ID and annotation event type
						(read more in https://labelstud.io/guide/webhook_reference.html#Annotation-Created)
		:return: dictionary with trained model artefacts that could be used further in code with self.train_output
		"""

		print('Collecting dataset ...')
		self._create_dataset(completions)

		self.model.train(data = DETECTION_YAML,
						imgsz = imgsz,
						epochs = num_epochs,
						batch = batch_size,
						name = MODEL_NAME,
						exist_ok = True,
						augment=False)

		return {'checkpoint': MODEL_LATEST}
