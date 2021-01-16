# Package Detector uses Mask R-CNN to detect people and packages

# common libraries
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import os
import cv2
import imageio # for saving frames as a gif

# custom
from MaskRCNN.getDetections import getDetections
from MaskRCNN.addDetectionsToImage import addDetectionsToImage
from removeUnnecessaryOverlaps import removeUnnecessaryOverlaps
from findOverlaps import findOverlaps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_gif_path = 'input.gif'
output_gif_path = 'output.gif'
mrcnn_model_path = 'MaskRCNN/pretrained'

# pretrained Mask R-CNN model
# model originally obtained via torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
mrcnn_model = torch.load(os.path.join(mrcnn_model_path, 'maskrcnn.pth')) # full model with weights
mrcnn_model.eval()

# load input gif and extract frames into a numpy array
gif = Image.open(input_gif_path)
input_frames = np.array([np.array(frame.copy().convert('RGB').getdata()).reshape(frame.size[1],frame.size[0],3) for frame in ImageSequence.Iterator(gif)])

red, green = (255, 0, 0), (0, 255, 0)
output_frames = []
delivered, taken = False, False

# add detections to input frames based on package history (not delivered -> delivered -> taken)
for i in range(len(input_frames)):
  input_img = input_frames[i].astype(np.uint8)

  # obtain masks, bounding boxes, and classes for detected object instances above a confidence level
  confidence = 0.45
  with torch.no_grad():
    masks, boxes, classes = getDetections(mrcnn_model, input_img, confidence, device)
  
  classes = list(classes)
  person_indices, package_indices = [], []
  
  # extract indices for persons and packages respectively
  for j in range(len(classes)):
    if classes[j] == 'person':
      person_indices.append(j)
    elif classes[j] == 'package':
      package_indices.append(j)

  # remove unnecessary detection overlaps
  removeUnnecessaryOverlaps(person_indices, boxes)
  removeUnnecessaryOverlaps(package_indices, boxes)

  # to track the sequence of events:
  # 1) not delivered (default): no package has been idenitifed away from a person yet
  # 2) delivered: when a package is identified away from a person
  # 3) taken: when a package is identified close to a person after being delivered
  if not delivered:
    color = green
    package_person_overlap = findOverlaps(package_indices, person_indices, boxes)
    if package_indices and not package_person_overlap:
      delivered = True
  elif not taken:
    package_person_overlap = findOverlaps(package_indices, person_indices, boxes)
    taken, color = (True, red) if package_person_overlap else (False, green)
  else:
    color = red

  # modify package label depending on package status
  if delivered and not taken:
    for i in package_indices:
      classes[i] = 'delivered package'
  elif taken:
    for i in package_indices:
      classes[i] = 'taken package'
  
  # add person detections and package detections respectively to the image
  output_img = addDetectionsToImage(input_img, masks, boxes, classes, color, person_indices)
  output_img = addDetectionsToImage(output_img, masks, boxes, classes, color, package_indices)

  output_frames.append(output_img)

imageio.mimsave(output_gif_path, output_frames)