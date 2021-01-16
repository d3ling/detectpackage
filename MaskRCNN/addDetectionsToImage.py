import cv2

def addDetectionsToImage(input_img, masks, boxes, classes, color, obj_indices, box_th=2, font_size=1, font_th=2):
  """
  addDetectionsToImage
    parameters:
      - input_img: input image as a numpy array
      - masks: masks of detected object instances as a numpy array
      - boxes: bounding boxes of detected object instances as a list containing
        [(x0, y0), (x1, y1)] per instance
      - classes: classes of detected object instances as a numpy array
      - color: color for box and font
      - obj_indices: indices of detected objects of interest
      - box_th: line thickness of box
      - font_size: font size
      - font_th: line thickness of text
    return:
      - image with intended detections
    method:
      - add the box and text 
  """
  for i in obj_indices:
    cv2.rectangle(input_img, boxes[i][0], boxes[i][1], color, thickness=box_th)
    cv2.putText(input_img, classes[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness=font_th)
  return input_img