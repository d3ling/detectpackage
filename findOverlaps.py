def findOverlaps(obj_list_1, obj_list_2, boxes):
  """
  findOverlaps
    parameters:
      - obj_list_1: list of detected object indices that will convert to object centers
      - obj_list_2: list of detected object indices that will convert to object bounding boxes
      - boxes: bounding boxes of detected object instances as a list containing
        [(x0, y0), (x1, y1)] per instance
      NOTE: indices in obj_list_1 and obj_list_2 are used to index into boxes
    return:
      - overlaps: list of object indices representing overlaps of objects between input lists
    method:
      - center of each object represented in obj_list_1 is obtained
      - each center is checked to determine if it is within the bounding box of each object
        represented in obj_list_2, aside from matching indices (refers to the same object)
        - if it is within, there is overlap and this is added to the overlaps list
  """
  # check if any object in obj list 1 overlaps with any in list 2
  overlaps = []

  for i in obj_list_1:
    cent_x = (boxes[i][1][0] + boxes[i][0][0]) / 2
    cent_y = (boxes[i][1][1] + boxes[i][0][1]) / 2

    for j in obj_list_2:
      if i != j:
        cent_x_in = boxes[j][0][0] <= cent_x and cent_x <= boxes[j][1][0]
        cent_y_in = boxes[j][0][1] <= cent_y and cent_y <= boxes[j][1][1]

        if cent_x_in and cent_y_in:
          overlaps.append(i)
  
  return overlaps