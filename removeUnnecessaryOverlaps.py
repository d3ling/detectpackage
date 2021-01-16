from findOverlaps import findOverlaps

def removeUnnecessaryOverlaps(obj_indices, boxes):
  """
  removeUnnecessaryOverlaps
    parameters:
      - obj_indices: list of object indices
      - boxes: bounding boxes of detected object instances as a list containing
        [(x0, y0), (x1, y1)] per instance
    return:
      - N/A
    method:
      - within overlapping detections, indices of all detected objects except the
        most prominent detection are obtained in a list
      - the indices of these less prominent detected objects are removed from
        obj_indices in-place
  """
  obj_overlaps = findOverlaps(obj_indices, obj_indices, boxes)
  for overlap in obj_overlaps:
    obj_indices.remove(overlap)