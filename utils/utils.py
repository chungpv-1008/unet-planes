import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd


def label_in_box(color, mask):
    # get the binary mask
    r, g, b = mask[..., 0], mask[..., 1], mask[..., 2]
    binary = (r == color[0]) & (g == color[1]) & (b == color[2])
    binary = np.array(binary).astype(np.uint8)

    # because the masks are "too good", we make them a bit rougher to get cleaner contours
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel)
    binary = cv2.erode(binary, kernel)
    return binary


def parse_xml_image(xml_fpath):
    image = ET.parse(xml_fpath).getroot()
    if image.find("object") is None:
        return

    image_metadata = parse_xml_image_metadata(image)
    annotations = []
    for obj in image.findall("object"):
        object_type = obj.find('category0').text
        if object_type != 'Airplane':
            continue

        annotation = parse_xml_annotation(obj)

        # check if annotation is not empty
        if annotation:
            annotations.append({**image_metadata, **annotation})

    # columns = list(image_metadata.keys()) + list(annotation.keys())
    df = pd.DataFrame(annotations)
    return df


def parse_xml_annotation(obj):
    """Parse an XML annotation.
    args:
        - obj (xml.etree.ElementTree.Element): the XML annotation object
    returns:
        - annotation (dict): annotation info as a flat dict
    """
    # get the bbox
    bbox = obj.find("bndbox2D")

    # get the polygon
    socket = obj.find("Sockets")
    nose = get_coordinates(socket.find('Bone_PlaneAnnotation_Nose'))
    right_wing = get_coordinates(socket.find('Bone_PlaneAnnotation_RightWing'))
    left_wing = get_coordinates(socket.find('Bone_PlaneAnnotation_LeftWing'))
    tail = get_coordinates(socket.find('Bone_PlaneAnnotation_Tail'))
    polygon = [nose, left_wing, tail, right_wing]
    polygon = [coord for point in polygon for coord in point]  # flatten the list

    color = obj.find("object_mask_color_rgba")
    color = color.text

    if bbox is None:
        return {}

    annotation = dict(make=get_category_full(obj),
                      xmin=int(bbox.find("xmin").text),
                      ymin=int(bbox.find("ymin").text),
                      xmax=int(bbox.find("xmax").text),
                      ymax=int(bbox.find("ymax").text),
                      segmentation=polygon,
                      object_mask_color_rgba=[int(c) for c in color.split(',')[:-1]])  # only care about rgb

    return annotation


def parse_xml_image_metadata(image):
    """Parse XML image metadata.
    args:
        - image (xml.etree.ElementTree.Element): the XML image object
    returns:
        - image_metadata (dict): image metadata as a flat dict
    """

    image_metadata = dict(
        image_filename=image.find("filename").text,
        width=int(image.find("image_resolution").find("width").text),
        height=int(image.find("image_resolution").find("height").text)
    )

    return image_metadata


def get_category_full(obj):
    """ small function to merge categories"""
    return '_'.join(filter(None, parse_xml_categories(obj)))


def get_coordinates(obj):
    """ small wrapper function to get the coordinates of the airplanes keypoints"""
    coordinates_str = obj.find('screen').text
    c_split = coordinates_str.split(' ')
    x = float(c_split[0].split('=')[1])
    y = float(c_split[1].split('=')[1])
    return x, y


def parse_xml_categories(obj):
    """ parse categories """
    cat3 = obj.find('category3')
    cat4 = obj.find('category4')
    return [cat.text if cat is not None else None for cat in [cat3, cat4]]
