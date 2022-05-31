import os
import shutil
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm

def extract_info(xml_file=""):
    root = ET.parse(xml_file).getroot()
    info_dict = {}
    info_dict['bboxes'] = []
    
    for elem in root:
        if elem.tag == 'filename':
            info_dict['filename'] = elem.text
        elif elem.tag == 'size':
            img_size = [int(subelem.text) for subelem in elem]
            info_dict['image_size'] =  tuple(img_size)
        elif elem.tag == 'object':
            bbox = {}
            for subelem in elem:
                if subelem.tag == 'name':
                    bbox['class'] = subelem.text
                elif subelem.tag == 'bndbox':
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict['bboxes'].append(bbox)
    return info_dict
    
def convert_pascal_to_yolov5(info_dict):
    __className_to_id_mapping = {"truck" : 0}
    print_buffer = []
    
    for bbox in info_dict['bboxes']:
        try:
            class_id = __className_to_id_mapping[bbox['class']]
        except KeyError:
            print('Invalid class. must be one from', __className_to_id_mapping.keys())

        bbox_x_center = (bbox['xmin'] + bbox['xmax']) /2
        bbox_y_center = (bbox['ymin'] + bbox['ymax']) /2
        bbox_width = (bbox['xmax'] - bbox['xmin'])
        bbox_height = (bbox['ymax'] - bbox['ymin'])
        
        image_w, image_h, _ = info_dict['image_size']
        bbox_x_center /= image_w
        bbox_y_center /= image_h
        bbox_width /= image_w
        bbox_height /= image_h
        
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, bbox_x_center, bbox_y_center, bbox_width, bbox_height))
    save_file_name = os.path.join(f"annotations/{info_dict['bboxes'][0]['class']}",info_dict['filename'].replace('jpg', 'txt'))
    print('\n'.join(print_buffer), file=open(save_file_name,'w'))
    
    
def data_generator(root_path: str):
    root_path = root_path
    annotations_dir = {}
    
    for classes_dir in os.listdir(root_path):
        if classes_dir == '.DS_store':
            continue
        else:
            annotations_dir[classes_dir] = os.path.join((root_path), f"{classes_dir}/pascal")
    print(f"annotations dirs {annotations_dir.keys}")
    
    for class_name in annotations_dir.keys():
        for xml_filename in tqdm(os.listdir(annotations_dir[class_name])):
            xml_file_path = os.path.join(annotations_dir[class_name], xml_filename)
            
            if os.path.exists(f"./annotations/{class_name}"):
                extracted_info = extract_info(xml_file_path)
                convert_pascal_to_yolov5(extracted_info)
            else:
                os.makedirs(f"./annotations/{class_name}")
    print("xml to yolo convert success")
                
    
def draw_boundinig_box(__win_name,image,annotation_text_file):
    with open(annotation_text_file,'r') as file:
        annotation = file.read().split('\n')[:-1]
        annotation = [x.split(" ") for x in annotation]
        annotation = [[float(y) for y in x] for x in annotation]
        
    annotations = np.array(annotation)
    print(f"annotation value before normalize : {annotations}")
    
    h,w,_ = image.shape
    annotations_cp = np.copy(annotations)
    annotations_cp[:,[1,3]] = annotations[:, [1, 3]] * w
    annotations_cp[:,[2,4]] = annotations[:, [2,4]] * h
    print(f"annotation value after normalize : {annotations_cp}")
    
    
    annotations_cp[:,1] = (annotations_cp[ :,1] - annotations_cp[:,3]/2)
    annotations_cp[:,2] = (annotations_cp[ :,2] - annotations_cp[:,4]/2)
    annotations_cp[:,3] = (annotations_cp[ :,1] + annotations_cp[:,3])
    annotations_cp[:,4] = (annotations_cp[ :,2] + annotations_cp[:,4])    
    
    for single_annotation in annotations_cp:
        obj_cls,x0,y0,x1,y1 = single_annotation
        cv2.rectangle(image, (int(x0), int(y0)) ,( int(x1), int(y1)), color=(255,0,0), thickness=2)
        cv2.imshow(__win_name,image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def move_file_to_filder(list_of_files,destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f,destination_folder)
        except:
            print(f)
            assert False