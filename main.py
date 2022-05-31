from __future__ import annotations
import os

from sklearn.model_selection import train_test_split
import setup as Setup

if __name__ == "__main__":
    root_path = r"C:\Users\prio\Desktop\Angeline\artificial\CarsClassification\image_data"
    root_annotation = r"C:\Users\prio\Desktop\Angeline\artificial\CarsClassification\annotations"
    # Setup.data_generator(root_path=root_path)
    
    data = {}
    path_to_annotation_files = []
    path_to_image_files = []
    
    for class_id in os.listdir(root_annotation):
        ann_class_dir = os.path.join(root_annotation,class_id)
        img_class_dir = os.path.join(root_path,f"{class_id}/images")
        
        for annotation_files in os.listdir(ann_class_dir):
            annotation_path = os.path.join(ann_class_dir,annotation_files)
            img_path = os.path.join(img_class_dir,annotation_files.replace("txt","jpg"))
            path_to_annotation_files.append(annotation_path)
            path_to_image_files.append(img_path)
            
    train_image, val_image, train_annotation, val_annotation = train_test_split(path_to_image_files,
                                                                                path_to_annotation_files,
                                                                                test_size=0.2,
                                                                                random_state= 1,
                                                                                )
    val_image, test_image, val_annotation, test_annotation = train_test_split(val_image,
                                                                              val_annotation,
                                                                              test_size= 0.5,
                                                                              random_state= 1)
    
    Setup.move_file_to_filder(train_image, './images/train')
    Setup.move_file_to_filder(val_image, './images/val')
    Setup.move_file_to_filder(test_image, './images/test')
    Setup.move_file_to_filder(train_annotation, './labels/train')
    Setup.move_file_to_filder(val_annotation, './labels/val')
    Setup.move_file_to_filder(test_annotation, './labels/test')