from __future__ import annotations
from dataclasses import replace
import os
from pathlib import Path
import random
import shutil
import time
import cv2
import numpy as np
from roboflow import Roboflow

class Utils():
    '''
    Class untuk melakukan custom data, berisikan banyak method bantuan yang bisa anda pakai
    
    Method
    ------
    `rename_image_file(*args)`     : untuk mengubah filename lama menjadi baru
    `create_live_model(*args)`     : untuk mengubah model baru dengan live webcame. Recomended untuk faces model
    '''
    
    def rename_image_file(self,_directory='',_filename='',_extention='.png'):
        '''
        Duplicate audio file yang sudah ada dan mengubah filename nya,
        fungsi ini dapat dikombinasikan dengan fungsi looping apabila dibutuhkan
        
        Arguments
        ---------
        _filename : filename baru yang ingin anda tentukan, Required format `String Kosong`
        _directory : lokasi directory audio lama yang ingin dibaca untuk di ubah filenamenya. default `None`
        _extention : type extention baru hasil rename, default `.png`
        
        Example
        -------
        >>> renames = rename_image_file(f"{_directory}","nama file baru")
        '''
        
        for i,file in enumerate(os.listdir(_directory), start=1): # kita panggil semua file audio yang ada pada sebuah directry folder audio kita
            # rename files dengan yang baru
            os.rename(os.path.join(_directory,file),os.path.join(_directory,_filename+"_"+str(i)+_extention))
        
    def create_live_model(self,_directory_model='',_filename='',_extention='.png',_nums_model=20):
        '''
        Membuat model baru dengan live webcame device komputer / laptop anda. Recommended untuk membuat face model
        
        Arguments
        ---------
        `_filename` : filename baru yang ingin anda tentukan, Required format `String Kosong`
        `_directory_model` : lokasi directory model yang menjadi tempat simpan model. default `None`
        `_extention` : type extention baru hasil rename, default `.png`
        `_nums_model`  : banyak model screenshot yang akan dibuat untuk model faces / lainnya. default `20 model`.
        
        Example
        -------
        >>> created = create_live_model(f"{_directory}","nama file baru")
        
        '''
        _model_path = os.path.join(_directory_model)
        _capture = cv2.VideoCapture(0) # ini untuk  open webcame. nilai 0 = webcame device
        for index in range(_nums_model):
            print("membuat model untuk {}, model nomer ke {}".format(_filename,index))
            _ret , _frame = _capture.read()
            _model_name = os.path.join(_model_path,_filename + "_" + str(index) + _extention)
            
            cv2.imwrite(_model_name,_frame)
            cv2.imshow(_filename,_frame)
            time.sleep(3)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        _capture.release()
        cv2.destroyAllWindows() 
        
    def split_dataset():
        all_filename = []
        for file in os.listdir("data/images"):
            if file.endswith(".png"):
                all_filename.append(file.replace(".png",""))
                
        print(all_filename)
            
    def create_labeling():
        with open('data/object_recognize.pbtxt','w') as rx:
            # for i,name in enumerate()
            
            rx.write    
        
    def generate_zip(self,path='',ziph=None):
        for root,dirs,files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root,file))
        
    def getting_dateset_from_roboflow(self):
        rf = Roboflow(api_key="YcP1cL5afNk3xkAcd0HS")
        project = rf.workspace("angeline-qh27d").project("trucking_1")
        dataset = project.version(2).download("voc")
        
    def extract_image_and_pascal(self, current_dir="",target_dir="",formates=""):
        for src_file in Path(current_dir).glob(f'*.{formates}*'):
            shutil.move(os.path.join(current_dir,src_file),target_dir)
            