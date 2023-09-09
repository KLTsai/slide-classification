import os
print(os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
import sys
sys.path.append(r'../') # add parent path
sys.path.append(r'./mmocr/mmocr/') # add mmocr parent path
import pandas as pd
import numpy as np
import cv2
from components.labels_correspoding import read_alignment_file, proc_labels
from mmocr.apis import MMOCRInferencer

def get_img_list(df):
    # create dict for image name and page number
    few_dir = os.path.join(os.getcwd(), 'output', 'FewWordImg')
    png_files = [file for file in os.listdir(few_dir) if file.endswith('.PNG') or file.endswith('.png')]
    png_dict = {}
    for _, file in enumerate(png_files):
        ktr, png_page = file.split('_')[0], file.split('_')[-1].split('.')[0]
        png_page = int(png_page)
        png_dict[(ktr, png_page)] = file

    # get image list
    png_files = []
    for _, i in df.iterrows():
        file = i['file_name']
        page = i['page_num']
        ktr_page = (file.split('\\')[-2], page)

        if ktr_page in png_dict:
            png_file = png_dict[ktr_page]
            png_files.append(os.path.join(few_dir, png_file))
            # print(f"Found match: {ktr_page[0]} - Page {ktr_page[1]} - File {png_file}")
    
    return png_files

def check_files(png_files):
    
    ''' 
    WARNING:    
    there are some issues with images in "FewWordImg" folder in terms of the file name.
    for example:
        22321_22321_Implementierung S&OP Prozess_Anlage_1_x.PNG
    but in the "mmocr_pre" folder, the file name is:
        22321_22321_Implementierung S_OP Prozess_Anlage_1_x.PNG

    align the name to "mmocr_pre" folder
    '''
    check_dir = os.path.join(os.getcwd(), 'output', 'mmocr_pre', 'preds')
    few_dir = os.path.join(os.getcwd(), 'output', 'FewWordImg')
    saved_json_files = [file for file in os.listdir(check_dir) if file.endswith('.json')]
    sorted_files = sorted(saved_json_files, key=lambda x: os.path.getmtime(os.path.join(check_dir, x)))
    # print(f'Numbers:{len(saved_files)}, the last index of file: {saved_files[1587]}')
    # print(saved_json_files)
    proc_files = []
    for _, file in enumerate(sorted_files):
        if file.endswith('.json'):
            new_name = file[:-4] + 'PNG'
            print(new_name)
            proc_files.append(os.path.join(few_dir, new_name))

    set1 = set(png_files)
    set2 = set(proc_files)
    difference=set1-set2
    continue_files = list(difference)
    print(f'Number of files to inference: {len(continue_files)}')

    # get prepared-files
    if len(os.listdir(check_dir))>0 and len(continue_files)>0:
        return continue_files
    else:
        return None

def mmocr_inference(imgs):
    # mmocr inference
    infer = MMOCRInferencer(det='TextSnake')
    results = infer(imgs,
                    return_vis=True,
                    out_dir='../components',
                    save_pred=True,
                    save_vis=True)

def get_mono_img(png_files):
    img = cv2.imread(png_files[0])
    mono_path = os.path.join(os.getcwd(), 'output', 'mmocr_pre', 'mono')
    json_path = os.path.join(os.getcwd(), 'output', 'mmocr_pre', 'preds')
    if not os.path.exists(mono_path):
        print('create folder mono')
        os.makedirs(mono_path)

    det_imgs = [file for file in os.listdir(json_path) if file.endswith('.json')]
    
    for file in det_imgs:
        # create blank image (960x540, one channel)
        solid_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        df_det = pd.read_json(os.path.join(json_path, file))
        new_name = file[:-4] + 'jpg'

        # Definition of polygon
        color = 255
        for polygon in df_det['det_polygons']:
            polygon = np.array(polygon, dtype=np.int32)
            polygon = polygon.reshape((-1, 2))
            # plot rectangle
            cv2.fillPoly(solid_img, [polygon], color)

        # Save the image to a new file
        print(f'--->[3.1] Save mono image: {os.path.join(mono_path, new_name)}')
        cv2.imwrite(os.path.join(mono_path, new_name), solid_img)
        
if  __name__ == "__main__":
    # read alignment file
    df_few_words = pd.DataFrame()
    df_few_words = read_alignment_file(df_few_words)
    df_few_words = proc_labels(df_few_words)
    
    # get image list
    png_files = get_img_list(df_few_words)
    # check files whether they have been processed
    if check_files(png_files) is None:
        print('All images have been processed!, please check the output folder')
    else:
        print('--->[1] Start inference!')
        mmocr_inference(png_files)
        print('--->[2] export results to inference json files and color images')
        get_mono_img(png_files)
        print('--->[3] Done!')
