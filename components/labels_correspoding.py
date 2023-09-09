import pandas as pd
import ast
import re
import os

def read_alignment_file(df_few):
    def convert_to_list(data):
        return ast.literal_eval(data)
    
    os.chdir('../code')
    cur_path = os.getcwd()
    text_path = os.path.join(cur_path, 'alignment_pptx_list.csv')
    assert os.path.exists(text_path) == True, 'File does not exist with path: {}'.format(os.path.join(os.getcwd(), 'alignment_pptx_list.csv'))
    
    df_da = pd.read_csv(text_path, converters={'shape_type': convert_to_list})
    # filter out less than 4 words
    df_da['word_count'] = df_da['contents'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df_few = df_da[~(df_da['word_count'] <= 4)]
    return df_few

def proc_labels(df_few_words, labels_json_name = 'backup_23148_4374.json'):
    
    cur_path = os.getcwd()
    labels_json_path = os.path.join(cur_path, 'output/LABELS', labels_json_name)
    assert os.path.exists(labels_json_path) == True, 'Lable file does not exist with path: {}'.format(labels_json_path)
    df_json = pd.read_json(labels_json_path)
    # have done for annotations
    df_json = df_json[df_json['annotations'].apply(lambda x: len(x) != 0)]
    # annotation results exist
    df_json = df_json[df_json['annotations'].apply(lambda x: len(x[0]['result']) != 0)]
    # 22348 for 13th slide
    count = 13
    df_few_words['label'] = ''

    for _, row in enumerate(df_json.iterrows()):
        slide = row[1]
        image_path = slide['data']['image']
        choice = slide['annotations'][0]['result'][0]['value']['choices'][0]
        # print(image_path, choice)

        if image_path.find('22348')!=-1:
            print(image_path, choice)
            new_image_name = image_path.split('.')[0] + '_' + str(count) + '.PNG'
            slide['data']['image'] = new_image_name
            image_path = new_image_name
            # print(image_path)
            count += 1

        # get page number
        if image_path.split('.')[0].split('_')[-1].isdigit():
            page = int(image_path.split('.')[0].split('_')[-1])
        else:
            page = 0

        # given 'Angebot\ktr' as a condition
        ktr = image_path.split('-')[1].split('_')[0]
        ktr = 'Angebot\\'+ ktr
        ktr = re.escape(ktr)

        # some issues that it should modify manually
        if (ktr == '22348' and page == 13):
            df_few_words.loc[(df_few_words['file_name'].str.contains(ktr)) & (df_few_words['page_num'] == page), 'label'] = 'Initial Situation'
            continue
        elif (ktr == '22348' and page == 14):
            df_few_words.loc[(df_few_words['file_name'].str.contains(ktr)) & (df_few_words['page_num'] == page), 'label'] = 'Target Situation'
            continue

        df_few_words.loc[(df_few_words['file_name'].str.contains(ktr)) & (df_few_words['page_num'] == page), 'label'] = choice

    df_few_words = df_few_words[(df_few_words['label']!='') & (df_few_words['label']!='Others')] # filter out empty labels and 'Others' ca
    
    return df_few_words
    
# if __name__ == '__main__':
#     df_few_words = pd.DataFrame()
#     df_few_words = read_alignment_file(df_few_words)
#     df_few_words = proc_labels(df_few_words)