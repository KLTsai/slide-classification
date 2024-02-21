import os
import sys
# sys.path.append(os.getcwd())
from components.blob_storage import BlobStorageManager
from configuration.secrets import get_blob_storage_account_key, get_blob_storage_account_name
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LayoutLMv3ImageProcessor, LayoutLMv3Processor
import time
import requests
from PIL import Image
import io
from fastapi import HTTPException

logger.add(
    os.path.join(os.getcwd(), 'code', 'output', 'logs', 'setup_model.log'),
    format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    backtrace = True,
    diagnose = True
)


def load_model():
    # TODO: load model from model_path

    # MODEL = './code/output/checkpoints/a_mul_e6_s256_lr6e-6'
    bs_account_key = get_blob_storage_account_key()
    bs_account_name = get_blob_storage_account_name()
    blob_manager = BlobStorageManager('slide-classification-models', bs_account_name, bs_account_key)
    container_client = blob_manager.container
    model_config_name = 'a_mul_e6_s256_lr6e-6'
    local_path = os.path.join(os.getcwd(), 'downloaded_model')
    download_path = os.path.join(local_path, model_config_name)
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    if not os.listdir(download_path):
        start_time = time.time()
        for blob in container_client.list_blobs():
            blob_client = container_client.get_blob_client(blob)
            if os.path.exists(download_path):
                with open(local_path + "/" + blob.name, "wb") as my_blob:
                    download_stream = blob_client.download_blob()
                    total_size = download_stream.size
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:  # create progress bar
                        for chunk in download_stream.chunks():
                            my_blob.write(chunk)
                            pbar.update(len(chunk))  # update progress
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'execution time: {execution_time} seconds')


    tokenizer = AutoTokenizer.from_pretrained(download_path, 
                                              cache_dir='./code/output/pretrained')

    feature_extractor = LayoutLMv3ImageProcessor(ocr_lang="eng+deu")
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)
    classifier = AutoModelForSequenceClassification.from_pretrained(download_path, 
                                                                    cache_dir='./code/output/pretrained')
    logger.info('Model Load Success!')
    return tokenizer, processor, classifier


async def load_filesOrImgUrls(files, image_urls):
    img_list = []   # list of PIL images
    name_list = []  # list of image names
    
    if files:
        logger.info(f'Number of files: {len(files)}')
        for file in files:
            request_object_content = await file.read()
            img = Image.open(io.BytesIO(request_object_content))
            img_list.append(img)
            name_list.append(file.filename)

    if image_urls:
        logger.info(f'Number of image URLs: {len(image_urls)}')
        try:
            for image_url in image_urls:
                response = requests.get(image_url)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                img_list.append(img)
                name_list.append(image_url)
        except Exception as e:
            logger.error(f'Error in loading image URLs: {e}')

    logger.info(f'Number of total input sources: {len(name_list)}')

    if not files and not image_urls:
        raise HTTPException(status_code=400, detail="Please provide either files or image URLs.")
    
    return img_list, name_list

def run_model(classifier, encoded_input):
    # Run through model and normalize
    try:
        outputs = classifier( input_ids = encoded_input['input_ids'], 
                              attention_mask = encoded_input['attention_mask'], 
                              bbox = encoded_input['bbox'])
    except Exception as e:
        print(f"An exception occurred: {e}")
    
    return outputs


# if __name__ == '__main__':

#     tokenizer_, processor_, classifier_ = load_model()
#     print(f'main model loaded')
