import torch
import os
import requests
from fastapi import FastAPI, UploadFile, File, Form
from components.preprocessing_helper import preprocess
from components.setup_model import load_model, load_filesOrImgUrls, run_model
from components.api_schemas import ResponseDict, PredictDict
from loguru import logger
from typing import List
import uvicorn


logger.add(
    os.path.join(os.getcwd(), 'code', 'output', 'logs', 'setup_model.log'),
    format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    backtrace = True,
    diagnose = True
)

# Start up app
app = FastAPI()
prob_torch = None
tokenizer, processor, model = load_model()

class_map = ['Competencies',
            'Consultant Profile',
            'Initial & Target Situation',
            'Initial Situation',
            'Offer Title',
            'Project Calculation',
            'Reference Details',
            'Reference Overview',
            'Target Situation',
            'Working Package Description',
            'Working Package Examples',
            'Working Package Overview',
        ]

@app.get("/")
def read_root():
    return "UNITY: Welcome to the slide classification API"

@app.post('/cls_score/', response_model = ResponseDict)
async def cls_score(files: List[UploadFile] = File([]),
              image_urls: List[str] = Form([], description="List of image URLs")):
    global prob_torch

    if prob_torch is not None:
        del prob_torch 

    img_list, name_list = await load_filesOrImgUrls(files, image_urls)
    logger.info(f'preprocessing image')
    encoded_inputs = preprocess(img_list, processor)
    logger.info(f'running model')
    # Run through model and normalize
    outputs = run_model(model, encoded_inputs)
    logger.info(f'inference done')

    logits = outputs.logits
    # deal with logits
    prob_torch = torch.softmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1).tolist()
    # organize output
    output_dict = {}
    for name, scores in zip(name_list, probabilities):
        score_dict = {label: score for label, score in zip(class_map, scores)}
        output_dict[name] = score_dict

    return {"output": output_dict} 

@app.get('/cls_score/predict/', response_model=PredictDict)
def cls_score_predict():
    global prob_torch
    
    try:
        predicted_cls_idx_list = prob_torch.argmax(-1).tolist()
        predicted_cls_score_list = prob_torch.max(-1).values.tolist()
        predicted_cls_list = [model.config.id2label[idx] for idx in predicted_cls_idx_list]
        logger.info(f'Number of files: {len(predicted_cls_list)}')
        output_list = []
        
        for cls, score in zip(predicted_cls_list, predicted_cls_score_list):
            output_list.append({'label':cls, 'score': score})
        
        # print(output_list)

    except AttributeError as e:
        logger.error(f'AttributeError: {e}')
        output_list = []

    return {"output": output_list}
    
if __name__ == '__main__':
    uvicorn.run("__main__:app", host="0.0.0.0", port=80)