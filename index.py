import torch
import os
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from components.preprocessing_helper import preprocess
from components.setup_model import load_model
from components.api_schemas import ResponseDict, PredictDict
from loguru import logger
from typing import List
import uvicorn
import io

logger.add(
    os.path.join(os.getcwd(), 'code', 'output', 'logs', 'setup_model.log'),
    format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    backtrace = True,
    diagnose = True
)

# Start up app
app = FastAPI(root_path = "/slide-classification")
prob_torch = None

# Ready model
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
async def cls_score(files: List[UploadFile] = File(...)):
    global prob_torch

    if prob_torch is not None:
        del prob_torch 

    img_list = []
    for file in files:
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content))
        img_list.append(img)

    logger.info(f'preprocessing image')
    encoded_inputs = preprocess(img_list, processor)

    logger.info(f'running model')
    # Run through model and normalize
    try:
        outputs = model(input_ids = encoded_inputs['input_ids'], 
                        attention_mask = encoded_inputs['attention_mask'], 
                        bbox = encoded_inputs['bbox'])
    except Exception as e:
        print(f"An exception occurred: {e}")

    logger.info(f'inference done')

    logits = outputs.logits
    
    prob_torch = torch.softmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1).tolist()


    output_dict = {}
    for name, scores in zip(files, probabilities):
        score_dict = {label: score for label, score in zip(class_map, scores)}
        output_dict[name.filename] = score_dict


    # img_list.clear()
    # gc.collect()
    # torch.cuda.empty_cache()
    
    # predicted_class_idx = probabilities.argmax(-1).item()
    # predicted_class = model.config.id2label[predicted_class_idx]
    # predicted_class_score = probabilities[0][predicted_class_idx].item()

    # print("Predicted class:", predicted_class)
    # print("Predicted class score:", predicted_class_score)
    # predicted_class_idx = outputs.logits.argmax(-1).item()
    # predicted_class = model.config.id2label[predicted_class_idx]
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
        
    except AttributeError as e:
        logger.error(f'AttributeError: {e}')
        output_list = []

    return {"output": output_list}
    
if __name__ == '__main__':
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True, workers=4)