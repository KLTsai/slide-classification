import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from components.preprocessing_helper import preprocess
from components.setup_model import load_model
from components.api_schemas import ResponseDict, PredictDict
from typing import List
import uvicorn
import io

# Start up app
app = FastAPI()

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

    img_list = []
    for file in files:
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content))
        img_list.append(img)
    
    encoded_inputs = preprocess(img_list, processor)

    # Process input images - make required strings subs and tokenize
    # encoding = tokenizer(words, boxes=boxes, return_tensors="pt")


    # Run through model and normalize
    outputs = model(input_ids = encoded_inputs['input_ids'], 
                    attention_mask = encoded_inputs['attention_mask'], 
                    bbox = encoded_inputs['bbox'])
    
    logits = outputs.logits
    
    prob_torch = torch.softmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1).tolist()

    output_dict = {}
    for name, scores in zip(files, probabilities):
        score_dict = {label: score for label, score in zip(class_map, scores)}
        output_dict[name.filename] = score_dict

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

    # print(prob_torch.argmax(-1).tolist())

    predicted_cls_idx_list = prob_torch.argmax(-1).tolist()
    predicted_cls_score_list = prob_torch.max(-1).values.tolist()
    predicted_cls_list = [model.config.id2label[idx] for idx in predicted_cls_idx_list]
    output_list = []
    
    for cls, score in zip(predicted_cls_list, predicted_cls_score_list):
        output_list.append({cls: score})



    return {"output": output_list}


if __name__ == '__main__':
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True, workers=1)