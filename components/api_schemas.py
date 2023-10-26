from pydantic import BaseModel
from PIL import Image
from typing import List, Union


# Define input and output schema
class Cls_Input(BaseModel):
    input_path_list: List[str]


class ResponseDict(BaseModel):
    output: dict[str, dict[str, float]]


    # For example
    class Config:
        schema_extra = {
            "example": {
                "output": {
                    "FILE1.PNG": {
                        "Competencies": 0.9584630131721497,
                        "Consultant Profile": 0.022195372730493546,
                        "Initial & Target Situation": 0.0011004683328792453,
                        "Initial Situation": 0.001856239978224039,
                        "Offer Title": 0.0016864242497831583,
                        "Project Calculation": 0.0022484734654426575,
                        "Reference Details": 0.0020657696295529604,
                        "Reference Overview": 0.0027718045748770237,
                        "Target Situation": 0.002087327418848872,
                        "Working Package Description": 0.001168415998108685,
                        "Working Package Examples": 0.0024193846620619297,
                        "Working Package Overview": 0.0019373849499970675
                    },
                    "FILE2.PNG": {
                        "Competencies": 0.002081812359392643,
                        "Consultant Profile": 0.9909632802009583,
                        "Initial & Target Situation": 0.0006276462227106094,
                        "Initial Situation": 0.0006962332990951836,
                        "Offer Title": 0.0008524305885657668,
                        "Project Calculation": 0.0005045356811024249,
                        "Reference Details": 0.00040568120311945677,
                        "Reference Overview": 0.0007785540074110031,
                        "Target Situation": 0.0003945175267290324,
                        "Working Package Description": 0.0013996601337566972,
                        "Working Package Examples": 0.0004961551749147475,
                        "Working Package Overview": 0.000799578323494643
                    }
                }
            }
        }

class PredictDict(BaseModel):
    output: List[dict[str, Union[str, float]]]

    # For example
    class Config:
        schema_extra = {
            "output": [
                        {
                            "label": "Competencies",
                            "score": "0.9584630131721497"
                        },
                        {
                            "label": "Consultant Profile",
                            "score": "0.9909632802009583"
                        }
                ]

        }