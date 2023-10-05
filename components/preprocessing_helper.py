# Preprocess images and labels
from PIL import Image
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3Processor, AutoTokenizer


def preprocess(img: Image, processor_ins: LayoutLMv3Processor):

    processor = processor_ins
    encoded_inputs = processor(img,
                               return_tensors="pt",
                               max_length = 512,    # Pad & truncate all sentences.
                               padding = 'max_length',
                               truncation = True,
                        )

    return encoded_inputs