import os
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LayoutLMv3ImageProcessor, LayoutLMv3Processor
 
logger.add(
    os.path.join(os.getcwd(), 'code', 'output', 'logs', 'setup_model.log'),
    format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    backtrace = True,
    diagnose = True
)


def load_model():
    # TODO: load model from model_path
    MODEL = 'kungeer/adv_offerslide_mul_lilt_v0.0'
    # MODEL = './code/output/checkpoints/a_mul_e6_s256_lr6e-6'
    tokenizer = AutoTokenizer.from_pretrained(MODEL, 
                                              use_auth_token='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', 
                                              cache_dir='./output/pretrained')
    feature_extractor = LayoutLMv3ImageProcessor(ocr_lang="eng+deu")
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)
    classifier = AutoModelForSequenceClassification.from_pretrained(MODEL, 
                                                                    use_auth_token='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', 
                                                                    cache_dir='./output/pretrained')
    logger.info('Model Load Success!')
    return tokenizer, processor, classifier

if __name__ == '__main__': 
    tokenizer_, processor_, classifier_ = load_model()
    print('main model loaded')
