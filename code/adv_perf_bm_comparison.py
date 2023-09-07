
from PIL import Image
from time import perf_counter
import numpy as np
from pathlib import Path
import platform
import datetime 
import os
from transformers import pipeline
from transformers import ViTForImageClassification
from transformers import LiltForSequenceClassification, LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast, LayoutLMv3Processor
import torchvision.transforms as trns
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json
from distutils.util import strtobool
from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser(description='Performance Benchmark for adv. models, if you do not have GPU and have a few checkpoints folders, it will take for a while.')
    parser.add_argument("--labels_json", "-la", type=str, default='backup_23148_4374.json', help='<labels json file name>')
    parser.add_argument("--checkpoints_dir", "-o", type=str, default='./output/checkpoints', help='<checkpoints directory (relative path)>')
    parser.add_argument("--perf_metrics_json", "-pm", type=str, default='perf_metrics.json', help='<performance metrics json file name>')
    parser.add_argument("--only_save_plot", "-sp", type=strtobool, default=True, help='<only_save_plot to the given "True" is without running performance evaluation from perf_metrics.json, as for "False", it will run both>')
    return parser

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x[0] for x in batch]),
        'labels': torch.tensor([x[1] for x in batch])
    }

def split_data_by_modality(modal = 'text'):
  if modal == 'text':
    test_dataloader = torch.load('./output/others/text_test_dataloader.pth')
  elif modal == 'vis':
    test_dataloader = torch.load('./output/others/vis_test_dataloader.pth')
  elif modal == 'mono':
    test_dataloader = torch.load('./output/others/mono_test_dataloader.pth')
  elif modal == 'mul':
    test_dataloader = torch.load('./output/others/mul_test_dataloader.pth')
  else:
    print('Given a one of modal types (text, vis, mono, mul)')
    return 0
  return test_dataloader

def save_performance_metrics(cur, perf_metrics_json):
    json_path = os.path.join(cur, 'others', perf_metrics_json)
    # Save the data to the JSON file
    with open(json_path, 'w') as json_file:
        json.dump(perf_metrics, json_file)

    return json_path

def plot_metrics(perf_metrics, current_optim_type, save_dir=None):
    
    df = pd.DataFrame.from_dict(perf_metrics, orient='index')
    for idx in df.index:
        df_opt = df.loc[idx]
        # Add a dashed circle around the current optimization type
        if idx == current_optim_type:
            plt.scatter(df_opt["time_avg_ms"], df_opt["accuracy"] * 100,
                        alpha=0.5, s=df_opt["size_mb"], label=idx,
                        marker='$\u25CC$',)
                        
        else:
            plt.scatter(df_opt["time_avg_ms"], df_opt["accuracy"] * 100,
                        s=df_opt["size_mb"], label=idx, alpha=0.5)
                      
    legend = plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    
    for handle in legend.legend_handles:
        handle.set_sizes([50])

    plt.ylim(75,99)
    xlim = int(perf_metrics["Adv_BERT(a_text_e6_s256_lr5e-5)"]["time_avg_ms"] * 60)
    plt.xlim(1, xlim)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Average latency (ms)")


    if save_dir is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"{timestamp}_perf_plot.png"
        save_path = os.path.join(save_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight')  # save plot and bbox_inches='tight' can be removed extra white spaces
    else:
        plt.show()

class UNITYDataset(Dataset):
    def __init__(self, transform, split='vis', img_size=384):
        self.data = []
        self.png_dict = {}
        self.img_size = img_size
        self.transform = transform
        self.split = split
        self.imgs_path = f'./code/others/mmocr_pre/{self.split}'
        file_list = [file for file in os.listdir(self.imgs_path)]
        # print(file_list)
        # create dictionary
        for i in file_list:
            ktr, page = i.split('/')[-1].split('_')[0], i.split('/')[-1].split('_')[-1].split('.')[0]
            png_page = int(page)
            self.png_dict[(ktr, png_page)] = i

        # since the string names are too long, we would like to ignore them. So pick snippet as key words
        long_name_list = ['20830_Further_Support_Enhancing']
        # corressponding
        for row in df_json.iterrows():
            jump = False
            image_path = row[1]['data']['image']
            for long_name in long_name_list:
                if long_name in image_path:
                # print(f'ignore: {image_path}')
                    jump = True
                    break

            # ignore long string file name
            if jump:
                continue
            # print(image_path)

            ktr = image_path.split('-')[1].split('_')[0]
            page_num = int(image_path.split('/')[-1].split('.')[0].split('_')[-1])
            choice = row[1]['annotations'][0]['result'][0]['value']['choices'][0]
            # we do not consider 'Others' category
            if choice!= 'Others':
                self.data.append([self.png_dict[(ktr, page_num)], choice])
                # print(self.png_dict[(ktr, page_num)], choice)


        self.class_map = {'Competencies': 0,
                        'Consultant Profile':1,
                        'Initial & Target Situation':2,
                        'Initial Situation':3,
                        'Offer Title':4,
                        'Project Calculation':5,
                        'Reference Details':6,
                        'Reference Overview':7,
                        'Target Situation':8,
                        'Working Package Description':9,
                        'Working Package Examples':10,
                        'Working Package Overview':11,
                    }

        self.img_dim = (self.img_size, self.img_size)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_name = self.data[idx]
        read_path = os.path.join(self.imgs_path, img_path)

        if self.split == 'vis':
            img = Image.open(read_path).convert('RGB')
        elif self.split == 'mono':
            img = Image.open(read_path).convert('L')
            img = Image.merge("RGB", [img]*3)

        if self.transform is not None:
            img = self.transform(img)

        class_id = self.class_map[label_name]

        return img, class_id

class PerformanceBenchmarkBase:
    def __init__(self, model, testset, optim_type, device, path):
        self.model = model
        self.dataloader = testset
        self.optim_type = optim_type
        self.device = device
        self.path = path

    def compute_accuracy(self):
        # Define this in the subclasses
        pass

    def compute_size(self):
        # Define this in the subclasses
        pass

    def time_pipeline(self):
        # Define this in the subclasses
        pass

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        # metrics[self.optim_type].update(self.compute_accuracy())
        return metrics

class TextPerformanceBenchmark(PerformanceBenchmarkBase):
    def __init__(self, pipeline, testset, optim_type, device, path):
        super().__init__(pipeline, testset, optim_type, device, path)
        self.pipeline = pipeline

    def compute_accuracy(self):
        model = self.pipeline.model
        model.eval()
        preds, labels = [], []
        device = self.device
        # Predict
        for batch in self.dataloader:
          # Unpack the inputs from our dataloader
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)


            logits = outputs.logits
            preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            labels.extend(b_labels.cpu().numpy())

        # Calculate accuracy
        accuracy = sum(pred == label for pred, label in zip(preds, labels)) / len(labels)
        print(f"Accuracy on test set - {accuracy:.3f}")
        return {"accuracy": accuracy}

    def compute_size(self):
        # Implementation for text method
        state_dict = self.pipeline.model.state_dict()
        tmp_path = Path(os.path.join(self.path, 'pytorch_model.bin'))
        # Calculate size in megabytes
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}

    def time_pipeline(self, query="iCert 20 Change Management Design Sprint Support Enclosure 1 Service Description to offer No 10831 Munich 20th June 2022 UNITY AG"):
        # Implementation for text method
        latencies = []
        # Warmup
        for _ in range(10):
            _ = self.pipeline(query)
        # Timed run
        for _ in range(100):
            start_time = perf_counter()
            _ = self.pipeline(query)
            latency = perf_counter() - start_time
            latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

class ImagePerformanceBenchmark(PerformanceBenchmarkBase):
    def __init__(self, model, testset, optim_type, device, path):
        super().__init__(model, testset, optim_type, device, path)
        self.model = model
        self.dataloader = testset

    def compute_accuracy(self):
        # Implementation for image method
        device = self.device
        model = self.model

        model.to(device)
        # Put model in evaluation mode
        model.eval()

        # Tracking variables
        preds, labels = [], []

        # Predict
        for batch in self.dataloader:
          # Add batch to GPU
          # batch = tuple(t.to(device) for t in batch)

          # Unpack the inputs from our dataloader
          # b_input_ids, b_input_mask, b_labels = batch

          b_pixels = batch['pixel_values'].to(device)
          b_labels = batch['labels'].to(device)

          # Telling the model not to compute or store gradients, saving memory and
          # speeding up prediction
          with torch.no_grad():
              # Forward pass, calculate logit predictions
              outputs = model(b_pixels,
                              labels=b_labels)


          logits = outputs.logits

          # Move logits and labels to CPU
          logits = torch.tensor(logits).to(device)
          label_ids = b_labels.to('cpu').numpy()

          # Store predictions and true labels
          preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
          labels.extend(label_ids)

        # Calculate accuracy
        accuracy = sum(pred == label for pred, label in zip(preds, labels)) / len(labels)
        print(f"Accuracy on test set - {accuracy:.3f}")

        return {"accuracy": accuracy}

    def compute_size(self):
        output_path = os.path.join(os.getcwd(), 'output')
        state_dict = self.model.state_dict()
        tmp_path = Path(os.path.join(self.path, 'pytorch_model.bin'))
        # Calculate size in megabytes
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        # Delete temporary file
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}

    def time_pipeline(self, img_path=None):
        # Implementation for image method
        latencies = []
        
        app_type = self.path.split('\\')[-1].split('_')[1]
        
        if app_type == 'vis':
            img_path = f'./single_img/{app_type}.jpg'
            image = Image.open(img_path).convert('RGB')
        else:
            img_path = f'./single_img/{app_type}.jpg'
            image = Image.open(img_path).convert('L')
            image = Image.merge("RGB", [image]*3)
        
        transform = trns.Compose([
            trns.Resize((384, 384)),
            trns.ToTensor(),
            trns.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        image = transform(image)
        image = image.unsqueeze(0)  # Add the batch dimension manually
        sequence_label = torch.tensor([4]) # 'Offer Title[4]'
        # Warmup
        for _ in range(10):
            _ = self.model(image,
                          labels=sequence_label)
        # Timed run
        for _ in range(100):
            start_time = perf_counter()
            _ = self.model(image,
                          labels=sequence_label)
            latency = perf_counter() - start_time
            latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

class MultimodalPerformanceBenchmark(PerformanceBenchmarkBase):
    def __init__(self, model, processor, runs_name, testset, optim_type, device, path):
        super().__init__(model, testset, optim_type, device, path)
                       
        self.model = model
        self.r_name = runs_name
        self.processor = processor


    def compute_accuracy(self):
        # Implementation for multimodal method
        device = self.device
        model = self.model
        model.to(device)
        # Put model in evaluation mode
        model.eval()
        preds, labels = [], []

        # Predict
        for batch in self.dataloader:
          # Unpack the inputs from our dataloader
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_box = batch[2].to(device)
          b_labels = batch[3].to(device)

          # Telling the model not to compute or store gradients, saving memory and
          # speeding up prediction
          with torch.no_grad():
              # Forward pass, calculate logit predictions
              outputs = model(b_input_ids,
                              token_type_ids=None,
                              attention_mask=b_input_mask,
                              bbox = b_box,
                              labels=b_labels)


          logits = outputs.logits
          preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
          labels.extend(b_labels.cpu().numpy())

        # Calculate accuracy
        accuracy = sum(pred == label for pred, label in zip(preds, labels)) / len(labels)
        print(f"Accuracy on test set - {accuracy:.3f}")
        return {"accuracy": accuracy}

    def compute_size(self):
        # Implementation for multimodal method
        state_dict = self.model.state_dict()
        tmp_path = Path(os.path.join(self.path, 'pytorch_model.bin'))
        # Calculate size in megabytes
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}

    def time_pipeline(self, img_path="./output/FewWordImg/10831_Enclosure 1 – Service Description to offer No_1.PNG"):
        # Implementation for multimodal method
        latencies = []
        # extract sequence length by file name, note that
        seq = self.r_name.split('_')[3].split('s')[-1]

        # one test image
        image = Image.open(img_path).convert('RGB')
        # process encoding
        encoded_inputs = self.processor(image,
                                        return_tensors="pt",
                                        max_length = int(seq),    # Pad & truncate all sentences.
                                        padding = 'max_length',
                                        truncation = True,)

        sequence_label = torch.tensor([4]) # 'Offer Title[4]'
        # Warmup
        for _ in range(10):
            _ = self.model(input_ids = encoded_inputs['input_ids'],
                          attention_mask = encoded_inputs['attention_mask'],
                          bbox = encoded_inputs['bbox'],
                          labels = sequence_label)
        # Timed run
        for _ in range(100):
            start_time = perf_counter()
            _ = self.model(input_ids = encoded_inputs['input_ids'],
                          attention_mask = encoded_inputs['attention_mask'],
                          bbox = encoded_inputs['bbox'],
                          labels = sequence_label)
            latency = perf_counter() - start_time
            latencies.append(latency)

        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()  
    
    # parameters for adjestment  
    # labels_json = 'backup_23148_4374.json'
    # checkpoints_dir = "./output/checkpoints"
    labels_json = args.labels_json
    checkpoints_dir = args.checkpoints_dir
    # Specify the path to the JSON file
    perf_metrics_json = args.perf_metrics_json
    os_type = platform.system().lower()
    only_save_plot = args.only_save_plot
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set current working directory to the given path
    # os.chdir('./code/')
    cwd = os.getcwd()
    print(cwd)

    # make completed paths
    ck_path = [os.path.join(cwd, 'output/checkpoints', folder) for folder in os.listdir(checkpoints_dir)]

    current_path = os.getcwd() + '/output'
    df_json = pd.read_json(os.path.join(current_path, 'LABELS', labels_json))
    # have done for annotations
    df_json = df_json[df_json['annotations'].apply(lambda x: len(x) != 0)]
    # annotation results exist
    df_json = df_json[df_json['annotations'].apply(lambda x: len(x[0]['result']) != 0)]
    # 22348 for 13th slide
    count = 13

    for idx, row in enumerate(df_json.iterrows()):
        slide = row[1]
        image_path = slide['data']['image']
        choice = slide['annotations'][0]['result'][0]['value']['choices'][0]
        # print(file_path, choice)
        if image_path.find('22348')!=-1:
            # print(image_path, choice)
            new_image_name = image_path.split('.')[0] + '_' + str(count) + '.PNG'
            # update new name that matches length limitation
            slide['data']['image'] = new_image_name
            image_path = new_image_name
            # print(image_path)
            count += 1

    create_json = False 

    if not os.path.exists(os.path.join(current_path, 'others', perf_metrics_json)):
        create_json = True
    else:
        # Read data from the JSON file
        with open(os.path.join(current_path, 'others', perf_metrics_json), 'r') as json_file:
            perf_metrics = json.load(json_file)

    
    if not only_save_plot:

        for path in ck_path:
            if os_type == 'windows':
                folder = path.split('\\')[-1]
            else:
                folder = path.split('/')[-1]

            app_ = folder.split('_')[1]
            print(f'Performance Benchmark: {folder} ---------------')
            print(path)

            if app_ == 'text':
                test_dataloader = split_data_by_modality(modal = app_)
                assert test_dataloader != 0, 'please check again'
                pipe = pipeline("text-classification", model=path)
                pb = TextPerformanceBenchmark(pipe, test_dataloader, f'Adv_BERT({folder})', device, path)
            elif (app_ == 'vis') or (app_ == 'mono'):
                test_dataloader = split_data_by_modality(modal = app_)
                assert test_dataloader != 0, 'please check again'
                model = ViTForImageClassification.from_pretrained(path)
                pb = ImagePerformanceBenchmark(model, test_dataloader, f'Adv_ViT({folder})', device, path)
            elif app_ == 'mul':
                test_dataloader = split_data_by_modality(modal = app_)
                assert test_dataloader != 0, 'please check again'
                feature_extractor = LayoutLMv3ImageProcessor(ocr_lang="eng+deu")
                tokenizer = LayoutLMv3TokenizerFast.from_pretrained(path)
                processor = LayoutLMv3Processor(feature_extractor, tokenizer)
                model = LiltForSequenceClassification.from_pretrained(path)
                pb = MultimodalPerformanceBenchmark(model, processor, folder, test_dataloader, f'Adv_LiLT({folder})', device, path)
            else:
                assert app_ == ('text' or 'vis' or 'mono' or 'mul' ), 'please check app_'

            # Run the benchmark first time to create the JSON file
            if create_json:
                perf_metrics = pb.run_benchmark()
                create_json = False
            else:
                perf_metrics.update(pb.run_benchmark())

            final_pth = save_performance_metrics(current_path, perf_metrics_json)
            print(f"{folder} data saved to {final_pth}")
            plot_metrics(perf_metrics, 'Adv_LiLT(a_mul_e6_s256_lr6e-6)', save_dir=os.path.join(current_path, 'others'))
    else:
        plot_dir=os.path.join(current_path, "others")
        print(f'only save plot in directory-------->{plot_dir}')
        plot_metrics(perf_metrics, 'Adv_LiLT(a_mul_e6_s256_lr6e-6)', save_dir=plot_dir)