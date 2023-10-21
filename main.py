# Code reference: https://huggingface.co/spaces/akhaliq/yolov7/blob/main/app.py
import gradio as gr
from PIL import Image
import time
from pathlib import Path

import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
                          scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, time_synchronized, TracedModel

from tqdm import tqdm

# model_names = ['YOLOv7', 'YOLOv7-X', 'YOLOv7-W6', 'YOLOv7-E6', 'YOLOv7-D6', 'YOLOv7-E6E']
# models = {model_name: }

def main():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        weights = gr.State('./weights/yolov7.pt')
        with gr.Row():
            gr.Markdown('# YOLOv7 Demo')
        with gr.Row():
            with gr.Column():
                with gr.Tab(label='Image'):
                    with gr.Row():
                        input_img = gr.Image(label='Input Image', type='pil')
                        input_img_web = gr.Image(label='Input Image', type='pil', source='webcam')
                with gr.Tab(label='Video'):
                    with gr.Row():
                        input_vid = gr.Video(label='Input Video',)
                        input_vid_web = gr.Video(label='Input Video', source='webcam')

                submit_btn = gr.Button('Submit')
                with gr.Accordion("Settings", open=False):
                    conf_thres_sl = gr.Slider(label="Confidence Threshold", value=0.25, minimum=0.01, maximum=1.0, step=0.01, interactive=True)
                    iou_thres_sl = gr.Slider(label="IOU Threshold", value=0.45, minimum=0.01, maximum=1.0, step=0.01, interactive=True)
            with gr.Column():
                output_img = gr.Image(label='Output Image', type='pil')
    
        submit_btn.click(detect,
                            inputs=[input_img, input_img_web, weights, conf_thres_sl, iou_thres_sl],
                            outputs=output_img)
        
    demo.queue().launch()

def detect(img, img_web, weights, conf_thres, iou_thres, progress=gr.Progress(track_tqdm=True)):
    progress(0.0, desc="Setting up detection...")
    imgsz_orig = 640
    imgsz = imgsz_orig
    classes_filter = None

    if img is None and img_web is None:
        raise gr.Error("Please upload an image to detect on!")
    elif img is not None and img_web is not None:
        raise gr.Error("Please upload only one image at a time!")
    elif img is None:
        img = img_web
    else:
        img = img

    # Quick way to trick source logic.
    img.save('uploaded_images/test.jpg')
    source = 'uploaded_images/'
    trace = False

    # Initialize
    set_logging() # Setups the logging for YoloV7   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    progress(0.1, desc="Model setup...")
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, imgsz_orig)

    if half:
        model.half()  # to FP16

    # Second-stage classifier

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    progress(0.2, desc="Loading data...")
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    progress(0.3, desc="Running inference...")
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes_filter)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for _, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    print(f'Done. ({time.time() - t0:.3f}s)')
    progress(1.0, desc="Done.")

    return Image.fromarray(im0[:,:,::-1])

main()