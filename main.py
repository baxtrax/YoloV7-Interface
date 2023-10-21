# Code reference: https://huggingface.co/spaces/akhaliq/yolov7/blob/main/app.py
import gradio as gr
from PIL import Image
import time
from pathlib import Path

import torch
from numpy import random
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
                          scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, time_synchronized, TracedModel

from tqdm import tqdm

# model_names = ['YOLOv7', 'YOLOv7-X', 'YOLOv7-W6', 'YOLOv7-E6', 'YOLOv7-D6', 'YOLOv7-E6E']
# models = {model_name: }

classes = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

def main():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        weights = gr.State('./weights/yolov7.pt')
        is_img = gr.State(True)
        with gr.Row():
            gr.Markdown('# YOLOv7 Demo\n Example interface to mess around with \
                        YOLOv7. Start by uploading a video or image')
        with gr.Row():
            with gr.Column():
                with gr.Tab(label='Image') as img_tab:
                    with gr.Row():
                        gr.Markdown("Upload or take a picture:")
                    with gr.Row():
                        input_img = gr.Image(label='Input Image', type='filepath')
                        input_img_web = gr.Image(label='Input Image', type='filepath', source='webcam')
                with gr.Tab(label='Video') as vid_tab:
                    with gr.Row():
                        gr.Markdown("Upload or record a video:")
                    with gr.Row():
                        input_vid = gr.Video(label='Input Video',)
                        input_vid_web = gr.Video(label='Input Video', source='webcam', format='mp4')

                submit_btn = gr.Button('Submit')
                with gr.Accordion("Settings", open=False):
                    with gr.Row(variant='panel'):
                        with gr.Accordion(label="Inference Settings", open=False):
                            gr.Markdown("Settings related to inference and classes")
                            class_filter_dd = gr.Dropdown(label="Class Filter", info="Which classes to filter for, unselected classes are ignored", multiselect=True, choices=classes)
                            augment_infer_cb = gr.Checkbox(label="Augmented Inference", info="Whether to augment the input image before inference", value=True, interactive=True)
                    with gr.Row(variant='panel'):
                        with gr.Accordion(label="Non-Maximum Suppression Settings", open=False):
                            gr.Markdown("Controls the non-maximum suppression algorithm.")
                            conf_thres_sl = gr.Slider(label="Confidence Threshold", info="Acts as a confidence filter", value=0.25, minimum=0.01, maximum=1.0, step=0.01, interactive=True)
                            iou_thres_sl = gr.Slider(label="IOU Threshold", info="How much the boxes need to overlap to merge them together", value=0.45, minimum=0.01, maximum=1.0, step=0.01, interactive=True)
                            agnostic_nms_cb = gr.Checkbox(label="Class Agnostic", value=True, interactive=True)
                    with gr.Row(variant='panel'):
                        with gr.Accordion(label="Image settings", open=False):
                            gr.Markdown("Image size, any value between 1 and 4096, recommended 640")
                            img_size_num = gr.Number(label="Image Size", value=640, minimum=32, maximum=4096, step=1, interactive=True)

            with gr.Column() as img_output_col:
                output_img = gr.Image(label='Output Image', type='pil')
                context_txt = gr.Textbox(label='Description', interactive=False)
                labels = gr.Label(label='Top 5 Scoring Objects', num_top_classes=5)

            with gr.Column(visible=False) as video_output_col:
                output_vid = gr.Video(label='Output Video', interactive=False, autoplay=True)
    
        submit_btn.click(detect,
                            inputs=[input_img, input_img_web, input_vid, input_vid_web, is_img, weights, conf_thres_sl, iou_thres_sl, class_filter_dd, img_size_num, agnostic_nms_cb, augment_infer_cb],
                            outputs=[output_img, output_vid, labels, context_txt])
        
        img_tab.select(lambda: (True, gr.update(visible=True), gr.update(visible=False)),
                       outputs=[is_img, img_output_col, video_output_col])
        vid_tab.select(lambda: (False, gr.update(visible=False), gr.update(visible=True)),
                       outputs=[is_img, img_output_col, video_output_col])
        
        
    demo.queue().launch()

def detect(img, img_web, vid, vid_web, is_img, weights, conf_thres, iou_thres, classes_filter, img_size, agnostic, augment, progress=gr.Progress(track_tqdm=True)):
    progress(0.0, desc="Setting up detection...")

    imgsz_orig = img_size
    imgsz = imgsz_orig # This value will be overridden
    save_path = './inference/output.mp4'

    if len(classes_filter) == 0:
        classes_filter = None

    if classes_filter is not None: # Convert to indexes
        classes_filter = [classes.index(c) for c in classes_filter]

    if is_img:
        if img is None and img_web is None:
            raise gr.Error("Please upload an image to detect on!")
        elif img is not None and img_web is not None:
            raise gr.Error("Please upload only one image at a time!")
        elif img is None:
            img = img_web
        else:
            img = img

        # TODO, gradio should already save img, so use this path?
        source = img

    else:
        if vid is None and vid_web is None:
            raise gr.Error("Please upload a video to detect on!")
        elif vid is not None and vid_web is not None:
            raise gr.Error("Please upload only one video at a time!")
        elif vid is None:
            vid = vid_web
        else:
            vid = vid

        # TODO, gradio should already save vid, so use this path?
        source = vid

    trace = False
    # Quick way to trick source logic.

    # Initialize
    set_logging() # Setups the logging for YoloV7   
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    gr.Info(f'Using {device} for inferenece.')
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

    # Set Dataloader
    progress(0.2, desc="Loading data...")
    vid_path, vid_writer = None, None # Used later if input is video
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
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes_filter, agnostic=agnostic)
        t3 = time_synchronized()

        # Process detections
        labels = {}
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
                    if (names[int(cls)] not in labels or labels[names[int(cls)]] < conf.item()) and conf is not None:
                        labels[names[int(cls)]] = conf.item()

                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            if not is_img:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if vid_writer is not None:
        vid_writer.release()  # release previous video writer
        vid_cap.release()

    print(f'Done. ({time.time() - t0:.3f}s)')
    progress(1.0, desc="Done.")

    if is_img:
        return Image.fromarray(im0[:,:,::-1]), None, labels, s
    else:
        print(save_path)
        return None, save_path, None, None

main()