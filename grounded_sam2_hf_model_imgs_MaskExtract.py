import cv2
import os
import torch
import numpy as np
import supervision as sv
from tqdm import tqdm
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

# environment settings
# use bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='input your path')
parser.add_argument('--text', type=str, default="foreground object.", help='input your text')
args = parser.parse_args()
path = args.path

text = args.text
# scene = "BlackBunny"
img_path = f'{path}/images/'
save_path = f'{path}/'
annotated_path = os.path.join(save_path, "annotated")
mask_path = os.path.join(save_path, "masks")
if not os.path.exists(annotated_path):
    os.makedirs(annotated_path)
if not os.path.exists(mask_path):
    os.makedirs(mask_path)

Images = sorted(os.listdir(img_path), key=lambda x: x.zfill(10))

num_no_detection = 0
for idx, img_name in tqdm(enumerate([img_name for img_name in Images if img_name.endswith('.png')]), total=len(Images)):
    image_path = os.path.join(img_path, img_name)
    image = Image.open(image_path)
    image = image.convert("RGB")

    sam2_predictor.set_image(np.array(image.convert("RGB")))

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )


    """
    Results is a list of dict with the following structure:
    [
        {
            'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
            'labels': ['car', 'tire', 'tire', 'tire'], 
            'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                            [1392.4701,  554.4064, 1628.6133,  777.5872],
                            [ 436.1182,  621.8940,  676.5255,  851.6897],
                            [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
        }
    ]
    """

    # get the box prompt for SAM 2
    input_boxes = results[0]["boxes"].cpu().numpy()
    if input_boxes.shape[0] == 0: # filter images with no detection
        save_mask_path = os.path.join(mask_path, os.path.basename(img_name).replace(".jpg", ".png"))
        cv2.imwrite(save_mask_path, (np.zeros_like(image)).astype(np.uint8))
        num_no_detection += 1
        continue

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )


    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]
    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    """
    Note that if you want to use default color map,
    you can set color=ColorPalette.DEFAULT
    """
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    save_annotated_path = os.path.join(annotated_path, os.path.basename(img_name).replace(".jpg", ".png"))
    cv2.imwrite(save_annotated_path, annotated_frame)

    save_mask_path = os.path.join(mask_path, os.path.basename(img_name).replace(".jpg", ".png"))
    cv2.imwrite(save_mask_path, (masks[0]*255).astype(np.uint8))

print(f"num_no_detection: {num_no_detection}")