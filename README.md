# Extract object masks by Grounded SAM 2

**Please note that we use the method of [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) and we modified to generate the masks of the objects in RGB images**

## Set up your datasets:
```bash
datasets/
│
├── ds01/
│   ├── images
```
## Extract object masks:
```bash
python grounded_sam2_hf_model_imgs_MaskExtract.py --path {PATH}
```
