# Body Fat Prediction Using Full Body Pictures

## Abstract

This project leverages AI to estimate body fat percentage using a frontal and side full-body image, along with the individual's height and sex. 

## Input Guidelines

To ensure accurate predictions, follow these guidelines when capturing the input images:

- **Frontal Image**: The individual should place their right hand on their forehead and ensure a gap between their left elbow and left side.
- **Side Image**: The individual should stand with their left side facing the camera and hold their right shoulder with their right hand.
- **Clothing**: The individual should be shirtless or wear tight-fitting clothing to improve prediction accuracy. The abdomen and neck should remain free of any obstructions.
- **Distance**: Maintain a distance of 1-2 meters from the background for optimal results.

### Sample Input Images

<div style="display: flex; justify-content: space-around;">
  <img src="README FILES/Front_sample.jpg" alt="Frontal Image Sample" style="width: 45%;"/>
  <img src="README FILES/Left_sample.jpg" alt="Side Image Sample" style="width: 45%;"/>
</div>

## Methodology

1. **Pixel Size Calculation**: Pixel size in centimeters is determined using the individual's height and the distance between their right wrist and right heel in the images.
2. **Depth Estimation**: Depth maps of the images are generated using [Depth Anything](https://github.com/LiheYoung/Depth-Anything/).
3. **Segmentation and Pose Estimation**: Human segmentation is performed using [CDCL](https://github.com/kevinlin311tw/CDCL-human-part-segmentation) and refined with depth estimation results.
4. **Measurement Extraction**: The neck and abdomen circumferences are calculated from the segmented images.
5. **Body Fat Prediction**: The model predicts body fat percentage using neck circumference, abdomen circumference, height, and sex as inputs.

### Depth Estimation Samples

<div style="display: flex; justify-content: space-around;">
  <img src="README FILES/Front_depth.png" alt="Frontal Depth Estimation" style="width: 45%;"/>
  <img src="README FILES/Left_depth.png" alt="Side Depth Estimation" style="width: 45%;"/>
</div>

### Improved Segmentation and Pose Estimation Samples

<div style="display: flex; justify-content: space-around;">
  <img src="README FILES/Front_segmented_sample.jpg" alt="Frontal Segmented Image" style="width: 45%;"/>
  <img src="README FILES/Left_segmented_sample.jpg" alt="Side Segmented Image" style="width: 45%;"/>
</div>

## Model Accuracy

The model achieves the following levels of accuracy:

- **Mean Absolute Error (MAE) for Women**: 2.89
- **Mean Absolute Error (MAE) for Men**: 3.13

### Bland–Altman Plot for Men

<div style="display: flex; justify-content: space-around;">
  <img src="README FILES/men_altman.png" alt="Bland–Altman Plot for Men"/>
</div>

### Bland–Altman Plot for Women

<div style="display: flex; justify-content: space-around;">
  <img src="README FILES/women_altman.png" alt="Bland–Altman Plot for Women"/>
</div>

## Usage

### Installation

To set up the project, follow these steps:

```bash
git clone https://github.com/itsjibel/Body-Fat-Estimation-Using-Pictures
cd Body-Fat-Estimation-Using-Pictures
mkdir input
pip install -r requirements.txt
chmod +x run.sh
```

Place your input images in the `input` folder.

### Running the Script

To estimate body fat, run the following command:

```bash
python estimate-body-fat.py height sex
# The height should be in cm and the sex should be 'male' or 'female'.
# For example:
python estimate-body-fat.py 190 male
```
