# ONNX_Microsoft_ML_Face_Recognition
This is an exercise project using ONNX format models with Microsoft.ML, several pre-trained models were used to achieve face recognition and prediction of emotion, age and gender.
Click the button on the main window and select an image file from the dialog to recognize and predict.

Build passed on Visual Studio 2022, .NET 6.0.

<img src="images/image000.png" width="800"/>
- https://pxhere.com/en/photo/1266816

The following models were used, download and copy models to /onnx-models under the output directory.

#### [UltraFace](https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface)

> **Model:** version-RFB-320

> **License:** The MIT License

#### [Emotion FerPlus](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus)

> **Model:** emotion-ferplus-8

> **License:** The MIT License

#### [Age and Gender Classification using Convolutional Neural Networks](https://github.com/onnx/models/tree/main/vision/body_analysis/age_gender)

> **Models:** age_googlenet, gender_googlenet

> **License:** The Apache License 2.0
