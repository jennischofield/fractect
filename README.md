# fractect
FracTect is a system that combines a classification model (ResNeXt) with a detection model (Faster R-CNN) to classify distal radius (wrist) fractures as Fractured or Not Fractured with degrees of confidence, as well as detect objects of interest in an X-ray.

The system runs locally using the Flask framework, and it'll be accessible on localhost after setting it up. 

### Classification Model

The classification model will take an X-ray input and then classify it into Fractured or Not Fractured but will give its degree of confidence based on pre-defined thresholds.

|Prediction Confidence | Description |
|:----------------------:|:-------------:|
| >=85%                |Fractured/Not Fractured|
| >=65%, <85%          | More Likely Fractured/Not Fractured|
| <65%                 | Unsure|

It will also generate a Grad-CAM heatmap based on where it is activated.

Example output:

![Grad-CAM classification example](https://github.com/jennischofield/fractect/blob/main/readme_images/Grad-CAM_Example.jpg?raw=true)

### Detection Model

The detection model also takes in an X-ray, as well as the user's detection threshold (how confident the model must be to display the bounding box) and which objects they'd like to see. The model expects fractured X-rays (and was exclusively trained on them), but has decent performance on not fractured X-rays as well. There are nine possible object types - Fracture, Metal, Periosteal Reaction, Pronator Sign, Soft Tissue, Text, Bone Anomalies, Bone Lesions, and Foreign Bodies. However, due to the dataset being focused on bone fractures (and therefore, the training data has underrepresented classes), the performance is best on commonly seen objects (Fracture, Metal, and Text). Full statistics on performance can be seen in the associated write-up. The detection model will produce an image with bounding boxes around objects of interest and with a confidence percentage next to each bounding box.

Example output: 

![Detection example](https://github.com/jennischofield/fractect/blob/main/readme_images/Detection_Example.jpg?raw=true)

### UI

The UI runs locally and includes documentation, links to this repo, contact details, and the interface itself. More details about the UI can be found in the write-up.

Example of FracTect Tab:

![UI example](https://github.com/jennischofield/fractect/blob/main/readme_images/UI_Example.png?raw=true)

### Setting Up
This repo includes all development work along the way, but to use FracTect, a tester package has been made, but due to the limitations of GitHub storage, the model files can't be uploaded. If you want access to the tester package, please reach out at schofieldjenni@gmail.com, and I'll get it to you. However, the instructions to start it up can be found at Fractect_tester_pack_instructions.txt in this repo, but the model files would still need to be sent directly to you. 

The formal writeup for FracTect can also be found in this repo, under FracTect_Writeup.pdf.

If there are any questions, feel free to contact me at schofieldjenni@gmail.com
