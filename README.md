# Waste_Segregation_Model

**Agenda** : Waste segregation, separating biodegradable and non-biodegradable materials, is crucial for effective waste management, environmental protection, and resource conservation.
Segregation enables efficient recycling, composting, and other waste processing methods, reducing landfill burden and minimizing environmental pollution. 

**About the Model** : This Waste_Segregation_Model is built using CNN transfer learning model (MobileNetV2), and Binary Classification to segregate Waste images into Biodegradable and Non-Biodegradable.

**MobileNetV2 Archietecture** :
1. **Input Size**: Images resized to **128x128px** (RGB) and normalized.  
2. **Feature Extraction**: MobileNetV2 uses **depthwise separable convolutions** (not traditional max pooling) to progressively downsample.  
3. **Spatial Reduction**:  
   - Starts at 128x128 → **64x64** (after 1st layer) → **32x32** → **16x16** → **8x8** → **4x4** (final feature map).  
4. **Bottleneck Layers**: Expands channels (e.g., 32→96→144) before compressing spatial dimensions.  
5. **Final Output**: Global averaging reduces 4x4 features → **1x1** vector for classification.


**Model Accuracy**:
Training Accuracy: 92.86%

Test Accuracy: 91.96%


**For Dataset** :https://www.kaggle.com/datasets/techsash/waste-classification-data


**Streamlit Link** :https://waste-segregation-model.streamlit.app/


**Sample Output**:

<img width="2537" height="1075" alt="Screenshot 2025-08-13 183252" src="https://github.com/user-attachments/assets/12e4236a-c5dc-409d-9196-87a1584c9915" />

Sample 1: <img width="1324" height="820" alt="Screenshot 2025-08-13 183259" src="https://github.com/user-attachments/assets/9e6e5539-eab4-4e1d-844c-ad4147c4677e" />


Sample 2:<img width="1378" height="1331" alt="Screenshot 2025-08-13 183903" src="https://github.com/user-attachments/assets/92bc49e5-ace9-49e2-bd44-fa74cc77285e" />


Sample 3:<img width="1363" height="907" alt="Screenshot 2025-08-13 183446" src="https://github.com/user-attachments/assets/d3cbf086-4160-41e1-a13b-6e6b78090862" />


Sample 4:<img width="1264" height="1001" alt="Screenshot 2025-08-13 183411" src="https://github.com/user-attachments/assets/db8328fd-ec1a-4576-8244-4f95d23e21e1" />


Sample 5:<img width="1314" height="1235" alt="Screenshot 2025-08-13 183346" src="https://github.com/user-attachments/assets/fdc4e9fd-5965-40dd-a603-b2eb37890668" />



