# Table of Contents

1.  [Repository Name](#repository-name)
2.  [Title of the Project](#title-of-the-project)
3.  [Project Overview and Objectives](#project-overview-and-objectives)
4.  [Dataset Details](#dataset-details)
5.  [Project Goal](#project-goal)
6.  [Requirements](#requirements)
7.  [How to use](#how-to-use)
8.  [Conclusion and Key Findings](#conclusion-and-key-findings)
9.  [Key Takeaways](#key-takeaways)
10. [Recommendation](#recommendation)
11. [Future Enhancements](#future-enhancements)
12. [Model Deployment Plan](#model-deployment-plan)
13. [Final Reflections](#final-reflections)
14. [Author](#author)
15. [Colab Link](#colab-link)
16. [References](#references)
----------------------------------------------

# Repository Name
aai-510-sound-of-feelings

----------------------------------------------

# Title of the Project
The Sound of Feelings

----------------------------------------------

# Project Overview and Objectives

- Speech is more than spoken words—it's the melody of emotions.
“The Sound of Feelings” explores vocal acoustics as emotional signatures. Combining signal processing, machine learning, and psychology, this project:

    - Uses the RAVDESS dataset to classify emotions in voice recordings.

    - Extracts acoustic features like MFCCs, pitch, energy, and ZCR.

    - Builds and compares multiple ML and DL models.

    - Investigates why certain vocal features correlate with specific emotions.
  
- Key questions explored:

    - Does a slow tempo imply sadness?

    - Do variations in pitch mean excitement?

The goal is not just to classify emotions but to understand the emotional cues behind how we speak.
----------------------------------------------

#  Dataset Details

- **Name of the Dataset:** RAVDESS Emotional Speech Audio
- **Description of the dataset:** Speech audio-only files (16bit, 48kHz .wav) from the RAVDESS. 

- Full dataset of speech and song, audio and video (24.8 GB) available from Zenodo. Construction and perceptual validation of the RAVDESS is described in our Open Access paper in PLoS ONE.
	- Zenodo Reference: https://zenodo.org/records/1188976
	- PLoS ONE Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391 

- **Dataset Files:** This portion of the RAVDESS contains 1012 files: 44 trials per actor x 23 actors = 1012. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Song emotions includes calm, happy, sad, angry, and fearful expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

- **File naming conventions:** Each of the 1012 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-02-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:

- **Filename identifiers:**

  - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
  - Vocal channel (01 = speech, 02 = song).
  - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
  - Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
  - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
  - Repetition (01 = 1st repetition, 02 = 2nd repetition).
  - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


- **Filename example:** 03-02-06-01-02-01-12.wav

  - Audio-only (03)
  - Song (02)
  - Fearful (06)
  - Normal intensity (01)
  - Statement "dogs" (02)
  - 1st Repetition (01)
  - 12th Actor (12)
  - Female, as the actor ID number is even.

----------------------------------------------

# Project Goal

- To close the gap between human emotional understanding and machine perception by teaching AI systems to:

    - Decode emotions from vocal cues

    - Identify patterns in how emotions manifest in audio (e.g., pitch = excitement)

    - Deliver emotion-aware predictions using interpretable models

    - The project not only builds emotion classifiers but seeks to understand why the emotion is being expressed through specific acoustic patterns.

----------------------------------------------

# Requirements

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tqdm
- joblib
- itertools
- json5
- librosa
- pyloudnorm
- xgboost
- lightgbm
- torch
- torchaudio
- tensorflow
- ipython
- notebook
- warnings

----------------------------------------------

# How to Use 

- Clone the Repo
  
git clone https://github.com/ShruForAI/aai-510-sound-of-feelings.git
cd aai-510-sound-of-feelings
pip install -r requirements.txt


- Google Colab Instructions
  
    - Download the dataset and upload it to your Google Drive.

    - Open the Colab notebook.

    - Update the dataset path in the Extract Audio Dataset section.

    - Run the notebook end-to-end.

----------------------------------------------

# Conclusion and Key Findings

| Model        | Accuracy (%) | F1-Score (%) | Notes                      |
| ------------ | ------------ | ------------ | -------------------------- |
| CatBoost     | **90.3**     | **90.2**     | Best overall performer     |
| DNN (Reg.)   | 85.5         | 85.2         | Suitable for real-time use |
| RandomForest | 84.3         | 84.0         | Strong classical baseline  |
| LogisticReg  | 66.2         | 64.9         | Weakest performer          |

- High performance achieved using CatBoost

- Deep Learning model offers lower latency for real-time applications

- No major overfitting observed

- Models are robust across speakers and emotion intensity
----------------------------------------------

# Key Takeaways

| Aspect           | CatBoost (ML)      | DNN (Reg.)       |
| ---------------- | ------------------ | ---------------- |
| Accuracy (%)     | **90.3**           | 85.5             |
| F1 Score (%)     | **90.2**           | 85.2             |
| Interpretability | High               | Moderate         |
| Inference Speed  | Slower (Batch)     | Fast (Real-time) |
| Use Case Fit     | Customer Analytics | Voice Assistants |


- MFCCs, pitch, energy, and ZCR are strong predictors of emotion.

- Ensemble ML models still outperform deep networks on structured features.

- DNN with dropout and batch normalization generalizes well for live use.

----------------------------------------------

# Recommendation

 - Deploy CatBoost for batch inference in emotion-aware analytics tools.

 - Use Regularized DNN for voice assistants and real-time emotion sensing.

 - Pilot test in customer support or telehealth environments.

 - Conduct annual fairness audits to avoid bias in gender, accent, or intensity.

 - Expand to multimodal systems by combining voice with facial and textual cues.

----------------------------------------------

# Future Enhancements

- Incorporate datasets like CREMA-D and TESS to improve generalization.

- Train end-to-end deep networks on raw waveforms using CNN-RNN or transformers.

- Integrate text sentiment and facial emotion for a complete multimodal model.

- Apply GridSearchCV, Optuna, and Keras Tuner for model optimization.

- Visualize feature importance using SHAP and interpretability libraries.

- Test latency and memory use for real-time readiness.
  
----------------------------------------------

# Model Deployment Plan

| Component    | Plan Description                                              |
| ------------ | ------------------------------------------------------------- |
| Use Case     | Voice assistant for live emotion feedback                     |
| Backend      | FastAPI-based Python API to serve model                       |
| Frontend     | HTML/JS or mobile app to capture and stream voice input       |
| Hosting      | AWS EC2 or Azure App Services                                 |
| Model Type   | CatBoost (Batch) & DNN (Real-time)                            |
| Latency Goal | Under 300 ms per inference                                    |
| Security     | HTTPS endpoints, token authentication, GDPR-compliant logging |

----------------------------------------------
# Final Reflections

- Classical ML models like CatBoost can outperform deep learning on structured audio features.

- Deep learning remains promising, especially for real-time systems, but needs careful tuning and infrastructure support.

- Understanding how a voice sounds is as powerful as what it says.

- Emotional intelligence in machines is achievable—and transformative.

----------------------------------------------

# Author

| Author            | Contact Details       |
|-------------------|-----------------------|
| Shruthi AK        | sak@sandiego.edu |

----------------------------------------------

# Colab Link

https://colab.research.google.com/drive/1S4_GhJBXFj0G1okKU8PK1Y-nmbp-qg8C?usp=sharing

----------------------------------------------

# References

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLOS ONE Paper

- Zenodo Dataset Access
