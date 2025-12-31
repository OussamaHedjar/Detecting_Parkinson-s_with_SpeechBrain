# Detecting_Parkinson-s_with_SpeechBrain

This project evaluates deep learning–based speech models for binary Parkinson’s Disease (PD) detection using the Italian Parkinson’s Voice and Speech Dataset. The focus is on **model robustness, stability under limited data, and reproducibility**, rather than peak accuracy alone.

## Dataset
- 831 speech recordings from 65 speakers
- 28 Parkinson’s Disease, 37 healthy controls
- **Speaker-level train/validation/test split** to prevent data leakage
- Two training regimes: **large** and **small** training sets

## Models
- **X-vector** (TDNN-based speaker embeddings)
- **ECAPA-TDNN** (channel-attentive TDNN)
- **Wav2Vec2** (self-supervised pretrained speech model)

## Experimental Setup
- Implemented using **SpeechBrain** and **HuggingFace**
- **Three random seeds** used for robustness evaluation: `1986`, `1234`, `2025`
- Early stopping based on validation performance
- Data augmentation evaluated for TDNN-based models
- Wav2Vec2 evaluated using **transfer learning** (frozen encoder)

## Results

| Model       | Training Size | Augmentation | Test Error (Mean ± Std) | Notes           |
|-------------|---------------|--------------|--------------------------|-----------------|
| X-vector    | Large         | No           | 0.0931 ± 0.0790          | Unstable        |
| X-vector    | Small         | No           | 0.0455 ± 0.0193          | Efficient       |
| X-vector    | Large         | Yes          | 0.0635 ± 0.0394          | Improved        |
| X-vector    | Small         | Yes          | 0.0461 ± 0.0153          | Most robust     |
| ECAPA-TDNN  | Large         | No           | 0.0703 ± 0.0171          | Stable          |
| ECAPA-TDNN  | Small         | No           | 0.1490 ± 0.0644          | Data-sensitive  |
| ECAPA-TDNN  | Large         | Yes          | 0.0681 ± 0.0361          | Slight gain     |
| ECAPA-TDNN  | Small         | Yes          | 0.1366 ± 0.0857          | High variance   |
| Wav2Vec2    | Small         | No           | 0.159 (single run)       | Exploratory     |

## Key Findings
- **X-vectors are highly robust** and perform well even with limited training data.
- **ECAPA-TDNN benefits from larger datasets** but degrades under data scarcity.
- **Data augmentation improves stability**, especially for X-vectors.
- **Wav2Vec2 requires more data and computation** to outperform TDNN-based models.

## Conclusion
X-vector architectures provide the best trade-off between performance, robustness, and computational efficiency for speech-based Parkinson’s Disease detection in low-resource settings.

## Tools
SpeechBrain · PyTorch · HuggingFace · Torchaudio · Scikit-learn
