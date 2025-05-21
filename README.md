<!-- ## Localized Weather Prediction Using Kolmogorov-Arnold Networks and Deep RNNs -->

# Localized Weather Prediction  Using Kolmogorov-Arnold Network-Based Models and Deep RNNs
This repository contains the code and resources for the research project titled "Localized Weather Prediction Using Kolmogorov-Arnold Network-Based Models and Deep RNNs." The project focuses on developing and comparing advanced deep learning models for accurate weather forecasting.

📝 Description
This research investigates the efficacy of novel Kolmogorov-Arnold Network (KAN) based architectures alongside traditional Deep Recurrent Neural Networks (RNNs) for weather prediction in specific geographical locations. The primary goal is to assess which model paradigm offers superior performance and interpretability for localized weather forecasting tasks. We explore various configurations, including different activation functions within Temporal KAN (TKAN) variants, and conduct a comprehensive comparative analysis.

✨ Key Features & Contributions
Comparative Analysis: A rigorous comparison of KAN, various TKAN architectures (including Simple TKAN with SiLU, Mish, and GELU activations), and traditional Deep RNNs (LSTM, BiLSTM, GRU, BiGRU, Ensemble Model) for localized temperature prediction.

Demonstrated KAN Superiority: Our findings overwhelmingly show that the KAN model significantly outperforms all other evaluated models in terms of predictive accuracy and data fit for temperature .

TKAN Variant Evaluation: Investigation into the impact of different activation functions (SiLU, Mish, GELU) within simplified TKAN architectures, revealing subtle performance differences and general improvements over standard TKAN.

Localized Prediction Focus: Application and evaluation of these models on real-world localized weather datasets (Abidjan and Kigali).



📊 Dataset
The models were trained and evaluated on precipitation, 2-meter temperature data and pression from two distinct locations:

Abidjan, Côte d'Ivoire

Kigali, Rwanda
