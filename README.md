# Deep Learning Cryptocurrency Trading Bot

## Description

A deep learning trading bot using cryptocurrency market data from the Binance API that can predict market movements on different time frames and also place orders.

## Dataset and Model Files Available:

- **Dataset:** BTCUSDT for the past 14 months.
- **Training Model File:** Includes the model architecture and weights used during training.
- **Testing Model File:** Contains the model setup for evaluating performance.
- **Screenshots:** Documenting the model training process.

## Features:

- **Data Collection:** Automatically collects data for any cryptocurrency pair and time frame.
- **Model Architectures:** Implements LSTM, GRU, BiLSTM, and BiGRU architectures.
- **Performance:** Achieves good training results but seeks improvement on unseen data.

## Dataset

- **BTCUSDT Data:** Comprehensive dataset covering the past 14 months.
- **Custom Data:** Ability to provide data for any cryptocurrency pair, time frame, and period (e.g., 1 min, 5 min, 1 hour, etc.) upon request.

## Issue

- **Model Challenge:** The model struggles to predict reasonable results on unseen data despite achieving good training results.
- **Seeking Help:** Looking for assistance to improve the model’s performance and overcome this challenge.

## Suggestion

- **Enhanced Data Preparation:** Consider using the first 5 columns (e.g., `Open`, `High`, `Low`, `Close`, and `Volume`) from the dataset and generate various technical indicators (like SMA, EMA, RSI, MACD, etc.). Create a new dataset with these indicators and train the model on this enriched dataset. This approach might help improve the model's performance on unseen data.


## Contribute

Help improve the model's performance on unseen data. Share your expertise in deep learning, cryptocurrency market analysis, or trading bot development.

## Contact

- **Email:** muhammadhasnainkayani@gmail.com
- **LinkedIn:** [Muhammad Hasnain Kayani](https://www.linkedin.com/in/muhammad-hasnain-kayani-820599273)
