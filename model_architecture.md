# Ensemble Model Architecture

```
Input Data [15 x 5]
[Open, High, Low, Close, Volume]
           ↓
    ┌──────┴──────┐
    ↓             ↓
┌─────────┐  ┌─────────┐
│   CNN   │  │  LSTM   │
│  (128)  │  │  (64)   │
└────┬────┘  └────┬────┘
     ↓            ↓
┌─────────┐  ┌─────────┐
│ MaxPool │  │ Dropout │
│   (2)   │  │  (0.3)  │
└────┬────┘  └────┬────┘
     ↓            ↓
┌─────────┐  ┌─────────┐
│ Conv1D  │  │ Linear  │
│  (64)   │  │   (1)   │
└────┬────┘  └────┬────┘
     ↓            │
┌─────────┐      │
│ Dropout │      │
│  (0.3)  │      │
└────┬────┘      │
     ↓           │
┌─────────┐      │
│ Linear  │      │
│   (1)   │      │
└────┬────┘      │
     └─────┬─────┘
           ↓
    ┌─────────┐
    │Weighted │
    │Average  │
    └────┬────┘
         ↓
  Final Prediction

## Component Details

### CNN Branch
- Input Shape: [batch_size, 1, sequence_length, features]
- Conv1D Layer 1: 128 filters, kernel_size=3, stride=1
- MaxPool Layer: kernel_size=2, stride=1
- Conv1D Layer 2: 64 filters, kernel_size=3, stride=1
- Dropout: 0.3
- Linear Layer: Output size 1

### LSTM Branch
- Input Shape: [batch_size, sequence_length, features]
- LSTM Layer: hidden_size=64
- Dropout: 0.3
- Linear Layer: Output size 1

### Ensemble Mechanism
- Weighted average of CNN and LSTM predictions
- Initial weights: CNN (0.5), LSTM (0.5)
- Weights updated during training

### Input Features
1. Open Price
2. High Price
3. Low Price
4. Close Price
5. Volume

### Output
- Single value prediction for next time step's price
