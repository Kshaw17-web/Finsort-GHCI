# Dataset Documentation

Because GHCI does not provide an official dataset, we created our own synthetic + semi-realistic dataset for FinSort.

## Files
- `data/finsort_train.csv`
- `data/finsort_test.csv`
- `data/retail_map.csv`

## Columns
- `transaction`: Raw financial text
- `cleaned`: Preprocessed version
- `tag`: Fine-grained label (ML training)
- `category`: Mapped high-level taxonomy

## Generation Method
- Used public patterns (UPI, POS, BBPS, Paytm, Amazon)
- Added synthetic variants with noise, spelling errors, merchant codes
- Ensured no personal data was used
- Balanced category labels

## Why Synthetic Data?
- No official dataset allowed
- Keeps our project reproducible
