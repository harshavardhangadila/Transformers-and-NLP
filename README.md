# Sentiment Inference with KerasNLP

This project demonstrates how to use a pretrained BERT classifier from the `keras-nlp` library to perform sentiment inference on text data without any additional training. It showcases how to load a model, run predictions, interpret logits, and visualize output probabilities.



## Architecture

- Uses `keras_nlp.models.BertClassifier` with the `bert_base_en` preset.
- The classifier includes:
  - A pretrained BERT backbone.
  - A classification head for binary classification.
- Tokenization, input packing (with `[CLS]`, `[SEP]`, and padding), and preprocessing are built into the model.
- The model outputs logits for two sentiment classes (positive or negative).
- Softmax is applied to convert logits into interpretable probabilities.



## Dataset

- A manually defined list of 17 sample movie review texts simulates realistic sentiment analysis inputs.
- Custom test cases (5 additional `"custom_texts"`) and a small mini-evaluation set with expected labels are used to explore model behavior.



## Inference

- Raw logits from the model are computed for each input sentence.
- A `classify_text()` helper function converts logits to probabilities and sentiment labels.
- Predictions are printed alongside probability distributions.

---

## Evaluation

- A mini labeled dataset of 5 samples is used to evaluate model prediction accuracy.
- The accuracy on this small test set is reported.
- A `batch_classify()` utility is used to process multiple inputs at once and return full probability vectors.



## Visualization

- Bar plots are generated using Matplotlib to show the predicted sentiment probabilities for the first five examples.
- These visualizations help interpret the confidence of the model’s predictions.



## Model Summary

- The model architecture includes:
  - A BERT tokenizer layer.
  - A BERT backbone with ~108 million parameters.
  - A dropout layer and a dense classification layer.
- Model summary is printed using `classifier.summary()` to inspect the layer structure and parameter count.



## Performance Timing

- The total inference time for 17 samples is measured using `time.time()`.



## Dependencies

- `keras-nlp`
- `tensorflow`
- `numpy`
- `matplotlib`

---

## Pretrained BERT-Based Sentiment Classification with TensorFlow and Hugging Face

This project demonstrates how to fine-tune a pretrained BERT model for sentiment classification on the IMDB movie reviews dataset using TensorFlow and Hugging Face Transformers.

## Architecture

- Uses `TFBertForSequenceClassification` from Hugging Face.
- Tokenization is handled by `BertTokenizer` with padding and truncation.
- The model outputs logits for binary classification (positive or negative sentiment).
- Softmax is applied to convert logits into class probabilities.

## Dataset

- IMDB movie review dataset is loaded from `tensorflow_datasets`.
- Reviews are tokenized into input IDs and attention masks.
- Data is batched using `tf.data.Dataset` for efficient training and validation.

## Training

- The model is compiled using the AdamW optimizer with a learning rate scheduler.
- Sparse categorical crossentropy is used as the loss function (from logits).
- Training runs for 3 epochs with validation at each step.
- Accuracy and loss are tracked manually due to Hugging Face training wrappers.

## Evaluation

- Evaluation includes accuracy, precision, recall, F1-score, and confusion matrix.
- Model confidence is analyzed through histogram and accuracy-vs-confidence plots.
- Misclassified samples are displayed for qualitative inspection.
- Boxplots show confidence distributions by class and correctness.

## Visualization

- Confusion matrix and classification report summarize performance.
- Additional plots include:
  - Training loss and accuracy (custom loop)
  - Confidence distribution histogram
  - Accuracy by confidence bins
  - Per-class confidence boxplot
  - Sample misclassified reviews

## Dependencies

- `tensorflow`
- `transformers`
- `tensorflow_datasets`
- `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

# Transformer Text Classification with Keras

This project demonstrates how to build a simple Transformer-based text classification model using Keras.

## Architecture

- A custom TokenAndPositionEmbedding layer combines token and position embeddings.
- A custom TransformerBlock implements multi-head self-attention and a feedforward network.
- Layer normalization, residual connections, and dropout are included for stability and regularization.
- The output is pooled and passed through dense layers for classification.

## Dataset

- A synthetic dataset is generated with random integer token sequences.
- Each sequence is of fixed length and assigned a binary label.
- Data is split into training and validation sets.

## Training

- The model is compiled using the Adam optimizer and sparse categorical crossentropy loss.
- Accuracy and loss are tracked during training over 5 epochs.
- Plots are generated for training and validation metrics.

## Evaluation

- Custom examples simulate sentiment patterns using synthetic token IDs.
- Model predictions include class probabilities, confidence, and entropy.
- Additional visualizations include confusion matrix, bar plots, and t-SNE projection.

---
Youtube: [Keras NLP Tasks](https://www.youtube.com/playlist?list=PLCGwaUpxPWO3XpuxgzVGpbem7glHbm7od)
