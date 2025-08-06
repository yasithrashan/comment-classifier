# Comment Classifier

A machine learning project that automatically identifies spam and toxic comments from YouTube using natural language processing and pre-trained transformer models.

## Overview

This project analyzes YouTube comments to classify them as spam or legitimate content by combining:
- **Toxicity detection** using a pre-trained transformer model
- **Pattern matching** for common spam indicators
- **Text preprocessing** and cleaning techniques

## Features

- **Automated Text Cleaning**: Removes URLs, mentions, hashtags, and special characters
- **Dual Classification Approach**: 
  - Toxicity scoring using `martin-ha/toxic-comment-model`
  - Pattern-based spam detection
- **Comprehensive Analysis**: Statistical reporting and visualization
- **Flexible Thresholds**: Customizable spam detection sensitivity

## Installation

### Requirements

```bash
pip install transformers torch torchvision
pip install scikit-learn pandas numpy matplotlib seaborn
pip install nltk wordcloud textblob
```

### Setup

1. Clone this repository
2. Install the required packages
3. Prepare your YouTube comments dataset in CSV format
4. Run the Jupyter notebook

## Usage

### 1. Data Preparation

Ensure your CSV file contains a `CONTENT` column with the comment text:

```csv
CONTENT,category
"Great video! Thanks for sharing",ham
"Subscribe to my channel! Check out www.spam.com",spam
```

### 2. Running the Classifier

```python
# Load and preprocess data
df = pd.read_csv('Youtube-Spam-Dataset.csv')
df['CLEAN_CONTENT'] = df['CONTENT'].apply(clean_text)

# Initialize the classifier
classifier = pipeline('text-classification', model='martin-ha/toxic-comment-model')

# Generate spam labels
df['CLASS_LABEL'] = create_spam_labels(df)
```

### 3. View Results

The classifier outputs:
- `CLASS_LABEL`: Binary classification (0 = Clean, 1 = Spam)
- Statistical summary of spam detection rates
- Visualization charts showing comment distribution

## Classification Logic

The spam detection algorithm considers a comment as spam if:

1. **High Pattern Match**: 2+ spam patterns detected
2. **Medium Pattern + Toxicity**: 1+ patterns AND toxicity score > 0.3
3. **High Toxicity**: Toxicity score > 0.7

### Spam Patterns Detected:
- `subscribe`, `channel`, `check out`, `follow me`
- `my channel`, `visit`, `website`
- URLs (`.com`, `www`)

## Text Preprocessing

The `clean_text()` function performs:
- Converts to lowercase
- Removes URLs (`http://`, `www.`, `https://`)
- Removes mentions (`@username`) and hashtags (`#tag`)
- Strips special characters and extra whitespace
- Handles missing/null values

## Model Performance

The classifier uses the pre-trained `martin-ha/toxic-comment-model` which provides:
- Toxicity probability scores
- Binary toxic/non-toxic classification
- Robust performance on social media content

## Output Visualization

The project generates:
- **Pie Chart**: Distribution of clean vs spam comments
- **Statistical Summary**: Total counts and spam percentage
- **Color-coded Results**: Green for clean, red for spam

## Customization

### Adjusting Sensitivity

Modify the thresholds in `create_spam_labels()`:

```python
# More sensitive detection
if pattern_matches >= 1:  # Lower threshold
    spam_label = 1
elif toxicity > 0.5:  # Lower toxicity threshold
    spam_label = 1
```

### Adding Custom Patterns

Extend the spam patterns list:

```python
spam_patterns = [
    r'subscribe', r'channel', r'check out',
    r'like and subscribe',  # Add custom patterns
    r'hit the bell',
    r'free download'
]
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- **martin-ha/toxic-comment-model** for the pre-trained toxicity classifier
- **Hugging Face Transformers** for the pipeline infrastructure
- **YouTube Spam Dataset** contributors

## Support

If you encounter any issues or have questions:
1. Check the [Issues](../../issues) section
2. Review the code comments and documentation
3. Submit a new issue with detailed information

---

**Note**: This classifier is designed for educational and research purposes. Always review automated classifications and consider implementing human moderation for production systems.
