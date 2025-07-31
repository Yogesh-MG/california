# California Housing Price Predictor

This project is a simple machine learning application that predicts the median house value in California based on various features using a linear regression model. The project includes a Streamlit web app for interactive predictions and a script for training and evaluating the model.

## Project Structure

- `app.py`: Streamlit app that loads the California housing dataset, trains a linear regression model, and provides a user interface to input feature values and predict house prices.
- `main.py`: Script that loads the dataset, splits it into training and testing sets, trains a linear regression model, and evaluates the model performance using metrics such as Mean Squared Error, Mean Absolute Error, and R² Score.
- `requirements.text`: List of Python dependencies required to run the project.

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
```

2. Activate the virtual environment:

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.text
```

## Usage

To run the Streamlit app:

```bash
streamlit run app.py
```

This will launch a web interface where you can input feature values and get a predicted median house value.

To run the model training and evaluation script:

```bash
python main.py
```

This will output the evaluation metrics of the linear regression model on the test dataset.

## Dataset

The project uses the California Housing dataset from the `sklearn.datasets` module.

## License

This project is open source and available under the MIT License.
