# Legal Reasoning - Logic etc

This project is designed to analyze legal cases with a focus on force majeure using a logic-based reasoning approach. The application processes a dataset of legal cases, annotates them for force majeure, and evaluates the arguments using an argumentation framework.

## Project Structure

legal-reasoning-app ├── src │ ├── init.py │ ├── main.py │ ├── logic_based_legal_reasoner.py │ ├── force_majeure_annotator.py │ ├── process_dataset.py │ └── utils │ ├── init.py │ └── logging_config.py ├── requirements.txt 

### Prerequisites

- Python 3.8+
- Install the required packages using `pip install -r requirements.txt`

### Running the Application

1. Set the `CSV_PATH` environment variable to the path of your French law dataset CSV file.
2. Run the application:

```sh
python src/main.py
