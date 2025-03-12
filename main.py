import os
import logging
from process_dataset import process_french_law_dataset

def main():
    csv_path = os.getenv("CSV_PATH", "path/to/your/french-law-dataset.csv")

    try:
        results = process_dataset_francais(csv_path)

        logging.info("Dataset Analysis Complete")
        logging.info(f"Total cases analyzed: {results['summary']['total_cases']}")
        logging.info(f"Force majeure detected in {results['summary']['force_majeure_cases']} cases ({results['summary']['force_majeure_percentage'] * 100:.2f}%)")
        logging.info(f"Consistent force majeure arguments in {results['summary']['consistent_cases']} cases ({results['summary']['consistent_percentage'] * 100:.2f}% of force majeure cases)")
        logging.info(f"Average sequent calculus validity: {results['summary']['average_sequent_validity'] * 100:.2f}%")

    except Exception as e:
        logging.error(f"Critical error in main: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
