import networkx as nx
import pyreason as pr
import time
import logging
from typing import Tuple, List, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum
import csv
import os
import sys
import pandas as pd
import faulthandler
import numba

csv.field_size_limit(sys.maxsize)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Enum for argument types
class ArgumentType(Enum):
    SUPPORT = "support"
    ATTACK = "attack"
    NEUTRAL = "neutral"

# Enum for sequent types
class SequentType(Enum):
    AXIOM = "axiom"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"
    CONTRAPOSITIVE = "contrapositive"

# Data class for legal arguments
@dataclass
class LegalArgument:
    id: str
    premise: str
    conclusion: str
    confidence: Tuple[float, float]
    supporting_args: Set[str]
    attacking_args: Set[str]
    sequent_type: SequentType = SequentType.HYPOTHESIS

# Data class for sequents
@dataclass
class ParsedSequent:
    antecedents: str  # Now strings, to be parsed
    consequents: str  # Now strings, to be parsed
    confidence: float
    is_contrapositive: bool = False

# -----------------------------------------------------------------------------
# AAF SEMANTICS IMPLEMENTATION (Separate from PyReason)
# -----------------------------------------------------------------------------

def calculate_grounded_extension(graph: nx.DiGraph, arguments: Dict[str, LegalArgument]) -> Set[str]:
    """Calculates the grounded extension of an argumentation framework."""
    in_set = set()
    out_set = set()
    un_set = set(arguments.keys())

    while True:
        new_in = {arg for arg in un_set if all(attacker not in in_set for attacker in get_attackers(graph, arg))}
        new_out = {arg for arg in un_set if any(attacker in in_set for attacker in get_attackers(graph, arg))}

        if not new_in and not new_out:
            break

        in_set.update(new_in)
        out_set.update(new_out)
        un_set = set(arguments.keys()) - in_set - out_set

    return in_set

def get_attackers(graph: nx.DiGraph, argument_id: str) -> Set[str]:
    """Returns the arguments that attack a given argument."""
    return {u for u, v, data in graph.in_edges(argument_id, data=True) if data["type"] == ArgumentType.ATTACK}

# -----------------------------------------------------------------------------
# Force Majeure Annotator class
# -----------------------------------------------------------------------------
class ForceMajeureAnnotator:
    def __init__(self, reasoner):
        self.reasoner = reasoner
        self.sequents = []
        self.program = pr.Program()
        self.timeline = pr.Timeline()
        self.verbose = None

    def extract_force_majeure_claims(self, text: str) -> List[Dict]:
        """Extract force majeure claims from legal text."""
        #Placeholder implementation of force majeure claims extraction
        #Here claims extraction will happen from the text based on NLP models or simpler methods
        return [
            {
                "is_valid": True,
                "premise": "An earthquake occurred.",
                "conclusion": "Performance is excused due to force majeure.",
                "confidence": 0.92
            },
            {
                "is_valid": True,
                "premise": "A flood made performance impossible.",
                "conclusion": "Performance is excused due to impossibility.",
                "confidence": 0.85
            }
        ]

    def construct_contrapositive_sequents(self, arguments: Dict[str, LegalArgument]) -> List[ParsedSequent]:
        """Construct contrapositive sequents for force majeure arguments."""
        sequents = []
        for arg_id, arg in arguments.items():
            if "force_majeure" in arg_id:
                # Add a sequent for the original argument implying performance excused
                sequents.append(ParsedSequent(
                    antecedents=arg_id,
                    consequents=','.join([id for id in arguments if "performance_excused" in id]),
                    confidence=arg.confidence[0]
                ))
                # Add a contrapositive sequent: not performance excused implies not force majeure
                sequents.append(ParsedSequent(
                    antecedents=','.join([id for id in arguments if "performance_required" in id]),
                    consequents=f"not_{arg_id}",
                    confidence=arg.confidence[0],
                    is_contrapositive=True
                ))
        return sequents

    def annotate_force_majeure(self, case_text: str) -> Dict:
        """Annotate force majeure arguments and create logical sequents."""
        claims = self.extract_force_majeure_claims(case_text)
        arguments = self._create_arguments(claims)
        self.sequents = self.construct_contrapositive_sequents(arguments)

        return {
            "arguments": arguments,
            "sequents": self.sequents,
        }

    def _create_arguments(self, claims: List[Dict]) -> Dict[str, LegalArgument]:
        """Create arguments from claims."""
        arguments = {}
        for i, claim in enumerate(claims):
            arg_id = f"force_majeure_{i}"
            # Create main argument and its negation
            arguments.update(self._create_argument_pair(i, claim))
            # Create performance-related arguments
            arguments.update(self._create_performance_args(i, claim, arg_id))
        return arguments

    def _create_argument_pair(self, idx: int, claim: Dict) -> Dict[str, LegalArgument]:
        """Create a pair of arguments (original and negation)."""
        arg_id = f"force_majeure_{idx}"
        not_arg_id = f"not_{arg_id}"
        return {
            arg_id: LegalArgument(
                id=arg_id,
                premise=claim["premise"],
                conclusion=claim["conclusion"],
                confidence=(claim["confidence"] * 0.8, claim["confidence"]),
                supporting_args=set(),
                attacking_args=set(),
                sequent_type=SequentType.HYPOTHESIS
            ),
            not_arg_id: LegalArgument(
                id=not_arg_id,
                premise=f"Negation of: {claim['premise']}",
                conclusion=f"Negation of: {claim['conclusion']}",
                confidence=(1.0 - claim["confidence"], 1.0 - claim["confidence"] * 0.8),
                supporting_args=set(),
                attacking_args={arg_id},
                sequent_type=SequentType.CONTRAPOSITIVE
            )
        }

    def _create_performance_args(self, idx: int, claim: Dict, main_arg_id: str) -> Dict[str, LegalArgument]:
        """Create performance-related arguments."""
        perf_excused_id = f"performance_excused_{idx}"
        perf_required_id = f"performance_required_{idx}"
        return {
            perf_excused_id: LegalArgument(
                id=perf_excused_id,
                premise=claim["conclusion"],
                conclusion="Performance is legally excused",
                confidence=(claim["confidence"] * 0.9, claim["confidence"]),
                supporting_args={main_arg_id},
                attacking_args=set(),
                sequent_type=SequentType.CONCLUSION
            ),
            perf_required_id: LegalArgument(
                id=perf_required_id,
                premise=f"Negation of: {claim['conclusion']}",
                conclusion="Performance is legally required",
                confidence=(1.0 - claim["confidence"], 1.0 - claim["confidence"] * 0.9),
                supporting_args={f"not_{main_arg_id}"},
                attacking_args={perf_excused_id},
                sequent_type=SequentType.CONTRAPOSITIVE
            )
        }

    def setup_pyreason_knowledge(self, arguments: Dict[str, LegalArgument], graph: nx.DiGraph):
        """Sets up PyReason graph attributes and relationships (but no facts or rules)."""
        # Set initial confidence as node attributes directly on the graph
        for arg_id, arg in arguments.items():
            graph.nodes[arg_id]["confidence_lower"] = arg.confidence[0]
            graph.nodes[arg_id]["confidence_upper"] = arg.confidence[1]

# -----------------------------------------------------------------------------
# Main Reasoning Class
# -----------------------------------------------------------------------------
class LogicBasedLegalReasoner:
    def __init__(self):
        self.force_majeure_annotator = ForceMajeureAnnotator(self)

    def analyze_legal_case(self, case_text: str) -> Dict:
        """Analyze a legal case with focus on force majeure."""
        try:
            # Apply force majeure specific annotation
            force_majeure_analysis = self.force_majeure_annotator.annotate_force_majeure(case_text)
            arguments = force_majeure_analysis["arguments"]

            # Build argument graph with ArgumentType annotations.
            graph = nx.DiGraph()
            for arg in arguments.values():
                graph.add_node(arg.id)
                for sup_id in arg.supporting_args:
                    graph.add_edge(sup_id, arg.id, type=ArgumentType.SUPPORT)
                for atk_id in arg.attacking_args:
                    graph.add_edge(atk_id, arg.id, type=ArgumentType.ATTACK)

            # AAF Evaluation (Completely Separate)
            grounded_extension = calculate_grounded_extension(graph, arguments) #Only calculate acceptable arg based on AAF rules

            # Gather arguments from the grounded extension
            justified_arguments = {arg_id: arguments[arg_id] for arg_id in grounded_extension}

            overall_consistency = all("force_majeure" not in arg_id for arg_id in arguments.keys()) #If all force majeure arguments are justified and vice versa
            #Calculate any other rules based on the AAF.
            return {
                "force_majeure_analysis": force_majeure_analysis,
                "grounded_extension": grounded_extension,
                "justified_arguments": justified_arguments,
                "overall_consistency": overall_consistency,
            }

        except Exception as e:
            logging.error(f"Error analyzing case: {e}")
            return {}

# Function to process the French law dataset
def process_dataset_francais(csv_path: str):
    """Process the French law dataset with force majeure annotation."""
    try:
        # Load dataset
        try:
            with open(csv_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                dataset = list(reader)
            logging.info(f"Dataset loaded successfully. First example: {dataset[0]}")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return {
                "case_results": [],
                "summary": {
                    "total_cases": 0,
                    "force_majeure_cases": 0,
                    "force_majeure_percentage": 0,
                    "consistent_cases": 0,
                    "consistent_percentage": 0,
                    "average_sequent_validity": 0
                }
            }

        # Inspect dataset structure
        if len(dataset) > 0:
            logging.info(f"Dataset structure: {dataset[0]}")

        # Initialize reasoner
        reasoner = LogicBasedLegalReasoner()

        # Process each case
        results = []
        for i, case in enumerate(dataset):
            case_text = case.get("text", "")  # Adjust based on actual dataset structure

            logging.info(f"Processing case {i + 1}/{len(dataset)}")

            # Analyze with force majeure focus
            analysis = reasoner.analyze_legal_case(case_text)

            # Store results
            results.append({
                "case_id": i,
                "force_majeure_detected": any("force_majeure" in arg_id for arg_id in analysis["force_majeure_analysis"]["arguments"]),
                "AAF_Grounded_Ext": analysis["argumentation_results"]["grounded_extension"] #AAF analysis here
            })

        # Summarize results
        force_majeure_cases = sum(1 for r in results if r["force_majeure_detected"])
        consistent_cases = sum(1 for r in results if any(r["AAF_Grounded_Ext"])) #If any argument was within the acceptable ones, then the cases are consistent

        summary = {
            "total_cases": len(results),
            "force_majeure_cases": force_majeure_cases,
            "force_majeure_percentage": force_majeure_cases / len(results) if results else 0,
            "consistent_cases": consistent_cases,
            "consistent_percentage": consistent_cases / force_majeure_cases if force_majeure_cases else 0,
            "average_sequent_validity": 0 #We won't be evaluating sequents anymore
        }

        return {
            "case_results": results,
            "summary": summary
        }

    except Exception as e:
        logging.error(f"Error processing dataset: {e}")
        return {
            "case_results": [],
            "summary": {
                "total_cases": 0,
                "force_majeure_cases": 0,
                "force_majeure_percentage": 0,
                "consistent_cases": 0,
                "consistent_percentage": 0,
                "average_sequent_validity": 0
            }
        }

# Main function
def main():
    CSV_PATH = os.getenv("CSV_PATH", "/Volumes/Crucial/cold-french-law.csv")

    try:
        # Process dataset with force majeure focus
        results = process_dataset_francais(CSV_PATH)

        logging.info("Dataset Analysis Complete")
        logging.info(f"Total cases analyzed: {results['summary']['total_cases']}")
        logging.info(f"Force majeure detected in {results['summary']['force_majeure_cases']} cases ({results['summary']['force_majeure_percentage'] * 100:.2f}%)")
        logging.info(f"Consistent force majeure arguments in {results['summary']['consistent_cases']} cases ({results['summary']['consistent_percentage'] * 100:.2f}% of force majeure cases)")
        logging.info(f"Average sequent calculus validity: {results['summary']['average_sequent_validity'] * 100:.2f}%")

    except Exception as e:
        logging.error(f"Critical error in main: {e}")

if __name__ == "__main__":
    main()
