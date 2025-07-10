import networkx as nx
import pyreason as pr
import time
import logging
from typing import Tuple, List, Dict, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import csv
import os
import sys
import pandas as pd
import itertools
from collections import defaultdict

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Constants ---
ORIGINALITY_KEYWORDS = {
    "empreinte de la personnalité", "author's personality",
    "choix créatifs libres", "free creative choices",
    "dictated by technical function"
}

# --- Argument Framework (AAF) Definition ---
class OptimizedAAF:
    """Efficient representation and analysis of Abstract Argumentation Frameworks."""
    
    def __init__(self, arguments: Set[str], attacks: Set[Tuple[str, str]]):
        """
        Initialize the AAF with arguments and attack relations.
        
        Args:
            arguments: Set of argument identifiers
            attacks: Set of attack relations as tuples (attacker, target)
        """
        self.arguments = arguments
        self.attack_graph = nx.DiGraph()
        
        # Initialize attack data structures
        self.attacks_dict = defaultdict(set)
        self.attackers_dict = defaultdict(set)
        
        # Build the attack graph and dictionaries
        self.attack_graph.add_nodes_from(arguments)
        for attacker, target in attacks:
            if attacker in arguments and target in arguments:
                self.attacks_dict[attacker].add(target)
                self.attackers_dict[target].add(attacker)
                self.attack_graph.add_edge(attacker, target)

    def is_conflict_free(self, S: Set[str]) -> bool:
        """Check if a set of arguments S is conflict-free."""
        subset_args = S.intersection(self.arguments)
        return all(
            subset_args.isdisjoint(self.attacks_dict.get(arg, set()))
            for arg in subset_args
        )

    def defends(self, S: Set[str], a: str) -> bool:
        """Check if set S defends argument 'a' against all attackers."""
        if a not in self.arguments:
            return False
            
        subset_args = S.intersection(self.arguments)
        attackers_of_a = self.attackers_dict.get(a, set())
        
        return all(
            any(attacker in self.attacks_dict.get(defender, set())
                for defender in subset_args)
            for attacker in attackers_of_a
        )

    def get_admissible_sets(self) -> List[Set[str]]:
        """Generate all admissible sets using power set iteration."""
        admissible = []
        arguments_list = list(self.arguments)
        
        for i in range(1, 1 << len(arguments_list)):
            subset = {
                arguments_list[j] 
                for j in range(len(arguments_list)) 
                if (i >> j) & 1
            }
            
            if (self.is_conflict_free(subset) and 
                all(self.defends(subset, arg) for arg in subset)):
                admissible.append(subset)
                
        return admissible

    def get_preferred_extensions(self) -> List[Set[str]]:
        """Find maximal admissible sets (preferred extensions)."""
        admissible_sets = self.get_admissible_sets()
        preferred = []
        
        for s1 in admissible_sets:
            if not any(s1.issubset(s2) for s2 in admissible_sets if s1 != s2):
                preferred.append(s1)
                
        return preferred

    def get_stable_extensions(self) -> List[Set[str]]:
        """Find stable extensions (conflict-free sets attacking all outside args)."""
        stable = []
        arguments_list = list(self.arguments)
        
        for i in range(1, 1 << len(arguments_list)):
            subset = {
                arguments_list[j] 
                for j in range(len(arguments_list)) 
                if (i >> j) & 1
            }
            
            if self.is_conflict_free(subset):
                arguments_outside = self.arguments - subset
                attacks_all_outside = all(
                    any(outside_arg in self.attacks_dict.get(attacker, set())
                        for attacker in subset)
                    for outside_arg in arguments_outside
                )
                
                if attacks_all_outside:
                    stable.append(subset)
                    
        return stable

    # ----- Graph Analysis Methods -----
    
    def has_cycle(self) -> bool:
        """Check if the argumentation graph contains cycles."""
        try:
            sccs = list(nx.strongly_connected_components(self.attack_graph))
            return any(len(comp) > 1 for comp in sccs)
        except Exception as e:
            logging.error(f"Error detecting cycle: {e}")
            return False

    def shortest_attack_path(self, start: str, goal: str) -> Optional[List[str]]:
        """Find the shortest attack path between arguments using BFS."""
        try:
            return nx.shortest_path(self.attack_graph, source=start, target=goal)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None
        except Exception as e:
            logging.error(f"Error finding shortest path: {e}")
            return None

    def pagerank_influence(self) -> Dict[str, float]:
        """Compute argument influence scores using PageRank."""
        try:
            return nx.pagerank(self.attack_graph, alpha=0.85)
        except Exception as e:
            logging.error(f"Error computing PageRank: {e}")
            return {arg: 0.0 for arg in self.arguments}

# --- Enums and Data Classes ---

class ArgumentType(Enum):
    SUPPORT = "support"
    ATTACK = "attack"
    NEUTRAL = "neutral"

class SequentType(Enum):
    AXIOM = "axiom"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"
    CONTRAPOSITIVE = "contrapositive"

@dataclass
class LegalArgument:
    """Represents a legal argument with its properties and relationships."""
    id: str
    premise: str
    conclusion: str
    confidence: Tuple[float, float]  # Belief interval [lower, upper]
    supporting_args: Set[str] = field(default_factory=set)
    attacking_args: Set[str] = field(default_factory=set)
    sequent_type: SequentType = SequentType.HYPOTHESIS

@dataclass
class Sequent:
    """Represents a logical sequent (implication) between arguments."""
    antecedents: Set[str]  # Set of argument IDs
    consequents: Set[str]  # Set of argument IDs
    confidence: float  # Confidence in the implication
    is_contrapositive: bool = False

# --- Originality Annotator Class ---

class OriginalityAnnotator:
    """Annotates and analyzes IP originality arguments using formal reasoning."""
    
    def __init__(self, reasoner):
        """
        Initialize the annotator with a reference to the main reasoner.
        
        Args:
            reasoner: The main LogicBasedLegalReasoner instance
        """
        self.reasoner = reasoner
        self.sequents: List[Sequent] = []
        self.program = pr.Program()
        self.timeline = pr.Timeline()
        self._current_arguments: Dict[str, LegalArgument] = {}
        self._current_attacks: Set[Tuple[str, str]] = set()

    def extract_originality_claims(self, text: str) -> List[Dict]:
        """
        Extract potential originality claims from legal text using keyword matching.
        
        Args:
            text: The legal text to analyze
            
        Returns:
            List of dictionaries representing extracted claims
        """
        claims = []
        text_lower = text.lower()
        
        # Check for supporting evidence of originality
        if any(keyword in text_lower for keyword in ORIGINALITY_KEYWORDS):
            if "empreinte de la personnalité" in text_lower or "author's personality" in text_lower:
                claims.append({
                    "is_valid": True,
                    "premise": "Work shows the imprint of the author's personality.",
                    "conclusion": "The work is original.",
                    "confidence": 0.88
                })
                
            if "choix créatifs libres" in text_lower or "free creative choices" in text_lower:
                claims.append({
                    "is_valid": True,
                    "premise": "Author made free and creative choices.",
                    "conclusion": "The work is original.",
                    "confidence": 0.91
                })
        
        # Check for challenges to originality
        if "dictated by technical function" in text_lower:
            claims.append({
                "is_valid": True,
                "premise": "The work's form is dictated by its technical function.",
                "conclusion": "The work is NOT original.",
                "confidence": 0.95
            })
            
            # Add counter-argument about functionality
            claims.append({
                "is_valid": True,
                "premise": "Functional elements lack originality.",
                "conclusion": "No copyright protection for functional aspects.",
                "confidence": 0.90
            })

        return claims

    def construct_originality_sequents(self) -> List[Sequent]:
        """Construct logical sequents for originality arguments."""
        sequents = []
        arguments = self._current_arguments
        
        # Group arguments by type
        originality_args = {k for k in arguments if "is_original" in k}
        protection_args = {k for k in arguments if "copyright_protected" in k}
        no_protection_args = {k for k in arguments if "copyright_not_protected" in k}
        not_original_args = {k for k in arguments if "not_original" in k}

        # Create sequents linking originality to protection
        for orig_arg in originality_args:
            conf = arguments[orig_arg].confidence[0]
            sequents.append(Sequent(
                antecedents={orig_arg},
                consequents=protection_args,
                confidence=conf * 0.9
            ))
            
            # Contrapositive
            sequents.append(Sequent(
                antecedents=no_protection_args,
                consequents=not_original_args,
                confidence=conf * 0.9,
                is_contrapositive=True
            ))

        return sequents

    def annotate_originality(self, case_text: str) -> Dict:
        """
        Main method to annotate originality arguments in a legal case.
        
        Args:
            case_text: The text of the legal case to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Reset state for new analysis
        self._reset_analysis_state()
        
        # Extract and process claims
        claims = self.extract_originality_claims(case_text)
        if not claims:
            logging.warning("No originality claims extracted from text.")
            return self._empty_analysis_result()

        # Create arguments and identify relationships
        self._current_arguments = self._create_arguments_from_claims(claims)
        self._identify_attacks()
        self._update_argument_relationships()
        
        # Build logical sequents
        self.sequents = self.construct_originality_sequents()

        # Add arguments to reasoner's knowledge base
        for arg in self._current_arguments.values():
            self.reasoner.add_argument(arg)

        # Set up and run PyReason analysis
        self.setup_pyreason_knowledge()
        consistency_results = self.evaluate_originality_consistency()

        return {
            "arguments": self._current_arguments,
            "attacks": self._current_attacks,
            "sequents": self.sequents,
            "pyreason_consistency_results": consistency_results
        }

    def _reset_analysis_state(self):
        """Reset internal state for a new analysis."""
        self.program = pr.Program()
        self.timeline = pr.Timeline()
        self.sequents = []
        self._current_arguments = {}
        self._current_attacks = set()

    def _empty_analysis_result(self):
        """Return an empty result structure when no claims are found."""
        return {
            "arguments": {},
            "attacks": set(),
            "sequents": [],
            "pyreason_consistency_results": {
                "originality_arguments": {},
                "contrapositive_arguments": {},
                "overall_consistency": True
            }
        }

    def _create_arguments_from_claims(self, claims: List[Dict]) -> Dict[str, LegalArgument]:
        """Create LegalArgument objects from extracted claims."""
        arguments = {}
        
        for i, claim in enumerate(claims):
            base_id = f"claim_{i}"
            conf_lower = claim["confidence"] * 0.8
            conf_upper = claim["confidence"]
            
            supports_originality = ("not original" not in claim["conclusion"].lower() and 
                                  "no copyright" not in claim["conclusion"].lower())

            if supports_originality:
                arguments.update(self._create_supporting_arguments(
                    i, claim, base_id, (conf_lower, conf_upper)
                )
            else:
                arguments.update(self._create_challenging_arguments(
                    i, claim, base_id, (conf_lower, conf_upper)
                ))
                
        return arguments

    def _create_supporting_arguments(self, idx: int, claim: Dict, base_id: str, 
                                   confidence: Tuple[float, float]) -> Dict[str, LegalArgument]:
        """Create arguments that support originality/copyright protection."""
        args = {}
        
        # Originality argument
        orig_id = f"is_original_{idx}"
        args[orig_id] = LegalArgument(
            id=orig_id,
            premise=claim["premise"],
            conclusion=f"The work is original based on: {claim['premise']}",
            confidence=confidence,
            sequent_type=SequentType.HYPOTHESIS
        )
        
        # Negated originality argument
        not_orig_id = f"not_original_{idx}"
        args[not_orig_id] = LegalArgument(
            id=not_orig_id,
            premise=f"Negation of: {claim['premise']}",
            conclusion=f"Work may not be original despite: {claim['premise']}",
            confidence=(1 - confidence[1], 1 - confidence[0]),
            attacking_args={orig_id},
            sequent_type=SequentType.CONTRAPOSITIVE
        )
        args[orig_id].attacking_args.add(not_orig_id)
        
        # Copyright protection argument
        prot_id = f"copyright_protected_{idx}"
        args[prot_id] = LegalArgument(
            id=prot_id,
            premise=f"Based on originality argument {orig_id}",
            conclusion="Copyright protection likely applies.",
            confidence=(confidence[0] * 0.9, confidence[1]),
            supporting_args={orig_id},
            sequent_type=SequentType.CONCLUSION
        )
        
        # Negated protection argument
        not_prot_id = f"copyright_not_protected_{idx}"
        args[not_prot_id] = LegalArgument(
            id=not_prot_id,
            premise=f"If work is not original (related to {not_orig_id})",
            conclusion="Copyright protection likely does not apply.",
            confidence=(1 - args[prot_id].confidence[1], 1 - args[prot_id].confidence[0]),
            supporting_args={not_orig_id},
            attacking_args={prot_id},
            sequent_type=SequentType.CONTRAPOSITIVE
        )
        args[prot_id].attacking_args.add(not_prot_id)
        
        return args

    def _create_challenging_arguments(self, idx: int, claim: Dict, base_id: str, 
                                    confidence: Tuple[float, float]) -> Dict[str, LegalArgument]:
        """Create arguments that challenge originality/copyright protection."""
        args = {}
        
        # Non-originality argument
        not_orig_id = f"not_original_{idx}"
        args[not_orig_id] = LegalArgument(
            id=not_orig_id,
            premise=claim["premise"],
            conclusion=f"The work is likely NOT original due to: {claim['premise']}",
            confidence=confidence,
            sequent_type=SequentType.HYPOTHESIS
        )
        
        # Originality counter-argument
        orig_id = f"is_original_{idx}"
        args[orig_id] = LegalArgument(
            id=orig_id,
            premise=f"Counter-argument to: {claim['premise']}",
            conclusion=f"Work might still be original despite: {claim['premise']}",
            confidence=(1 - confidence[1], 1 - confidence[0]),
            attacking_args={not_orig_id},
            sequent_type=SequentType.CONTRAPOSITIVE
        )
        args[not_orig_id].attacking_args.add(orig_id)
        
        # Non-protection argument
        not_prot_id = f"copyright_not_protected_{idx}"
        args[not_prot_id] = LegalArgument(
            id=not_prot_id,
            premise=f"Based on non-originality argument {not_orig_id}",
            conclusion="Copyright protection likely does NOT apply.",
            confidence=(confidence[0] * 0.9, confidence[1]),
            supporting_args={not_orig_id},
            sequent_type=SequentType.CONCLUSION
        )
        
        # Protection counter-argument
        prot_id = f"copyright_protected_{idx}"
        args[prot_id] = LegalArgument(
            id=prot_id,
            premise=f"Counter-argument regarding protection based on {orig_id}",
            conclusion="Copyright protection might still apply.",
            confidence=(1 - args[not_prot_id].confidence[1], 1 - args[not_prot_id].confidence[0]),
            supporting_args={orig_id},
            attacking_args={not_prot_id},
            sequent_type=SequentType.CONTRAPOSITIVE
        )
        args[not_prot_id].attacking_args.add(prot_id)
        
        return args

    def _identify_attacks(self):
        """Identify attack relationships between generated arguments."""
        arg_ids = list(self._current_arguments.keys())
        
        for i in range(len(arg_ids)):
            arg1_id = arg_ids[i]
            arg1 = self._current_arguments[arg1_id]
            
            # Add pre-defined attacks from argument relationships
            self._current_attacks.update(
                (arg1_id, atk_id) 
                for atk_id in arg1.attacking_args 
                if atk_id in self._current_arguments
            )
            
            # Identify substantive attacks based on conclusions
            for j in range(i + 1, len(arg_ids)):
                arg2_id = arg_ids[j]
                arg2 = self._current_arguments[arg2_id]
                
                # Skip if these are direct negations (already handled)
                if (f"not_{arg1_id}" == arg2_id or 
                    f"not_{arg2_id}" == arg1_id):
                    continue
                    
                # Check for opposing conclusions about originality
                if self._have_opposing_conclusions(arg1, arg2):
                    self._add_mutual_attack(arg1_id, arg2_id)
                    
                # Check for opposing conclusions about protection
                elif self._have_opposing_protection(arg1, arg2):
                    self._add_mutual_attack(arg1_id, arg2_id)

    def _have_opposing_conclusions(self, arg1: LegalArgument, arg2: LegalArgument) -> bool:
        """Check if arguments have opposing conclusions about originality."""
        c1 = arg1.conclusion.lower()
        c2 = arg2.conclusion.lower()
        
        return (("original" in c1 or "considered original" in c1) and 
                "not original" in c2) or \
               (("original" in c2 or "considered original" in c2) and 
                "not original" in c1)

    def _have_opposing_protection(self, arg1: LegalArgument, arg2: LegalArgument) -> bool:
        """Check if arguments have opposing conclusions about copyright protection."""
        c1 = arg1.conclusion.lower()
        c2 = arg2.conclusion.lower()
        
        return ("protection likely applies" in c1 and 
                "protection likely does not apply" in c2) or \
               ("protection likely applies" in c2 and 
                "protection likely does not apply" in c1)

    def _add_mutual_attack(self, arg1_id: str, arg2_id: str):
        """Add mutual attack relationship between two arguments."""
        self._current_attacks.add((arg1_id, arg2_id))
        self._current_attacks.add((arg2_id, arg1_id))

    def _update_argument_relationships(self):
        """Update supporting/attacking_args in LegalArgument objects."""
        for attacker, target in self._current_attacks:
            if target in self._current_arguments:
                self._current_arguments[target].attacking_args.add(attacker)

    def setup_pyreason_knowledge(self):
        """Set up PyReason knowledge base with facts and rules."""
        current_time = 0
        self.timeline.add_time_point(current_time)

        # Add facts for each argument
        for arg_id, arg in self._current_arguments.items():
            try:
                self.program.add_fact(
                    pr.Fact(
                        name=f"{arg_id}_fact",
                        component=arg_id,
                        value=pr.Interval(arg.confidence[0], arg.confidence[1]),
                        start_time=current_time,
                        end_time=current_time,
                        static=False
                    )
                )
            except Exception as e:
                logging.error(f"Error adding fact for {arg_id}: {e}")

        # Add support rules
        for arg_id, arg in self._current_arguments.items():
            for support_id in arg.supporting_args:
                if support_id in self._current_arguments:
                    self._add_support_rule(support_id, arg_id)

        # Add attack rules
        for attacker_id, target_id in self._current_attacks:
            if (attacker_id in self._current_arguments and 
                target_id in self._current_arguments):
                self._add_attack_rule(attacker_id, target_id)

        # Add sequent rules
        for i, sequent in enumerate(self.sequents):
            self._add_sequent_rule(sequent, i)

    def _add_support_rule(self, source: str, target: str):
        """Add a support rule to the PyReason program."""
        rule_name = f"support_{source}_to_{target}"
        try:
            self.program.add_rule(
                pr.Rule(
                    rule_text=f"{source} => {target}",
                    name=rule_name,
                    infer_edges=True,
                    set_static=False,
                    custom_thresholds=[0.5],
                    weights=[1.0],
                    interval=pr.Interval(0.7, 1.0)
                )
            )
            logging.debug(f"Added support rule: {rule_name}")
        except Exception as e:
            logging.error(f"Error adding support rule {rule_name}: {e}")

    def _add_attack_rule(self, source: str, target: str):
        """Add an attack rule to the PyReason program."""
        rule_name = f"attack_{source}_to_{target}"
        try:
            self.program.add_rule(
                pr.Rule(
                    rule_text=f"{source} => ¬{target}",
                    name=rule_name,
                    infer_edges=True,
                    set_static=False,
                    custom_thresholds=[0.6],
                    weights=[1.0],
                    interval=pr.Interval(0.8, 1.0)
                )
            )
            logging.debug(f"Added attack rule: {rule_name}")
        except Exception as e:
            logging.error(f"Error adding attack rule {rule_name}: {e}")

    def _add_sequent_rule(self, sequent: Sequent, index: int):
        """Add a sequent rule to the PyReason program."""
        if not sequent.antecedents or not sequent.consequents:
            logging.warning(f"Skipping sequent {index} due to empty antecedents/consequents.")
            return

        antecedent_part = ' & '.join(sequent.antecedents)
        consequent_part = ' & '.join(sequent.consequents)
        rule_name = f"sequent_{index}_{'contra' if sequent.is_contrapositive else 'fwd'}"

        try:
            self.program.add_rule(
                pr.Rule(
                    rule_text=f"{antecedent_part} => {consequent_part}",
                    name=rule_name,
                    infer_edges=False,
                    set_static=False,
                    custom_thresholds=[sequent.confidence * 0.9],
                    weights=[1.0],
                    interval=pr.Interval(sequent.confidence * 0.9, sequent.confidence)
                )
            )
            logging.debug(f"Added sequent rule: {rule_name}")
        except Exception as e:
            logging.error(f"Error adding sequent rule {rule_name}: {e}")

    def evaluate_originality_consistency(self) -> Dict:
        """Evaluate consistency of originality arguments using PyReason."""
        # Build the argument graph
        graph = nx.DiGraph()
        facts_node = {}
        facts_edge = {}
        
        # Add nodes (arguments)
        for arg_id, arg in self._current_arguments.items():
            graph.add_node(arg_id)
            facts_node[arg_id] = {"value": pr.Interval(arg.confidence[0], arg.confidence[1])}

        # Add edges (attacks and supports)
        edge_counter = 0
        for source, target in self._current_attacks:
            if graph.has_node(source) and graph.has_node(target):
                edge_id = f"edge_{edge_counter}"
                edge_counter += 1
                graph.add_edge(source, target, type=ArgumentType.ATTACK.value, weight=1.0)
                facts_edge[edge_id] = {"type": ArgumentType.ATTACK.value, "weight": 1.0}

        for arg_id, arg_data in self._current_arguments.items():
            for supporter in arg_data.supporting_args:
                if graph.has_node(supporter) and graph.has_node(arg_id):
                    edge_id = f"edge_{edge_counter}"
                    edge_counter += 1
                    graph.add_edge(supporter, arg_id, type=ArgumentType.SUPPORT.value, weight=1.0)
                    facts_edge[edge_id] = {"type": ArgumentType.SUPPORT.value, "weight": 1.0}

        # Initialize PyReason Reasoner
        try:
            reasoner = pr.Reasoner(
                program=self.program,
                timeline=self.timeline,
                graph=graph,
                facts_node=facts_node,
                facts_edge=facts_edge,
                rules=getattr(self.program, '_rules', []),
                ipl=True,
                reverse_graph=graph.reverse(copy=True),
                atom_trace=True,
                save_graph_attributes_to_rule_trace=True,
                inconsistency_check=True,
                store_interpretation_changes=True,
                verbose=False
            )
        except Exception as e:
            logging.error(f"Failed to initialize PyReason Reasoner: {e}")
            return self._error_consistency_result(f"PyReason initialization failed: {e}")

        # Run reasoning
        try:
            interpretation = reasoner.reason(iterations=5, convergence_threshold=0.01)
            logging.info("PyReason reasoning completed.")
        except Exception as e:
            logging.error(f"Error during PyReason reasoning: {e}")
            return self._error_consistency_result(f"PyReason reasoning failed: {e}")

        # Analyze results
        return self._analyze_interpretation(interpretation)

    def _error_consistency_result(self, error_msg: str) -> Dict:
        """Return a consistency result dictionary for error cases."""
        return {
            "error": error_msg,
            "originality_arguments": {},
            "contrapositive_arguments": {},
            "overall_consistency": False
        }

    def _analyze_interpretation(self, interpretation) -> Dict:
        """Analyze PyReason interpretation results."""
        originality_consistency = {}
        contrapositive_consistency = {}
        
        for arg_id, arg in self._current_arguments.items():
            node_interpretation = interpretation.nodes.get(arg_id)
            result = self._get_argument_consistency(arg_id, node_interpretation)
            
            # Classify argument type
            if "is_original" in arg_id or "copyright_protected" in arg_id:
                originality_consistency[arg_id] = result
            if (arg.sequent_type == SequentType.CONTRAPOSITIVE or 
                "not_original" in arg_id or "copyright_not_protected" in arg_id):
                contrapositive_consistency[arg_id] = result

        # Check overall consistency
        overall_consistent = self._check_overall_consistency(
            interpretation, originality_consistency, contrapositive_consistency
        )

        return {
            "originality_arguments": originality_consistency,
            "contrapositive_arguments": contrapositive_consistency,
            "overall_consistency": overall_consistent
        }

    def _get_argument_consistency(self, arg_id: str, node_interpretation) -> Dict:
        """Get consistency analysis for a single argument."""
        if node_interpretation and hasattr(node_interpretation, 'value'):
            if isinstance(node_interpretation.value, pr.Interval):
                is_consistent = (node_interpretation.value.lower > 0.5 and 
                                (node_interpretation.value.upper - node_interpretation.value.lower) < 0.7)
                value_repr = (node_interpretation.value.lower, node_interpretation.value.upper)
            else:
                is_consistent = node_interpretation.value > 0.5 if isinstance(node_interpretation.value, (int, float)) else False
                value_repr = node_interpretation.value
                
            return {"value": value_repr, "is_consistent": is_consistent}
        else:
            logging.warning(f"Argument {arg_id} not found in PyReason interpretation.")
            return {"value": None, "is_consistent": False}

    def _check_overall_consistency(self, interpretation, orig_consistency, contra_consistency) -> bool:
        """Check for overall consistency between arguments and their negations."""
        checked_pairs = set()
        
        for arg_id in self._current_arguments:
            if arg_id.startswith("not_"): 
                continue
                
            neg_arg_id = f"not_{arg_id}"
            if arg_id not in checked_pairs and neg_arg_id in self._current_arguments:
                checked_pairs.update({arg_id, neg_arg_id})
                
                arg_val = interpretation.nodes.get(arg_id)
                neg_val = interpretation.nodes.get(neg_arg_id)
                
                arg_lower = arg_val.value.lower if arg_val and isinstance(arg_val.value, pr.Interval) else 0
                neg_lower = neg_val.value.lower if neg_val and isinstance(neg_val.value, pr.Interval) else 0
                
                if arg_lower > 0.7 and neg_lower > 0.7:
                    logging.warning(f"Inconsistency between {arg_id} and {neg_arg_id}")
                    return False
                    
        return True

# --- Logic Based Legal Reasoner ---

class LogicBasedLegalReasoner:
    """Main class for legal reasoning about IP originality."""
    
    def __init__(self):
        self.arguments: Dict[str, LegalArgument] = {}
        self._setup_ip_rules()
        self.originality_annotator = OriginalityAnnotator(self)

    def _setup_ip_rules(self):
        """Define high-level IP originality rules."""
        self.ip_rules = {
            "originality_criteria": {
                "imprint_of_personality": lambda args: self._evaluate_argument_confidence(args.get("personality_arg", "")) > 0.7,
                "creative_choices": lambda args: self._evaluate_argument_confidence(args.get("choices_arg", "")) > 0.7,
            },
            "originality_defeaters": {
                "technical_function": lambda args: self._evaluate_argument_confidence(args.get("function_arg", "")) > 0.8,
            }
        }
        logging.info("IP-specific high-level rules defined.")

    def add_argument(self, argument: LegalArgument):
        """Add a legal argument to the knowledge base."""
        if argument.id in self.arguments:
            logging.warning(f"Argument {argument.id} already exists. Overwriting.")
        self.arguments[argument.id] = argument

    def get_argument(self, arg_id: str) -> Optional[LegalArgument]:
        """Retrieve an argument by its ID."""
        return self.arguments.get(arg_id)

    def _evaluate_argument_confidence(self, arg_id: str) -> float:
        """Get the average confidence of an argument."""
        argument = self.get_argument(arg_id)
        return (argument.confidence[0] + argument.confidence[1]) / 2 if argument else 0.0

    def apply_ip_rules(self, argument_ids: List[str]) -> Dict:
        """Apply high-level IP rules to check for key argument types."""
        results = {"criteria_met": [], "defeaters_present": []}
        
        for arg_id in argument_ids:
            arg = self.get_argument(arg_id)
            if not arg:
                continue
                
            # Check originality criteria
            if "personality" in arg.premise and self.ip_rules["originality_criteria"]["imprint_of_personality"]({"personality_arg": arg_id}):
                results["criteria_met"].append(f"{arg_id} (Personality)")
                
            if "creative choices" in arg.premise and self.ip_rules["originality_criteria"]["creative_choices"]({"choices_arg": arg_id}):
                results["criteria_met"].append(f"{arg_id} (Creative Choices)")
                
            # Check originality defeaters
            if "technical function" in arg.premise and self.ip_rules["originality_defeaters"]["technical_function"]({"function_arg": arg_id}):
                results["defeaters_present"].append(f"{arg_id} (Technical Function)")
                
        return results

    def analyze_legal_case(self, case_text: str) -> Dict:
        """Analyze a legal case for IP originality arguments."""
        try:
            logging.info("Starting analysis for new case.")
            
            # Use the OriginalityAnnotator
            annotation_result = self.originality_annotator.annotate_originality(case_text[:50000])  # Limit text length

            if not annotation_result["arguments"]:
                logging.warning("No arguments generated for this case.")
                return {"error": "No arguments generated."}

            # Perform AAF analysis
            aaf_results = self._perform_aaf_analysis(
                set(annotation_result["arguments"].keys()),
                annotation_result["attacks"]
            )

            # Evaluate sequent calculus
            sequent_results = self._evaluate_sequents(annotation_result["sequents"])

            # Apply high-level IP rules
            rule_results = self.apply_ip_rules(list(annotation_result["arguments"].keys()))

            # Combine results
            return {
                "arguments_identified": list(annotation_result["arguments"].keys()),
                "attack_relations": list(annotation_result["attacks"]),
                "pyreason_analysis": annotation_result["pyreason_consistency_results"],
                "aaf_analysis": aaf_results,
                "sequent_calculus_evaluation": sequent_results,
                "high_level_ip_rules": rule_results
            }

        except Exception as e:
            logging.exception(f"Critical error analyzing case: {e}")
            return {"error": f"Unexpected error during analysis: {e}"}

    def _perform_aaf_analysis(self, arguments: Set[str], attacks: Set[Tuple[str, str]]) -> Dict:
        """Perform Abstract Argumentation Framework analysis."""
        try:
            analyzer = OptimizedAAF(arguments, attacks)
            return {
                "preferred_extensions": [list(ext) for ext in analyzer.get_preferred_extensions()],
                "stable_extensions": [list(ext) for ext in analyzer.get_stable_extensions()],
                "has_cycle": analyzer.has_cycle(),
                "influence_pagerank": analyzer.pagerank_influence()
            }
        except Exception as e:
            logging.error(f"Error during AAF analysis: {e}")
            return {"error": f"AAF analysis failed: {e}"}

    def _evaluate_sequents(self, sequents: List[Sequent]) -> Dict:
        """Evaluate the validity of generated sequents."""
        results = {
            "total_sequents": len(sequents),
            "valid_sequents": 0,
            "invalid_sequents": 0,
            "contrapositive_valid": 0,
            "contrapositive_total": 0,
            "details": []
        }

        for i, sequent in enumerate(sequents):
            results["contrapositive_total"] += 1 if sequent.is_contrapositive else 0

            # Get minimum confidence bounds
            min_ant = min(
                [self.get_argument(a).confidence[0] for a in sequent.antecedents if self.get_argument(a)] + [1.0]
            )
            min_con = min(
                [self.get_argument(c).confidence[0] for c in sequent.consequents if self.get_argument(c)] + [1.0]
            )

            # Check validity
            is_valid = min_con >= min_ant - 0.2
            if is_valid:
                results["valid_sequents"] += 1
                if sequent.is_contrapositive:
                    results["contrapositive_valid"] += 1

            results["details"].append({
                "sequent_index": i,
                "antecedents": list(sequent.antecedents),
                "consequents": list(sequent.consequents),
                "min_antecedent_lb": min_ant,
                "min_consequent_lb": min_con,
                "is_valid": is_valid,
                "is_contrapositive": sequent.is_contrapositive
            })

        # Calculate ratios
        results["validity_ratio"] = (
            results["valid_sequents"] / results["total_sequents"] 
            if results["total_sequents"] > 0 else 0
        )
        results["contrapositive_validity_ratio"] = (
            results["contrapositive_valid"] / results["contrapositive_total"] 
            if results["contrapositive_total"] > 0 else 0
        )

        return results

# --- Dataset Processing Function ---

def process_ip_dataset(csv_path: str, max_cases: Optional[int] = None) -> Dict:
    """
    Process a dataset of legal cases for IP originality analysis.
    
    Args:
        csv_path: Path to the CSV file containing legal cases
        max_cases: Maximum number of cases to process (None for all)
        
    Returns:
        Dictionary containing processing results
    """
    try:
        # Load dataset
        df, text_col = _load_dataset(csv_path)
        if df is None:
            return {"error": "Dataset loading failed."}

        # Initialize reasoner
        reasoner = LogicBasedLegalReasoner()
        num_to_process = min(max_cases, len(df)) if max_cases is not None else len(df)
        logging.info(f"Processing {num_to_process} cases...")

        # Process cases
        all_results = []
        for i in range(num_to_process):
            case_result = _process_single_case(df, i, text_col, reasoner)
            all_results.append(case_result)
            
            # Clear state between cases if needed
            reasoner.arguments = {}
            reasoner.originality_annotator._reset_analysis_state()

        return _summarize_results(all_results, num_to_process)

    except Exception as e:
        logging.exception("Fatal error during dataset processing.")
        return {"error": f"Fatal error in process_ip_dataset: {e}"}

def _load_dataset(csv_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load the dataset and identify the text column."""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='warn')
        text_col = next(
            (col for col in ['text', 'decision_text', 'content', 'case_body', 'full_text'] 
            if col in df.columns),
            None
        )
        if not text_col:
            logging.error("No text column found in the CSV.")
            return None, None
        logging.info(f"Dataset loaded successfully using column '{text_col}'. Shape: {df.shape}")
        return df, text_col
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None, None

def _process_single_case(df: pd.DataFrame, idx: int, text_col: str, 
                        reasoner: LogicBasedLegalReasoner) -> Dict:
    """Process a single case from the dataset."""
    case_row = df.iloc[idx]
    case_text = str(case_row[text_col]) if pd.notna(case_row[text_col]) else ""
    case_id = case_row.get('id', idx)
    
    logging.info(f"--- Processing Case {case_id} ({idx + 1}/{len(df)}) ---")
    if not case_text:
        logging.warning(f"Case {case_id} has empty text. Skipping.")
        return {"case_id": case_id, "status": "skipped_empty_text"}
    
    # Analyze the case
    analysis_result = reasoner.analyze_legal_case(case_text)
    
    # Prepare case summary
    summary = {
        "case_id": case_id,
        "status": "processed",
        "originality_arguments_found": len(analysis_result.get("arguments_identified", [])),
        "attacks_found": len(analysis_result.get("attack_relations", [])),
        "pyreason_overall_consistency": analysis_result.get("pyreason_analysis", {}).get("overall_consistency"),
        "aaf_preferred_extensions_count": len(analysis_result.get("aaf_analysis", {}).get("preferred_extensions", [])),
        "aaf_stable_extensions_count": len(analysis_result.get("aaf_analysis", {}).get("stable_extensions", [])),
        "sequent_validity_ratio": analysis_result.get("sequent_calculus_evaluation", {}).get("validity_ratio"),
    }
    
    if "error" in analysis_result:
        summary.update({
            "status": "error",
            "error_message": analysis_result["error"]
        })
        
    return summary

def _summarize_results(results: List[Dict], total_cases: int) -> Dict:
    """Generate summary statistics from case processing results."""
    processed = [r for r in results if r["status"] == "processed"]
    errors = [r for r in results if r["status"] == "error"]
    num_processed = len(processed)
    
    summary = {
        "total_cases_attempted": total_cases,
        "cases_successfully_processed": num_processed,
        "cases_with_errors": len(errors),
        "cases_skipped": len(results) - num_processed - len(errors),
        "average_originality_args": (
            sum(r["originality_arguments_found"] for r in processed) / num_processed 
            if num_processed else 0
        ),
        "average_attacks": (
            sum(r["attacks_found"] for r in processed) / num_processed 
            if num_processed else 0
        ),
        "cases_pyreason_consistent": sum(1 for r in processed if r.get("pyreason_overall_consistency") is True),
        "percent_pyreason_consistent": (
            sum(1 for r in processed if r.get("pyreason_overall_consistency") is True) / num_processed * 100 
            if num_processed else 0
        ),
        "average_sequent_validity": (
            sum(r["sequent_validity_ratio"] for r in processed if r["sequent_validity_ratio"] is not None) / num_processed 
            if num_processed else 0
        ),
    }
    
    logging.info("--- Dataset Processing Summary ---")
    for key, value in summary.items():
        logging.info(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    return {
        "case_summaries": results,
        "overall_summary": summary
    }

# --- Main Execution ---

def main():
    """Main execution function."""
    CSV_PATH = os.getenv("CSV_PATH", "path/to/your/french_law_dataset.csv")  # UPDATE THIS PATH
    MAX_CASES_TO_PROCESS = 10  # Set to None to process all cases
    
    if not os.path.exists(CSV_PATH):
        logging.error(f"Dataset not found at: {CSV_PATH}")
        logging.error("Please set CSV_PATH or update the default path.")
        return
    
    try:
        results = process_ip_dataset(CSV_PATH, max_cases=MAX_CASES_TO_PROCESS)
        
        if "error" in results:
            logging.error(f"Processing failed: {results['error']}")
        else:
            logging.info("--- IP Originality Analysis Complete ---")
            _log_summary(results.get('overall_summary', {}))
            
            # Optionally save results to file
            # import json
            # with open("ip_analysis_results.json", "w") as f:
            #     json.dump(results, f, indent=2)
            # logging.info("Results saved to ip_analysis_results.json")

    except Exception as e:
        logging.exception("Critical error in main execution.")

def _log_summary(summary: Dict):
    """Log the summary statistics."""
    if not summary:
        logging.warning("No summary generated.")
        return
        
    logging.info("Overall Summary:")
    for key, value in summary.items():
        logging.info(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

if __name__ == "__main__":
    main()
