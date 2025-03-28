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
# import numba # Removed as it wasn't used directly in the logic shown
import pandas as pd
import itertools # Added for OptimizedAAF

csv.field_size_limit(sys.maxsize)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Argument Framework (AAF) Definition ---
class OptimizedAAF:
    """Represents and analyzes an Abstract Argumentation Framework."""
    def __init__(self, arguments: Set[str], attacks: Set[Tuple[str, str]]):
        self.arguments = arguments
        # Store attacks efficiently: attacker -> set of targets
        self.attacks_dict = {arg: set() for arg in arguments}
        # Store attackers efficiently: target -> set of attackers
        self.attackers_dict = {arg: set() for arg in arguments}
        for attacker, target in attacks:
            if attacker in self.arguments and target in self.arguments:
                self.attacks_dict[attacker].add(target)
                self.attackers_dict[target].add(attacker)

        # NetworkX graph for algorithms that benefit from it
        self.attack_graph = nx.DiGraph()
        self.attack_graph.add_nodes_from(arguments)
        self.attack_graph.add_edges_from(attacks)

    def is_conflict_free(self, S: Set[str]) -> bool:
        """Checks if a set of arguments S is conflict-free."""
        subset_args = S.intersection(self.arguments) # Ensure we only consider valid args
        for arg1 in subset_args:
            attacked_by_arg1 = self.attacks_dict.get(arg1, set())
            if not subset_args.isdisjoint(attacked_by_arg1):
                return False
        return True

    def defends(self, S: Set[str], a: str) -> bool:
        """Checks if set S defends argument 'a'."""
        if a not in self.arguments:
            return False # Cannot defend an argument not in the framework
        subset_args = S.intersection(self.arguments)
        attackers_of_a = self.attackers_dict.get(a, set())
        for attacker in attackers_of_a:
            # Check if any argument in S attacks this attacker
            is_attacker_attacked_by_S = False
            for defender in subset_args:
                if attacker in self.attacks_dict.get(defender, set()):
                    is_attacker_attacked_by_S = True
                    break
            if not is_attacker_attacked_by_S:
                return False # Attacker 'attacker' is not counter-attacked by S
        return True

    def get_admissible_sets(self) -> List[Set[str]]:
        """Generates all admissible sets."""
        admissible = []
        # Iterate through the power set of arguments
        for i in range(1 << len(self.arguments)):
            subset = set()
            temp_args = list(self.arguments) # Consistent ordering
            for j in range(len(self.arguments)):
                if (i >> j) & 1:
                    subset.add(temp_args[j])

            if self.is_conflict_free(subset):
                is_admissible = True
                for arg_in_subset in subset:
                    if not self.defends(subset, arg_in_subset):
                        is_admissible = False
                        break
                if is_admissible:
                    admissible.append(subset)
        return admissible

    def get_preferred_extensions(self) -> List[Set[str]]:
        """Finds preferred extensions (maximal admissible sets)."""
        admissible_sets = self.get_admissible_sets()
        preferred = []
        for s1 in admissible_sets:
            is_maximal = True
            for s2 in admissible_sets:
                if s1 != s2 and s1.issubset(s2):
                    is_maximal = False
                    break
            if is_maximal:
                preferred.append(s1)
        return preferred

    def get_stable_extensions(self) -> List[Set[str]]:
        """Finds stable extensions (conflict-free sets attacking all outside args)."""
        stable = []
        # Iterate through the power set (similar to admissible)
        for i in range(1 << len(self.arguments)):
            subset = set()
            temp_args = list(self.arguments) # Consistent ordering
            for j in range(len(self.arguments)):
                if (i >> j) & 1:
                    subset.add(temp_args[j])

            if self.is_conflict_free(subset):
                attacks_all_outside = True
                arguments_outside = self.arguments - subset
                for outside_arg in arguments_outside:
                    is_attacked = False
                    for attacker_in_subset in subset:
                        if outside_arg in self.attacks_dict.get(attacker_in_subset, set()):
                            is_attacked = True
                            break
                    if not is_attacked:
                        attacks_all_outside = False
                        break
                if attacks_all_outside:
                    stable.append(subset)
        return stable

    # ----- Graph Traversal Methods (using self.attack_graph) -----

    def dfs_detect_cycle(self) -> bool:
        """Detects cycles in the argumentation graph using DFS."""
        try:
            # Find strongly connected components. A cycle exists if any component > 1 node.
            sccs = list(nx.strongly_connected_components(self.attack_graph))
            return any(len(comp) > 1 for comp in sccs)
        except Exception as e:
             logging.error(f"Error detecting cycle: {e}")
             return False # Assume no cycle if error occurs

    def bfs_shortest_attack_path(self, start: str, goal: str) -> Optional[List[str]]:
        """Finds the shortest attack path from start to goal using BFS."""
        if start not in self.attack_graph or goal not in self.attack_graph:
            return None
        try:
            return nx.shortest_path(self.attack_graph, source=start, target=goal)
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
             logging.error(f"Error finding shortest path: {e}")
             return None


    def pagerank_influence(self) -> Dict[str, float]:
        """Computes the influence of arguments using PageRank on the attack graph."""
        try:
             # Higher alpha means more weight on links, less on random jumps.
             # Personalization can bias towards certain nodes if needed.
            return nx.pagerank(self.attack_graph, alpha=0.85)
        except Exception as e:
             logging.error(f"Error computing PageRank: {e}")
             return {arg: 0.0 for arg in self.arguments}

# --- Enums and Data Classes (Mostly Unchanged) ---

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
    id: str
    premise: str # Description of the reason/evidence
    conclusion: str # What the argument asserts
    confidence: Tuple[float, float] # Belief interval [lower, upper]
    supporting_args: Set[str] = field(default_factory=set)
    attacking_args: Set[str] = field(default_factory=set)
    sequent_type: SequentType = SequentType.HYPOTHESIS

@dataclass
class Sequent:
    antecedents: Set[str] # Set of argument IDs
    consequents: Set[str] # Set of argument IDs
    confidence: float # Confidence in the implication (used for rule thresholds)
    is_contrapositive: bool = False


# --- Originality Annotator Class ---

class OriginalityAnnotator:
    """Annotates IP originality arguments and sets up PyReason reasoning."""
    def __init__(self, reasoner):
        self.reasoner = reasoner # The main LogicBasedLegalReasoner instance
        self.sequents: List[Sequent] = []
        self.program = pr.Program()
        self.timeline = pr.Timeline()
        # Internal storage for the current analysis
        self._current_arguments: Dict[str, LegalArgument] = {}
        self._current_attacks: Set[Tuple[str, str]] = set()

    def extract_originality_claims(self, text: str) -> List[Dict]:
        """
        Extract potential originality claims from legal text.
        (This is a placeholder - needs a real NLP model or rule-based extractor)
        """
        # Placeholder: Look for keywords - Replace with actual extraction logic
        claims = []
        if "empreinte de la personnalité" in text or "author's personality" in text:
            claims.append({
                "is_valid": True, # Placeholder validity
                "premise": "Work shows the imprint of the author's personality.",
                "conclusion": "The work is original.",
                "confidence": 0.88 # Initial confidence estimate
            })
        if "choix créatifs libres" in text or "free creative choices" in text:
             claims.append({
                "is_valid": True,
                "premise": "Author made free and creative choices.",
                "conclusion": "The work is original.",
                "confidence": 0.91
            })
        if "dictated by technical function" in text:
             claims.append({
                "is_valid": True, # This claim *attacks* originality
                "premise": "The work's form is dictated by its technical function.",
                "conclusion": "The work is NOT original.", # Attacks originality
                "confidence": 0.95
            })
        # Add more extraction rules based on French IP law criteria

        # Simulate finding a counter-argument based on functionality
        if claims and any("technical function" in c["premise"] for c in claims):
             claims.append({
                 "is_valid": True,
                 "premise": "Functional elements lack originality.",
                 "conclusion": "No copyright protection for functional aspects.",
                 "confidence": 0.90
             })

        return claims


    def construct_originality_sequents(self, arguments: Dict[str, LegalArgument]) -> List[Sequent]:
        """Construct sequents (and contrapositives) for originality arguments."""
        sequents = []
        originality_args = {arg_id for arg_id, arg in arguments.items() if "is_original" in arg_id}
        protection_args = {arg_id for arg_id, arg in arguments.items() if "copyright_protected" in arg_id}
        no_protection_args = {arg_id for arg_id, arg in arguments.items() if "copyright_not_protected" in arg_id}
        not_original_args = {arg_id for arg_id, arg in arguments.items() if "not_original" in arg_id}

        # Rule: Originality -> Copyright Protection
        if originality_args and protection_args:
            # Simple version: any originality arg implies any protection arg
            # A more refined version could link specific originality args to protection
            for orig_arg in originality_args:
                 sequent_conf = arguments[orig_arg].confidence[0] # Use lower bound as base confidence
                 sequents.append(Sequent(
                     antecedents={orig_arg},
                     consequents=protection_args, # Leads to all protection conclusions
                     confidence=sequent_conf * 0.9 # Rule confidence slightly lower than premise
                 ))
                 # Contrapositive: No Protection -> Not Original
                 sequents.append(Sequent(
                     antecedents=no_protection_args, # If any 'no protection' holds
                     consequents=not_original_args, # Then all 'not original' might hold
                     confidence=sequent_conf * 0.9,
                     is_contrapositive=True
                 ))

        # Add more domain-specific sequents if needed
        return sequents

    def annotate_originality(self, case_text: str) -> Dict:
        """Annotate originality arguments, create sequents, and set up PyReason."""
        self.program = pr.Program() # Reset program
        self.timeline = pr.Timeline() # Reset timeline
        self._current_arguments = {}
        self._current_attacks = set()

        claims = self.extract_originality_claims(case_text)
        if not claims:
             logging.warning("No originality claims extracted from text.")
             # Return empty/default structure if no claims found
             return {
                "arguments": {},
                "attacks": set(),
                "sequents": [],
                "pyreason_consistency_results": {
                    "originality_arguments": {},
                    "contrapositive_arguments": {},
                    "overall_consistency": True # Vacuously true
                }
            }

        arguments = self._create_arguments_from_claims(claims)
        self._current_arguments = arguments

        # Identify attacks based on argument conclusions
        self._identify_attacks()
        # Update attacking_args in LegalArgument objects
        for attacker, target in self._current_attacks:
            if target in self._current_arguments:
                self._current_arguments[target].attacking_args.add(attacker)
            # We could also add symmetric attacks for negations, but let's keep it simple
            # if attacker in self._current_arguments and f"not_{target}" in self._current_arguments:
            #     self._current_arguments[f"not_{target}"].supporting_args.add(attacker)


        self.sequents = self.construct_originality_sequents(arguments)

        # Add arguments to the main reasoner's knowledge base
        for arg in arguments.values():
            self.reasoner.add_argument(arg) # Assumes reasoner has this method

        # Setup PyReason facts and rules based on the generated arguments and relationships
        self.setup_pyreason_knowledge(arguments, self._current_attacks)

        # Evaluate consistency using PyReason
        consistency_results = self.evaluate_originality_consistency(arguments, self._current_attacks)

        return {
            "arguments": arguments,
            "attacks": self._current_attacks,
            "sequents": self.sequents,
            "pyreason_consistency_results": consistency_results
        }

    def _create_arguments_from_claims(self, claims: List[Dict]) -> Dict[str, LegalArgument]:
        """Create LegalArgument objects from extracted claims."""
        arguments = {}
        for i, claim in enumerate(claims):
            base_id = f"claim_{i}" # Base identifier for this claim
            confidence_interval = (claim["confidence"] * 0.8, claim["confidence"]) # Example interval

            # Determine if the claim supports or denies originality/protection
            supports_originality = "not original" not in claim["conclusion"].lower() and \
                                   "no copyright" not in claim["conclusion"].lower()

            if supports_originality:
                 # Argument for originality
                 orig_arg_id = f"is_original_{i}"
                 arguments[orig_arg_id] = LegalArgument(
                     id=orig_arg_id,
                     premise=claim["premise"],
                     conclusion="The work is considered original based on: " + claim["premise"],
                     confidence=confidence_interval,
                     sequent_type=SequentType.HYPOTHESIS
                 )
                 # Corresponding negated argument
                 not_orig_arg_id = f"not_original_{i}"
                 arguments[not_orig_arg_id] = LegalArgument(
                     id=not_orig_arg_id,
                     premise=f"Negation of: {claim['premise']}",
                     conclusion=f"Work may not be original despite: {claim['premise']}",
                     confidence=(1.0 - confidence_interval[1], 1.0 - confidence_interval[0]), # Inverted confidence
                     attacking_args={orig_arg_id}, # Attacks the positive claim
                     sequent_type=SequentType.CONTRAPOSITIVE # Conceptually related
                 )
                 arguments[orig_arg_id].attacking_args.add(not_orig_arg_id) # Mutual attack for negation

                 # Argument for copyright protection (if originality implies protection)
                 prot_arg_id = f"copyright_protected_{i}"
                 arguments[prot_arg_id] = LegalArgument(
                     id=prot_arg_id,
                     premise=f"Based on originality argument {orig_arg_id}",
                     conclusion="Copyright protection likely applies.",
                     confidence=(confidence_interval[0] * 0.9, confidence_interval[1]), # Slightly derived confidence
                     supporting_args={orig_arg_id},
                     sequent_type=SequentType.CONCLUSION
                 )
                 # Corresponding negated argument
                 not_prot_arg_id = f"copyright_not_protected_{i}"
                 arguments[not_prot_arg_id] = LegalArgument(
                     id=not_prot_arg_id,
                     premise=f"If work is not original (related to {not_orig_arg_id})",
                     conclusion="Copyright protection likely does not apply.",
                     confidence=(1.0 - arguments[prot_arg_id].confidence[1], 1.0 - arguments[prot_arg_id].confidence[0]),
                     supporting_args={not_orig_arg_id},
                     attacking_args={prot_arg_id},
                     sequent_type=SequentType.CONTRAPOSITIVE
                 )
                 arguments[prot_arg_id].attacking_args.add(not_prot_arg_id)

            else: # Claim argues against originality or protection
                # Argument against originality
                not_orig_arg_id = f"not_original_{i}"
                arguments[not_orig_arg_id] = LegalArgument(
                    id=not_orig_arg_id,
                    premise=claim["premise"],
                    conclusion="The work is likely NOT original due to: " + claim["premise"],
                    confidence=confidence_interval,
                    sequent_type=SequentType.HYPOTHESIS
                )
                # Corresponding negated argument (i.e., argument FOR originality despite the claim)
                orig_arg_id = f"is_original_{i}"
                arguments[orig_arg_id] = LegalArgument(
                     id=orig_arg_id,
                     premise=f"Counter-argument to: {claim['premise']}",
                     conclusion=f"Work might still be original despite: {claim['premise']}",
                     confidence=(1.0 - confidence_interval[1], 1.0 - confidence_interval[0]),
                     attacking_args={not_orig_arg_id},
                     sequent_type=SequentType.CONTRAPOSITIVE
                 )
                arguments[not_orig_arg_id].attacking_args.add(orig_arg_id)

                # Argument against copyright protection
                not_prot_arg_id = f"copyright_not_protected_{i}"
                arguments[not_prot_arg_id] = LegalArgument(
                     id=not_prot_arg_id,
                     premise=f"Based on non-originality argument {not_orig_arg_id}",
                     conclusion="Copyright protection likely does NOT apply.",
                     confidence=(confidence_interval[0] * 0.9, confidence_interval[1]),
                     supporting_args={not_orig_arg_id},
                     sequent_type=SequentType.CONCLUSION
                 )
                 # Corresponding negated argument (i.e., protection applies despite the claim)
                prot_arg_id = f"copyright_protected_{i}"
                arguments[prot_arg_id] = LegalArgument(
                     id=prot_arg_id,
                     premise=f"Counter-argument regarding protection based on {orig_arg_id}",
                     conclusion="Copyright protection might still apply.",
                     confidence=(1.0 - arguments[not_prot_arg_id].confidence[1], 1.0 - arguments[not_prot_arg_id].confidence[0]),
                     supporting_args={orig_arg_id},
                     attacking_args={not_prot_arg_id},
                     sequent_type=SequentType.CONTRAPOSITIVE
                 )
                arguments[not_prot_arg_id].attacking_args.add(prot_arg_id)

        return arguments

    def _identify_attacks(self):
        """Identify attack relationships between generated arguments."""
        self._current_attacks = set()
        arg_ids = list(self._current_arguments.keys())

        for i in range(len(arg_ids)):
            arg1_id = arg_ids[i]
            arg1 = self._current_arguments[arg1_id]

            # 1. Direct negation attacks (already added in _create_arguments_from_claims)
            # Example: is_original_0 attacks not_original_0 and vice-versa.
            self._current_attacks.update((arg1_id, atk_id) for atk_id in arg1.attacking_args if atk_id in self._current_arguments)


            # 2. Substantive attacks (e.g., 'functionality' attacks 'originality')
            for j in range(i + 1, len(arg_ids)):
                arg2_id = arg_ids[j]
                arg2 = self._current_arguments[arg2_id]

                # Simple check: if one concludes "original" and the other "NOT original"
                conclusion1_lower = arg1.conclusion.lower()
                conclusion2_lower = arg2.conclusion.lower()

                # If arg1 asserts originality and arg2 asserts non-originality
                if ("is original" in conclusion1_lower or "considered original" in conclusion1_lower) and \
                   ("not original" in conclusion2_lower):
                    # Check if they are not just negations of each other (already handled)
                    if f"not_{arg1_id}" != arg2_id and f"not_{arg2_id}" != arg1_id:
                         self._current_attacks.add((arg2_id, arg1_id)) # arg2 attacks arg1

                # If arg2 asserts originality and arg1 asserts non-originality
                elif ("is original" in conclusion2_lower or "considered original" in conclusion2_lower) and \
                     ("not original" in conclusion1_lower):
                     if f"not_{arg1_id}" != arg2_id and f"not_{arg2_id}" != arg1_id:
                         self._current_attacks.add((arg1_id, arg2_id)) # arg1 attacks arg2

                # Similar logic for "protection applies" vs "protection does not apply"
                if ("protection likely applies" in conclusion1_lower) and \
                   ("protection likely does not apply" in conclusion2_lower):
                    if f"not_{arg1_id}" != arg2_id and f"not_{arg2_id}" != arg1_id:
                         self._current_attacks.add((arg2_id, arg1_id))
                elif ("protection likely applies" in conclusion2_lower) and \
                     ("protection likely does not apply" in conclusion1_lower):
                     if f"not_{arg1_id}" != arg2_id and f"not_{arg2_id}" != arg1_id:
                         self._current_attacks.add((arg1_id, arg2_id))

        logging.debug(f"Identified attacks: {self._current_attacks}")


    def setup_pyreason_knowledge(self, arguments: Dict[str, LegalArgument], attacks: Set[Tuple[str, str]]):
        """Set up PyReason knowledge base with facts and rules for originality."""
        current_time = 0 # Using a single time point for simplicity
        self.timeline.add_time_point(current_time)

        # Add facts for each argument
        for arg_id, arg in arguments.items():
            try:
                self.program.add_fact(
                    pr.Fact(
                        name=f"{arg_id}_fact", # Unique name for fact
                        component=arg_id, # Link fact to the argument node
                        value=pr.Interval(arg.confidence[0], arg.confidence[1]),
                        start_time=current_time,
                        end_time=current_time,
                        static=False # Beliefs can change
                    )
                )
            except Exception as e:
                logging.error(f"Error adding fact for {arg_id}: {e}")


        # Add rules for SUPPORT relationships (based on supporting_args)
        for arg_id, arg in arguments.items():
            for support_id in arg.supporting_args:
                if support_id in arguments: # Ensure supporter exists
                    # Rule: supporter -> arg_id
                    self._add_support_rule(support_id, arg_id)

        # Add rules for ATTACK relationships
        for attacker_id, target_id in attacks:
             if attacker_id in arguments and target_id in arguments:
                  # Rule: attacker -> ¬target
                  self._add_attack_rule(attacker_id, target_id)

        # Add rules for SEQUENTS (including contrapositives)
        for i, sequent in enumerate(self.sequents):
             # Rule: antecedent1 ∧ antecedent2 ... => consequent1 ∧ consequent2 ...
             self._add_sequent_rule(sequent, i)


    def _add_support_rule(self, source: str, target: str):
        """Add a support rule to the PyReason program."""
        rule_name = f"support_{source}_to_{target}"
        # PyReason rule syntax: source => target
        rule_text = f"{source} :- {source}_fact(1)." # Trigger based on fact
        target_rule = f"{target} += {source}." # How support propagates belief

        # Placeholder: Add basic rules. PyReason's exact syntax for interval propagation might differ.
        # This needs refinement based on how PyReason handles interval updates via rules.
        # For now, let's assume a simplified rule structure.
        # We might need custom annotation functions for precise interval logic.

        # Let's use infer_edges=True for simplicity, assuming edges represent propagation
        try:
            self.program.add_rule(
                pr.Rule(
                    rule_text=f"{source} => {target}", # Simplified representation
                    name=rule_name,
                    infer_edges=True, # Let PyReason manage graph edges based on this rule
                    set_static=False, # Support can be dynamic
                    custom_thresholds=[0.5], # Example threshold for propagation
                    weights=[1.0], # Example weight
                    interval=pr.Interval(0.7, 1.0) # Confidence in the support link itself
                )
            )
            logging.debug(f"Added support rule: {rule_name}")
        except Exception as e:
            logging.error(f"Error adding support rule {rule_name}: {e}")


    def _add_attack_rule(self, source: str, target: str):
        """Add an attack rule to the PyReason program."""
        rule_name = f"attack_{source}_to_{target}"
        # PyReason rule syntax: source => ¬target
        try:
            self.program.add_rule(
                pr.Rule(
                    rule_text=f"{source} => ¬{target}", # Negation operator
                    name=rule_name,
                    infer_edges=True, # Manage graph edges
                    set_static=False, # Attacks can be dynamic
                    custom_thresholds=[0.6], # Example threshold for attack effectiveness
                    weights=[1.0],
                    interval=pr.Interval(0.8, 1.0) # Confidence in the attack link
                )
            )
            logging.debug(f"Added attack rule: {rule_name}")
        except Exception as e:
            logging.error(f"Error adding attack rule {rule_name}: {e}")

    def _add_sequent_rule(self, sequent: Sequent, index: int):
        """Add a sequent rule (implication) to the PyReason program."""
        if not sequent.antecedents or not sequent.consequents:
            logging.warning(f"Skipping sequent {index} due to empty antecedents or consequents.")
            return

        # Format: ant1 ∧ ant2 ... => cons1 ∧ cons2 ...
        antecedent_part = ' & '.join(sequent.antecedents)
        consequent_part = ' & '.join(sequent.consequents) # Assuming PyReason handles conjunction on RHS

        rule_name = f"sequent_{index}_{'contra' if sequent.is_contrapositive else 'fwd'}"
        rule_text = f"{antecedent_part} => {consequent_part}"

        try:
            self.program.add_rule(
                pr.Rule(
                    rule_text=rule_text,
                    name=rule_name,
                    infer_edges=False, # This is a logical implication, not necessarily a direct graph edge
                    set_static=False, # Logical rules can have dynamic effects
                    custom_thresholds=[sequent.confidence * 0.9], # Lower bound for rule trigger
                    weights=[1.0],
                    interval=pr.Interval(sequent.confidence * 0.9, sequent.confidence) # Confidence interval of the rule
                )
            )
            logging.debug(f"Added sequent rule: {rule_name}")
        except Exception as e:
             logging.error(f"Error adding sequent rule {rule_name}: {e}")


    def evaluate_originality_consistency(self, arguments: Dict[str, LegalArgument], attacks: Set[Tuple[str, str]]) -> Dict:
        """Evaluate consistency of originality arguments using PyReason."""

        # Build the graph for PyReason (nodes are arguments, edges are attacks/supports)
        graph = nx.DiGraph()
        facts_node = {}
        facts_edge = {} # PyReason can use edge facts too

        # Add nodes (arguments)
        for arg_id, arg in arguments.items():
            graph.add_node(arg_id)
            # Node facts represent initial confidence
            facts_node[arg_id] = {"value": pr.Interval(arg.confidence[0], arg.confidence[1])}

        # Add edges (attacks and supports from argument relationships)
        edge_counter = 0
        for source, target in attacks:
            if graph.has_node(source) and graph.has_node(target):
                edge_id = f"edge_{edge_counter}"
                edge_counter += 1
                graph.add_edge(source, target, type=ArgumentType.ATTACK.value, weight=1.0)
                facts_edge[edge_id] = {"type": ArgumentType.ATTACK.value, "weight": 1.0} # Example edge fact

        for arg_id, arg_data in arguments.items():
            for supporter in arg_data.supporting_args:
                if graph.has_node(supporter) and graph.has_node(arg_id):
                     edge_id = f"edge_{edge_counter}"
                     edge_counter += 1
                     graph.add_edge(supporter, arg_id, type=ArgumentType.SUPPORT.value, weight=1.0)
                     facts_edge[edge_id] = {"type": ArgumentType.SUPPORT.value, "weight": 1.0}


        # Ensure the program has rules added
        if not hasattr(self.program, '_rules') or not self.program._rules:
             logging.warning("PyReason program has no rules defined before reasoning.")
             # Attempt to re-run setup if necessary, or return default failure
             # self.setup_pyreason_knowledge(arguments, attacks) # Careful about re-adding
             # if not self.program._rules:
             #      return {"error": "No rules could be added to PyReason program."}

        # Get rules (ensure they exist)
        rules = getattr(self.program, '_rules', [])

        # Create reverse graph (needed by some PyReason modes)
        reverse_graph = graph.reverse(copy=True)

        # Initialize PyReason Reasoner
        try:
             reasoner = pr.Reasoner(
                 program=self.program,
                 timeline=self.timeline,
                 graph=graph,
                 facts_node=facts_node,
                 facts_edge=facts_edge,
                 rules=rules,
                 ipl=True,  # Interval-based probabilistic logic
                 reverse_graph=reverse_graph,
                 atom_trace=True,
                 save_graph_attributes_to_rule_trace=True,
                 inconsistency_check=True, # Crucial for finding contradictions
                 store_interpretation_changes=True,
                 verbose=False # Keep console clean unless debugging
             )
        except Exception as e:
             logging.error(f"Failed to initialize PyReason Reasoner: {e}")
             return {
                 "error": f"PyReason initialization failed: {e}",
                 "originality_arguments": {},
                 "contrapositive_arguments": {},
                 "overall_consistency": False
             }


        # Run reasoning
        try:
            interpretation = reasoner.reason(iterations=5, convergence_threshold=0.01) # Run for a few steps
            logging.info("PyReason reasoning completed.")
            # You can access interpretation.nodes, interpretation.edges etc.
        except Exception as e:
            logging.error(f"Error during PyReason reasoning: {e}")
            return {
                "error": f"PyReason reasoning failed: {e}",
                "originality_arguments": {},
                "contrapositive_arguments": {},
                "overall_consistency": False
            }

        # Analyze results from the interpretation
        originality_consistency = {}
        contrapositive_consistency = {}

        for arg_id, arg in arguments.items():
            node_interpretation = interpretation.nodes.get(arg_id) # Get final state
            node_value = None
            is_consistent = False # Default to not consistent if no value found

            if node_interpretation and hasattr(node_interpretation, 'value'):
                 node_value = node_interpretation.value
                 if isinstance(node_value, pr.Interval):
                     # Consistent if the lower bound is reasonably high (e.g., > 0.5)
                     # And the interval isn't too wide (indicating uncertainty)
                     is_consistent = node_value.lower > 0.5 and (node_value.upper - node_value.lower) < 0.7
                     node_value_repr = (node_value.lower, node_value.upper)
                 else: # Handle cases where value might not be an interval
                     is_consistent = node_value > 0.5 if isinstance(node_value, (int, float)) else False
                     node_value_repr = node_value

                 result_entry = {"value": node_value_repr, "is_consistent": is_consistent}

                 if "is_original" in arg_id or "copyright_protected" in arg_id:
                     originality_consistency[arg_id] = result_entry
                 if arg.sequent_type == SequentType.CONTRAPOSITIVE or \
                    "not_original" in arg_id or "copyright_not_protected" in arg_id:
                     contrapositive_consistency[arg_id] = result_entry

            else:
                 # Argument not found in interpretation or has no value
                 result_entry = {"value": None, "is_consistent": False}
                 logging.warning(f"Argument {arg_id} not found or has no value in PyReason interpretation.")
                 if "is_original" in arg_id or "copyright_protected" in arg_id:
                      originality_consistency[arg_id] = result_entry
                 if arg.sequent_type == SequentType.CONTRAPOSITIVE or \
                    "not_original" in arg_id or "copyright_not_protected" in arg_id:
                      contrapositive_consistency[arg_id] = result_entry

        # Check for overall consistency: e.g., no argument and its negation are both strongly believed
        overall_consistent = True
        checked_pairs = set()
        for arg_id in arguments:
             if arg_id.startswith("not_"): continue # Check pairs from the positive side

             neg_arg_id = f"not_{arg_id}"
             if arg_id not in checked_pairs and neg_arg_id in arguments:
                 checked_pairs.add(arg_id)
                 checked_pairs.add(neg_arg_id)

                 arg_val = interpretation.nodes.get(arg_id)
                 neg_val = interpretation.nodes.get(neg_arg_id)

                 arg_lower = arg_val.value.lower if arg_val and isinstance(arg_val.value, pr.Interval) else 0
                 neg_lower = neg_val.value.lower if neg_val and isinstance(neg_val.value, pr.Interval) else 0

                 # Inconsistent if both an argument and its negation have high lower bounds
                 if arg_lower > 0.7 and neg_lower > 0.7:
                      overall_consistent = False
                      logging.warning(f"Inconsistency detected between {arg_id} (>{arg_lower:.2f}) and {neg_arg_id} (>{neg_lower:.2f})")
                      break # Found one inconsistency

        # Also consider PyReason's internal inconsistency detection if available
        # (Need to check PyReason's API for how inconsistency flags are exposed)

        return {
            "originality_arguments": originality_consistency,
            "contrapositive_arguments": contrapositive_consistency,
            "overall_consistency": overall_consistent
        }


# --- Logic Based Legal Reasoner (Adapted for Originality) ---

class LogicBasedLegalReasoner:
    def __init__(self):
        self.arguments: Dict[str, LegalArgument] = {} # Stores all arguments across cases if needed
        self.setup_ip_rules() # Changed method name
        # Initialize the annotator, passing self (the reasoner)
        self.originality_annotator = OriginalityAnnotator(self)

    def setup_ip_rules(self):
        """Define logical rules specific to IP originality."""
        # These rules are high-level checks, PyReason handles the detailed logic.
        self.ip_rules = {
            "originality_criteria": {
                "imprint_of_personality": lambda args: self.evaluate_confidence_avg(args.get("personality_arg", "")) > 0.7,
                "creative_choices": lambda args: self.evaluate_confidence_avg(args.get("choices_arg", "")) > 0.7,
            },
            "originality_defeaters": {
                 "technical_function": lambda args: self.evaluate_confidence_avg(args.get("function_arg", "")) > 0.8,
            }
        }
        logging.info("IP-specific high-level rules defined.")

    def add_argument(self, argument: LegalArgument):
        """Add a new legal argument to the central knowledge base."""
        if argument.id in self.arguments:
             logging.warning(f"Argument {argument.id} already exists. Overwriting.")
        self.arguments[argument.id] = argument

    def get_argument(self, arg_id: str) -> Optional[LegalArgument]:
        return self.arguments.get(arg_id)

    def evaluate_confidence_avg(self, arg_id: str) -> float:
        """Evaluate the average confidence of an argument."""
        argument = self.get_argument(arg_id)
        if not argument:
            return 0.0
        # Return the midpoint of the confidence interval
        return (argument.confidence[0] + argument.confidence[1]) / 2

    def apply_ip_rules(self, argument_ids: List[str]) -> Dict:
        """Apply high-level IP rules to check for presence of key argument types."""
        results = {"criteria_met": [], "defeaters_present": []}
        for arg_id in argument_ids:
            arg = self.get_argument(arg_id)
            if not arg: continue

            # Check if it matches any rule criteria (using premise/conclusion text)
            if "personality" in arg.premise and self.ip_rules["originality_criteria"]["imprint_of_personality"]({"personality_arg": arg_id}):
                 results["criteria_met"].append(f"{arg_id} (Personality)")
            if "creative choices" in arg.premise and self.ip_rules["originality_criteria"]["creative_choices"]({"choices_arg": arg_id}):
                 results["criteria_met"].append(f"{arg_id} (Creative Choices)")
            if "technical function" in arg.premise and self.ip_rules["originality_defeaters"]["technical_function"]({"function_arg": arg_id}):
                 results["defeaters_present"].append(f"{arg_id} (Technical Function)")
        return results

    def analyze_legal_case(self, case_text: str) -> Dict:
        """Analyze a legal case focusing on IP originality."""
        try:
            logging.info("Starting analysis for new case.")
            # Reset arguments for this specific case analysis if desired
            # self.arguments = {} # Uncomment if each case should be isolated

            # Use the OriginalityAnnotator
            # This generates arguments, identifies attacks, sets up PyReason, and runs it.
            annotation_result = self.originality_annotator.annotate_originality(case_text)

            # The annotator now holds the arguments and attacks for this case
            current_arguments = annotation_result["arguments"]
            current_attacks = annotation_result["attacks"]
            current_sequents = annotation_result["sequents"]
            pyreason_results = annotation_result["pyreason_consistency_results"]

            if not current_arguments:
                 logging.warning("No arguments generated for this case.")
                 return {"error": "No arguments generated."}

            # --- Optional: Perform AAF analysis using OptimizedAAF ---
            aaf_analyzer = None
            aaf_results = {}
            if current_arguments:
                 arg_ids_set = set(current_arguments.keys())
                 aaf_analyzer = OptimizedAAF(arg_ids_set, current_attacks)
                 try:
                      aaf_results = {
                          "preferred_extensions": [list(ext) for ext in aaf_analyzer.get_preferred_extensions()], # Convert sets to lists for JSON later
                          "stable_extensions": [list(ext) for ext in aaf_analyzer.get_stable_extensions()],
                          "has_cycle": aaf_analyzer.dfs_detect_cycle(),
                          "influence_pagerank": aaf_analyzer.pagerank_influence()
                      }
                 except Exception as e:
                      logging.error(f"Error during OptimizedAAF analysis: {e}")
                      aaf_results = {"error": f"AAF analysis failed: {e}"}


            # --- Evaluate Sequent Calculus ---
            sequent_results = self.evaluate_sequent_calculus(current_sequents)

            # --- Apply high-level IP rules ---
            rule_application_results = self.apply_ip_rules(list(current_arguments.keys()))

            # --- Combine Results ---
            final_results = {
                "arguments_identified": list(current_arguments.keys()),
                "attack_relations": list(current_attacks), # Convert set of tuples to list
                "pyreason_analysis": pyreason_results,
                "aaf_analysis": aaf_results, # Include AAF semantics results
                "sequent_calculus_evaluation": sequent_results,
                "high_level_ip_rules": rule_application_results
            }

            logging.info(f"Analysis complete. Overall PyReason consistency: {pyreason_results.get('overall_consistency', 'N/A')}")
            return final_results

        except Exception as e:
            logging.exception(f"Critical error analyzing case: {e}") # Use exception for traceback
            return {"error": f"Unexpected error during analysis: {e}"}


    def evaluate_sequent_calculus(self, sequents: List[Sequent]) -> Dict:
        """Evaluate the validity of generated sequents based on current argument confidences."""
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

            # Get lower bounds of confidence for antecedents and consequents
            # Use 1.0 if list is empty (vacuously true antecedent/consequent base)
            min_antecedent_lb = min(
                [self.get_argument(a).confidence[0] for a in sequent.antecedents if self.get_argument(a)] + [1.0]
            )
            min_consequent_lb = min(
                [self.get_argument(c).confidence[0] for c in sequent.consequents if self.get_argument(c)] + [1.0]
            )

            # Validity check: Does the minimum confidence of antecedents support the minimum confidence of consequents?
            # Let's consider a sequent valid if the consequent lower bound is not significantly lower than the antecedent lower bound.
            is_valid = min_consequent_lb >= min_antecedent_lb - 0.2 # Allow some drop-off

            if is_valid:
                results["valid_sequents"] += 1
                if sequent.is_contrapositive:
                    results["contrapositive_valid"] += 1
            else:
                results["invalid_sequents"] += 1

            results["details"].append({
                "sequent_index": i,
                "antecedents": list(sequent.antecedents),
                "consequents": list(sequent.consequents),
                "min_antecedent_lb": min_antecedent_lb,
                "min_consequent_lb": min_consequent_lb,
                "is_valid": is_valid,
                "is_contrapositive": sequent.is_contrapositive
            })

        # Add overall ratios
        results["validity_ratio"] = results["valid_sequents"] / results["total_sequents"] if results["total_sequents"] > 0 else 0
        results["contrapositive_validity_ratio"] = results["contrapositive_valid"] / results["contrapositive_total"] if results["contrapositive_total"] > 0 else 0

        return results


# --- Dataset Processing Function (Adapted for Originality) ---

def process_ip_dataset(csv_path: str, max_cases: Optional[int] = None):
    """Process a dataset of legal cases for IP originality analysis."""
    try:
        # Load dataset using pandas for robustness
        try:
            # Consider using chunking for very large files
            df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='warn')
            # Assume the relevant text is in a column named 'text' or similar
            # Find potential text columns (common names)
            potential_cols = ['text', 'decision_text', 'content', 'case_body', 'full_text']
            text_col = next((col for col in potential_cols if col in df.columns), None)
            if not text_col:
                 logging.error(f"Could not find a text column (e.g., 'text', 'decision_text') in the CSV: {csv_path}")
                 return {"error": "No text column found."}
            logging.info(f"Dataset loaded successfully using column '{text_col}'. Shape: {df.shape}")
        except Exception as e:
            logging.error(f"Error loading dataset from {csv_path}: {e}")
            return {"error": f"Dataset loading failed: {e}"}

        # Initialize reasoner
        reasoner = LogicBasedLegalReasoner()

        # Process cases
        all_results = []
        num_to_process = min(max_cases, len(df)) if max_cases is not None else len(df)
        logging.info(f"Processing {num_to_process} cases...")

        for i in range(num_to_process):
            case_row = df.iloc[i]
            case_text = str(case_row[text_col]) if pd.notna(case_row[text_col]) else ""
            case_id = case_row.get('id', i) # Use an 'id' column if available, otherwise index

            logging.info(f"--- Processing Case {case_id} ({i + 1}/{num_to_process}) ---")
            if not case_text:
                logging.warning(f"Case {case_id} has empty text. Skipping analysis.")
                all_results.append({"case_id": case_id, "status": "skipped_empty_text"})
                continue

            # Analyze the case
            analysis_result = reasoner.analyze_legal_case(case_text[:50000]) # Limit text length if needed

            # Store results for this case
            case_summary = {
                "case_id": case_id,
                "status": "processed",
                "originality_arguments_found": len(analysis_result.get("arguments_identified", [])),
                "attacks_found": len(analysis_result.get("attack_relations", [])),
                "pyreason_overall_consistency": analysis_result.get("pyreason_analysis", {}).get("overall_consistency", None),
                "aaf_preferred_extensions_count": len(analysis_result.get("aaf_analysis", {}).get("preferred_extensions", [])),
                "aaf_stable_extensions_count": len(analysis_result.get("aaf_analysis", {}).get("stable_extensions", [])),
                "sequent_validity_ratio": analysis_result.get("sequent_calculus_evaluation", {}).get("validity_ratio", None),
            }
            # Include error messages if analysis failed
            if "error" in analysis_result:
                 case_summary["status"] = "error"
                 case_summary["error_message"] = analysis_result["error"]

            all_results.append(case_summary)

            # Optional: Clear reasoner state if cases are independent
            reasoner.arguments = {}
            reasoner.originality_annotator.program = pr.Program()
            reasoner.originality_annotator.timeline = pr.Timeline()


        # Summarize overall results
        processed_cases = [r for r in all_results if r["status"] == "processed"]
        error_cases = [r for r in all_results if r["status"] == "error"]
        num_processed = len(processed_cases)

        summary = {
            "total_cases_attempted": num_to_process,
            "cases_successfully_processed": num_processed,
            "cases_with_errors": len(error_cases),
            "cases_skipped": len(all_results) - num_processed - len(error_cases),
            "average_originality_args": sum(r["originality_arguments_found"] for r in processed_cases) / num_processed if num_processed else 0,
            "average_attacks": sum(r["attacks_found"] for r in processed_cases) / num_processed if num_processed else 0,
            "cases_pyreason_consistent": sum(1 for r in processed_cases if r["pyreason_overall_consistency"] is True),
            "percent_pyreason_consistent": (sum(1 for r in processed_cases if r["pyreason_overall_consistency"] is True) / num_processed * 100) if num_processed else 0,
            "average_sequent_validity": sum(r["sequent_validity_ratio"] for r in processed_cases if r["sequent_validity_ratio"] is not None) / num_processed if num_processed else 0,
        }

        logging.info("--- Dataset Processing Summary ---")
        for key, value in summary.items():
             logging.info(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

        return {
            "case_summaries": all_results,
            "overall_summary": summary
        }

    except Exception as e:
        logging.exception("Fatal error during dataset processing.") # Log traceback
        return {"error": f"Fatal error in process_ip_dataset: {e}"}

# --- Main Execution ---

def main():
    # Use environment variable or default path
    # Make sure this path points to your actual CSV file.
    CSV_PATH = os.getenv("CSV_PATH", "path/to/your/french_law_dataset.csv") # UPDATE THIS PATH
    MAX_CASES_TO_PROCESS = 10 # Limit number of cases for faster testing, set to None to process all

    if not os.path.exists(CSV_PATH):
         logging.error(f"Dataset not found at specified path: {CSV_PATH}")
         logging.error("Please set the CSV_PATH environment variable or update the default path in the script.")
         return

    try:
        # Process dataset with IP originality focus
        results = process_ip_dataset(CSV_PATH, max_cases=MAX_CASES_TO_PROCESS)

        if "error" in results:
            logging.error(f"Dataset processing failed: {results['error']}")
        else:
            logging.info("--- IP Originality Analysis Complete ---")
            # Output overall summary again for clarity
            summary = results.get('overall_summary', {})
            if summary:
                 logging.info("Overall Summary:")
                 for key, value in summary.items():
                      logging.info(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
            else:
                 logging.warning("No overall summary generated.")

            # You could save results['case_summaries'] to a file here if needed
            # import json
            # with open("ip_analysis_results.json", "w") as f:
            #     json.dump(results, f, indent=2)
            # logging.info("Saved detailed results to ip_analysis_results.json")


    except Exception as e:
        logging.exception("Critical error in main execution flow.")

if __name__ == "__main__":
    main()