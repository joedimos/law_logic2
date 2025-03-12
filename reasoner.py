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
            grounded_extension = calculate_grounded_extension(graph, arguments) 

            # Gather arguments from the grounded extension
            justified_arguments = {arg_id: arguments[arg_id] for arg_id in grounded_extension}

            overall_consistency = all("force_majeure" not in arg_id for arg_id in arguments.keys()) 
            return {
                "force_majeure_analysis": force_majeure_analysis,
                "grounded_extension": grounded_extension,
                "justified_arguments": justified_arguments,
                "overall_consistency": overall_consistency,
            }

        except Exception as e:
            logging.error(f"Error analyzing case: {e}")
            return {}
