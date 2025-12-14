from typing import Iterator
from graphrag_sdk.document import Document
from graphrag_sdk.helpers import extract_name_from_uri


class TTLLoader:
    """
    TTL (Terse RDF Triple/Turtle Language) loader
    
    Parses RDF triples from TTL files and groups them by subject.
    The entire file is processed as a single document.
    """

    def __init__(self, path: str):
        """
        Initialize loader

        Args:
            path (str): Path to the TTL file
        """
        try:
            import rdflib
        except ImportError:
            raise ImportError(
                "rdflib package not found, please install it with " "`pip install rdflib`"
            )
        
        self.path = path

    def load(self) -> Iterator[Document]:
        """
        Load TTL file

        Returns:
            Iterator[Document]: document iterator
        """
        # Parse the TTL file using shared method
        graph = self.get_rdf_graph()
        
        # Group triples by subject
        subjects_dict = {}
        for subject, predicate, obj in graph:
            # Strip namespaces - get local name only
            subj_str = extract_name_from_uri(str(subject))
            pred_str = extract_name_from_uri(str(predicate))
            obj_str = extract_name_from_uri(str(obj))
            
            if subj_str not in subjects_dict:
                subjects_dict[subj_str] = []
            
            # Store as readable triple text
            subjects_dict[subj_str].append(f"{subj_str} {pred_str} {obj_str}")

        # Convert to list of (subject, triples) tuples
        subjects_list = list(subjects_dict.items())
        
        if len(subjects_list) == 0:
            # Empty file - yield empty document
            yield Document("", self.path)
            return

        # Build document content: group all triples by subject
        content_lines = []
        for subject, triples in subjects_list:
            content_lines.append(f"Subject: {subject}")
            for triple in triples:
                content_lines.append(f"  {triple}")
            content_lines.append("")  # Empty line between subjects
        
        content = "\n".join(content_lines)
        yield Document(content, self.path)

    def get_rdf_graph(self):
        """
        Get the parsed RDF graph for ontology extraction
        
        Returns:
            rdflib.Graph: The parsed RDF graph
            
        Raises:
            Exception: If the file cannot be parsed
        """
        from rdflib import Graph as RDFGraph
        
        try:
            graph = RDFGraph()
            graph.parse(self.path, format="turtle")
            return graph
        except Exception as e:
            raise Exception(f"Failed to parse TTL file: {e}, please ensure the file is valid TTL format.")

