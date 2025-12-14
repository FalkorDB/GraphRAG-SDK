"""
RDF Ontology Extractor for extracting ontologies from RDF/OWL schema files.
"""

import logging
from typing import List, Optional, Dict, Set, Tuple
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD

from graphrag_sdk.entity import Entity
from graphrag_sdk.relation import Relation
from graphrag_sdk.attribute import Attribute, AttributeType
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.helpers import extract_name_from_uri


logger = logging.getLogger(__name__)


class RDFOntologyExtractor:
    """
    Extracts ontology (entities, relations, attributes) from RDF/OWL schema graphs.
    
    This extractor processes RDF graphs containing RDFS or OWL class and property
    definitions to generate a GraphRAG-compatible ontology. It enforces that only
    schema definitions are present (no individual instances).
    
    Key Features:
    - Extracts classes as entities with labels and comments
    - Extracts object properties as relations with domain/range
    - Extracts datatype properties as entity attributes
    - Validates that no individuals are present in the schema
    - Supports RDFS and OWL vocabularies
    """
    
    def __init__(self, graph: Graph):
        """
        Initialize the RDF Ontology Extractor.
        
        Args:
            graph: An rdflib.Graph containing the RDF schema
            
        Raises:
            ValueError: If the graph contains individual instances
        """
        self.graph = graph
        self.entities: Dict[str, Entity] = {}
        self._check_for_individuals()
    
    def extract(self) -> Ontology:
        """
        Extract complete ontology from the RDF graph.
        
        Returns:
            Ontology object containing all entities and relations
        """
        entities = self.extract_entities()
        relations = self.extract_relations()
        
        return Ontology(
            entities=entities,
            relations=relations
        )

    def extract_entities(self) -> List[Entity]:
        """
        Extract all classes as entities.
        
        Returns:
            List of Entity objects
        """
        entity_list = []
        seen_classes = set()
        
        # Find all classes (RDFS and OWL)
        classes = set()
        for s in self.graph.subjects(RDF.type, RDFS.Class):
            classes.add(s)
        for s in self.graph.subjects(RDF.type, OWL.Class):
            classes.add(s)
            
        for class_uri in classes:
            # Skip blank nodes
            if not isinstance(class_uri, URIRef):
                continue
                
            if class_uri in seen_classes:
                continue
            seen_classes.add(class_uri)

            # Extract label and comment
            label = self._extract_label(class_uri)
            comment = self._extract_comment(class_uri)
            local_name = extract_name_from_uri(str(class_uri))
            
            # Use label if available, otherwise local name
            entity_name = label if label else local_name
            
            # Extract attributes (properties with this class as domain)
            attributes = self._infer_attributes_for_entity(class_uri)
            
            entity = Entity(
                label=entity_name,
                attributes=attributes,
                description=comment or ""
            )

            self.entities[str(class_uri)] = entity
            entity_list.append(entity)
            
        logger.info(f"Extracted {len(entity_list)} entities from RDF schema")
        return entity_list

    def extract_relations(self) -> List[Relation]:
        """
        Extract all object properties as relations.
        
        Returns:
            List of Relation objects
        """
        relations = []
        
        # Find all properties that look like relations
        properties = set()
        for s in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            properties.add(s)
        # Also check generic properties that link two classes
        for s in self.graph.subjects(RDF.type, RDF.Property):
            properties.add(s)
            
        for prop_uri in properties:
            if not isinstance(prop_uri, URIRef):
                continue

            domain = self._get_domain(prop_uri)
            range_val = self._get_range(prop_uri)
            
            # Skip if domain or range is missing
            if not domain or not range_val:
                # Only warn if it's explicitly an ObjectProperty
                if (prop_uri, RDF.type, OWL.ObjectProperty) in self.graph:
                    label = self._extract_label(prop_uri) or extract_name_from_uri(str(prop_uri))
                    logger.warning(
                        f"Property '{label}' is missing domain or range. "
                        f"This relation will be skipped."
                    )
                continue
                
            # Check if domain/range are properties (property chains)
            if self._is_property(domain) or self._is_property(range_val):
                logger.warning(f"Property {extract_name_from_uri(str(prop_uri))} has property as domain/range - skipping (property chains not supported)")
                continue
                
            # Check if domain and range are known classes (or at least not datatypes)
            if self._is_datatype(range_val):
                continue

            # We need to map URIs to Entity names
            # If we have processed entities, we can use their names
            # Otherwise we extract names from URIs
            
            domain_str = str(domain)
            range_str = str(range_val)
            
            source_name = self.entities[domain_str].label if domain_str in self.entities else (self._extract_label(domain) or extract_name_from_uri(domain_str))
            target_name = self.entities[range_str].label if range_str in self.entities else (self._extract_label(range_val) or extract_name_from_uri(range_str))
            
            label = self._extract_label(prop_uri)
            local_name = extract_name_from_uri(str(prop_uri))
            relation_name = label if label else local_name
            
            relation = Relation(
                label=relation_name,
                source=source_name,
                target=target_name
            )
            relations.append(relation)
                
        logger.info(f"Extracted {len(relations)} relations from RDF schema")
        return relations

    def _infer_attributes_for_entity(self, entity_uri: URIRef) -> List[Attribute]:
        """
        Find properties where rdfs:domain is this entity.
        """
        attributes = []
        
        # Find properties with this domain
        for prop_uri in self.graph.subjects(RDFS.domain, entity_uri):
            # Skip if not a URIRef
            if not isinstance(prop_uri, URIRef):
                continue
                
            # Check if it's a datatype property (attribute) or object property (relation)
            # If range is a literal/datatype, it's an attribute
            range_val = self._get_range(prop_uri)
            
            if not range_val:
                # Default to string attribute if no range
                attr_type = AttributeType.STRING
            elif self._is_datatype(range_val):
                attr_type = self._map_xsd_type(range_val)
            else:
                # It's likely a relation (range is a class), skip here
                continue
                
            label = self._extract_label(prop_uri)
            local_name = extract_name_from_uri(str(prop_uri))
            attr_name = label if label else local_name
            
            attributes.append(Attribute(name=attr_name, attr_type=attr_type, required=False))
            
        return attributes

    def _check_for_individuals(self) -> None:
        """
        Check if the graph contains any individual instances.
        
        Raises:
            ValueError: If individuals are found in the graph
        """
        # Look for instances (subjects that are instances of a class)
        individuals = set()
        
        # Get all classes defined in the schema
        classes = set()
        for s in self.graph.subjects(RDF.type, RDFS.Class):
            classes.add(s)
        for s in self.graph.subjects(RDF.type, OWL.Class):
            classes.add(s)
        
        # Now check if there are any subjects that are instances of these classes
        for cls in classes:
            for individual in self.graph.subjects(RDF.type, cls):
                # Make sure this is not a class definition itself
                if not (individual, RDF.type, RDFS.Class) in self.graph and \
                   not (individual, RDF.type, OWL.Class) in self.graph:
                    individuals.add(individual)
        
        if individuals:
            individual_list = [extract_name_from_uri(str(ind)) for ind in list(individuals)[:5]]
            error_msg = (
                f"The RDF file contains {len(individuals)} individual instance(s), "
                f"but only schema definitions (classes and properties) are allowed. "
                f"Examples: {', '.join(individual_list)}"
                f"{'...' if len(individuals) > 5 else ''}. "
                f"Please provide an RDF schema file (RDFS/OWL) without individual instances."
            )
            raise ValueError(error_msg)

    def _get_domain(self, prop_uri: URIRef) -> Optional[URIRef]:
        for obj in self.graph.objects(prop_uri, RDFS.domain):
            if isinstance(obj, URIRef):
                return obj
        return None

    def _get_range(self, prop_uri: URIRef) -> Optional[URIRef]:
        for obj in self.graph.objects(prop_uri, RDFS.range):
            if isinstance(obj, URIRef):
                return obj
        return None

    def _is_property(self, uri: URIRef) -> bool:
        """Check if URI is a property definition"""
        return (uri, RDF.type, RDF.Property) in self.graph or \
               (uri, RDF.type, OWL.ObjectProperty) in self.graph or \
               (uri, RDF.type, OWL.DatatypeProperty) in self.graph

    def _is_datatype(self, uri: URIRef) -> bool:
        """Check if URI is a datatype (XSD)"""
        return str(uri).startswith(str(XSD))

    def _map_xsd_type(self, xsd_uri: URIRef) -> AttributeType:
        """Map XSD types to AttributeType"""
        uri_str = str(xsd_uri)
        
        if uri_str in [str(XSD.integer), str(XSD.int), str(XSD.long), 
                      str(XSD.float), str(XSD.decimal), str(XSD.double),
                      str(XSD.nonNegativeInteger), str(XSD.positiveInteger),
                      str(XSD.unsignedLong), str(XSD.unsignedInt),
                      str(XSD.unsignedShort), str(XSD.unsignedByte),
                      str(XSD.nonPositiveInteger), str(XSD.negativeInteger),
                      str(XSD.short), str(XSD.byte)]:
            return AttributeType.NUMBER
        elif uri_str == str(XSD.boolean):
            return AttributeType.BOOLEAN
        else:
            return AttributeType.STRING

    def _extract_label(self, uri: URIRef) -> Optional[str]:
        """Extract rdfs:label"""
        for label in self.graph.objects(uri, RDFS.label):
            return str(label)
        return None

    def _extract_comment(self, uri: URIRef) -> Optional[str]:
        """Extract rdfs:comment"""
        for comment in self.graph.objects(uri, RDFS.comment):
            return str(comment)
        return None

