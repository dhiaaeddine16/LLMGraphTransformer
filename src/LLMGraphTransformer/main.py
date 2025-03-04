from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json_repair
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from LLMGraphTransformer.prompt_generation import generate_prompt
from LLMGraphTransformer.schema import NodeSchema, RelationshipSchema, Node, Relationship, GraphDocument

DEFAULT_NODE_TYPE = "Node"

def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id.title() if isinstance(el.id, str) else el.id,
            type=el.type.capitalize()
            if el.type
            else DEFAULT_NODE_TYPE,
            properties=el.properties,
        )
        for el in nodes
    ]

def _format_relationships(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
            properties=el.properties,
        )
        for el in rels
    ]

def _format_graph(nodes: List[Node], relationships: List[Relationship]):
    return _format_nodes(nodes),_format_relationships(relationships)

def process_properties(props: Dict[str, List[str]]) -> Dict[str, Union[str, List[str]]]:
    """Process a properties dictionary:
       - Remove keys where the value is an empty list.
       - If the value is a list of length 1, replace it with its single element.
    """
    keys_to_delete = []
    for key, value in list(props.items()):
        if isinstance(value, list):
            if len(value) == 0:
                keys_to_delete.append(key)
            elif len(value) == 1:
                props[key] = value[0]
    for key in keys_to_delete:
        del props[key]
    return props

def _parse_and_clean_json(
    parsed_json: Dict[str, Any],
) -> Tuple[List[Node], List[Relationship]]:
    node_map = {}
    nodes = []
    for node_data in parsed_json["nodes"]:
        if "properties" in node_data:
            node_data["properties"] = process_properties(node_data["properties"])
        node = Node(**node_data)
        nodes.append(node)
        node_map[node.id] = node

    relationships = []
    for rel_data in parsed_json["relationships"]:
        if "properties" in rel_data:
            rel_data["properties"] = process_properties(rel_data["properties"])
        # Extract the source and target IDs
        source_id = rel_data.pop("source")
        target_id = rel_data.pop("target")
        # Lookup the Node objects (assumes nodes have been processed already)
        source_node = node_map.get(source_id)
        target_node = node_map.get(target_id)
        # Create the Relationship object with the resolved nodes
        relationship = Relationship(source=source_node, target=target_node, **rel_data)
        relationships.append(relationship)

    return nodes,relationships
    
class LLMGraphTransformer:
    """Transform documents into graph-based documents using a LLM.

    It allows specifying constraints on the types of nodes and relationships to include
    in the output graph. The class supports extracting properties for both nodes and
    relationships.

    Args:
        llm (BaseLanguageModel): An instance of a language model supporting structured
          output.
        allowed_nodes (List[str], optional): Specifies which node types are
          allowed in the graph. Defaults to an empty list, allowing all node types.
        allowed_relationships (List[str], optional): Specifies which relationship types
          are allowed in the graph. Defaults to an empty list, allowing all relationship
          types.
        prompt (Optional[ChatPromptTemplate], optional): The prompt to pass to
          the LLM with additional instructions.
        strict_mode (bool, optional): Determines whether the transformer should apply
          filtering to strictly adhere to `allowed_nodes` and `allowed_relationships`.
          Defaults to True.
        node_properties (Union[bool, List[str]]): If True, the LLM can extract any
          node properties from text. Alternatively, a list of valid properties can
          be provided for the LLM to extract, restricting extraction to those specified.
        relationship_properties (Union[bool, List[str]]): If True, the LLM can extract
          any relationship properties from text. Alternatively, a list of valid
          properties can be provided for the LLM to extract, restricting extraction to
          those specified.
        additional_instructions (str): Allows you to add additional instructions
          to the prompt without having to change the whole prompt.

    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[NodeSchema] = [],
        allowed_relationships: List[RelationshipSchema] = [],
        prompt: Optional[ChatPromptTemplate] = None,
        strict_mode: bool = True,
        additional_instructions: str = "",
    ) -> None:
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self.prompt = prompt or generate_prompt(
            allowed_nodes,
            allowed_relationships,
            additional_instructions
        )
        self.chain = self.prompt | llm

    def graph_strict_mode_filtering(self,nodes,relationships):
        # Strict mode filtering
        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.type.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if (
                        (
                            rel.source.type.lower(),
                            rel.type.lower(),
                            rel.target.type.lower(),
                        )
                        in [
                            (t.source.lower(), t.type.lower(), t.target.lower())
                            for t in self.allowed_relationships
                        ]
                    )
                ]
        return nodes,relationships

    def convert_to_graph_document(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints.
        """
        text = document.page_content
        raw_schema = self.chain.invoke({"input_text": text}, config=config)
        #print(raw_schema.content)
        if not isinstance(raw_schema, str):
            raw_schema = raw_schema.content
        parsed_json = json_repair.loads(raw_schema)
        
        parsedNodes, parsedRelationships = _parse_and_clean_json(parsed_json)
        nodes, relationships = _format_graph(parsedNodes, parsedRelationships)
        nodes, relationships = self.graph_strict_mode_filtering(nodes, relationships)
    
        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    def convert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        #TODo: add correfence resolution
        return [self.convert_to_graph_document(document, config) for document in documents]
    
   


