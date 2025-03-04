from typing import List, Union, Optional
from pydantic import Field
from langchain_core.documents import Document
from langchain_core.load.serializable import Serializable

class NodeSchema:
    def __init__(self, type: str, properties: List[str]= ["name"], description: str= ""):
        self.type = type
        self.description = description 
        self.properties = properties 

class RelationshipSchema:
    def __init__(self, source: str, type: str, target: str, properties: List[str]= []):
        self.source = source
        self.target = target
        self.type = type
        self.properties = properties 

class Node(Serializable):
    id: Union[str, int]
    type: str = "Node"
    properties: dict = Field(default_factory=dict)

class Relationship(Serializable):
    source: Node
    target: Node
    type: str
    properties: dict = Field(default_factory=dict)

class GraphDocument(Serializable):
    nodes: List[Node]
    relationships: List[Relationship]
    source: Optional[Document] = None

