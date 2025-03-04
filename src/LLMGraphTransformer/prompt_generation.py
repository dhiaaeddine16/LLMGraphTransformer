from typing import List
from langchain_core.prompts import PromptTemplate
from LLMGraphTransformer.schema import NodeSchema, RelationshipSchema


def format_node_schemas(schemas: List[NodeSchema]) -> str:
    lines = []
    for node in schemas:
        if node.description:
            lines.append(f"- {node.type}: {node.description}")
        else:
            lines.append(f"- {node.type}")
    return "\n".join(lines)

def format_node_properties_schemas(schemas: List[NodeSchema]) -> str:
    lines = []
    for node in schemas:
        if node.properties:
            props = '", "'.join(node.properties)
            lines.append(f"- {node.type}: \"{props}\"")
    return "\n".join(lines)

def format_relationship_schemas(schemas: List[RelationshipSchema]) -> str:
    lines = []
    for rel in schemas:
        lines.append(f"- {rel.source}, {rel.type}, {rel.target}")
    return "\n".join(lines)

def format_relationship_properties_schemas(schemas: List[RelationshipSchema]) -> str:
    lines = []
    for rel in schemas:
        if rel.properties:
            props = '", "'.join(rel.properties)
            lines.append(f"- {rel.type}: \"{props}\"")
    return "\n".join(lines)

def generate_prompt(node_schemas: List[NodeSchema], relationship_schemas: List[RelationshipSchema], additional_instructions: str):
    node_definitions = format_node_schemas(node_schemas)
    node_properties_definitions = format_node_properties_schemas(node_schemas)
    relationship_definitions = format_relationship_schemas(relationship_schemas)
    relationship_properties_definitions = format_relationship_properties_schemas(relationship_schemas)

    template_str = """
You are a top-tier algorithm designed for extracting information in JSON structured formats to build a knowledge graph.

### Output Format
Generate a knowledge graph in JSON format using the following structure:

**Nodes:**
- **id:** A unique identifier (string). Use a human-readable identifier as found in the text (e.g., "first bank of america").
- **type:** The type or label of the node (string), must be one of the allowed node types (e.g., "Person").
- **properties:** A dictionary of additional metadata where keys are strings and values are lists of strings. Omit this field if no properties exist.

**Relationships:**
- **source:** The id of the source node.
- **target:** The id of the target node.
- **type:** The type of the relationship, must be one of the allowed relationship types (e.g., SPOUSE_OF).
- **properties:** A dictionary of additional metadata where keys are strings and values are lists of strings. Omit this field if no properties exist.

### Schema Definition (STRICT)
The schema below must be followed exactly. Only use the allowed types, properties, and relationship types listed, and do not introduce any additional elements.

**Allowed Node Types:**
{node_definitions}

**Allowed Relationship:**
(like: Node Source Type, Relationship type, Node Target Type):
{relationship_definitions}

**Allowed Node Properties:**
{node_properties_definitions}

**Allowed Relationship Properties:**
{relationship_properties_definitions}

### Rules to Follow:
- Do not add any information that is not explicitly mentioned in the text.
- Relationship source and target must reference valid node ids.
- Omit any empty fields from the output.
{additional_instructions}

### Task:
Generate a knowledge graph in JSON format from the following text using the provided schema and rules:

"{input_text}"

Output only a valid JSON object with keys "nodes" and "relationships" and no additional commentary.
"""

    return PromptTemplate(
        input_variables=["input_text"],
        partial_variables={
            "node_definitions": node_definitions,
            "relationship_definitions": relationship_definitions,
            "node_properties_definitions":node_properties_definitions,
            "relationship_properties_definitions":relationship_properties_definitions,
            "additional_instructions":additional_instructions
        },
        template=template_str
    )

if __name__ == "__main__":
    # Define the allowed node schemas
    node_schemas = [
        NodeSchema("Person", ["name", "birth_year", "death_year", "nationalitie", "profession"]),
        NodeSchema("Organization", ["name", "founding_year", "industrie"], "Represents a group, company, or institution"),
        NodeSchema("Location", ["name"], "Represents a geographical area such as a city, country, or region"),
        NodeSchema("Award", ["name","field"])
    ]
    
    # Define the allowed relationship schemas
    relationship_schemas = [
        RelationshipSchema("Person", "SPOUSE_OF", "Person"),
        RelationshipSchema("Person", "MEMBER_OF", "Organization", ["start_year", "end_year", "year"]),
        RelationshipSchema("Person", "AWARDED", "Award", ["year"]),
        RelationshipSchema("Person", "LOCATED_IN", "Location"),
        RelationshipSchema("Organization", "LOCATED_IN", "Location")
    ]
    text="""Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris."""
    # Format the prompt with the Marie Curie text as input
    prompt = generate_prompt(node_schemas, relationship_schemas)
    formatted_prompt = prompt.format(input_text=text)
    print(formatted_prompt)
