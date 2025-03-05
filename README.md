# 🧠 LLMGraphTransformer

LLMGraphTransformer is a Python library designed to extract structured knowledge graphs from unstructured text using LLMs. It allows users to define schemas for nodes and relationships, ensuring that the extracted graph follows a strict format. 🔗📊

## 🚀 Installation

Install LLMGraphTransformer from PyPI:
```bash
pip install LLMGraphTransformer
```

## 🛠️ Usage

### 📥 Importing the Required Modules
```python
from LLMGraphTransformer import LLMGraphTransformer
from LLMGraphTransformer.schema import NodeSchema, RelationshipSchema
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


from dotenv import load_dotenv
import os
load_dotenv(".env")  
```

### 🏗️ Defining the Schema

#### 🏷️ Node Schemas
Node schemas define the types of entities that can be extracted from the text. Each node has:
- A **type** (e.g., "Person", "Organization")
- A list of **properties** that store additional information (e.g., "name", "birth_year")
- An optional **description** to describe the node type

📌 Example:
```python
node_schemas = [
    NodeSchema("Person", ["name", "birth_year", "death_year", "nationalitie", "profession"], "Represents an individual"),
    NodeSchema("Organization", ["name", "founding_year", "industrie"], "Represents a group, company, or institution"),
    NodeSchema("Location", ["name"], "Represents a geographical area such as a city, country, or region"),
    NodeSchema("Award", ["name", "field"], "Represents an honor, prize, or recognition")
]
```

#### 🔗 Relationship Schemas
Relationship schemas define the allowed connections between entities. Each relationship has:
- A **source** node type
- A **target** node type
- A **relationship type**
- A list of optional **properties** (e.g., "year")

📌 Example:
```python
relationship_schemas = [
    RelationshipSchema("Person", "SPOUSE_OF", "Person"),
    RelationshipSchema("Person", "MEMBER_OF", "Organization", ["start_year", "end_year", "year"]),
    RelationshipSchema("Person", "AWARDED", "Award", ["year"]),
    RelationshipSchema("Person", "LOCATED_IN", "Location"),
    RelationshipSchema("Organization", "LOCATED_IN", "Location")
]
```

### ⚙️ Defining Additional Instructions
You can specify additional rules for extraction:
```python
additional_instructions="""- all names must be extracted as uppercase"""
```

### 📜 Defining the Input Text
Provide the text from which the knowledge graph should be extracted:
```python
text="""Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris."""
```

### 🤖 Initializing the LLM Model
Use OpenAI's API (or a compatible model) to process the text:
```python
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")

llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model=model_name,
    temperature=0,
)
```

### 🔄 Initializing the Transformer
Create an instance of `LLMGraphTransformer`:
```python
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=node_schemas,
    allowed_relationships=relationship_schemas,
    additional_instructions=additional_instructions
)
```

### 🔍 Converting Text to a Knowledge Graph
Process the text into a structured knowledge graph:
```python
document = Document(page_content=text)
graph_document = llm_transformer.convert_to_graph_document(document)

print(f"Nodes: {graph_document.nodes}")
print(f"Relationships: {graph_document.relationships}")
```

## 📊 Output Format
The extracted knowledge graph will be represented in JSON format with `nodes` and `relationships`:
```json
{
  "nodes": [
    {
      "id": "Marie Curie",
      "type": "Person",
      "properties": {
        "name": "Marie Curie",
        "birth_year": "1867",
        "nationalitie": ["Polish", "French"],
        "profession": ["physicist", "chemist"]
      }
    },
    ...
  ],
  "relationships": [
    {
      "source": "Marie Curie",
      "target": "Pierre Curie",
      "type": "SPOUSE_OF"
    },
    ...
  ]
}
```

## 📜 License
This project is licensed under the MIT License.

## 🤝 Contributing
Pull requests and feature suggestions are welcome! Open an issue for bug reports or improvements. 🚀

