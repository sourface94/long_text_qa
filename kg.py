import json
from typing import List, Optional

import guidance
import numpy as np
import spacy
from guidance import models
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from prompts import node_and_rel_extraction_prompt
from models import KGList, Entity, Relationship

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_kg(model: models.Model, text: str, kg: Optional[KGList] = None):
    """Extracts nodes and edges from text using a guidance model"""
    lm = model + node_and_rel_extraction_prompt.format(chunk=text) + guidance.json(name="kg_list", schema=KGList)
    return KGList(**json.loads(lm['kg_list']))


def clean_extracted_kg(kg: KGList, timestamp: int) -> KGList:
    """Cleans KG output from LLM"""
    entites_clean = clean_entities(kg.entities)
    relationships_clean = clean_relationships(kg.relationships, entites_clean)
    for r in relationships_clean:
        r.timestep = timestamp
    return KGList(entities=entites_clean, relationships=relationships_clean)


def clean_entities(entities: List[Entity]) -> List[Entity]:
    """Removes duplicate entities from a list of Entity"""
    entities = set(entities)
    return entities


def clean_relationships(relationships: List[Relationship], entities: List[Entity]) -> List[Relationship]:
    """Removes invalid relationships from a list of Relationship"""
    entity_names = [e.entity_name for e in entities]
    # remove duplicate relationships
    relationships = list(set(relationships))
    # remove relationships where an entity in the relationship isnt an existing Entity
    new_relationships = []
    for r in relationships:
        if r.entity_subject_name not in entity_names or r.entity_object_name not in entity_names:
            continue
        new_relationships.append(r)
    return new_relationships


def get_entities(text: str) -> List[str]:
    """Get entities from text"""
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents]
    return ents


def get_subkg(kg: KGList, entities: List[str], threshold: float = 0.8) -> KGList:
    """Get sub graph that contains given entities"""
    # get embeddings
    entity_embeddings = model.encode([e for e in entities])
    node_embeddings = model.encode([e.entity_name + ' ' + e.entity_description for e in kg.entities])
    
    # get kg entities that appear in enetities by using simialrity threshold as a proxy
    sim = cosine_similarity(node_embeddings, entity_embeddings)
    indices = np.argwhere(np.max(sim, axis=1) >= threshold).flatten()
    kg_entities = [kg.entities[i] for i in indices]
    kg_relationships = []
    for r in kg.relationships:
        for e in kg_entities:
            if r.contains_entity(e):
                kg_relationships.append(r)
                break
    return KGList(entities=kg_entities, relationships=kg_relationships)


def kg_to_nl(kg: KGList) -> str:
    rep = ''
    for r in kg.relationships:
        rep += str(r) + '. '
    return rep


def merge_kg(main_kg: KGList, sub_kg: KGList) -> KGList:
    """"Merges sub_kg in to main_kg"""
    main_kg_entity_embeddings = model.encode([e.entity_name + ' ' + e.entity_description for e in main_kg.entities])
    sub_kg_entity_embeddings = model.encode([e.entity_name + ' ' + e.entity_description for e in sub_kg.entities])

    # check if there are matching entities
    # for the matching entities then replace the entitiy names in the sub graph with the name in the main graph
    # for each relationship check if a duplicate (minus timestamp) exists in the subgraph. if it doesnt add it. if it does chck if the most recent relationship between the two nodes is the same, if it is dont add it, if it isnt add it
    # for entities that dont match, add the entity. for the entities relationsships check if identical relationship alrerady exists, if it does dont add it, otherwise add it
    
    raise NotImplementedError

