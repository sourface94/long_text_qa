import json
from typing import List, Optional

import guidance
from guidance import models

from prompts import node_and_rel_extraction_prompt
from models import KGList, Entity, Relationship


def extract_kg(model: models.Model, text: str, kg: Optional[KGList] = None):
    """Extracts nodes and edges from text using a guidance model"""
    lm = model + node_and_rel_extraction_prompt.format(chunk=text) + guidance.json(name="kg_list", schema=KGList)
    return KGList(**json.loads(lm['kg_list']))


def clean_extracted_kg(kg: KGList) -> KGList:
    """Cleans KG output from LLM"""
    entites_clean = clean_entities(KGList['entities'])
    relationships_clean = clean_relationships(KGList['relationships'], entites_clean)
    return KGList(entities=entites_clean, relatonships=relationships_clean)


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
    raise NotImplementedError


def get_subkg(kg: KGList, entities: str) -> KGList:
    """Get sub graph that contains given entities"""
    raise NotImplementedError
