from enum import Enum
from typing import Annotated, List

from pydantic import BaseModel
from annotated_types import Len


class EntityType(str, Enum):
    person = 'person'
    location = 'location'
    event = 'event'
    other = 'other'


class RelationshipType(str, Enum):
    friend = 'friend'
    enemy = 'enemy'
    family = 'family'
    is_in = 'is_in'
    other = 'other'


class Entity(BaseModel):
    entity_name: str
    entity_description: str
    entity_type: EntityType

    def __eq__(self, other):
        if self.entity_name == other.entity_name:
            return True
        return False

    def __hash__(self):
        return hash(self.entity_name)


class Relationship(BaseModel):
    entity_subject_name: str
    entity_predicate_name: RelationshipType
    entity_object_name: str
    timestep: int = -1

    def contains_entity(self, e: Entity):
        if e.entity_name == self.entity_object_name or e.entity_name == self.entity_subject_name:
            return True
        return False

    def __eq__(self, other):
        if self.entity_subject_name == other.entity_subject_name and self.entity_predicate_name == other.entity_predicate_name and self.entity_object_name == other.entity_object_name:
            return True
        return False

    def __hash__(self):
        return hash(self.entity_subject_name+self.entity_predicate_name+self.entity_object_name)

    def __str__(self):
        return f"At timestep {self.timestep}, {self.entity_subject_name} has a relation type called '{self.entity_predicate_name.replace('_', ' ')}' to {self.entity_object_name }"
    
    def __repr__(self):
        return f"At timestep {self.timestep}, {self.entity_subject_name} has a relation type called '{self.entity_predicate_name.replace('_', ' ')}' to {self.entity_object_name }"


class KGList(BaseModel):
    entities: Annotated[list[Entity], Len(max_length=50)]
    relationships: Annotated[list[Relationship], Len(max_length=20)]


class NodeListStr(BaseModel):
    node_name: List[str]
