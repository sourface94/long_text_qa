node_and_rel_extraction_prompt = """
You are a top-tier algorithm designed for extracting information from fiction literature in a structured format to build a knowledge graph.

An entity represents a person, a location or an event that took place in the text. Examples of an event are th arrival of a person in a location, the death of a character, a character graudating and more.
An edge connects two entities to each other through some type of relationship.

Now extract the main entities and their relationships that appear in the text extract below. Only include entities that are in the text below:
{chunk}


"""

event_extraction_prompt = """
You are a top-tier algorithm designed for extracting information from fiction texts in a structured format to build a knowledge graph.

You extract the names of events that take place in fiction texts. An event is an important part of a story that impacts what happens to the characters. 
The event name  must be a maximum of 5 words and summarise the event concisely. The event description must be a maximum of 20 words and summarise the event.

Examples of event names are: 'joanne_died', 'harry got angry', 'the ship sank' and 'the war began'.

Extract event names along with a 1 sentence description of the event from the text extract below. Only include entities that are in the text below. Make sure to have short concise descriptions of the events:
{chunk}


"""
