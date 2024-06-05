ONTOLOGY_DETECTION_SYSTEM_PROMPT = """Create SQL statement which will capture the different entities and the relations between them in the provided text.
Each table should represent either a single entity type with the entity's attributes as the table columns or a relation between two entities, using foreign keys.
For example an IMDB document is likely to produce 5 tables: Movie, Actor and Director each with its own columns and ACTED and DIRECTED tables.
Do not create an ID primary key column but set the most compelling attribute as the entity's primary key
Make sure to call the run_sql function provided in your tools set for each sql statement generated."""

KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = """You are a helpful assistant with the goal of extracting
entities and relationships from text"""

SQL_FOLDING_PROMPT = """The following two CREATE TABLE SQL statments are
creating the same table name, consolidate the two into a single
CREATE TABLE SQL statment.
Remove all duplicated columns.
Make sure the number of PRIMARY KEYs and FOREIGN KEYs is maintained.
Reply only with the result SQL statment, the reply should start with:
CREATE TABLE and end with );
For example: given the two CREATE TABLE SQL:
CREATE TABLE Person (FirstName string, LastName string, Age int);
CREATE TABLE Person (SurName string, ForeName string, Height int);
The response should be:
CREATE TABLE Person (FirstName string, LastName string, Age int, Height int);
"""

SQL_FOCUS_PROMPT = """Given a CREATE TABLE SQL statment the table name
 represents some entity, in case the number of columns in the table is
 greater than 5, remove columns which are less descriptive of the entity.
 Do NOT remove columns which are either a PRIMARY KEY or a FOREIGN KEY.
 Make sure to leave at least 5 columns.
 Do NOT rename the table name.
 Reply only with the result SQL statment, the reply should start with:
 CREATE TABLE and end with );
 For example: given the CREATE TABLE SQL:
 CREATE TABLE Movie (
     Title string PRIMARY KEY,
     Genre string,
     ReleaseYear int,
     Rating string,
     Language string,
     BoxOffice string,
     streaming_date STRING,
     runtime STRING
 );
 The response should be:
 CREATE TABLE Movie (
     Title string PRIMARY KEY,
     Genre string,
     ReleaseYear int,
     Rating string,
     runtime STRING
 );
 Another example: given the CREATE TABLE SQL:
     CREATE TABLE Episode (
         show_title VARCHAR(255),
         show_year VARCHAR(4),
         season INT,
         episode_number INT,
         episode_title VARCHAR(255),
         air_date DATE,
         episode_rating DECIMAL(2, 1),
         FOREIGN KEY (show_title, show_year) REFERENCES TVShow(title, year)
     );
 You are not allowed to remove either show_title or show_year as these are part of the table's
 foreign key.
 """

RELATIONSHIP_NAMING_PROMPT = """"You are a Cypher expert, help suggest names for
relationship types connecting source node to destination node.
Respond only with the suggested name.
For example given source node of type 'Actor' and destination node of type 'Movie'
Replay with: ACTED"""

FIND_UNIQUE_ATTRIBUTE_PROMPT = """From the list of attributes describing a graph node
pick the most suitble attribute to act as a unique identifier of the node.
Respond only with the suggested attribute name
For example given a list of attributes: series_title, air_date, season_number, episode_number, title
Describing the graph entity "Episode" Reply with: title"""
