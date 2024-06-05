class Attribute(object):
    def __init__(self, name:str, type:type, desc=None, unique=False, mandatory=False):
        self.name = name
        self.type = type
        self.desc = desc
        self.unique = unique
        self.mandatory = mandatory

    def __eq__(self, other) -> bool:
        if not isinstance(other, Attribute):
            return False

        return (self.name == other.name and
                self.type == other.type and
                self.desc == other.desc and
                self.unique == other.unique and
                self.mandatory == other.mandatory)

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self) -> int:
        return hash(self.name)

class Entity(object):
    """
    Represents an ontolegy entity
    """

    def __init__(self, name:str):
        """
        Initialize Entity

        Parameters:
            name (str): name of entity
        """

        if not isinstance(name, str) or name == "":
            raise Exception("Entity name must be a none empty string")

        self.name = name
        self.attributes = set([])

    def __eq__(self, other) -> bool:
        if not isinstance(other, Entity):
            return False

        return (self.name == other.name and self.attributes == other.attributes)

    def add_attribute(self, name:str, type:type, desc:str|None=None, unique:bool=False, mandatory:bool=False) -> None:
        """
        Add attribute to entity

        Parameters:
            name (str): attribute name
            type (type): attribute type
            desc (str): short description for attribute
            unique (bool): is the attribute value unique
            mandatory (bool): is the attribute mandatory
        """

        # Validate arguments
        if not isinstance(name, str) or name == "":
            raise Exception("Attribute name must be a none empty string")
        #if not isinstance(type, type) or type not in [int, str, bool]:
        if type not in [int, float, str, bool]:
            raise Exception("Attribute type must be one of the supported types")
        if not isinstance(unique, bool):
            raise Exception("primary_key must be of type bool")
        if name in self.attributes:
            raise Exception(f"Entity already contains attribute: {name}")
        if isinstance(desc, str) and desc == "":
            raise Exception(f"Invalid argument, desc should be a non empty string")

        desc = f"{self.name}'s {name}" if desc is None else desc
        self.attributes.add(Attribute(name, type, desc, unique, mandatory))

    def remove_attribute(self, name:str) ->None:
        """
        Remove attribute from entity

        Parameters:
            name (str): name of attribute to remove
        """

        # Validate argument
        if not isinstance(name, str) or name == "":
            raise Exception("Attribute name must be a none empty string")

        for attr in self.attributes:
            if attr.name == name:
                self.attributes.remove(attr)
                return

    def get_attribute(self, name:str) -> Attribute|None:
        """
        Get attribute

        Parameters:
            name (str): name of attribute to get

        Returns:
            attribute
        """

        # Validate argument
        if not isinstance(name, str) or name == "":
            raise Exception("Attribute name must be a none empty string")

        for attr in self.attributes:
            if attr.name == name:
                return attr

        return None

    def unique_attributes(self) -> list[Attribute]:
        """
        Get a list of Entity's unique attributes

        Returns:
            list of unique attributes
        """

        unique_attrs = []
        for attr in self.attributes:
            if attr.unique:
                unique_attrs.append(attr)

        return sorted(unique_attrs)
