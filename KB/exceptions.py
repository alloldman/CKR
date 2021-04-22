from typing import Iterable


class EntityUnavailable(Exception):
    def __init__(self, uri):
        super().__init__(f"Entity {uri} is unavailable.")


class EmbeddingNotFound(Exception):
    def __init__(self, needed_entities: str):
        super().__init__(f"Embedding for entity {self.needed_entity} not found.")


if __name__ == '__main__':
    raise EntityUnavailable('/c/en/teapot')
