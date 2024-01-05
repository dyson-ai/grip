import pybullet as p
from .entity import Entity
from .types import BulletContactInfo


def get_collision_info(entity_a: Entity, entity_b: Entity, max_distance: float = 0.0):
    """Returns collision information from pair of entities

    Args:
        entity_a (Entity): the first entity
        entity_b (Entity): the second entity
        max_ditance (float, optional): If the distance between objects is greater than this maximum distance, no points are be returned.
    Returns:
        (List[BulledContactInfo]): a list of contacts if any exists
    """

    p.performCollisionDetection()
    contacts = []

    links_a = entity_a.get_link_ids()
    links_b = entity_b.get_link_ids()

    for l_a in links_a:
        for l_b in links_b:
            contacts += p.getClosestPoints(
                entity_a.id,
                entity_b.id,
                linkIndexA=l_a,
                linkIndexB=l_b,
                distance=max_distance,
                physicsClientId=entity_a.phys_id,
            )

    return [BulletContactInfo(*c) for c in contacts]


def is_colliding(entity_a: Entity, entity_b: Entity, max_distance: float = 0.0):
    """Returns collision information from pair of entities

    Args:
        entity_a (Entity): the first entity
        entity_b (Entity): the second entity
        max_ditance (float, optional): If the distance between objects is greater than this maximum distance, no points are be returned.
    Returns:
        (bool): whether or not collision contacts exist
    """

    return len(get_collision_info(entity_a, entity_b, max_distance)) != 0
