import random
from src.config import SEEDS

semillas_nuevas = SEEDS.copy()
random.seed(42)


def create_seed(n_semillas: int = 6):
    while len(semillas_nuevas) < n_semillas:
        sem_nueva = random.randint(1, 999999)
        if sem_nueva not in semillas_nuevas:
            semillas_nuevas.append(sem_nueva)
    return (semillas_nuevas[:n_semillas])
