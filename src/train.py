import engine
from model import PlantDiseaseClassifierModel

if __name__ == "__main__":
    model = PlantDiseaseClassifierModel()
    engine.train_fn(model)
