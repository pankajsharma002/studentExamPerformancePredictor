from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # Data ingestion
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    # Data transformation
    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_obj = transformation.initiate_data_transformation(train_data_path, test_data_path)

    # Model training
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
