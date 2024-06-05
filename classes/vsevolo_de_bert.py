from sentence_transformers.cross_encoder import CrossEncoder


class VsevoloDeBERT:
    def __init__(self, model_path: str) -> None:
        """
        model_path: path to dir with weights in transformers format
        """
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self) -> None:
        """
        Load model weights
        """
        try:
            model = CrossEncoder(
                self.model_path,
                num_labels=1,
                automodel_args={
                    "ignore_mismatched_sizes": True
                    }
                )
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, data: list[str]) -> int:
        """
        Model prediction on one sample
        """
        if not self.model:
            print("Model is not loaded")
            return None

        try:
            predictions = int(self.model.predict(data) > 0.2)
            return predictions
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None


# Usage example:
if __name__ == "__main__":
    import polars as pl

    raw = pl.read_csv("../../../Data/Raw Data/train.csv").to_dicts()
    test_data = raw[0]["summary"] + " Ответ: " + raw[0]["answer"]

    model = VsevoloDeBERT(
        "../vsevolo_de_bert"
    )

    predictions = model.predict([test_data])
    if predictions is not None:
        print(predictions)
