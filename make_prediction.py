from pipeline import pipeline
import polars as pl


def main() -> None:
    test_df = pl.read_csv("data\share\\test.csv")
    test_df_dicts = test_df.to_dicts()

    predict = pipeline("vsevolo_de_bert", test_df_dicts, use_NER=True)

    submission_df = pl.DataFrame({
        "line_id": test_df["line_id"].to_numpy(),
        "is_hallucination": predict
    })
    submission_df.write_csv("data/submission.csv")


if __name__ == "__main__":
    main()

# please work
