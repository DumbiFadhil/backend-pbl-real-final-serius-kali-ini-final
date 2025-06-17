from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch
import pandas as pd

# Load model and tokenizer once
tokenizer = TapasTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")
model = TapasForQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq")

def answer_table_question(table, query):
    """
    Process a question on a given Pandas DataFrame table using TAPAS.
    """
    inputs = tokenizer(table=table, queries=[query], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_answer_coordinates, _ = tokenizer.convert_logits_to_predictions(
        inputs,
        outputs.logits.detach(),
        outputs.logits_aggregation.detach()
    )

    if predicted_answer_coordinates[0]:
        answers = [table.iat[coord] for coord in predicted_answer_coordinates[0]]
        return answers
    else:
        return ["No answer"]
