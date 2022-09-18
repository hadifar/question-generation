from transformers import AutoModel,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('runs/qg_mrm8488_openqg_test')
model = AutoModel.from_pretrained('runs/qg_mrm8488_openqg_test')

tokenizer.push_to_hub("dutch_qg")
model.push_to_hub("dutch_qg")


