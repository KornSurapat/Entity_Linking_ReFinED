from refined.data_types.base_types import Span
from refined.inference.processor_11 import Refined

import json

refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                  entity_set='wikipedia',
                                  use_precomputed_descriptions=False)

# # Difficult disambiguation example
# text = 'Michael Jordan is a Professor of Computer Science at UC Berkeley.'
# spans = refined.process_text(text)
# print('\n' + '****' * 10 + '\n')
# print(text)
# print(spans)
# print('\n' + '****' * 10 + '\n')

# Example where entity mention spans are provided
text = "Joe Biden was born in Scranton."
spans = refined.process_text(text, spans=[Span(text='Joe Biden', start=0, ln=9),
                                          Span(text='Scranton', start=22, ln=8)])
print(text)
print(spans)
print('\n' + '****' * 10 + '\n')

###
def read_jsonl(file_path):
    test_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for l in f:
            json_object = json.loads(l.strip())
            test_data.append(json_object)
    return test_data

def transform_input(test_data, problem):
    text = test_data[problem]["text"][ : -1]
    spans = []
    for mention in test_data[problem]["gold_spans"]:
        span = Span(text = test_data[problem]["text"][mention["start"] : mention["start"] + mention["length"]],
                    start = mention["start"],
                    ln = mention["length"])
        spans.append(span)
    return text, spans

def write_jsonl(file_path, added_data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in added_data:
            f.write(json.dumps(item) + '\n')

def transform_output(added_data, problem, output):
    for mention in range(len(output)):
        added_data[problem]["gold_spans"][mention]["predictions_refined"] = output[mention]
    return added_data

def process(file_path_read, file_path_write):
    # Read data from file
    test_data = read_jsonl(file_path_read)
    # Prepare container
    added_data = test_data
    for i in range(len(test_data)):
        # Transform input
        text, spans = transform_input(test_data, i)
        # Run model
        output = refined.process_text(text, spans)
        # Transform output
        added_data = transform_output(added_data, i, output)
    # Store data to file
    write_jsonl(file_path_write, added_data)

test_data = read_jsonl("ace2004.jsonl")
added_data = test_data
for i in range(len(test_data)):
    text, spans = transform_input(test_data, i)
    output = refined.process_text(text, spans)
    # Just show
    print(output)
    added_data = transform_output(added_data, i, output)
    # Just show
    print(len(test_data[i]["gold_spans"][0]))
    print(len(added_data[i]["gold_spans"][0]))
    print("\n" + "----------" * 10 + "\n")

# process("ace2004.jsonl", "ace2004_pred.jsonl")
# print("1 done ____________________________________________________________________________________________________")
# process("aida.jsonl", "aida_pred.jsonl")
# print("2 done ____________________________________________________________________________________________________")
# process("aquaint.jsonl", "aquaint_pred.jsonl")
# print("3 done ____________________________________________________________________________________________________")
# process("cweb.jsonl", "cweb_pred.jsonl")
# print("4 done ____________________________________________________________________________________________________")
# process("graphq.jsonl", "graphq_pred.jsonl")
# print("5 done ____________________________________________________________________________________________________")
# process("mintaka.jsonl", "mintaka_pred.jsonl")
# print("6 done ____________________________________________________________________________________________________")
# process("msnbc.jsonl", "msnbc_pred.jsonl")
# print("7 done ____________________________________________________________________________________________________")
# process("reddit_comments.jsonl", "reddit_comments_pred.jsonl")
# print("8 done ____________________________________________________________________________________________________")
# process("reddit_posts.jsonl", "reddit_posts_pred.jsonl")
# print("9 done ____________________________________________________________________________________________________")
# process("shadow.jsonl", "shadow_pred.jsonl")
# print("10 done ____________________________________________________________________________________________________")
# process("tail.jsonl", "tail_pred.jsonl")
# print("11 done ____________________________________________________________________________________________________")
# process("top.jsonl", "top_pred.jsonl")
# print("12 done ____________________________________________________________________________________________________")
# process("tweeki.jsonl", "tweeki_pred.jsonl")
# print("13 done ____________________________________________________________________________________________________")
# process("webqsp.jsonl", "webqsp_pred.jsonl")
# print("14 done ____________________________________________________________________________________________________")
# process("wiki.jsonl", "wiki_pred.jsonl")
# print("15 done ____________________________________________________________________________________________________")
###

# # Example with numeric value
# text = 'The population of England is 55,000,000.'
# spans = refined.process_text(text)
# print(text)
# print(spans)
# print('\n' + '****' * 10 + '\n')

# # Example with currency
# text = "The net worth of Elon Musk is $200B."
# spans = refined.process_text(text)
# print(text)
# print(spans)
# print('\n' + '****' * 10 + '\n')

# # Example with time
# text = "It takes 60 minutes bake a potato."
# spans = refined.process_text(text)
# print(text)
# print(spans)
# print('\n' + '****' * 10 + '\n')

# # Example with an ordinal
# text = "The first book in the Harry Potter series is Harry Potter and the Philosopher's Stone."
# spans = refined.process_text(text)
# print(text)
# print(spans)
# print('\n' + '****' * 10 + '\n')

# # Example with age
# text = "Barack Obama was 48 years old when he became president of the United States."
# spans = refined.process_text(text)
# print(text)
# print(spans)
# print('\n' + '****' * 10 + '\n')

# # Example with percentage
# text = "The rural population of England was 10% in 2020."
# spans = refined.process_text(text)
# print(text)
# print(spans)
# print('\n' + '****' * 10 + '\n')

# # Example with height (quantity)
# text = "Joe Biden is 1.82m tall."
# spans = refined.process_text(text)
# print(text)
# print(spans)
# print('\n' + '****' * 10 + '\n')

# # Example with Wikidata entity that is not in Wikipedia
# text = "Andreas Hecht is a professor."
# spans = refined.process_text(text)
# print(text)
# print(spans)
# print('\n' + '****' * 10 + '\n')

# # Batched example
# texts = ["Andreas Hecht is a professor.", "Michael Jordan is a Professor of Computer Science at UC Berkeley."]
# docs = refined.process_text_batch(texts)
# for doc in docs:
#     print(f'Document: {doc.text}, spans: {doc.spans}')
# print('\n' + '****' * 10 + '\n\n')

# Batched example with spans
# texts = ["Joe Biden was born in Scranton."] * 2
# deep copy the Spans otherwise in-place modifications can cause issues
# spanss = [[Span(text='Joe Biden', start=0, ln=10), Span(text='Scranton', start=22, ln=8)] for _ in range(2)]
# docs = refined.process_text_batch(texts=texts, spanss=spanss)
# for doc in docs:
#     print(f'Document: {doc.text}, spans: {doc.spans}')
# print('\n' + '****' * 10 + '\n')
