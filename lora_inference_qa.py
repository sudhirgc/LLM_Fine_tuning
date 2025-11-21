from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'outputs/TinyLlama-1.1B-Chat-v1.0_qa/merged'
#model_name = 'outputs/Phi-3-mini-128k-instruct_qa/merged'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

context = """The library system of the university is divided between the main library and each of the colleges and schools. The main building is the 14-story Theodore M. Hesburgh Library, completed in 1963, which is the third building to house the main collection of books. The front of the library is adorned with the Word of Life mural designed by artist Millard Sheets. This mural is popularly known as "Touchdown Jesus" because of its proximity to Notre Dame Stadium and Jesus' arms appearing to make the signal for a touchdown."""
question = "Who created mural at the front of Library?"

messages = [{'role':'system', 'content':'Read the following context and concisely answer the question'},
            {'role':'user', 'content':f'context:{context}, question: {question}'}]

input = tokenizer.apply_chat_template(messages , tokenize=False, add_generation_token=True)
print(input)
tokenized_input = tokenizer(input, return_tensors='pt').to('cuda')
print('Tokenized Input', tokenized_input)

output = model.generate(**tokenized_input)

print(output)

generate_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generate_text)