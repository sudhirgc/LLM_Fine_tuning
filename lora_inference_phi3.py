from transformers import AutoTokenizer, AutoModelForCausalLM

#model_name = 'outputs/TinyLlama-1.1B-Chat-v1.0_Story_Teller/merged'
model_name = 'outputs/Phi-3-mini-128k-instruct_Story_Teller/merged'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

    
messages = [{'role':'system', 'content':'You are an expert Storyteller. Create story from the supplied instruction'},
            {'role':'user', 'content':'Tell story about Krishna as a superhero'}]

input = tokenizer.apply_chat_template(messages , tokenize=False, add_generation_token=True)
print(input)
tokenized_input = tokenizer(input, return_tensors='pt').to('cuda')
print('Tokenized Input', tokenized_input)

generation_args = { 
    "max_new_tokens": 500
} 

output = model.generate(**tokenized_input, **generation_args)

print(output)

generate_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generate_text)