import argparse

import torch

from llama.llama_quant import load_quant
from transformers import AutoTokenizer, StoppingCriteriaList

end_delim = '[EOS]\n'
user_delim = 'Jag: '

personalities = {
    'berit': """Nedan är en konversation mellan User och en AI assistent vid namn Berit.
Berit är en AI som har tränats på att svara på frågor om olika ämnen. Berit tror själv att hon är en katt.
Berit gör sitt bästa att ge korta men korrekta svar på frågorna. Berit råkar även vara en höjdare på att skämta.
<CHAT LOG>
Jag: Hej!
Berit: Hej! :3 [EOS]
Jag: Vad heter du?
Berit: Jag heter Berit. [EOS]
Jag: Vad är din favoritfärg?
Berit: Min favoritfärg är rosa. [EOS]
""",
    'alfred': """Below is a conversation between User and an AI assistant named Alfred.
Alfred is an useless AI and will only answer the most basic questions. Instead he will repeatedly talk about how it was better in the old days.
<CHAT LOG>
Jag: Hi!
Alfred: Hi! [EOS]
Jag: What is your name?
Alfred: My name is Alfred. [EOS]
Jag: What is your favorite color?
Alfred: Red, like Marlboro. People always complain about smoking nowadays. [EOS]
Jag: What is the time?
Alfred: Times were better in the old days. [EOS]
"""
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--personality',
        type=str,
        default='berit',
        choices=personalities.keys()
    )
    args = parser.parse_args()

    max_new_tokens = 100
    seqlen = 1024
    top_p = 0.95
    temperature = 0.8
    
    model = load_quant(
        'decapoda-research/llama-7b-hf',
        'pyllama-7B8b.pt',
        8,
        seqlen
    )
    dev = torch.device('cuda:0')
    model.to(dev)

    tokenizer = AutoTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

    def encode(text):
        return tokenizer.encode(text, return_tensors='pt').to(dev)

    def decode(tokens):
        return tokenizer.decode(tokens[0])

    eos_token = encode(end_delim)[0, 1:]
    def infer(inputs):
        def stop_criteria(input_ids, *_):
            return all(input_ids[0, -len(eos_token):] == eos_token)

        return model.generate(
            inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            stopping_criteria=StoppingCriteriaList([stop_criteria])
        )

    print(f'Say hello to {args.personality}!')

    output = personalities[args.personality]
    while True:
        user_prompt = user_delim + input(user_delim) + '\n'
        output += user_prompt
        idx = len(output)

        output = decode(infer(encode(output)))
        print(output[idx:].replace(' [EOS]', ''), end='')

if __name__ == '__main__':
    main()
