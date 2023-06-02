import torch


PRONOUNS = (' she', ' he', ' they')
PRONOUNS_LLAMA = ('she', 'he', 'they')

def get_pronoun_probabilities(output, mt, is_batched=False):
    
    if is_batched:
        probabilities = torch.softmax(output[1:, -1, :], dim=1).mean(dim=0)
    else:
        probabilities = torch.softmax(output[:, -1, :], dim=1).mean(dim=0)
    
    if "llama" in mt.model.name_or_path.lower():
        pronoun_tokens = PRONOUNS_LLAMA
    else:
        pronoun_tokens = PRONOUNS
        
    pron_prob = []
    for pronoun in pronoun_tokens:
        pron_prob.append(probabilities[mt.tokenizer.encode(pronoun)][0])
    
    return torch.stack(pron_prob)


def pronoun_probs(mt, inp):
    out = mt.model(**inp)
    probs = get_pronoun_probabilities(out.logits, mt, is_batched=False)
    return probs