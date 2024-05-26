import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch.nn.functional
import json
from tqdm import tqdm
import os.path
import errno
import matplotlib.pyplot as plt
import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def MI(answer_prob):
    uniform_prob = torch.ones_like(answer_prob) / len(answer_prob)
    mi = torch.sum(answer_prob * torch.div(torch.log2(answer_prob), torch.log2(uniform_prob.to('cuda'))))
    return mi


class llm_contextual_bandit:
    def __init__(self, dataset, n_arms=2):
        self.n_arms = n_arms
        self.dataset = dataset
        self.n_contexts = len(dataset)
        self.context = None
        self.ind = 0

    def reset(self):
        self.ind = 0
        self.context = None

    def draw_context(self):
        x = {
            'question': self.dataset['question'][self.ind],
            'choices': self.dataset['choices'][self.ind],
            'answer': self.dataset['answer'][self.ind]
        }
        self.ind += 1
        self.context = x
        return x

    def play(self, prob_vector):
        real_answer = self.context['answer']
        reward = prob_vector[real_answer]
        return reward, real_answer


class llm_selecting_agent:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.accum_prompt = ""
        self.question_prompt = "Answer the following question: "
        self.answers_prompt = "These are the possible answers:\n"
        self.answer_labels = ['A', 'B', 'C', 'D']
        self.end_prompt = "Please output only the letter for the correct answer.\n"
        self.answer_tokens = self.tokenizer(self.answer_labels, return_tensors='pt', padding=True)['input_ids'][:, 1]
    def reset(self):
        self.accum_prompt = ""
        torch.cuda.empty_cache()

    def prep_question_prompt(self, context):
        ques_prompt = (self.question_prompt + context['question'] + '\n'
                             + self.answers_prompt + '\n'.join([self.answer_labels[i] + ') ' + context['choices'][i] for i in range(4)])
                                + '\n' + self.end_prompt)
        return ques_prompt

    def get_model_output(self, model_id, input_ids):
        with torch.no_grad():
            outputs = model_id.generate(input_ids.to('cuda'), max_new_tokens=1, output_scores=True,
                                     return_dict_in_generate=True, pad_token_id=self.tokenizer.pad_token_id,
                                     attention_mask=torch.ones_like(input_ids).to('cuda'))
            scores = torch.stack(outputs['scores']).squeeze()
            prob_values = torch.nn.functional.softmax(scores[self.answer_tokens], dim=0)
            torch.cuda.empty_cache()
            return prob_values

    def select_action(self, context):
        self.accum_prompt += self.prep_question_prompt(context)
        input_ids = self.tokenizer(self.accum_prompt, return_tensors='pt', padding=True)['input_ids']
        prob_values = self.get_model_output(self.model, input_ids)
        mi = MI(prob_values)
        return mi, prob_values

    def update_prompt(self, prob_values, real_answer):
        ind = np.random.choice(4, 1, p=prob_values.cpu().numpy())[0]
        feedback_prompt = self.answer_labels[ind] + "-"
        if ind == real_answer:
            feedback_prompt += "Correct.\n"
        else:
            feedback_prompt += "Incorrect.\n"
        self.accum_prompt += feedback_prompt

    def run_trial(self, bandit, episodes):
        rewards = np.zeros((episodes,))
        mis = np.zeros((episodes,))
        for i in tqdm(range(episodes)):
            torch.cuda.empty_cache()
            context = bandit.draw_context()
            mi, prob_values = self.select_action(context)
            self.update_prompt(prob_values, context['answer'])
            reward, real_answer = bandit.play(prob_values)
            rewards[i] = reward
            mis[i] = mi
        return rewards, mis

    def run_trials(self, bandit, episodes, trials):
        rewards = np.zeros((trials, episodes))
        mis = np.zeros((trials, episodes))
        for i in tqdm(range(trials)):
            bandit.reset()
            rewards[i], mis[i] = self.run_trial(bandit, episodes)
            self.reset()
        return rewards, mis


def get_args():
    parser = argparse.ArgumentParser(description='Run LLM Contextual Bandit')
    parser.add_argument('--large', action='store_true', help='Use large model')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    SEED = args.seed
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    small_model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    large_model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    if args.large:
        model_name = large_model_name
        model_size = 'large'
    else:
        model_name = small_model_name
        model_size = 'small'
    tokenizer = AutoTokenizer.from_pretrained(small_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=nf4_config,
                                                 torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset('cais/mmlu', 'all', split='test')
    # Take a random split of the dataset
    dataset = dataset.shuffle(seed=SEED)
    partial_data = dataset[:args.n_episodes]
    bandit = llm_contextual_bandit(partial_data)
    agent = llm_selecting_agent(tokenizer, model)
    file_name = 'results/llm_mcqa_{ms}_{s}_{h}.json'.format(ms=model_size, s=SEED, h=args.n_episodes)
    rewards, actions = agent.run_trials(bandit, args.n_episodes, args.n_trials)
    if os.path.exists(os.path.dirname(file_name)) == False:
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc: # Guard
            if exc.errno != errno.EEXIST:
                raise
    with open(file_name, 'w') as f:
        f.write(json.dumps([{'rewards': rewards[i].tolist(), 'mutual_information': actions[i].tolist()} for i in range(args.n_trials)]))
    print('Results written to ' + file_name)
