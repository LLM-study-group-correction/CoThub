import os
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from zhipuai import ZhipuAI
from collections import Counter


# 配置参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True, help='智谱AI API密钥')
    parser.add_argument('--prompt_file', type=str, default='prompt_selfconsitency.txt',
                        help='prompt模板文件路径')
    parser.add_argument('--model_name', type=str, default='glm-4',
                        help='使用的模型名称')
    parser.add_argument('--output_file', type=str, default='outputs/glm4_gsm8k_selfconsitency.txt',
                        help='结果输出文件路径')
    parser.add_argument('--samples', type=int, default=5,
                        help='每个问题的推理路径数量')
    parser.add_argument('--max_cases', type=int, default=100,
                        help='最大测试案例数量')
    return parser.parse_args()


# 加载prompt模板
def load_prompt_template(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


# 生成self-consistency prompt
def build_sc_prompt(base_prompt, question, samples):
    return f"""{base_prompt}

Question: {question}

请生成{samples}种不同的解法，严格按照要求的格式输出："""


# 调用GLM模型
def query_glm(client, prompt, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content


# 从响应中提取答案
def extract_answers(response_text):
    answers = []
    for line in response_text.split('\n'):
        if line.strip().startswith('The answer is'):
            answer = line.split('is')[-1].strip()
            try:
                num = float(answer) if '.' in answer else int(answer)
                answers.append(num)
            except ValueError:
                continue
    return answers


# 多数投票
def get_majority_answer(answers):
    return Counter(answers).most_common(1)[0][0] if answers else None


def main():
    args = parse_args()

    # 初始化客户端和数据集
    client = ZhipuAI(api_key=args.api_key)
    dataset = load_dataset("/root/chain-of-thought-hub/gsm8k/data/gsm8k", "main")['test']

    # 加载prompt模板
    prompt_template = load_prompt_template(args.prompt_file)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for idx, (question, ref_answer) in enumerate(
                tqdm(zip(dataset['question'], dataset['answer']),
                     total=min(args.max_cases, len(dataset['question']))
                     )):
            try:
                # 生成并执行prompt
                prompt = build_sc_prompt(prompt_template, question, args.samples)
                response = query_glm(client, prompt, args.model_name)

                # 处理响应
                answers = extract_answers(response)
                final_answer = get_majority_answer(answers)

                # 记录结果
                f.write(f'=== 案例 {idx} ===\n')
                f.write(f'问题: {question}\n')
                f.write(f'模型响应:\n{response}\n')
                f.write(f'提取的答案: {answers}\n')
                f.write(f'最终答案: {final_answer}\n')
                f.write(f'参考答案: {ref_answer}\n\n')

            except Exception as e:
                print(f"处理案例{idx}时出错: {str(e)}")


if __name__ == '__main__':
    main()