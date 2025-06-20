import re
from pathlib import Path
from typing import List
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from tabulate import tabulate  # 导入 tabulate 库


class CoTConsistencyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        steps = self._extract_steps(test_case.actual_output)
        if not steps:
            return 0.0

        valid_steps = sum(1 for step in steps if self._validate_step(step))
        self.score = valid_steps / len(steps)
        return self.score

    def _extract_steps(self, text: str) -> List[str]:
        """精确提取GLM-4的思维链步骤"""
        step_lines = []
        in_steps = False

        print(f"实际输出：\n{text}")  # 打印实际输出，帮助调试

        # 去除多余的空白字符和换行符
        text = text.replace("\n", " ").strip()

        # 检查是否包含关键字 "Let's think step by step"
        if "Let's think step by step" in text:
            in_steps = True

        # 逐行提取步骤
        if in_steps:
            # 将步骤分割为若干个子字符串
            step_texts = re.split(
                r'(\d+\s*[+\-*/]\s*\d+|\s*(left|sold|makes|total|per|multiply|subtract|add|divided|equals))', text)

            # 如果正则表达式没有匹配到有效的步骤，则返回空列表
            if not step_texts:
                return []

            # 移除空字符串或 None 值，防止出现错误
            step_lines = [s.strip() for s in step_texts if s and s.strip()]

        print(f"提取的步骤：{step_lines}")  # 打印提取的步骤，帮助调试
        return step_lines if step_lines else []

    def _validate_step(self, step: str) -> bool:
        """验证单个步骤的逻辑正确性"""
        if math_match := re.search(r'(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)', step):
            a, op, b, res = map(int, math_match.groups())
            if op == '+':
                return a + b == res
            elif op == '-':
                return a - b == res
            elif op == '*':
                return a * b == res
            elif op == '/' and b != 0:
                return a / b == res

        if money_match := re.search(r'(\d+)\s*\*\s*\$(\d+)\s*=\s*\$(\d+)', step):
            qty, price, total = map(int, money_match.groups())
            return qty * price == total

        return True  # 默认认为是合理文本陈述

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "CoT Consistency"


def parse_test_cases(file_path: str) -> List[LLMTestCase]:
    content = Path(file_path).read_text(encoding='utf-8')
    case_blocks = re.split(r'===== CASE \d+ =====\n', content)[1:]

    test_cases = []
    for block in case_blocks:
        try:
            question = re.search(r'Question:\n(.+?)\nGLM-4 Answer:', block, re.DOTALL).group(1).strip()
            actual = re.search(r'GLM-4 Answer:\n(.+?)\nReference Answer:', block, re.DOTALL).group(1).strip()
            expected = re.search(r'#### (\d+)', block).group(1).strip()

            test_cases.append(LLMTestCase(
                input=question,
                actual_output=actual,
                expected_output=expected
            ))
        except AttributeError as e:
            print(f"解析失败: {e}\n当前block:\n{block[:200]}...")

    return test_cases


def generate_table(test_cases: List[LLMTestCase], metrics: List[BaseMetric]) -> str:
    """生成并返回评估结果的表格"""
    table_data = []

    for i, case in enumerate(test_cases):
        row = [f"Case {i}", case.input.strip()[:30], case.actual_output.strip()[:30]]  # 输入和输出（截取前30个字符）

        # 计算每个度量指标的得分
        for metric in metrics:
            score = metric.measure(case)
            row.append(f"{score:.2f}")

        table_data.append(row)

    headers = ["# Case", "Question", "Answer"] + [metric.__name__ for metric in metrics]  # 表头
    return tabulate(table_data, headers=headers, tablefmt="grid")


if __name__ == "__main__":
    # 加载测试案例
    test_cases = parse_test_cases("/root/chain-of-thought-hub/gsm8k/outputs/glm4_gsm8k_test.txt")

    # 定义度量指标
    metrics = [
        CoTConsistencyMetric(threshold=0.75)  # 使用思维链一致性评估
    ]

    # 生成表格
    table = generate_table(test_cases, metrics)

    # 打印表格
    print(table)

    # 错误诊断输出
    print("\n错误诊断:")
    for i, case in enumerate(test_cases):
        metric = CoTConsistencyMetric()
        metric.measure(case)
        if not metric.is_successful():
            print(f"\nCase {i} 思维链错误:")
            steps = metric._extract_steps(case.actual_output)
            for j, step in enumerate(steps):
                valid = metric._validate_step(step)
                print(f"{'✅' if valid else '❌'} 步骤{j + 1}: {step}")
