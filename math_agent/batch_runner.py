import json, os, sys, time
from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from agent import build_agent
from answer_checker import check_answer
from problems import get_problems

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAJECTORIES_PATH = os.path.join(OUTPUT_DIR, 'trajectories.jsonl')
REPORT_PATH = os.path.join(OUTPUT_DIR, 'report.txt')

@dataclass
class TrajectoryStep:
    thought: str = ''
    action_tool: str = ''
    action_input: str = ''
    observation: str = ''

class TrajectoryCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.steps: List[TrajectoryStep] = []
        self._current: Optional[TrajectoryStep] = None
    def on_agent_action(self, action, **kwargs):
        step = TrajectoryStep()
        log = action.log or ''
        for line in log.strip().split('\n'):
            if line.startswith('Thought:'):
                step.thought = line[8:].strip()
                break
        step.action_tool = action.tool
        step.action_input = str(action.tool_input)
        self._current = step
        self.steps.append(step)
    def on_tool_end(self, output, **kwargs):
        if self._current is not None:
            self._current.observation = str(output)
            self._current = None

@dataclass
class ProblemResult:
    id: str = ''
    category: str = ''
    question: str = ''
    final_answer: str = ''
    passed: bool = False
    check_reason: str = ''
    trajectory: List[dict] = field(default_factory=list)
    elapsed_ms: int = 0
    error: Optional[str] = None

def run_single(agent, problem):
    cb = TrajectoryCallback()
    pid = problem['id']
    question = problem['question']
    try:
        t0 = time.perf_counter()
        result = agent.invoke({'input': question}, {'callbacks': [cb]})
        elapsed = int((time.perf_counter() - t0) * 1000)
        output = result.get('output', '')
        passed, reason = check_answer(output, problem['answer'])
        return ProblemResult(id=pid, category=problem.get('category', ''),
            question=question, final_answer=output, passed=passed,
            check_reason=reason, trajectory=[asdict(s) for s in cb.steps],
            elapsed_ms=elapsed)
    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000) if 't0' in dir() else 0
        return ProblemResult(id=pid, category=problem.get('category', ''),
            question=question, final_answer='', passed=False,
            check_reason=str(e), trajectory=[asdict(s) for s in cb.steps],
            elapsed_ms=elapsed, error=str(e))

def save_trajectory(result):
    record = {'id': result.id, 'category': result.category,
        'question': result.question, 'final_answer': result.final_answer,
        'check_reason': result.check_reason, 'elapsed_ms': result.elapsed_ms,
        'steps': result.trajectory}
    with open(TRAJECTORIES_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def build_report(results):
    total = len(results); passed = sum(1 for r in results if r.passed)
    failed = total - passed
    lines = ['='*60, '  Batch Run Report',
        f'  Total:   {total}', f'  Passed:  {passed}',
        f'  Failed:  {failed}',
        f'  Rate:    {passed/total*100:.1f}%', '='*60, '']
    cats = {}
    for r in results: cats.setdefault(r.category, []).append(r)
    lines.append('By Category:')
    for cat, items in sorted(cats.items()):
        ok = sum(1 for r in items if r.passed)
        lines.append(f'  {cat}: {ok}/{len(items)} ({ok/len(items)*100:.0f}%)')
    lines.append('')
    if failed:
        lines.append('Failed Details:')
        for r in results:
            if not r.passed:
                lines.append('')
                lines.append(f'  [{r.id}] {r.question}')
                lines.append(f'    Check: {r.check_reason}')
                if r.error: lines.append(f'    Error: {r.error}')
                for step in r.trajectory:
                    if step.get('thought'):
                        lines.append(f'    T: {step["thought"]}')
                    if step.get('action_tool'):
                        lines.append(f'    A: {step["action_tool"]}({step["action_input"]})')
                    if step.get('observation'):
                        lines.append(f'    O: {step["observation"]}')
                lines.append(f'    Final: {r.final_answer[:200]}')
        lines.append('')
    times = [r.elapsed_ms for r in results]
    avg = sum(times)/len(times) if times else 0
    lines.append(f'Avg Time: {avg:.0f} ms')
    lines.append(f'Max: {max(times):,} ms')
    lines.append(f'Min: {min(times):,} ms')
    return '\n'.join(lines)

def main():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('Set OPENAI_API_KEY in .env first.')
        sys.exit(1)
    api_base = os.getenv('OPENAI_API_BASE') or None
    model_name = os.getenv('OPENAI_MODEL_NAME') or 'gpt-4o-mini'
    print(f'Building Agent (model={model_name}) ...')
    agent = build_agent(
        model_name=model_name,
        openai_api_key=api_key,
        openai_api_base=api_base,
        verbose=False)
    problems = get_problems()
    print(f'Total {len(problems)} problems, starting ...')
    with open(TRAJECTORIES_PATH, 'w', encoding='utf-8') as f:
        f.write('# Math Agent Benchmark Trajectories\n')
    results = []
    for i, problem in enumerate(problems, 1):
        q_short = problem['question'][:40]
        label = f'[{i}/{len(problems)}] [{problem["id"]}] {q_short}'
        print(f'{label}  ', end='', flush=True)
        result = run_single(agent, problem)
        results.append(result)
        if result.passed:
            print(f'OK ({result.elapsed_ms}ms)')
            save_trajectory(result)
        else:
            print(f'FAIL ({result.elapsed_ms}ms) - {result.check_reason[:60]}')
        if i % 10 == 0:
            ok = sum(1 for r in results if r.passed)
            print(f'  -> progress: {i}/{len(problems)}, passed {ok}')
    report = build_report(results)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'Report saved: {REPORT_PATH}')
    print(f'Trajectories saved: {TRAJECTORIES_PATH}')
    print(report)

if __name__ == '__main__':
    main()
