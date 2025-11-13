"""
DS-STAR: Data Science Agent via Iterative Planning and Verification (v5 - Optimized)
論文 https://arxiv.org/abs/2509.21825v3 に準拠した最適化実装

エージェント構成:
- Analyzer: データファイルの分析
- Planner: 計画の作成・拡張（初期計画と追加計画を統合）
- Coder: コード実装（通常コードと最終コードを統合）
- Verifier: 結果の検証
- Router: 計画の修正戦略を決定
"""

import asyncio
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_framework import (
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatAgent,
    ChatMessage,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.observability import setup_observability
from agent_framework.observability import get_tracer

from dotenv import load_dotenv
from typing_extensions import Never
from opentelemetry import trace

load_dotenv(override=True)

# Observability export may complain when the local collector is unavailable.
# Disable metrics export by default to avoid noisy errors in common setups.
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"

# OpenTelemetryのログレベルを上げてエラーメッセージを抑止
logging.getLogger("opentelemetry.exporter.otlp.proto.grpc.exporter").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry").setLevel(logging.WARNING)

# Suppress noisy pandas deprecation warnings surfaced during agent execution.
warnings.filterwarnings(
    "ignore",
    "The argument 'infer_datetime_format' is deprecated",
    UserWarning,
)
warnings.filterwarnings(
    "ignore",
    "Could not infer format, so each element will be parsed individually",
    UserWarning,
)

# 定数
MAX_ITERATIONS = 10
STATE_KEY_QUERY = "query"
STATE_KEY_FILE_DESCS = "file_descriptions"
STATE_KEY_PLAN_STEPS = "plan_steps"
STATE_KEY_CURRENT_CODE = "current_code"
STATE_KEY_ITERATION = "iteration"
STATE_KEY_CURRENT_FILE = "current_file"
STATE_KEY_LAST_RESULT = "last_result"
STATE_KEY_ROUTER_ACTION = "router_action"
STATE_KEY_EXECUTION_PATH = "execution_path"

# ===== Data Classes =====


@dataclass
class FileDescription:
    """ファイル説明"""
    filename: str
    summary: str


@dataclass
class PlanStep:
    """計画ステップ"""
    step_number: int
    description: str


@dataclass
class WorkflowState:
    """ワークフロー状態"""
    query: str
    file_descriptions: list[FileDescription] = field(default_factory=list)
    plan_steps: list[PlanStep] = field(default_factory=list)
    current_code: str = ""
    last_result: str = ""
    iteration: int = 0
    is_sufficient: bool = False
    is_final: bool = False  # 最終コード生成モード


@dataclass
class CodeExecution:
    """コード実行結果"""
    code: str
    result: str
    success: bool
    error: str | None


@dataclass
class VerificationDecision:
    """検証結果"""
    is_sufficient: bool
    state: WorkflowState


# ===== Utility Functions =====


async def get_shared_state_safe(ctx: WorkflowContext, key: str, default: Any = None) -> Any:
    """共有状態を安全に取得（キーが存在しない場合はデフォルト値を返す）"""
    try:
        return await ctx.get_shared_state(key)
    except KeyError:
        return default


async def track_execution_path(ctx: WorkflowContext, step_name: str) -> None:
    """実行パスを記録"""
    execution_path: list[str] = await get_shared_state_safe(ctx, STATE_KEY_EXECUTION_PATH, [])
    execution_path.append(step_name)
    await ctx.set_shared_state(STATE_KEY_EXECUTION_PATH, execution_path)


async def log_shared_state_to_trace(ctx: WorkflowContext, executor_name: str) -> None:
    """すべての共有状態をトレースに出力"""
    try:
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            # 各状態キーの値を取得してトレース属性に設定
            query = await get_shared_state_safe(ctx, STATE_KEY_QUERY, "")
            if query:
                current_span.set_attribute(f"{executor_name}.query", str(query))
            
            iteration = await get_shared_state_safe(ctx, STATE_KEY_ITERATION, 0)
            current_span.set_attribute(f"{executor_name}.iteration", iteration)
            
            file_descriptions = await get_shared_state_safe(ctx, STATE_KEY_FILE_DESCS, [])
            current_span.set_attribute(f"{executor_name}.file_count", len(file_descriptions))
            
            plan_steps = await get_shared_state_safe(ctx, STATE_KEY_PLAN_STEPS, [])
            current_span.set_attribute(f"{executor_name}.plan_step_count", len(plan_steps))
            if plan_steps:
                # すべての計画ステップ
                for idx, step in enumerate(plan_steps, start=1):
                    step_desc = step.description if hasattr(step, 'description') else str(step)
                    current_span.set_attribute(f"{executor_name}.plan_step_{idx}", step_desc)

            current_code = await get_shared_state_safe(ctx, STATE_KEY_CURRENT_CODE, "")
            if current_code:
                current_span.set_attribute(f"{executor_name}.code_length", len(current_code))
            
            last_result = await get_shared_state_safe(ctx, STATE_KEY_LAST_RESULT, "")
            if last_result:
                current_span.set_attribute(f"{executor_name}.last_result", str(last_result))
            
            router_action = await get_shared_state_safe(ctx, STATE_KEY_ROUTER_ACTION, None)
            if router_action:
                current_span.set_attribute(f"{executor_name}.router_action", str(router_action))
            
            execution_path = await get_shared_state_safe(ctx, STATE_KEY_EXECUTION_PATH, [])
            current_span.set_attribute(f"{executor_name}.execution_step_count", len(execution_path))
            
    except Exception as e:
        # トレース出力の失敗はワークフローを止めない
        print(f"Warning: Failed to log shared state to trace: {e}")


def extract_code_block(text: str) -> str | None:
    """マークダウンコードブロックからコードを抽出"""
    patterns = [
        r"```python\n(.*?)\n```",
        r"```\n(.*?)\n```",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    return None


def execute_python_code(code: str) -> tuple[str, bool, str | None]:
    """Pythonコードを実行"""
    from io import StringIO
    import contextlib

    output_buffer = StringIO()

    try:
        import pandas as pd
        import numpy as np
        import json
        from pathlib import Path

        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "json": json,
            "Path": Path,
        }

        with contextlib.redirect_stdout(output_buffer):
            exec(code, exec_globals)

        output = output_buffer.getvalue()
        return (output, True, None)

    except Exception:
        import traceback
        error_msg = traceback.format_exc()
        return (output_buffer.getvalue(), False, error_msg)


# ===== Phase 1: Initialization & File Analysis =====


@executor(id="initialize")
async def initialize(query: str, ctx: WorkflowContext[WorkflowState]) -> None:
    """初期化: データファイルをリスト化"""
    await track_execution_path(ctx, "initialize")
    await log_shared_state_to_trace(ctx, "initialize")
    
    print(f"\n{'='*80}")
    print("DS-STAR v5 (Optimized): Iterative Planning and Verification")
    print(f"Query: {query}")
    print(f"{'='*80}\n")
    print("Phase 1: Analyzing data files...")

    data_dir = Path("data")
    state = WorkflowState(query=query)
    
    if data_dir.exists():
        files = list(data_dir.glob("*"))
        print(f"Found {len(files)} files")
        
        for filepath in files:
            if filepath.is_file():
                state.file_descriptions.append(
                    FileDescription(filename=filepath.name, summary="")
                )
    
    # 共有状態初期化
    await ctx.set_shared_state(STATE_KEY_QUERY, query)
    await ctx.set_shared_state(STATE_KEY_ITERATION, 0)
    await ctx.set_shared_state(STATE_KEY_PLAN_STEPS, [])
    await ctx.set_shared_state(STATE_KEY_CURRENT_CODE, "")
    await ctx.set_shared_state(STATE_KEY_FILE_DESCS, state.file_descriptions)
    await ctx.set_shared_state(STATE_KEY_LAST_RESULT, "")
    await ctx.set_shared_state(STATE_KEY_ROUTER_ACTION, None)
    await ctx.set_shared_state(STATE_KEY_EXECUTION_PATH, [])
    
    await ctx.send_message(state)


@executor(id="analyze_files")
async def analyze_files(state: WorkflowState, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    """ファイル分析（リクエスト作成・実行・処理を統合）"""
    await track_execution_path(ctx, "analyze_files")
    await log_shared_state_to_trace(ctx, "analyze_files")
    
    # 常に共有状態から取得
    file_descriptions = await get_shared_state_safe(ctx, STATE_KEY_FILE_DESCS, [])
    unanalyzed = [f for f in file_descriptions if not f.summary]
    
    if not unanalyzed:
        print("✓ All files analyzed\n")
        # 分析完了時は次のステップへスキップ
        return
    
    file_to_analyze = unanalyzed[0]
    print(f"Analyzing: {file_to_analyze.filename}")
    
    # プロンプト作成
    prompt = f"""あなたは専門的なデータアナリストです。data/{file_to_analyze.filename}の内容を読み込み、説明するPythonコードを生成してください。

# 要件
- ファイルは非構造化データまたは構造化データのいずれかです。
- 構造化データが多すぎる場合は、いくつかの例だけを出力してください。
- 重要な情報を出力してください。例えば、すべての列名を出力してください。
- Pythonコードはdata/{file_to_analyze.filename}の内容を出力する必要があります。
- コードは自己完結型の単一ファイルPythonプログラムで、そのまま実行可能である必要があります。
- 応答には単一のコードブロックのみを含めてください。
- 重要: エラーが発生した場合はデバッグするため、ダミーの内容を含めないでください。
- エラーを防ぐためにtry:とexcept:を使用しないでください。後でデバッグします。"""

    # Analyzerエージェント呼び出し
    request = AgentExecutorRequest(
        messages=[ChatMessage(Role.USER, text=prompt)],
        should_respond=True
    )
    await ctx.send_message(request)


@executor(id="process_analysis")
async def process_analysis(response: AgentExecutorResponse, ctx: WorkflowContext[WorkflowState]) -> None:
    """ファイル分析結果を処理"""
    await track_execution_path(ctx, "process_analysis")
    await log_shared_state_to_trace(ctx, "process_analysis")
    
    query = await get_shared_state_safe(ctx, STATE_KEY_QUERY, "")
    file_descriptions: list[FileDescription] = await get_shared_state_safe(ctx, STATE_KEY_FILE_DESCS, [])
    
    # 現在分析中のファイルを特定（最初の未分析ファイル）
    current_file = next((f.filename for f in file_descriptions if not f.summary), None)
    
    code = extract_code_block(response.agent_run_response.text)
    summary_text = ""
    
    if code:
        print(f"Generated analysis code for {current_file}:")
        print("-" * 60)
        print(code)
        print("-" * 60)
        
        result, success, error = execute_python_code(code)
        
        if success:
            summary_text = result.strip() or "Execution completed with no output."
            print(f"✓ Analysis successful")
        else:
            failure_detail = (error or "Unknown error").strip()
            summary_text = f"Execution failed: {failure_detail}"
            print(f"✗ Analysis failed")
    else:
        summary_text = "Analyzer response did not contain code."
        print("✗ No code found")

    # ファイル説明を更新
    if current_file:
        for desc in file_descriptions:
            if desc.filename == current_file:
                desc.summary = summary_text
                break

    await ctx.set_shared_state(STATE_KEY_FILE_DESCS, file_descriptions)
    
    # まだ未分析ファイルがあるかチェック
    unanalyzed = [f for f in file_descriptions if not f.summary]
    if unanalyzed:
        print(f"Remaining files: {len(unanalyzed)}\n")
    else:
        print("✓ All files analyzed\n")

    state = WorkflowState(
        query=query,
        file_descriptions=file_descriptions,
        last_result=summary_text,
    )

    await ctx.send_message(state)


# ===== Phase 2: Planning =====


@executor(id="plan")
async def plan(state: WorkflowState, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    """計画作成（初期計画と追加計画を統合）"""
    await track_execution_path(ctx, "plan")
    await log_shared_state_to_trace(ctx, "plan")
    
    query = await get_shared_state_safe(ctx, STATE_KEY_QUERY, "")
    # state から直接取得せず、常に共有状態から取得
    file_descriptions: list[FileDescription] = await get_shared_state_safe(ctx, STATE_KEY_FILE_DESCS, [])
    plan_steps: list[PlanStep] = await get_shared_state_safe(ctx, STATE_KEY_PLAN_STEPS, [])
    last_result = await get_shared_state_safe(ctx, STATE_KEY_LAST_RESULT, "")
    
    file_info = "\n".join([
        f"- data/{desc.filename}: {desc.summary}" if desc.summary else f"- data/{desc.filename}"
        for desc in file_descriptions
    ]) or "- (no files provided)"
    
    # 初期計画 vs 追加計画
    if not plan_steps:
        # 初期計画
        print("\nPhase 2: Creating initial plan...")
        prompt = f"""あなたは専門的なデータサイエンス計画者です。与えられたデータに基づいて事実に関する質問に答えるには、まず効果的に計画を立てる必要があります。

# 質問
{query}

# 与えられたデータ:
{file_info}

# あなたのタスク
- 上記の質問に答えるための最初のステップを提案してください。
- 最初のステップは質問に答えるのに十分である必要はありません。
- 質問に答えるための良い出発点となる、非常にシンプルな最初のステップを提案してください。
- 応答には初期ステップのみを含めてください。"""
    else:
        # 追加計画
        print("\nGenerating next plan step...")
        plan_text = "\n".join([f"{step.step_number}. {step.description}" for step in plan_steps])
        result_text = last_result.strip() if last_result else "(no result captured)"
        
        prompt = f"""あなたは専門的なデータサイエンス計画者です。与えられたデータに基づいて事実に関する質問に答えるには、まず効果的に計画を立てる必要があります。あなたのタスクは、質問に答えるために次に行う計画を提案することです。

# 質問
{query}

# 与えられたデータ:
{file_info}

# 現在の計画
{plan_text}

# 現在の計画から得られた結果:
{result_text}

# あなたのタスク
- 上記の質問に答えるための次のステップを提案してください。
- 次のステップは質問に答えるのに十分である必要はありませんが、最後の単純なステップのみが必要な場合は、それを提案してもかまいません。
- 質問に答えるための良い中間点となる、非常にシンプルな次のステップを提案してください。
- もちろん、あなたの応答は質問に直接答えることができる計画でもかまいません。
- 応答には説明なしで次のステップのみを含めてください。"""
    
    # Plannerエージェント呼び出し
    request = AgentExecutorRequest(
        messages=[ChatMessage(Role.USER, text=prompt)],
        should_respond=True
    )
    await ctx.send_message(request)


@executor(id="process_plan")
async def process_plan(response: AgentExecutorResponse, ctx: WorkflowContext[WorkflowState]) -> None:
    """計画結果を処理"""
    await track_execution_path(ctx, "process_plan")
    await log_shared_state_to_trace(ctx, "process_plan")
    
    plan_text = response.agent_run_response.text.strip()
    plan_steps: list[PlanStep] = await get_shared_state_safe(ctx, STATE_KEY_PLAN_STEPS, [])
    
    print(f"\n{'='*60}")
    if not plan_steps:
        print("Initial Plan Generated:")
    else:
        print("Follow-up Plan Step Generated:")
    print(f"{'='*60}")
    print(plan_text)
    print(f"{'='*60}\n")
    
    # 番号を削除してステップを追加
    clean_text = re.sub(r"^\s*\d+[\.|\)]\s*", "", plan_text)
    next_number = len(plan_steps) + 1
    plan_steps.append(PlanStep(step_number=next_number, description=clean_text))
    
    await ctx.set_shared_state(STATE_KEY_PLAN_STEPS, plan_steps)
    await ctx.set_shared_state(STATE_KEY_ROUTER_ACTION, None)
    
    query = await get_shared_state_safe(ctx, STATE_KEY_QUERY, "")
    file_descriptions: list[FileDescription] = await get_shared_state_safe(ctx, STATE_KEY_FILE_DESCS, [])
    last_result = await get_shared_state_safe(ctx, STATE_KEY_LAST_RESULT, "")
    
    state = WorkflowState(
        query=query,
        file_descriptions=file_descriptions,
        plan_steps=plan_steps,
        last_result=last_result,
    )
    
    await ctx.send_message(state)


# ===== Phase 3: Code Generation & Execution =====


@executor(id="generate_and_execute_code")
async def generate_and_execute_code(message: WorkflowState | VerificationDecision, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    """コード生成と実行（通常コードのみ）"""
    await track_execution_path(ctx, "generate_and_execute_code")
    await log_shared_state_to_trace(ctx, "generate_and_execute_code")

    # すべて共有状態から取得するため、message型のチェックは不要
    # （型互換性のためだけにUnion型を受け入れる）

    iteration = await get_shared_state_safe(ctx, STATE_KEY_ITERATION, 0)
    print(f"\nPhase 3: Generating code (Iteration {iteration + 1})...")

    query = await get_shared_state_safe(ctx, STATE_KEY_QUERY, "")
    plan_steps: list[PlanStep] = await get_shared_state_safe(ctx, STATE_KEY_PLAN_STEPS, [])
    current_code = await get_shared_state_safe(ctx, STATE_KEY_CURRENT_CODE, "")
    file_descriptions: list[FileDescription] = await get_shared_state_safe(ctx, STATE_KEY_FILE_DESCS, [])
    last_result = await get_shared_state_safe(ctx, STATE_KEY_LAST_RESULT, "")

    file_info = "\n".join([
        f"- data/{desc.filename}: {desc.summary}" if desc.summary else f"- data/{desc.filename}"
        for desc in file_descriptions
    ]) or "- (no files provided)"

    if not plan_steps:
        print("✗ No plan steps available")
        execution = CodeExecution(code="", result="", success=False, error="No plan")
        await ctx.send_message(execution)
        return

    previous_plan_lines = [f"{step.step_number}. {step.description}" for step in plan_steps[:-1]]
    previous_plans_text = "\n".join(previous_plan_lines) if previous_plan_lines else "None"
    current_plan = plan_steps[-1].description if plan_steps else "Analyze the data"
    result_text = last_result.strip() if last_result else "(no execution result captured)"

    if iteration == 0:
        # 初回コード生成
        prompt = f"""# 与えられたデータ:
{file_info}

# 計画
{current_plan}

# 質問
{query}

# あなたのタスク
- 与えられたデータで計画を実装してください。
- **重要: 必ずprint()関数を使用して、計算結果や中間結果を標準出力に出力してください。**
- スクリプトが上記の質問に答えるために必要な出力をprintするようにしてください。
- 応答は単一のマークダウンPythonコード(```pythonでラップされたもの)である必要があります。
- 応答には追加の見出しやテキストを含めないでください。"""
    else:
        # 反復コード生成
        prompt = f"""あなたは専門的なデータアナリストです。あなたのタスクは、与えられたデータで次の計画を実装することです。

# 与えられたデータ:
{file_info}

# ベースコード
```python
{current_code}
```

# これまでの計画
{previous_plans_text}

# 実装する現在の計画
{current_plan}

# 最新の実行出力
{result_text or "(以前の実行出力はありません)"}

# 質問
{query}

# あなたのタスク
- ベースコードを更新して、現在の計画を完全に達成するようにしてください。
- **重要: 必ずprint()関数を使用して、計算結果や分析結果を標準出力に出力してください。**
- スクリプトが、集計されたメトリクスや結論を含む、ユーザーの質問に答えるために必要な出力をprintするようにしてください。
- 単にベースコードを繰り返すのではなく、新しい計画を組み込むように修正してください。
- 与えられたデータで現在の計画を実装してください。
- 実装はベースコードに基づいて行う必要があります。
- ベースコードはこれまでの計画の実装です。
- 応答は単一のマークダウンPythonコード(```pythonでラップされたもの)である必要があります。
- 応答には追加の見出しやテキストを含めないでください。"""

    # Coderエージェント呼び出し
    request = AgentExecutorRequest(
        messages=[ChatMessage(Role.USER, text=prompt)],
        should_respond=True
    )
    await ctx.send_message(request)


@executor(id="execute_generated_code")
async def execute_generated_code(response: AgentExecutorResponse, ctx: WorkflowContext[CodeExecution]) -> None:
    """生成されたコードを実行"""
    await track_execution_path(ctx, "execute_generated_code")
    await log_shared_state_to_trace(ctx, "execute_generated_code")
    code = extract_code_block(response.agent_run_response.text)
    
    if not code:
        print("✗ No code found in response")
        execution = CodeExecution(code="", result="", success=False, error="No code found")
        await ctx.send_message(execution)
        return
    
    print(f"\n{'='*60}")
    print(f"Generated Code ({len(code)} chars):")
    print(f"{'='*60}")
    print(code)
    print(f"{'='*60}")
    print("Executing...")
    
    # コード保存
    await ctx.set_shared_state(STATE_KEY_CURRENT_CODE, code)
    
    # コード実行
    result, success, error = execute_python_code(code)
    await ctx.set_shared_state(STATE_KEY_LAST_RESULT, result if success else f"ERROR: {error}")
    
    print(f"\n{'='*60}")
    if success:
        print("✓ Execution Success")
        print(f"{'='*60}")
        print("Output:")
        print(result)
        if not result.strip():
            print("⚠ WARNING: No output produced - print() statements may be missing")
    else:
        print("✗ Execution Failed")
        print(f"{'='*60}")
        print("Error:")
        print(error if error else 'Unknown error')
    print(f"{'='*60}\n")
    
    execution = CodeExecution(code=code, result=result, success=success, error=error)
    await ctx.send_message(execution)


# ===== Phase 4: Verification =====


@executor(id="verify")
async def verify(execution: CodeExecution, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    """検証（リクエスト作成と処理を統合） - 通常コードのみ対象"""
    await track_execution_path(ctx, "verify")
    await log_shared_state_to_trace(ctx, "verify")
    
    print("\nPhase 4: Verification...")
    
    query = await get_shared_state_safe(ctx, STATE_KEY_QUERY, "")
    plan_steps: list[PlanStep] = await get_shared_state_safe(ctx, STATE_KEY_PLAN_STEPS, [])
    plan_text = "\n".join([f"{step.step_number}. {step.description}" for step in plan_steps]) or "None"
    
    # 実行結果の取得と警告
    result_text = execution.result if execution.success else f"エラー: {execution.error}"
    if execution.success and not execution.result.strip():
        result_text = "(コードは成功したが出力がありません - print()文が不足している可能性があります)"
        print("⚠ WARNING: Code executed but produced no output")
    
    # 検証プロンプト
    prompt = f"""あなたは専門的なデータアナリストです。あなたのタスクは、現在の計画とそのコード実装が質問に答えるのに十分かどうかを確認することです。

# 計画
{plan_text}

# コード
```python
{execution.code}
```

# コードの実行結果
{result_text}

# 質問
{query}

# あなたのタスク
- 現在の計画とそのコード実装が質問に答えるのに十分かどうかを検証してください。
- **重要: 実行結果が空の場合や、具体的な数値が含まれていない場合は'No'と答えてください。**
- 応答は'Yes'または'No'のいずれかである必要があります。
- 質問に答えるのに十分であれば、'Yes'と答えてください。
- それ以外の場合は、'No'と答えてください。"""

    # Verifierエージェント呼び出し
    request = AgentExecutorRequest(
        messages=[ChatMessage(Role.USER, text=prompt)],
        should_respond=True
    )
    await ctx.send_message(request)


@executor(id="process_verification")
async def process_verification(response: AgentExecutorResponse, ctx: WorkflowContext[VerificationDecision]) -> None:
    """検証結果を処理"""
    await track_execution_path(ctx, "process_verification")
    await log_shared_state_to_trace(ctx, "process_verification")
    
    raw_text = response.agent_run_response.text.strip()
    text = raw_text.upper()
    is_sufficient = "YES" in text and "NO" not in text
    
    query = await get_shared_state_safe(ctx, STATE_KEY_QUERY, "")
    iteration = await get_shared_state_safe(ctx, STATE_KEY_ITERATION, 0)
    file_descriptions: list[FileDescription] = await get_shared_state_safe(ctx, STATE_KEY_FILE_DESCS, [])
    plan_steps: list[PlanStep] = await get_shared_state_safe(ctx, STATE_KEY_PLAN_STEPS, [])
    last_result = await get_shared_state_safe(ctx, STATE_KEY_LAST_RESULT, "")
    
    print(f"\n{'='*60}")
    print("Verifier Response:")
    print(f"{'='*60}")
    print(raw_text)
    print(f"{'='*60}")
    print(f"Result: {'✓ SUFFICIENT (Yes)' if is_sufficient else '✗ INSUFFICIENT (No)'}")
    print(f"{'='*60}\n")
    
    # 反復回数インクリメント
    iteration += 1
    await ctx.set_shared_state(STATE_KEY_ITERATION, iteration)
    
    # 最大反復チェック
    if iteration >= MAX_ITERATIONS:
        print(f"\n⚠ Max iterations ({MAX_ITERATIONS}) reached")
        is_sufficient = True
    
    state = WorkflowState(
        query=query,
        file_descriptions=file_descriptions,
        plan_steps=plan_steps,
        last_result=last_result,
        iteration=iteration,
        is_sufficient=is_sufficient
    )
    
    decision = VerificationDecision(is_sufficient=is_sufficient, state=state)
    await ctx.send_message(decision)


# ===== Phase 5: Routing =====


@executor(id="route")
async def route(decision: VerificationDecision, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    """ルーティング（リクエスト作成と処理を統合）"""
    await track_execution_path(ctx, "route")
    await log_shared_state_to_trace(ctx, "route")
    
    print("\nPhase 5: Routing - deciding next action...")
    
    file_descriptions: list[FileDescription] = await get_shared_state_safe(ctx, STATE_KEY_FILE_DESCS, decision.state.file_descriptions)
    plan_steps: list[PlanStep] = await get_shared_state_safe(ctx, STATE_KEY_PLAN_STEPS, decision.state.plan_steps)
    last_result = await get_shared_state_safe(ctx, STATE_KEY_LAST_RESULT, decision.state.last_result)
    
    file_info = "\n".join([
        f"- data/{desc.filename}: {desc.summary}" if desc.summary else f"- data/{desc.filename}"
        for desc in file_descriptions
    ]) or "- (no files provided)"
    plan_text = "\n".join([f"{step.step_number}. {step.description}" for step in plan_steps]) or "None"
    result_text = last_result.strip() if last_result else "(no result captured)"
    
    # ルータープロンプト
    prompt = f"""あなたは専門的なデータアナリストです。現在の計画は質問に答えるには不十分なので、あなたのタスクは質問に答えるために計画をどのように改善するかを決定することです。

# 質問
{decision.state.query}

# 与えられたデータ:
{file_info}

# 現在の計画
{plan_text}

# 現在の計画から得られた結果:
{result_text}

# あなたのタスク
- 現在の計画のステップの1つが間違っていると思う場合は、次のオプションから答えてください: Step 1, Step 2, ..., Step K。
- 新しい次のステップを実行する必要があると思う場合は、'Add Step'と答えてください。
- 応答はStep 1 - Step Kまたは'Add Step'のみである必要があります。"""

    # Routerエージェント呼び出し
    request = AgentExecutorRequest(
        messages=[ChatMessage(Role.USER, text=prompt)],
        should_respond=True
    )
    await ctx.send_message(request)


@executor(id="process_routing")
async def process_routing(response: AgentExecutorResponse, ctx: WorkflowContext[WorkflowState]) -> None:
    """ルーティング結果を処理"""
    await track_execution_path(ctx, "process_routing")
    await log_shared_state_to_trace(ctx, "process_routing")
    
    text = response.agent_run_response.text.strip()
    
    query = await get_shared_state_safe(ctx, STATE_KEY_QUERY, "")
    file_descriptions: list[FileDescription] = await get_shared_state_safe(ctx, STATE_KEY_FILE_DESCS, [])
    plan_steps: list[PlanStep] = await get_shared_state_safe(ctx, STATE_KEY_PLAN_STEPS, [])
    last_result = await get_shared_state_safe(ctx, STATE_KEY_LAST_RESULT, "")
    
    print(f"\n{'='*60}")
    print("Router Response:")
    print(f"{'='*60}")
    print(text)
    print(f"{'='*60}")
    
    normalized = text.lower()
    step_match = re.search(r"step\s*(\d+)", normalized)
    router_action = "add"

    if "add" in normalized:
        router_action = "add"
        print("→ Action: Add new step")
    elif step_match:
        target_index = int(step_match.group(1))
        if 1 <= target_index <= len(plan_steps):
            router_action = "truncate"
            plan_steps = plan_steps[: target_index - 1]
            for idx, step in enumerate(plan_steps, start=1):
                step.step_number = idx
            print(f"→ Action: Truncate to step {target_index - 1}")
        else:
            router_action = "add"
            print(f"⚠ Invalid step {target_index}, defaulting to Add Step")
    else:
        router_action = "add"
        print("⚠ Unrecognized response, defaulting to Add Step")

    await ctx.set_shared_state(STATE_KEY_PLAN_STEPS, plan_steps)
    await ctx.set_shared_state(STATE_KEY_ROUTER_ACTION, router_action)
    
    state = WorkflowState(
        query=query,
        file_descriptions=file_descriptions,
        plan_steps=plan_steps,
        last_result=last_result,
    )
    
    await ctx.send_message(state)


# ===== Conditional Functions =====


def is_sufficient(message: Any) -> bool:
    """十分条件"""
    if isinstance(message, VerificationDecision):
        return message.is_sufficient
    return False


def is_insufficient(message: Any) -> bool:
    """不十分条件"""
    return not is_sufficient(message)


def has_unanalyzed_files(state: WorkflowState) -> bool:
    """未分析ファイルが存在するか判定"""
    return any(not desc.summary for desc in state.file_descriptions)


def analysis_complete(state: WorkflowState) -> bool:
    """ファイル分析が完了したか判定"""
    return not has_unanalyzed_files(state)


# ===== Finalization =====


@executor(id="prepare_final_answer")
async def prepare_final_answer(decision: VerificationDecision, ctx: WorkflowContext[VerificationDecision]) -> None:
    """最終コード実行後の処理"""
    await track_execution_path(ctx, "prepare_final_answer")
    await log_shared_state_to_trace(ctx, "prepare_final_answer")

    query = await get_shared_state_safe(ctx, STATE_KEY_QUERY, decision.state.query)
    plan_steps: list[PlanStep] = await get_shared_state_safe(ctx, STATE_KEY_PLAN_STEPS, decision.state.plan_steps)
    file_descriptions: list[FileDescription] = await get_shared_state_safe(ctx, STATE_KEY_FILE_DESCS, decision.state.file_descriptions)
    iteration = await get_shared_state_safe(ctx, STATE_KEY_ITERATION, decision.state.iteration)
    last_result = await get_shared_state_safe(ctx, STATE_KEY_LAST_RESULT, decision.state.last_result)

    answer = last_result.strip()
    if not answer:
        answer = "(Code executed successfully but produced no output.)"

    state = WorkflowState(
        query=query,
        plan_steps=plan_steps,
        file_descriptions=file_descriptions,
        last_result=answer,
        iteration=iteration,
        is_sufficient=True,
        is_final=True
    )

    final_decision = VerificationDecision(is_sufficient=True, state=state)
    await ctx.send_message(final_decision)


@executor(id="finalize")
async def finalize(decision: VerificationDecision, ctx: WorkflowContext[Never, str]) -> None:
    """最終化"""
    await track_execution_path(ctx, "finalize")
    await log_shared_state_to_trace(ctx, "finalize")
    
    execution_path: list[str] = await get_shared_state_safe(ctx, STATE_KEY_EXECUTION_PATH, [])
    answer = decision.state.last_result or "(No answer generated)"
    
    print(f"\n{'='*80}")
    print("DS-STAR v5 Analysis Complete")
    print(f"Total iterations: {decision.state.iteration}")
    print(f"Final answer:\n{answer}")
    print(f"{'='*80}")
    
    # 実行パス表示
    print(f"\n{'='*80}")
    print("EXECUTION PATH TRACE")
    print(f"{'='*80}")
    print(f"Total steps executed: {len(execution_path)}")
    print("\nExecution sequence:")
    for i, step in enumerate(execution_path, 1):
        print(f"  {i:3d}. {step}")
    print(f"{'='*80}\n")

    report = (
        f"Analysis completed in {decision.state.iteration} iterations for query: {decision.state.query}.\n"
        f"Answer:\n{answer}\n\n"
        f"Execution Path ({len(execution_path)} steps):\n" + 
        "\n".join([f"  {i}. {step}" for i, step in enumerate(execution_path, 1)])
    )
    await ctx.yield_output(report)


# ===== Workflow Builder =====

def build_workflow() -> Any:
    """ワークフローを構築（v5 最適化版）"""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    deployment_name = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    chat_client = AzureOpenAIChatClient(
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key,
    )
    
    # 5つの統合エージェント
    analyzer = ChatAgent(
        id="analyzer",
        name="analyzer",
        instructions=(
            "あなたは専門的なデータアナリストです。与えられたデータファイルを読み込んで説明するPythonスクリプトを生成してください。"
            "重要な情報を出力し、ダミーの内容を避け、スクリプトをtry/exceptでラップしないでください。"
        ),
        chat_client=chat_client,
    )
    
    planner = ChatAgent(
        id="planner",
        name="planner",
        instructions=(
            "あなたは専門的なデータサイエンス計画者です。"
            "初期計画の作成、および以前の結果に基づく計画の拡張・修復を行います。"
            "シンプルで実行可能な分析計画をステップバイステップで作成してください。"
        ),
        chat_client=chat_client,
    )
    
    coder = ChatAgent(
        id="coder",
        name="coder",
        instructions=(
            "あなたは専門的なデータサイエンティストです。"
            "通常のコード生成と最終回答コードの両方を生成できます。"
            "データ分析のためのクリーンで正確なPythonコードを生成してください。"
            "重要: 生成するコードには必ずprint()文を含めて、計算結果や中間結果を標準出力に表示してください。"
            "結果を変数に代入するだけでなく、必ずprint()で出力してください。"
        ),
        chat_client=chat_client,
    )
    
    verifier = ChatAgent(
        id="verifier",
        name="verifier",
        instructions="あなたは検証の専門家です。分析結果がクエリに十分に答えているかどうかを評価してください。'Yes'または'No'のみで答えてください。",
        chat_client=chat_client,
    )
    
    router = ChatAgent(
        id="router",
        name="router",
        instructions="あなたは計画の専門家です。分析計画に新しいステップを追加するか、既存のステップを修正するかを決定してください。",
        chat_client=chat_client,
    )
    
    # ワークフロー構築（v5 最適化版）
    workflow = (
        WorkflowBuilder()
        .set_start_executor(initialize)
        
        # Phase 1: File Analysis Loop
        .add_edge(initialize, analyze_files, condition=has_unanalyzed_files)
        .add_edge(initialize, plan, condition=analysis_complete)
        .add_edge(analyze_files, analyzer)
        .add_edge(analyzer, process_analysis)
        .add_edge(process_analysis, analyze_files, condition=has_unanalyzed_files)
        .add_edge(process_analysis, plan, condition=analysis_complete)
        
        # Phase 2: Planning
        .add_edge(plan, planner)
        .add_edge(planner, process_plan)
        
        # Phase 3: Code Generation & Execution
        .add_edge(process_plan, generate_and_execute_code)
        .add_edge(generate_and_execute_code, coder)
        .add_edge(coder, execute_generated_code)
        
        # Phase 4: Verification (通常コードの場合)
        .add_edge(execute_generated_code, verify)
        .add_edge(verify, verifier)
        .add_edge(verifier, process_verification)
        
        # Conditional: Sufficient (最終コード生成へ) or Insufficient (ルーティングへ)
        .add_edge(process_verification, prepare_final_answer, condition=is_sufficient)
        .add_edge(process_verification, route, condition=is_insufficient)
        
        # Phase 5: Routing & Plan Refinement
        .add_edge(route, router)
        .add_edge(router, process_routing)
        .add_edge(process_routing, plan)  # 新しい計画ステップ生成へ
        
        # Phase 6: Final Answer (最終コード実行後)
        .add_edge(prepare_final_answer, finalize)
        
        .build()
    )
    
    return workflow


# ===== Main =====


async def main() -> None:
    """メイン関数"""
    query = "各カテゴリの総売上高はいくらですか？また、最も売れているカテゴリを特定してください。"

    # エージェントトレース オブザーバビリティ設定
    os.environ["OTEL_METRICS_EXPORTER"] = "none"
    os.environ["OTEL_LOGS_EXPORTER"] = "none"
    
    try:
        setup_observability(
            enable_sensitive_data=True,
            otlp_endpoint="http://localhost:4317",
        )
    except Exception as e:
        print(f"Warning: Observability setup failed: {e}")
        print("Continuing without observability...")

    tracer = get_tracer()

    workflow = build_workflow()
    with tracer.start_as_current_span("DSStar_v5"):
        events = await workflow.run(query)
        outputs = events.get_outputs()

        if outputs:
            print(f"\nFinal Output:\n{outputs[0]}\n")
        
        print("✓ DS-STAR v5 Workflow completed")


if __name__ == "__main__":
    asyncio.run(main())
