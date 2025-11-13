# DS-STAR Data Science Multi-Agent Samples

データサイエンスエージェント - 反復的な計画と検証による実装

## 概要

このプロジェクトは、論文「[DS-STAR: Data Science Agent via Iterative Planning and Verification](https://arxiv.org/abs/2509.21825v3)」に基づいたデータ分析ワークフローの最適化実装です。

Azure OpenAI Service と Microsoft Agent Framework を活用し、データファイルの分析から最終回答の生成まで、複数の AI エージェントが協調して動作します。

## 主な機能

### 5つの専門エージェント

1. **Analyzer（分析者）**
   - データファイルの内容を分析
   - 構造化データと非構造化データに対応
   - ファイル情報の要約を生成

2. **Planner（計画者）**
   - データ分析の計画を立案
   - 段階的な分析ステップを生成
   - 結果に基づき計画を拡張

3. **Coder（コーダー）**
   - Pythonコードの生成
   - 分析計画をコード実装に変換
   - 結果を標準出力に出力するコードを作成

4. **Verifier（検証者）**
   - 実行結果の検証
   - クエリへの回答が十分かどうかを判定
   - Yes/Noで判定結果を返す

5. **Router（ルーター）**
   - 計画の修正戦略を決定
   - 新しいステップの追加または既存ステップの修正を判断
   - ワークフローの方向性を制御

### ワークフローの流れ

```
1. 初期化 & ファイル分析
   ↓
2. 計画の作成
   ↓
3. コード生成 & 実行
   ↓
4. 検証
   ↓
5. 十分? → Yes → 最終回答
          → No  → ルーティング → 計画修正 → (3に戻る)
```

## セットアップ

### 前提条件

- Python 3.10 以上
- Azure OpenAI Service モデル
- OpenTelemetry Collector（オプション、トレーシング用）

### インストール

1. リポジトリのクローン:
```bash
git clone <repository-url>
cd ds_star_microsoft_agent_framework
```

2. 依存パッケージのインストール:
```bash
pip install agent-framework --pre
pip install python-dotenv
pip install pandas numpy
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc opentelemetry-instrumentation-openai
```

3. 環境変数の設定:
```bash
# .env.sampleをコピー
copy .env.sample .env

# .envファイルを編集して、Azure OpenAIの認証情報を設定
```

### 環境変数

`.env`ファイルに以下の環境変数を設定してください:

```env
AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=<your-deployment-name>
AZURE_OPENAI_API_KEY=<your-api-key>
```

詳細は `.env.sample` を参照してください。

## 使用方法

### 基本的な使い方

1. **データファイルの準備**:
   ```bash
   # dataディレクトリを作成
   mkdir data
   
   # 分析したいデータファイルを配置
   # 例: data/sales.csv, data/products.json など
   ```

2. **クエリの設定**:
   `ds_star_workflow.py`の`main()`関数内でクエリを設定:
   ```python
   async def main() -> None:
       query = "各カテゴリの総売上高はいくらですか？また、最も売れているカテゴリを特定してください。"
       # ...
   ```

3. **ワークフローの実行**:
   ```bash
   python ds_star_workflow.py
   ```

### 実行例

```bash
$ python ds_star_workflow.py

================================================================================
DS-STAR v5 (Optimized): Iterative Planning and Verification
Query: 各カテゴリの総売上高はいくらですか？また、最も売れているカテゴリを特定してください。
================================================================================

Phase 1: Analyzing data files...
Found 1 files
Analyzing: sales.csv
...

Phase 2: Creating initial plan...
...

Phase 3: Generating code (Iteration 1)...
...

Phase 4: Verification...
...

================================================================================
DS-STAR v5 Analysis Complete
Total iterations: 3
Final answer:
カテゴリA: 15000円
カテゴリB: 23000円
カテゴリC: 18500円
最も売れているカテゴリ: カテゴリB
================================================================================
```

## 設定

### 最大反復回数

ワークフローの最大反復回数は `MAX_ITERATIONS` 定数で設定できます（デフォルト: 10）:

```python
MAX_ITERATIONS = 10
```

### オブザーバビリティ

OpenTelemetry によるトレーシングが有効化されています。

- デフォルトエンドポイント: `http://localhost:4317`
- トレース、ログ、メトリクスの詳細な記録
- 各エージェントの実行パスを追跡

オブザーバビリティを無効化する場合は、`main()`関数の`setup_observability()`呼び出しをコメントアウトしてください。

#### 参考
https://qiita.com/nohanaga/items/6e0a42716e86eea58091

## ライセンス

MIT License

## 参考文献

- [DS-STAR: Data Science Agent via Iterative Planning and Verification](https://arxiv.org/abs/2509.21825v3)
- [Microsoft Agent Framework Documentation](https://learn.microsoft.com/ja-jp/agent-framework/overview/agent-framework-overview)

## 貢献

バグ報告や機能リクエストは、GitHub の Issue でお願いします。

---

**Note**: このプロジェクトは Microsoft Agent Framework を使用しており、Azure OpenAI Service へのアクセスが必要です。
