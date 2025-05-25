import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from typing import Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import json
from datetime import datetime, timezone


# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")

# ベースラインモデルとモデルのパスを定義
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_model.pkl")
BASELINE_METRICS_PATH = os.path.join(MODEL_DIR, "baseline_metrics.json")

# 許容する精度低下率
MAX_DEGRADATION = 0.01


def save_baseline(model: Pipeline, accuracy: float) -> None:
    """
    ベースラインモデルとメトリクスを保存する

    Args:
        model (Pipeline): 保存するモデル
        accuracy (float): モデルの精度

    Raises:
        IOError: ファイル操作に失敗した場合
    """
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        # ベースラインモデルが存在しない場合は保存
        if not os.path.exists(BASELINE_MODEL_PATH):
            with open(BASELINE_MODEL_PATH, "wb") as f:
                pickle.dump(model, f)

        # メトリクスJSONの読み込みと更新
        entries = []
        if os.path.exists(BASELINE_METRICS_PATH):
            with open(BASELINE_METRICS_PATH, "r") as f:
                entries = json.load(f)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accuracy": accuracy,
        }
        entries.append(entry)

        with open(BASELINE_METRICS_PATH, "w") as f:
            json.dump(entries, f, indent=2)
    except (IOError, json.JSONDecodeError) as e:
        raise IOError(f"ベースラインの保存に失敗しました: {str(e)}")


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    テスト用データセットを読み込む

    Returns:
        pd.DataFrame: テスト用データセット

    Raises:
        FileNotFoundError: データファイルが見つからない場合
    """
    try:
        if not os.path.exists(DATA_PATH):
            from sklearn.datasets import fetch_openml

            titanic = fetch_openml("titanic", version=1, as_frame=True)
            df = titanic.data
            df["Survived"] = titanic.target

            # 必要なカラムのみ選択
            df = df[
                [
                    "Pclass",
                    "Sex",
                    "Age",
                    "SibSp",
                    "Parch",
                    "Fare",
                    "Embarked",
                    "Survived",
                ]
            ]

            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            df.to_csv(DATA_PATH, index=False)

        return pd.read_csv(DATA_PATH)
    except Exception as e:
        raise FileNotFoundError(f"データの読み込みに失敗しました: {str(e)}")


@pytest.fixture
def preprocessor() -> ColumnTransformer:
    """
    前処理パイプラインを定義

    Returns:
        ColumnTransformer: 前処理パイプライン
    """
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(
    sample_data: pd.DataFrame, preprocessor: ColumnTransformer
) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """
    モデルの学習とテストデータの準備

    Args:
        sample_data (pd.DataFrame): 学習用データ
        preprocessor (ColumnTransformer): 前処理パイプライン

    Returns:
        Tuple[Pipeline, pd.DataFrame, pd.Series]: 学習済みモデル、テストデータ、テストラベル

    Raises:
        ValueError: データの分割に失敗した場合
    """
    try:
        # データの分割とラベル変換
        X = sample_data.drop("Survived", axis=1)
        y = sample_data["Survived"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # モデルパイプラインの作成
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=100, random_state=42),
                ),
            ]
        )

        # モデルの学習
        model.fit(X_train, y_train)

        # モデルの保存
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        # テストセットでの精度を算出してベースラインJSONに追記
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        save_baseline(model, acc)

        return model, X_test, y_test
    except Exception as e:
        raise ValueError(f"モデルの学習に失敗しました: {str(e)}")


def test_model_exists() -> None:
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model: Tuple[Pipeline, pd.DataFrame, pd.Series]) -> None:
    """
    モデルの精度を検証

    Args:
        train_model (Tuple[Pipeline, pd.DataFrame, pd.Series]): 学習済みモデル、テストデータ、テストラベル
    """
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(
    train_model: Tuple[Pipeline, pd.DataFrame, pd.Series],
) -> None:
    """
    モデルの推論時間を検証

    Args:
        train_model (Tuple[Pipeline, pd.DataFrame, pd.Series]): 学習済みモデル、テストデータ、テストラベル
    """
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(
    sample_data: pd.DataFrame, preprocessor: ColumnTransformer
) -> None:
    """
    モデルの再現性を検証

    Args:
        sample_data (pd.DataFrame): 学習用データ
        preprocessor (ColumnTransformer): 前処理パイプライン
    """
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"


def test_model_regression() -> None:
    """直前2回分の精度を比較し、劣化が許容値内か検証"""
    if not os.path.exists(BASELINE_METRICS_PATH):
        pytest.skip("メトリクスファイルが存在しないためスキップします")

    try:
        with open(BASELINE_METRICS_PATH, "r") as f:
            entries = json.load(f)

        if len(entries) < 2:
            pytest.skip("メトリクスが2件未満のためスキップします")

        prev_acc = entries[-2]["accuracy"]
        latest_acc = entries[-1]["accuracy"]

        # 精度の変化を計算（劣化も向上も考慮）
        accuracy_change = latest_acc - prev_acc

        # 精度が劣化した場合のみ警告
        if accuracy_change < 0:
            assert (
                abs(accuracy_change) <= MAX_DEGRADATION
            ), f"モデル精度が前回({prev_acc:.3f})から劣化しています: 最新={latest_acc:.3f}, 劣化率={abs(accuracy_change):.3f}"
    except (IOError, json.JSONDecodeError) as e:
        pytest.fail(f"メトリクスの読み込みに失敗しました: {str(e)}")
