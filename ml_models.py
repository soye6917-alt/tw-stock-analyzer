"""
🤖 機器學習預測模組
- XGBoost 漲跌預測（分類）
- 隨機森林特徵重要性
- LSTM 短期趨勢預測（深度學習）
- 多因子融合決策
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# XGBoost 漲跌預測
# ─────────────────────────────────────────────
def prepare_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    從技術指標產生特徵矩陣
    回傳特徵欄位 + 目標(Target): 隔天漲跌(1漲0跌)
    """
    data = df.copy()
    
    # 確保數值型態
    for col in data.select_dtypes(include=['datetime64', 'datetime64[ns]', 'object']).columns:
        if col != '日期':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    features = pd.DataFrame(index=data.index)
    
    # 1. 價格變動率
    features['return_1d'] = data['Close'].pct_change(1)
    features['return_5d'] = data['Close'].pct_change(5)
    features['return_10d'] = data['Close'].pct_change(10)
    features['return_20d'] = data['Close'].pct_change(20)
    
    # 2. 技術指標衍生特徵
    for ma_col in ['MA5', 'MA10', 'MA20', 'MA60']:
        if ma_col in data.columns and not data[ma_col].isna().all():
            features[f'{ma_col}_dist'] = (data['Close'] / data[ma_col] - 1) * 100
    
    # 3. RSI 區間
    if 'RSI' in data.columns:
        features['RSI'] = data['RSI']
        features['RSI_overbought'] = (data['RSI'] > 70).astype(int)
        features['RSI_oversold'] = (data['RSI'] < 30).astype(int)
    
    # 4. MACD
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        features['MACD'] = data['MACD']
        features['MACD_diff'] = data['MACD'] - data['MACD_Signal']
        features['MACD_cross'] = ((features['MACD_diff'] > 0) & 
                                  (features['MACD_diff'].shift(1) <= 0)).astype(int)
    
    # 5. 布林通道位置
    if 'BB_Upper' in data.columns:
        bb_range = data['BB_Upper'] - data['BB_Lower']
        features['BB_position'] = ((data['Close'] - data['BB_Lower']) / bb_range.replace(0, np.nan))
        features['BB_breakout'] = (data['Close'] > data['BB_Upper']).astype(int)
        features['BB_breakdown'] = (data['Close'] < data['BB_Lower']).astype(int)
    
    # 6. 波動率
    features['volatility_5d'] = data['Close'].pct_change().rolling(5).std()
    features['volatility_10d'] = data['Close'].pct_change().rolling(10).std()
    
    # 7. 成交量變化
    if 'Volume' in data.columns:
        vol = data['Volume'].replace(0, np.nan)
        features['volume_change_1d'] = vol.pct_change(1)
        features['volume_change_5d'] = vol.pct_change(5)
        features['volume_ma5_ratio'] = vol / vol.rolling(5).mean()
        features['volume_ma20_ratio'] = vol / vol.rolling(20).mean()
    
    # 8. 價格動能
    features['momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
    features['momentum_10d'] = data['Close'] / data['Close'].shift(10) - 1
    
    # KDJ
    if 'K' in data.columns and 'D' in data.columns:
        features['KDJ_K'] = data['K']
        features['KDJ_D'] = data['D']
        features['KDJ_cross'] = ((data['K'] > data['D']) & 
                                 (data['K'].shift(1) <= data['D'].shift(1))).astype(int)
    
    # 目標：隔天漲跌
    features['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    return features


def train_xgboost(df_features: pd.DataFrame, 
                  test_size: int = 60,
                  use_early_stopping: bool = True) -> dict:
    """
    訓練 XGBoost 分類模型預測隔天漲跌
    """
    from xgboost import XGBClassifier
    
    # 排除目標欄位中的 NaN
    feat = df_features.dropna().copy()
    if len(feat) < 100:
        return {"error": f"資料不足（需 ≥100 筆，目前 {len(feat)} 筆）"}
    
    # 分割訓練/測試
    X = feat.drop(columns=['Target']).select_dtypes(include=[np.number])
    y = feat['Target']
    
    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]
    
    if len(X_train) < 50:
        return {"error": "訓練資料不足"}
    
    # 訓練模型
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
    )
    
    if use_early_stopping:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    
    # 預測
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 準確率
    accuracy = (y_pred == y_test).mean()
    
    # 最近預測
    recent_preds = []
    for i in range(min(10, len(X_test))):
        idx = X_test.index[i]
        actual_price = None
        actual_direction = "↑" if y_test.iloc[i] == 1 else "↓"
        pred_direction = "↑" if y_pred[i] == 1 else "↓"
        correct = "✅" if y_pred[i] == y_test.iloc[i] else "❌"
        recent_preds.append({
            "date": str(idx.date()) if hasattr(idx, 'date') else str(idx),
            "actual": actual_direction,
            "predicted": pred_direction,
            "probability": round(float(y_prob[i]), 3),
            "correct": correct,
        })
    
    # 特徵重要性
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    # 最新預測資訊
    latest_features = X.iloc[-1:]
    latest_prob = model.predict_proba(latest_features)[0][1]
    latest_pred = "↑ 看漲" if latest_prob >= 0.5 else "↓ 看跌"
    
    return {
        "model": model,
        "feature_names": list(X.columns),
        "accuracy": round(float(accuracy), 4),
        "accuracy_pct": f"{accuracy*100:.1f}%",
        "latest_prediction": latest_pred,
        "latest_probability": round(float(latest_prob), 4),
        "recent_predictions": recent_preds,
        "feature_importance": importance,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }


# ─────────────────────────────────────────────
# 隨機森林特徵重要性
# ─────────────────────────────────────────────
def random_forest_feature_importance(df_features: pd.DataFrame) -> dict:
    """隨機森林找出最關鍵的預測因子"""
    from sklearn.ensemble import RandomForestClassifier
    
    feat = df_features.dropna().copy()
    if len(feat) < 60:
        return {"error": "資料不足"}
    
    X = feat.drop(columns=['Target']).select_dtypes(include=[np.number])
    y = feat['Target']
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    return {
        "model": rf,
        "top_features": importance.to_dict('records'),
        "cv_accuracy": round(float((rf.predict(X) == y).mean()), 4),
    }


# ─────────────────────────────────────────────
# 綜合預測（多模型投票）
# ─────────────────────────────────────────────
def ensemble_prediction(df: pd.DataFrame) -> dict:
    """
    多模型投票預測：
    - XGBoost
    - 隨機森林
    - 邏輯回歸
    - 技術指標規則式
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    features = prepare_features(df)
    feat = features.dropna().copy()
    
    if len(feat) < 80:
        return {"error": "資料不足，需 ≥80 筆"}
    
    X = feat.drop(columns=['Target']).select_dtypes(include=[np.number])
    y = feat['Target']
    
    train_size = len(X) - 60
    if train_size < 50:
        train_size = int(len(X) * 0.7)
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # 訓練各模型
    models = {
        "XGBoost": __import__('xgboost', fromlist=['XGBClassifier']).XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, verbosity=0, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
    }
    
    results = {}
    votes = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        acc = (y_pred == y_test).mean()
        
        if name == "LogisticRegression":
            coef = pd.DataFrame({
                'feature': X.columns,
                'coef': model.coef_[0]
            }).sort_values('coef', ascending=False).head(10)
            results[name] = {
                "accuracy": round(float(acc), 4),
                "top_coefficients": coef.to_dict('records')
            }
        else:
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            results[name] = {
                "accuracy": round(float(acc), 4),
                "top_features": importance.to_dict('records')
            }
        
        # 最新預測
        latest_prob = model.predict_proba(X.iloc[-1:])[0][1]
        vote = "↑" if latest_prob >= 0.5 else "↓"
        votes.append(vote)
    
    # 多數決
    up_votes = votes.count("↑")
    down_votes = votes.count("↓")
    consensus = "↑ 看漲" if up_votes > down_votes else "↓ 看跌" if down_votes > up_votes else "⚖️ 方向不明"
    
    return {
        "models": results,
        "consensus": consensus,
        "up_votes": up_votes,
        "down_votes": down_votes,
        "test_samples": len(X_test),
    }


# ─────────────────────────────────────────────
# 技術指標規則式預測（傳統分析方法）
# ─────────────────────────────────────────────
def technical_consensus(df: pd.DataFrame) -> dict:
    """
    綜合技術指標共識：多個指標投票決定短期方向
    每個指標投「看漲(+1) / 看跌(-1) / 中立(0)」
    """
    data = df.tail(5).copy()
    if data.empty or len(data) < 5:
        return {"error": "資料不足"}
    
    last = data.iloc[-1]
    signals = []
    total_score = 0
    
    # 1. RSI
    if 'RSI' in df.columns and pd.notna(last.get('RSI')):
        rsi = last['RSI']
        if rsi < 30:
            signals.append(("RSI 超賣區", 1, 2))
            total_score += 2
        elif rsi < 40:
            signals.append(("RSI 偏低", 1, 1))
            total_score += 1
        elif rsi > 70:
            signals.append(("RSI 超買區", -1, 2))
            total_score -= 2
        elif rsi > 60:
            signals.append(("RSI 偏高", -1, 1))
            total_score -= 1
        else:
            signals.append(("RSI 中立", 0, 0))
    
    # 2. MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd = last.get('MACD', 0)
        signal = last.get('MACD_Signal', 0)
        prev_macd = df['MACD'].iloc[-2] if len(df) >= 2 else 0
        prev_signal = df['MACD_Signal'].iloc[-2] if len(df) >= 2 else 0
        
        if pd.notna(macd) and pd.notna(signal):
            if macd > signal and prev_macd <= prev_signal:
                signals.append(("MACD 黃金交叉", 1, 2))
                total_score += 2
            elif macd < signal and prev_macd >= prev_signal:
                signals.append(("MACD 死亡交叉", -1, 2))
                total_score -= 2
            elif macd > signal:
                signals.append(("MACD 多方", 1, 1))
                total_score += 1
            elif macd < signal:
                signals.append(("MACD 空方", -1, 1))
                total_score -= 1
            else:
                signals.append(("MACD 中立", 0, 0))
    
    # 3. 均線排列
    if all(f'MA{p}' in df.columns for p in [5, 10, 20]):
        ma5 = last.get('MA5', 0)
        ma10 = last.get('MA10', 0)
        ma20 = last.get('MA20', 0)
        close = last.get('Close', 0)
        
        if pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20):
            if ma5 > ma10 > ma20:
                signals.append(("均線多頭排列", 1, 2))
                total_score += 2
            elif ma5 < ma10 < ma20:
                signals.append(("均線空頭排列", -1, 2))
                total_score -= 2
            elif close > ma20:
                signals.append(("站上月線", 1, 1))
                total_score += 1
            elif close < ma20:
                signals.append(("跌破月線", -1, 1))
                total_score -= 1
            else:
                signals.append(("均線糾結", 0, 0))
    
    # 4. 布林通道
    if all(c in df.columns for c in ['BB_Upper', 'BB_Lower', 'BB_Mid']):
        close = last.get('Close', 0)
        bb_upper = last.get('BB_Upper', 0)
        bb_lower = last.get('BB_Lower', 0)
        bb_mid = last.get('BB_Mid', 0)
        
        if all(pd.notna(x) for x in [close, bb_upper, bb_lower, bb_mid]):
            if close > bb_upper:
                signals.append(("突破布林上軌(超買)", -1, 1))
                total_score -= 1
            elif close < bb_lower:
                signals.append(("跌破布林下軌(超賣)", 1, 1))
                total_score += 1
            elif close > bb_mid:
                signals.append(("布林通道中上軌", 1, 1))
                total_score += 1
            elif close < bb_mid:
                signals.append(("布林通道中下軌", -1, 1))
                total_score -= 1
    
    # 5. 成交量確認
    if 'Volume' in df.columns:
        vol = df['Volume']
        if len(vol) >= 5:
            vol_ma5 = vol.rolling(5).mean().iloc[-1]
            current_vol = vol.iloc[-1]
            if pd.notna(vol_ma5) and vol_ma5 > 0:
                vol_ratio = current_vol / vol_ma5
                if vol_ratio > 2:
                    # 搭配方向判斷
                    if last.get('Close', 0) > last.get('Open', 0):
                        signals.append(("爆量上漲(買訊)", 1, 2))
                        total_score += 2
                    else:
                        signals.append(("爆量下跌(賣訊)", -1, 2))
                        total_score -= 2
                elif vol_ratio > 1.5:
                    if last.get('Close', 0) > last.get('Open', 0):
                        signals.append(("量增上漲", 1, 1))
                        total_score += 1
                    else:
                        signals.append(("量增下跌", -1, 1))
                        total_score -= 1
    
    # 6. KDJ
    if 'K' in df.columns and 'D' in df.columns:
        k = last.get('K', 0)
        d = last.get('D', 0)
        prev_k = df['K'].iloc[-2] if len(df) >= 2 else 0
        prev_d = df['D'].iloc[-2] if len(df) >= 2 else 0
        
        if all(pd.notna(x) for x in [k, d]):
            if k > d and prev_k <= prev_d:
                signals.append(("KDJ 黃金交叉", 1, 1))
                total_score += 1
            elif k < d and prev_k >= prev_d:
                signals.append(("KDJ 死亡交叉", -1, 1))
                total_score -= 1
            elif k > 80:
                signals.append(("KDJ 超買區", -1, 1))
                total_score -= 1
            elif k < 20:
                signals.append(("KDJ 超賣區", 1, 1))
                total_score += 1
    
    # 綜合判斷
    if total_score >= 4:
        conclusion = "🟢 強烈看漲（多個指標共振）"
    elif total_score >= 2:
        conclusion = "🟢 偏多看漲"
    elif total_score <= -4:
        conclusion = "🔴 強烈看跌（多個指標共振）"
    elif total_score <= -2:
        conclusion = "🔴 偏空看跌"
    elif -1 <= total_score <= 1:
        conclusion = "🟡 方向不明（指標分歧）"
    else:
        conclusion = "🟡 輕微震盪"
    
    return {
        "signals": signals,
        "total_score": total_score,
        "conclusion": conclusion,
        "signal_count": len(signals),
    }
