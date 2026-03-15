"""
ML Data Analysis & AI-oriented Editing Advisor.
Analyzes uploaded CSV data and gives specific edit instructions
to make it more suitable for machine learning.
"""
import math
import csv
import io


def _parse_csv(text: str):
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], []
    headers = list(rows[0].keys())
    return headers, rows


def _try_float(v):
    try:
        return float(v), True
    except (ValueError, TypeError):
        return None, False


def _column_stats(values):
    nums = []
    nulls = 0
    cats = set()
    for v in values:
        if v is None or str(v).strip() == "" or str(v).strip().lower() in ("nan", "null", "none", "na"):
            nulls += 1
            continue
        fv, ok = _try_float(v)
        if ok:
            nums.append(fv)
        else:
            cats.add(str(v).strip())

    stat = {
        "count": len(values),
        "nulls": nulls,
        "null_rate": nulls / max(len(values), 1),
        "is_numeric": len(nums) > 0 and len(cats) == 0,
        "unique_cats": list(cats)[:20],
        "n_unique": len(cats),
    }
    if stat["is_numeric"] and nums:
        mu = sum(nums) / len(nums)
        var = sum((x - mu) ** 2 for x in nums) / max(len(nums), 1)
        stat["mean"] = mu
        stat["std"] = math.sqrt(var)
        stat["min"] = min(nums)
        stat["max"] = max(nums)
        # detect outliers (|z| > 3)
        if stat["std"] > 1e-10:
            stat["outlier_count"] = sum(1 for x in nums if abs(x - mu) / stat["std"] > 3)
        else:
            stat["outlier_count"] = 0
        stat["range"] = stat["max"] - stat["min"]
    return stat


def _skewness(values):
    nums = [float(v) for v in values if _try_float(v)[1]]
    if len(nums) < 3:
        return 0.0
    n = len(nums)
    mu = sum(nums) / n
    std = math.sqrt(sum((x - mu) ** 2 for x in nums) / n)
    if std < 1e-10:
        return 0.0
    return sum(((x - mu) / std) ** 3 for x in nums) / n


def analyze_and_advise(csv_text: str) -> dict:
    """
    Returns full analysis + prioritized edit instructions for ML.
    """
    headers, rows = _parse_csv(csv_text)
    if not headers:
        return {"error": "CSVが空またはパースできません"}

    n = len(rows)
    col_data = {h: [r.get(h, "") for r in rows] for h in headers}
    col_stats = {h: _column_stats(col_data[h]) for h in headers}

    issues = []
    instructions = []

    # ---- 1. Missing values ----
    for col, st in col_stats.items():
        rate = st["null_rate"]
        if rate > 0.5:
            issues.append({"col": col, "type": "missing_high", "rate": rate})
            instructions.append({
                "priority": "HIGH",
                "column": col,
                "issue": f"欠損率 {rate:.1%} (50%超)",
                "action": f"「{col}」列を削除するか、ドメイン知識に基づいた値で補完してください。"
                          f" 欠損が多すぎるため、平均補完では誤りが大きくなります。",
                "why": "欠損率50%超の特徴量は、モデルにノイズを与えます。"
            })
        elif rate > 0.1:
            issues.append({"col": col, "type": "missing_moderate", "rate": rate})
            st_info = ""
            if st["is_numeric"]:
                st_info = f"（中央値 ≈ {st.get('mean', 0):.3g}）"
            instructions.append({
                "priority": "MEDIUM",
                "column": col,
                "issue": f"欠損率 {rate:.1%}",
                "action": f"「{col}」の欠損値{st_info}を補完してください：数値なら中央値、カテゴリなら最頻値を使用。",
                "why": "欠損のある行はそのままでは学習に使えません。"
            })

    # ---- 2. Scaling ----
    for col, st in col_stats.items():
        if st["is_numeric"]:
            rng = st.get("range", 0)
            if rng > 1000:
                instructions.append({
                    "priority": "HIGH",
                    "column": col,
                    "issue": f"値域が大きい (range={rng:.1f})",
                    "action": f"「{col}」を正規化してください: Min-Max → (x - {st['min']:.3g}) / {rng:.3g}  "
                              f"または標準化 → (x - {st['mean']:.3g}) / {st['std']:.3g}",
                    "why": "値域が大きいとグラジェント爆発・学習不安定の原因になります。"
                })
            elif st.get("std", 1) > 0 and rng > 0 and (st["max"] > 1 or st["min"] < 0):
                instructions.append({
                    "priority": "MEDIUM",
                    "column": col,
                    "issue": f"[0,1]範囲外 (min={st['min']:.3g}, max={st['max']:.3g})",
                    "action": f"「{col}」を Min-Max 正規化してください: (x - {st['min']:.3g}) / {rng:.3g}",
                    "why": "ニューラルネットは [0,1] または [-1,1] の入力で安定して学習します。"
                })

    # ---- 3. Outliers ----
    for col, st in col_stats.items():
        if st["is_numeric"] and st.get("outlier_count", 0) > 0:
            pct = st["outlier_count"] / max(n, 1)
            instructions.append({
                "priority": "MEDIUM" if pct > 0.02 else "LOW",
                "column": col,
                "issue": f"外れ値 {st['outlier_count']} 件 ({pct:.1%})",
                "action": f"「{col}」の |z-score| > 3 のサンプルをクリッピング（IQR法）または除去してください。",
                "why": "外れ値はモデルの汎化性能を下げ、損失関数に過剰に影響します。"
            })

    # ---- 4. Skewness ----
    for col, st in col_stats.items():
        if st["is_numeric"]:
            skew = _skewness([v for v in col_data[col] if _try_float(v)[1]])
            if abs(skew) > 2:
                transform = "log1p(x)" if st.get("min", 0) >= 0 else "Box-Cox変換"
                instructions.append({
                    "priority": "LOW",
                    "column": col,
                    "issue": f"歪み (skewness={skew:.2f})",
                    "action": f"「{col}」に {transform} を適用してください。",
                    "why": "強い歪みは線形・非線形モデル両方の学習精度に影響します。"
                })

    # ---- 5. Categorical encoding ----
    for col, st in col_stats.items():
        if not st["is_numeric"] and st["n_unique"] > 0:
            if st["n_unique"] <= 10:
                cats = st["unique_cats"]
                instructions.append({
                    "priority": "HIGH",
                    "column": col,
                    "issue": f"カテゴリ列 ({st['n_unique']} 種類)",
                    "action": f"「{col}」を One-Hot Encoding してください。値: {cats}  "
                              f"→ {st['n_unique']} 列に展開（列名: {col}_{{値}}）。",
                    "why": "ニューラルネットは文字列を直接扱えないため数値化が必要です。"
                })
            else:
                instructions.append({
                    "priority": "HIGH",
                    "column": col,
                    "issue": f"高カーディナリティカテゴリ ({st['n_unique']} 種類)",
                    "action": f"「{col}」は種類が多すぎるため、Target Encoding または Embedding を使用してください。"
                              f" 頻度が 1% 未満のカテゴリは 'OTHER' にまとめることを推奨。",
                    "why": "One-Hot では次元爆発が起きます。"
                })

    # ---- 6. Duplicate rows ----
    seen = set()
    dup_count = 0
    for r in rows:
        key = tuple(r.values())
        if key in seen:
            dup_count += 1
        seen.add(key)
    if dup_count > 0:
        instructions.append({
            "priority": "MEDIUM",
            "column": "(全体)",
            "issue": f"重複行 {dup_count} 件",
            "action": "重複行を削除してください。",
            "why": "同一データの繰り返しはモデルを過学習させます。"
        })

    # ---- 7. Class imbalance (last column heuristic) ----
    last_col = headers[-1]
    last_st = col_stats[last_col]
    if not last_st["is_numeric"] and last_st["n_unique"] >= 2:
        from collections import Counter
        counts = Counter(col_data[last_col])
        vals = list(counts.values())
        if vals:
            ratio = min(vals) / max(vals)
            if ratio < 0.3:
                instructions.append({
                    "priority": "HIGH",
                    "column": last_col,
                    "issue": f"クラス不均衡 (min/max={ratio:.2f})",
                    "action": f"「{last_col}」（目的変数と思われる列）が不均衡です。"
                              f" 少数クラスをオーバーサンプリング（SMOTE等）または"
                              f" 多数クラスをアンダーサンプリングしてください。",
                    "why": "不均衡データはモデルを多数クラスに偏らせます。"
                })

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    instructions.sort(key=lambda x: priority_order.get(x["priority"], 3))

    # Column summary
    col_summary = []
    for h in headers:
        st = col_stats[h]
        entry = {
            "name": h,
            "type": "数値" if st["is_numeric"] else "カテゴリ",
            "nulls": st["nulls"],
            "null_rate": f"{st['null_rate']:.1%}",
        }
        if st["is_numeric"]:
            entry.update({
                "min": round(st.get("min", 0), 4),
                "max": round(st.get("max", 0), 4),
                "mean": round(st.get("mean", 0), 4),
                "std": round(st.get("std", 0), 4),
                "outliers": st.get("outlier_count", 0),
            })
        else:
            entry["unique"] = st["n_unique"]
            entry["samples"] = st["unique_cats"][:5]
        col_summary.append(entry)

    return {
        "rows": n,
        "columns": len(headers),
        "headers": headers,
        "column_summary": col_summary,
        "total_issues": len(instructions),
        "high_priority": sum(1 for i in instructions if i["priority"] == "HIGH"),
        "instructions": instructions,
    }


def apply_fixes(csv_text: str) -> tuple:
    """
    Apply automatic data transformations:
    - Remove duplicate rows
    - Fill missing numeric values with median
    - Fill missing categorical values with mode
    - Clip outliers using IQR method
    - Min-Max normalize numeric columns with large range or outside [0,1]
    - One-Hot encode categorical columns with 2-10 unique values

    Returns (new_csv_text, transform_log)
    """
    headers, rows = _parse_csv(csv_text)
    if not headers or not rows:
        return csv_text, []

    new_headers = list(headers)
    new_rows = [dict(r) for r in rows]
    log = []

    # Step 1: Remove duplicate rows
    seen = set()
    unique_rows = []
    dup_count = 0
    for r in new_rows:
        key = tuple(r.get(h, "") for h in new_headers)
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
            unique_rows.append(r)
    if dup_count > 0:
        new_rows = unique_rows
        log.append(f"重複行 {dup_count} 件を削除しました")

    # Step 2: Process each column
    cols_to_encode = []  # (col_name, sorted_unique_cats)
    for col in list(new_headers):
        st = _column_stats([r.get(col, "") for r in new_rows])
        values = [r.get(col, "") for r in new_rows]

        if st["is_numeric"]:
            nums = sorted([float(v) for v in values if _try_float(v)[1]])

            # Fill missing with median
            if st["nulls"] > 0 and nums:
                n = len(nums)
                median = nums[n // 2] if n % 2 else (nums[n // 2 - 1] + nums[n // 2]) / 2
                filled = 0
                for r in new_rows:
                    v = r.get(col, "")
                    if v is None or str(v).strip() == "" or str(v).strip().lower() in ("nan", "null", "none", "na"):
                        r[col] = str(median)
                        filled += 1
                log.append(f"「{col}」の欠損値 {filled} 件を中央値 {median:.4g} で補完しました")
                nums = sorted([float(r[col]) for r in new_rows])

            # IQR outlier clipping
            if nums:
                n = len(nums)
                q1 = nums[n // 4]
                q3 = nums[min(3 * n // 4, n - 1)]
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    clipped = 0
                    for r in new_rows:
                        v, ok = _try_float(r.get(col, ""))
                        if ok:
                            if v < lower:
                                r[col] = str(lower)
                                clipped += 1
                            elif v > upper:
                                r[col] = str(upper)
                                clipped += 1
                    if clipped > 0:
                        log.append(f"「{col}」の外れ値 {clipped} 件を IQR法でクリッピングしました")
                    nums = [float(r[col]) for r in new_rows]

            # Min-Max normalization
            if nums:
                mn = min(nums)
                mx = max(nums)
                rng = mx - mn
                if rng > 1e-10 and (rng > 1000 or mx > 1 or mn < 0):
                    for r in new_rows:
                        v, ok = _try_float(r.get(col, ""))
                        if ok:
                            r[col] = str((v - mn) / rng)
                    log.append(f"「{col}」を Min-Max 正規化しました [{mn:.4g}, {mx:.4g}] → [0, 1]")

        else:
            # Categorical: fill missing with mode
            cats = [str(v).strip() for v in values
                    if v is not None and str(v).strip() not in ("", "nan", "null", "none", "na")]
            if st["nulls"] > 0 and cats:
                from collections import Counter
                mode_val = Counter(cats).most_common(1)[0][0]
                filled = 0
                for r in new_rows:
                    v = r.get(col, "")
                    if v is None or str(v).strip() == "" or str(v).strip().lower() in ("nan", "null", "none", "na"):
                        r[col] = mode_val
                        filled += 1
                if filled > 0:
                    log.append(f"「{col}」の欠損値 {filled} 件を最頻値 '{mode_val}' で補完しました")

            # Queue for One-Hot encoding if 2-10 unique values
            unique_cats = sorted(set(str(r.get(col, "")).strip() for r in new_rows))
            if 2 <= len(unique_cats) <= 10:
                cols_to_encode.append((col, unique_cats))

    # Step 3: One-Hot encoding (must be done after all numeric processing)
    for col, unique_cats in cols_to_encode:
        if col not in new_headers:
            continue
        col_idx = new_headers.index(col)
        new_col_names = [f"{col}_{cat}" for cat in unique_cats]
        for r in new_rows:
            v = str(r.get(col, "")).strip()
            for cat in unique_cats:
                r[f"{col}_{cat}"] = "1" if v == cat else "0"
            del r[col]
        new_headers = new_headers[:col_idx] + new_col_names + new_headers[col_idx + 1:]
        log.append(f"「{col}」を One-Hot Encoding しました → {len(unique_cats)} 列")

    # Generate new CSV text
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=new_headers)
    writer.writeheader()
    for r in new_rows:
        writer.writerow({h: r.get(h, "") for h in new_headers})

    return out.getvalue(), log
