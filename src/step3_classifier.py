"""Step 3: direct 3-way classifier + ensemble with the Dixon-Coles Poisson.

Why: Steps 1-2 showed the Poisson grid is draw-blind (pred-draw rate ~0.000) and
Dixon-Coles only recovers ~0.003 log-loss because the predicted goal means rarely
land in draw territory at all. A model that predicts the 0/1/2 label DIRECTLY
optimises our actual metric (multiclass log-loss) and naturally produces draw
probabilities. We then ensemble it with the DC-Poisson — two different inductive
biases usually beat either alone — and temperature-calibrate the blend.

Everything is evaluated through the SAME 5-fold walk-forward as Steps 1-2, so the
numbers are directly comparable. Baselines to beat (walk-forward log-loss):
  Poisson rho=0            -> 0.9120
  Poisson + DC (rho=-0.15) -> 0.9091
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

from harness import load, outcomes, brier_multi, grid_probs, expected_goals, fit

# Per-match features for the direct classifier (one row per match, not stacked).
# abs_elo_diff: draws peak when teams are evenly matched (|elo_diff|~0). A linear
# model can't express "draw most likely at the centre" from signed elo_diff alone,
# so we hand it the U-shape explicitly.
CLF_FEATS = ["elo_diff", "abs_elo_diff", "home_gf", "home_ga", "away_gf", "away_ga", "home_advantage"]
BEST_RHO = -0.15  # winner of the Step-2 sweep


def match_features(d):
    elo_diff = d.home_elo - d.away_elo
    return pd.DataFrame({
        "elo_diff": elo_diff,
        "abs_elo_diff": elo_diff.abs(),
        "home_gf": d.home_gf, "home_ga": d.home_ga,
        "away_gf": d.away_gf, "away_ga": d.away_ga,
        "home_advantage": d.home_advantage,
    })[CLF_FEATS]


def make_clf(kind):
    if kind == "logit":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=1000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, min_samples_leaf=50, random_state=0)
    raise ValueError(kind)


def clf_probs(model, d):
    """predict_proba reordered to [away(0), draw(1), home(2)]."""
    P = model.predict_proba(match_features(d))
    col = {c: i for i, c in enumerate(model.classes_)}
    return P[:, [col[0], col[1], col[2]]]


# ---- predictor factories: given a train df, return a (test df -> P) function ----
def poisson_predictor(rho=BEST_RHO, alpha=0.1):
    def make(train):
        m = fit(train, alpha=alpha)
        return lambda test: grid_probs(*expected_goals(m, test), rho=rho)
    return make


def clf_predictor(kind):
    def make(train):
        m = make_clf(kind).fit(match_features(train), outcomes(train))
        return lambda test: clf_probs(m, test)
    return make


def ensemble_predictor(w, rho=BEST_RHO, kind="hgb", alpha=0.1, temp=1.0):
    """Blend: w * Poisson + (1-w) * classifier, then optional temperature `temp`."""
    def make(train):
        pm = fit(train, alpha=alpha)
        cm = make_clf(kind).fit(match_features(train), outcomes(train))

        def predict(test):
            Pp = grid_probs(*expected_goals(pm, test), rho=rho)
            Pc = clf_probs(cm, test)
            P = w * Pp + (1 - w) * Pc
            if temp != 1.0:
                P = np.clip(P, 1e-12, 1.0) ** (1.0 / temp)
                P = P / P.sum(axis=1, keepdims=True)
            return P
        return predict
    return make


def walk_forward_generic(make_predictor, n_splits=5, label="", verbose=True):
    df = load()
    tss = TimeSeriesSplit(n_splits=n_splits)
    ll, br, ac, pred_draw, true_draw = [], [], [], [], []
    mean_p = []  # mean predicted prob per class -> probability calibration
    cm = np.zeros((3, 3), dtype=int)
    for tr_idx, te_idx in tss.split(df):
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        predict = make_predictor(train)
        P = predict(test)
        y = outcomes(test)
        ll.append(log_loss(y, P, labels=[0, 1, 2]))
        br.append(brier_multi(y, P))
        ac.append(accuracy_score(y, P.argmax(1)))
        cm += confusion_matrix(y, P.argmax(1), labels=[0, 1, 2])
        pred_draw.append((P.argmax(1) == 1).mean())
        true_draw.append((y == 1).mean())
        mean_p.append(P.mean(axis=0))
    res = {"log_loss": np.mean(ll), "brier": np.mean(br), "accuracy": np.mean(ac),
           "pred_draw_rate": np.mean(pred_draw), "true_draw_rate": np.mean(true_draw),
           "mean_p": np.mean(mean_p, axis=0), "cm": cm}
    if verbose:
        print(f"\n{label}")
        print(f"  log-loss : {res['log_loss']:.4f}")
        print(f"  brier    : {res['brier']:.4f}")
        print(f"  accuracy : {res['accuracy']:.4f}")
        print(f"  draw rate: pred {res['pred_draw_rate']:.3f}  vs true {res['true_draw_rate']:.3f}")
    return res


if __name__ == "__main__":
    print("=" * 64)
    print("STEP 3 — direct 3-way classifier + ensemble (walk-forward, 5 folds)")
    print("=" * 64)

    print("\n--- references (Steps 1-2) ---")
    walk_forward_generic(poisson_predictor(rho=0.0), label="Poisson (rho=0)")
    walk_forward_generic(poisson_predictor(rho=BEST_RHO), label=f"Poisson + DC (rho={BEST_RHO})")

    print("\n--- direct 3-way classifiers (now with abs_elo_diff draw feature) ---")
    for kind, name in [("logit", "Multinomial logistic"), ("hgb", "Gradient boosting (HGB)")]:
        r = walk_forward_generic(clf_predictor(kind), label=name)
        print(f"  mean P [away,draw,home] = {np.round(r['mean_p'], 3)}  (draw base rate {r['true_draw_rate']:.3f})")

    # pick the better classifier as the ensemble partner
    KIND = min(["logit", "hgb"],
               key=lambda k: walk_forward_generic(clf_predictor(k), verbose=False)["log_loss"])
    print(f"\n--- ensemble weight sweep: w*Poisson(DC) + (1-w)*{KIND} ---")
    best = None
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        r = walk_forward_generic(ensemble_predictor(w, kind=KIND), label=f"w={w:.1f}", verbose=False)
        print(f"  w={w:.1f}  log-loss {r['log_loss']:.4f}  brier {r['brier']:.4f}  acc {r['accuracy']:.4f}")
        if best is None or r["log_loss"] < best[1]:
            best = (w, r["log_loss"])
    print(f"  -> best weight w={best[0]:.1f}  (log-loss {best[1]:.4f})")

    print("\n--- temperature calibration on best ensemble ---")
    bestT = None
    for T in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]:
        r = walk_forward_generic(ensemble_predictor(best[0], kind=KIND, temp=T),
                                 label=f"T={T}", verbose=False)
        print(f"  T={T:<4}  log-loss {r['log_loss']:.4f}  brier {r['brier']:.4f}  acc {r['accuracy']:.4f}")
        if bestT is None or r["log_loss"] < bestT[1]:
            bestT = (T, r["log_loss"])
    print(f"  -> best temperature T={bestT[0]}  (log-loss {bestT[1]:.4f})")

    print("\n" + "=" * 64)
    print(f"FINAL: ensemble w={best[0]:.1f} ({KIND}), T={bestT[0]}")
    r = walk_forward_generic(ensemble_predictor(best[0], kind=KIND, temp=bestT[0]),
                             label="FINAL ensemble (DC-Poisson + classifier, temp-calibrated)")
    print(f"  mean P [away,draw,home] = {np.round(r['mean_p'], 3)}  (draw base rate {r['true_draw_rate']:.3f})")
    print(f"\n  vs Poisson rho=0 baseline: 0.9120 -> {r['log_loss']:.4f}  "
          f"({100*(0.9120-r['log_loss'])/0.9120:+.1f}% log-loss)")
