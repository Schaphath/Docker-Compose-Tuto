#=======================#
#   Import librairies   #
#=======================#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier




#=============================#
#   Fonction d'entrainement   #
#=============================#
def compare_models(
    df,
    target_col="diagnosis",
    test_size=0.2,
    random_state=42,
    save_models=True,
    output_dir="Save_models",
):
    """
    Compare plusieurs algorithmes de classification sur le dataset cancer du sein.

    Retourne
    --------
    results_df, fig, rapport_meilleur_model, fig_learning, fig_cm,
    model_filename, scaler_filename
    """

    #=============================#
    #   Preparation des donnees   #
    #=============================#

    # Selectionner les features (_worst en priorite, sinon _mean)
    worst_cols   = [c for c in df.columns if c.endswith("_worst")]
    mean_cols    = [c for c in df.columns if c.endswith("_mean")]
    feature_cols = worst_cols if worst_cols else mean_cols

    print(f"Features selectionnees : {len(feature_cols)} colonnes "
          f"({'_worst' if worst_cols else '_mean'})")

    X = df[feature_cols]
    y = df[target_col]

    if y.dtype == "object":
        y = y.map({"M": 1, "B": 0})

    print(f"\nDistribution des classes :")
    print(f"  Benin  (B=0) : {(y == 0).sum()}  ({(y == 0).mean()*100:.1f}%)")
    print(f"  Malin  (M=1) : {(y == 1).sum()}  ({(y == 1).mean()*100:.1f}%)")


    #=====================#
    #  Split train/test   #
    #=====================#
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain set : {X_train.shape[0]} echantillons")
    print(f"Test set  : {X_test.shape[0]} echantillons")


    #============================#
    #  Mise a l echelle MinMax   #
    #============================#
    scaler         = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("Mise a l echelle MinMax appliquee\n")


    #=============================#
    #   Definition des modeles    #
    #=============================#
    
    models = {
        "Logistic Regression"    : LogisticRegression(max_iter=1000, random_state=random_state),
        #"Decision Tree"          : DecisionTreeClassifier(random_state=random_state),
        "Random Forest"          : RandomForestClassifier(random_state=random_state),
        #"Extra Trees"            : ExtraTreesClassifier(random_state=random_state)
        "XGBoost"                : XGBClassifier(random_state=random_state, eval_metric="logloss"),
        #"Gradient Boosting"      : GradientBoostingClassifier(random_state=random_state),
        #"KNN"                    : KNeighborsClassifier(),
        #"AdaBoost"               : AdaBoostClassifier(random_state=random_state)
        #"Hist Gradient Boosting" : HistGradientBoostingClassifier(random_state=random_state),
    }

    
    
    """
    models = {
        "Logistic Regression"    : LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest"          : RandomForestClassifier(random_state=random_state),
        "XGBoost"                : XGBClassifier(random_state=random_state, eval_metric="logloss"),
        "Gradient Boosting"      : GradientBoostingClassifier(random_state=random_state),
        "KNN"                    : KNeighborsClassifier(),
        "AdaBoost"               : AdaBoostClassifier(random_state=random_state),
        "Hist Gradient Boosting" : HistGradientBoostingClassifier(random_state=random_state)
    }

    

    
    """
    

    #================================#
    #   Entrainement et evaluation   #
    #================================#
    results        = {}
    trained_models = {}

    print("Entrainement des modeles...\n")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred               = model.predict(X_test_scaled)
        recall               = recall_score(y_test, y_pred)
        results[name]        = recall
        trained_models[name] = model
        print(f"{name:<30} | Recall: {recall:.4f}")

    print("-" * 60)


    #==========================================#
    #   Creation du DataFrame de resultats     #
    #==========================================#
    results_df = pd.DataFrame({
        "Model"  : list(results.keys()),
        "Recall" : list(results.values()),
    }).sort_values("Recall", ascending=True)


    #=====================#
    #    Visualisation    #
    #=====================#
    fig, ax = plt.subplots(figsize=(12, 8))
    colors  = plt.cm.RdYlGn(results_df["Recall"])

    ax.barh(results_df["Model"], results_df["Recall"],
            color=colors, edgecolor="black", linewidth=1.2)

    for i, (model, recall) in enumerate(zip(results_df["Model"], results_df["Recall"])):
        ax.text(recall + 0.01, i, f"{recall:.4f}",
                va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Recall Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Comparaison des modeles - Metrique: Recall\n(Du moins performant au plus performant)",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.axvline(x=0.9, color="green", linestyle="--", linewidth=2, alpha=0.5, label="Seuil 90%")
    ax.legend()
    plt.tight_layout()

    print(f"\nMeilleur modele : {results_df.iloc[-1]['Model']} "
          f"(Recall: {results_df.iloc[-1]['Recall']:.4f})")


    #============================================#
    #    Rapport detaille du meilleur modele     #
    #============================================#
    best_model_name = results_df.iloc[-1]["Model"]
    best_model      = trained_models[best_model_name]
    y_pred_best     = best_model.predict(X_test_scaled)

    print(f"\n{'='*60}")
    print(f"RAPPORT DETAILLE - {best_model_name}")
    print(f"{'='*60}\n")

    rapport_meilleur_model = classification_report(
        y_test, y_pred_best,
        target_names=["Benin (0)", "Malin (1)"],
        digits=4,
        zero_division=0,
    )
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(rapport_meilleur_model)
    print("=" * 50)


    #==================================#
    #         MATRICE DE CONFUSION     #
    #==================================#
    print(f"\n{'='*60}")
    print(f"MATRICE DE CONFUSION - {best_model_name}")
    print(f"{'='*60}\n")

    cm             = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    total          = tn + fp + fn + tp

    # Affichage texte (lisible dans les logs Jenkins)
    print(f"                     Predit Benin   Predit Malin")
    print(f"  Reel Benin  (0) |  {tn:^13}  |  {fp:^13}  |")
    print(f"  Reel Malin  (1) |  {fn:^13}  |  {tp:^13}  |")
    print()
    print(f"  TN = {tn:>4}  Vrais Negatifs  -- Benins bien classes")
    print(f"  FP = {fp:>4}  Faux  Positifs  -- Benins classes malins   (fausses alarmes)")
    print(f"  FN = {fn:>4}  Faux  Negatifs  -- Malins MANQUES          *** DANGEREUX ***")
    print(f"  TP = {tp:>4}  Vrais Positifs  -- Malins correctement detectes")
    print()

    recall_cm    = tp / (tp + fn)  if (tp + fn) > 0 else 0
    precision_cm = tp / (tp + fp)  if (tp + fp) > 0 else 0
    specificity  = tn / (tn + fp)  if (tn + fp) > 0 else 0
    accuracy     = (tp + tn) / total

    print(f"  Accuracy    : {accuracy:.4f}   ({tp+tn}/{total} bien classes)")
    print(f"  Recall      : {recall_cm:.4f}   ({tp}/{tp+fn} malins detectes)")
    print(f"  Precision   : {precision_cm:.4f}   ({tp}/{tp+fp} alertes justifiees)")
    print(f"  Specificite : {specificity:.4f}   ({tn}/{tn+fp} benins ignores a raison)")

    taux_fn = fn / (fn + tp) if (fn + tp) > 0 else 0
    if taux_fn > 0.10:
        print(f"\n  /!\\ ALERTE : {fn} malins manques ({taux_fn*100:.1f}% de FN).")
        print("       Envisage class_weight='balanced' pour reduire les FN.")
    else:
        print(f"\n  OK : {fn} malins manques ({taux_fn*100:.1f}%) -- dans les seuils medicaux.")

    # Figure : deux panneaux cote a cote
    fig_cm, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig_cm.suptitle(
        f"Matrice de Confusion -- {best_model_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Panneau gauche : valeurs absolues
    disp_raw = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benin (0)", "Malin (1)"],
    )
    disp_raw.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Valeurs absolues",  fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Classe predite",   fontsize=11)
    axes[0].set_ylabel("Classe reelle",    fontsize=11)
    for text in disp_raw.text_.ravel():
        text.set_fontsize(18)
        text.set_fontweight("bold")

    # Panneau droit : normalise par classe reelle (taux de detection)
    cm_norm   = confusion_matrix(y_test, y_pred_best, normalize="true")
    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=["Benin (0)", "Malin (1)"],
    )
    disp_norm.plot(ax=axes[1], colorbar=False, cmap="Greens")
    axes[1].set_title("Normalisee -- % par classe reelle", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Classe predite", fontsize=11)
    axes[1].set_ylabel("Classe reelle",  fontsize=11)
    for text in disp_norm.text_.ravel():
        val = float(text.get_text())
        text.set_text(f"{val*100:.1f}%")
        text.set_fontsize(18)
        text.set_fontweight("bold")

    legend_elems = [
        mpatches.Patch(color="#d0e8ff", label=f"TN={tn}  Benins bien classes"),
        mpatches.Patch(color="#fdd49e", label=f"FP={fp}  Benins -> fausse alarme"),
        mpatches.Patch(color="#d7191c", label=f"FN={fn}  Malins manques  !!!"),
        mpatches.Patch(color="#2b83ba", label=f"TP={tp}  Malins bien detectes"),
    ]
    fig_cm.legend(
        handles=legend_elems, loc="lower center", ncol=2,
        fontsize=10, bbox_to_anchor=(0.5, -0.12), frameon=True,
    )
    plt.tight_layout()
    print("\nMatrice de confusion generee avec succes")


    #=================================================#
    #    Courbe d apprentissage du meilleur modele    #
    #=================================================#
    print(f"\n{'='*60}")
    print("Generation de la courbe d apprentissage...")
    print(f"{'='*60}\n")

    best_model_fresh = models[best_model_name]

    train_sizes, train_scores, val_scores = learning_curve(
        best_model_fresh, X_train_scaled, y_train,
        cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_log_loss",
        random_state=random_state,
    )

    train_scores = -train_scores
    val_scores   = -val_scores
    train_mean   = np.mean(train_scores, axis=1)
    train_std    = np.std(train_scores,  axis=1)
    val_mean     = np.mean(val_scores,   axis=1)
    val_std      = np.std(val_scores,    axis=1)

    fig_learning, ax_learning = plt.subplots(figsize=(10, 6))
    ax_learning.plot(train_sizes, train_mean, "o-", color="blue",
                     label="Score d entrainement", linewidth=2)
    ax_learning.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                              alpha=0.2, color="blue")
    ax_learning.plot(train_sizes, val_mean, "o-", color="red",
                     label="Score de validation", linewidth=2)
    ax_learning.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                              alpha=0.2, color="red")
    ax_learning.set_xlabel("Nombre d echantillons d entrainement", fontsize=12, fontweight="bold")
    ax_learning.set_ylabel("Log-Loss (plus bas = meilleur)",       fontsize=12, fontweight="bold")
    ax_learning.set_title(
        f"Courbe d apprentissage - {best_model_name}\n(Metrique: Log-Loss)",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax_learning.legend(loc="upper right", fontsize=11)
    ax_learning.grid(True, alpha=0.3, linestyle="--")
    ax_learning.set_ylim(bottom=0)
    plt.tight_layout()
    print("Courbe d apprentissage generee avec succes")


    #====================================================#
    #    Sauvegarde du meilleur modele et du scaler      #
    #====================================================#
    if save_models:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_filename  = output_path / f"{best_model_name.lower().replace(' ', '_')}_best.pkl"
        scaler_filename = output_path / "MinMax_scaler.pkl"

        with open(model_filename, "wb") as f:
            pickle.dump(best_model, f)
        with open(scaler_filename, "wb") as f:
            pickle.dump(scaler, f)

        print(f"\n{'='*60}")
        print("Modeles sauvegardes")
        print(f"{'='*60}")
        print(f"Meilleur modele : {model_filename}")
        print(f"Scaler MinMax   : {scaler_filename}")
        print(f"{'='*60}")

    return (
        results_df,
        fig,
        rapport_meilleur_model,
        fig_learning,
        fig_cm,
        model_filename,
        scaler_filename,
    )



#  Lancer l entrainement
if __name__ == "__main__":
    df = pd.read_csv("Data/process/cancer_clean.csv", sep=",")

    results, fig, rapport, fig_learning, fig_cm, model_filename, scaler_filename = compare_models(df)

    plt.show()