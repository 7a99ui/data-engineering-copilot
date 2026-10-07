# Ce fichier lit les résultats sauvegardés et les affiche
# de façon claire pour comparer les 3 stratégies

import json
import os


def load_results() -> list[dict]:
    """
    Charge les résultats depuis le fichier JSON sauvegardé
    par evaluate.py.
    """
    results_path = "eval/results.json"

    # Vérifier que le fichier existe
    if not os.path.exists(results_path):
        print("❌ Pas de résultats trouvés.")
        print("   Lance d'abord : python -m eval.evaluate")
        return []

    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_comparison(results: list[dict]):
    """
    Affiche un tableau comparatif des 3 stratégies.
    Met en évidence la meilleure stratégie pour chaque métrique.
    """

    if not results:
        return

    print("\n" + "="*70)
    print("COMPARAISON DES STRATÉGIES RAG")
    print("="*70)

    # En-tête du tableau
    print(f"\n{'Stratégie':<25} {'Faithfulness':>14} {'Relevancy':>12} {'Precision':>12} {'Overall':>10}")
    print("-"*70)

    # Trouver les meilleurs scores pour les mettre en évidence
    best_faith = max(r["faithfulness"]      for r in results)
    best_relev = max(r["answer_relevancy"]  for r in results)
    best_prec  = max(r["context_precision"] for r in results)
    best_over  = max(r["overall"]           for r in results)

    for r in results:
        # Ajouter ✅ pour les meilleurs scores
        faith_str = f"{r['faithfulness']:.3f}"     + (" ✅" if r["faithfulness"]      == best_faith else "")
        relev_str = f"{r['answer_relevancy']:.3f}" + (" ✅" if r["answer_relevancy"]  == best_relev else "")
        prec_str  = f"{r['context_precision']:.3f}"+ (" ✅" if r["context_precision"] == best_prec  else "")
        over_str  = f"{r['overall']:.3f}"          + (" ✅" if r["overall"]           == best_over  else "")

        print(f"{r['strategy']:<25} {faith_str:>14} {relev_str:>12} {prec_str:>12} {over_str:>10}")

    print("-"*70)

    # Trouver la meilleure stratégie globale
    best = max(results, key=lambda r: r["overall"])
    print(f"\n🏆 Meilleure stratégie : {best['strategy']} (Overall: {best['overall']:.3f})")

    # Analyse des améliorations
    if len(results) >= 2:
        baseline = results[0]   # Dense seul = référence
        best_r   = results[-1]  # Hybride+Reranker = notre meilleure

        improvement = (best_r["overall"] - baseline["overall"]) / baseline["overall"] * 100
        print(f"📈 Amélioration vs Dense seul : +{improvement:.1f}%")

    print("\n")


def print_per_question_analysis(results: list[dict]):
    """
    Affiche les scores par question pour identifier
    les questions difficiles pour chaque stratégie.
    """
    from eval.dataset import EVALUATION_DATASET

    print("="*70)
    print("ANALYSE PAR QUESTION")
    print("="*70)

    for i, item in enumerate(EVALUATION_DATASET):
        print(f"\nQ{i+1}: {item['question'][:65]}...")
        print(f"{'Stratégie':<25} {'Faith':>8} {'Relev':>8} {'Prec':>8}")
        print("-"*50)

        for r in results:
            faith = r["details"]["faithfulness_per_question"][i]
            relev = r["details"]["relevancy_per_question"][i]
            prec  = r["details"]["precision_per_question"][i]
            print(f"{r['strategy']:<25} {faith:>8.2f} {relev:>8.2f} {prec:>8.2f}")


if __name__ == "__main__":
    results = load_results()
    if results:
        print_comparison(results)
        print_per_question_analysis(results)