# httpx — pour faire des requêtes HTTP vers Ollama
# même bibliothèque que client.py et retriever.py — cohérence totale
import httpx

# settings — pour récupérer OLLAMA_BASE_URL et LLM_MODEL depuis le .env
from app.config import settings


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Construit le prompt RAG complet selon la hiérarchie :
    Rôle → Instruction → Contexte → Question → Format de sortie

    question : la question posée par l'utilisateur
    chunks   : la liste des chunks retournés par retriever.py
               chaque chunk contient "score" et "payload" (texte + métadonnées)
    -> str   : retourne le prompt complet prêt à être envoyé au LLM
    """

    # ── Construction du bloc contexte ──

    # On trie les chunks par score décroissant
    # POURQUOI : les LLMs font plus attention aux premières informations
    # du prompt (biais de position) — on met le plus pertinent en premier
    # key=lambda x: x["score"] → trier selon le champ "score" de chaque chunk
    # reverse=True             → ordre décroissant (0.94 avant 0.67)
    sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)

    # Liste vide qui va accumuler les blocs de contexte formatés
    # un bloc par chunk — ils seront joints avec "\n\n" à la fin
    context_parts = []

    # enumerate() donne l'index i (0, 1, 2...) ET le chunk en même temps
    # on utilise i pour numéroter les passages (Passage 1, Passage 2...)
    # la numérotation aide le LLM à référencer précisément chaque source
    for i, chunk in enumerate(sorted_chunks):

        # Extraire les informations utiles du payload
        # payload = le dictionnaire stocké dans Qdrant au Sprint 2
        # contient : text, source, page, chunk_index, file_type...
        source = chunk["payload"]["source"]  # ex: "spark.md"
        score  = round(chunk["score"], 2)    # ex: 0.94 (arrondi à 2 décimales)
        text   = chunk["payload"]["text"]    # le texte brut du chunk

        # .get("page", 1) → si "page" n'existe pas dans le payload
        # retourne 1 par défaut au lieu de lever une erreur KeyError
        # sécurité supplémentaire pour les documents sans numéro de page
        page = chunk["payload"].get("page", 1)

        # Construire le bloc formaté pour ce chunk
        # format : [Passage N | Source: fichier | Page: N | Score: 0.XX]
        #          texte du chunk
        # le LLM voit clairement d'où vient chaque information
        # et peut citer la source précise dans sa réponse
        context_parts.append(
            f"[Passage {i+1} | Source: {source} | Page: {page} | Relevance score: {score}]\n{text}"
        )

    # Joindre tous les blocs avec une ligne vide entre chacun
    # "\n\n".join([bloc1, bloc2, bloc3]) → "bloc1\n\nbloc2\n\nbloc3"
    # la ligne vide améliore la lisibilité pour le LLM
    context = "\n\n".join(context_parts)

    # ── Construction du prompt complet ──
    # Structure : Rôle → Instructions → Contexte → Question → Format
    # C'est une f-string multiligne — les {} sont remplacés par les valeurs
    return f"""## ROLE
You are an expert Data Engineering assistant helping engineers solve
technical problems related to pipelines, Spark, Airflow, databases,
and data infrastructure.
# Définit QUI est le LLM — fixe le ton et le domaine d'expertise
# "expert Data Engineering" → le LLM adopte le comportement d'un expert
# et comprend le vocabulaire technique (Spark, Airflow, DAG...)

## INSTRUCTIONS
- Answer ONLY based on the context passages provided below.
# Règle 1 — ANCRAGE STRICT : interdit au LLM d'utiliser sa culture générale
# sans cette règle → hallucinations possibles

- Prioritize passages with higher relevance scores.
# Règle 2 — RETRIEVAL-AWARE : le LLM doit faire plus confiance
# aux passages avec un score élevé qu'aux passages avec un score faible

- If the answer is not found in the context, respond EXACTLY with:
  "I could not find information about this topic in the available documents."
# Règle 3 — GESTION DES INCERTITUDES : si le LLM ne trouve pas la réponse
# dans le contexte, il doit l'admettre plutôt qu'inventer
# "EXACTLY with" → force une réponse standardisée facile à détecter

- Do NOT add any external knowledge beyond what is in the context.
# Règle 4 — RENFORCEMENT de l'ancrage strict — dit explicitement
# "n'ajoute rien de ta propre connaissance"

- Always cite the source file in brackets [source_name] when using a passage.
# Règle 5 — CITATION DES SOURCES : force le LLM à mentionner [spark.md]
# etc. dans sa réponse — permet à l'utilisateur de vérifier

- If information seems contradictory between passages, mention it explicitly
  and indicate which source you trust more based on its relevance score.
# Règle 6 — GESTION DES CONTRADICTIONS : si deux chunks disent des choses
# différentes, le LLM doit le signaler et se fier au score le plus haut

- Be concise and structured. Use bullet points for lists of solutions.
# Règle 7 — FORMAT DE SORTIE : réponses courtes et structurées
# "bullet points" → évite les longues réponses verboses

## CONTEXT
{context}
# Le contexte est injecté ici — les chunks formatés avec leurs sources
# le LLM "lit" ces passages comme s'il lisait un document

## QUESTION
{question}
# La question originale de l'utilisateur
# posée APRÈS le contexte — le LLM sait quoi chercher dans ce contexte

## ANSWER
Provide a clear, structured answer based strictly on the context above.
Cite sources in brackets. If multiple solutions exist, list them clearly:"""
# L'instruction finale déclenche la génération
# "Cite sources in brackets" → renforce la règle 5
# ":" à la fin → invite le LLM à commencer sa réponse directement


async def generate(question: str, chunks: list[dict]) -> str:
    """
    Construit le prompt et appelle le LLM via Ollama.
    Retourne la réponse générée en texte.

    question : la question de l'utilisateur
    chunks   : les chunks retournés par retriever.py (peut être vide)
    -> str   : la réponse du LLM
    """

    # ── Cas particulier : aucun chunk pertinent trouvé ──
    # Si retriever.py n'a trouvé aucun chunk avec score >= 0.5
    # chunks sera une liste vide []
    # "not chunks" est True quand la liste est vide
    # POURQUOI cette vérification : appeler le LLM sans contexte
    # lui ferait inventer une réponse — exactement ce qu'on veut éviter
    # On retourne directement un message clair sans gaspiller de tokens
    if not chunks:
        return (
            "I could not find relevant information about this topic "
            "in the available documents. Please make sure the relevant "
            "documents have been ingested."
        )

    # ── Construire le prompt enrichi ──
    # On appelle build_prompt() définie juste au-dessus
    # elle assemble Rôle + Instructions + Contexte + Question
    # et retourne une longue chaîne de texte prête à envoyer au LLM
    prompt = build_prompt(question, chunks)

    # ── Appel à Ollama via HTTP ──
    # "async with" → ouvre la connexion HTTP et la ferme automatiquement
    # à la fin du bloc, même si une erreur survient
    # timeout=120 → le LLM peut prendre du temps pour un long prompt RAG
    # (le contexte + les instructions sont plus longs qu'un simple chat)
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(

            # Même endpoint que client.py — /api/generate d'Ollama
            f"{settings.OLLAMA_BASE_URL}/api/generate",

            json={
                # Le modèle défini dans le .env — ex: "qwen2.5:3b"
                "model": settings.LLM_MODEL,

                # Le prompt complet construit par build_prompt()
                # contient : rôle + instructions + contexte + question
                "prompt": prompt,

                # False = attendre la réponse complète avant de retourner
                # True  = streaming mot par mot (pas encore implémenté)
                "stream": False,

                # ── Paramètres de génération ──
                "options": {

                    # temperature : contrôle la créativité du LLM
                    # 0.0 = toujours la même réponse, 100% déterministe
                    # 0.1 = quasi-déterministe, légère variation possible
                    # 1.0 = très créatif, hallucine facilement
                    # Pour le RAG → 0.1 est idéal : factuel sans être robotique
                    "temperature": 0.1,

                    # top_p : diversité des tokens que le LLM peut choisir
                    # 0.9 = considère les tokens qui couvrent 90% de probabilité
                    # travaille avec temperature pour contrôler la créativité
                    # valeur standard recommandée pour le RAG
                    "top_p": 0.9,

                    # num_predict : nombre maximum de tokens à générer
                    # 512 tokens ≈ ~380 mots — suffisant pour une réponse
                    # structurée avec sources sans être trop verbose
                    # évite les réponses infinies qui consomment trop de mémoire
                    "num_predict": 512
                }
            }
        )

        # Si Ollama répond avec une erreur → exception claire immédiatement
        response.raise_for_status()

        # Ollama retourne un JSON de cette forme :
        # {
        #   "model"   : "qwen2.5:3b",
        #   "response": "According to [spark.md], the OutOfMemory...",
        #   "done"    : true
        # }
        # On extrait uniquement le texte de la réponse avec ["response"]
        # c'est ce texte qu'on retourne à pipeline.py → main.py → utilisateur
        return response.json()["response"]