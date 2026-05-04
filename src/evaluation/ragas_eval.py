# src/evaluation/ragas_eval.py
import logging
logger = logging.getLogger(__name__)

TEST_DATASET = [
    {"question": "What was RBI's policy repo rate decision in February 2025?",
     "ground_truth": "The RBI reduced the policy repo rate by 25 basis points to 6.25 per cent in February 2025.",
     "filter": {"doc_type": "rbi"}},
    {"question": "What stance did the MPC adopt in October 2024?",
     "ground_truth": "The MPC changed stance from withdrawal of accommodation to neutral in October 2024.",
     "filter": {"doc_type": "rbi"}},
    {"question": "What was the CRR decision made by RBI in December 2024?",
     "ground_truth": "The cash reserve ratio was reduced to 4.0 per cent of NDTL in December 2024.",
     "filter": {"doc_type": "rbi"}},
    {"question": "What was global GDP growth in 2024 according to RBI?",
     "ground_truth": "Global GDP grew by 3.3 per cent in 2024.",
     "filter": {"doc_type": "rbi"}},
    {"question": "What was global inflation in 2024 according to RBI?",
     "ground_truth": "Global inflation eased to 5.7 per cent in 2024 from 6.6 per cent a year ago.",
     "filter": {"doc_type": "rbi"}},
    {"question": "What was India's headline inflation trend in 2024-25?",
     "ground_truth": "Headline inflation exhibited gradual easing in 2024-25, interrupted by volatile food inflation.",
     "filter": {"doc_type": "rbi"}},
    {"question": "What was the repo rate from February 2023 to early 2025?",
     "ground_truth": "The policy repo rate was maintained at 6.50 per cent from February 2023 until February 2025.",
     "filter": {"doc_type": "rbi"}},
    {"question": "What is TCS's primary business?",
     "ground_truth": "TCS is an IT services, consulting and business solutions organization partnering with businesses for over 56 years.",
     "filter": {"company": "TCS"}},
    {"question": "How many employees does TCS have?",
     "ground_truth": "TCS has over 601,000 consultants in 54 countries.",
     "filter": {"company": "TCS"}},
    {"question": "What are Reliance Industries' main business segments?",
     "ground_truth": "Reliance's key segments include Oil to Chemicals, Oil and Gas, Retail, Digital Services, and Others.",
     "filter": {"company": "Reliance"}},
    {"question": "What is Infosys's business model?",
     "ground_truth": "Infosys is a global technology company providing IT services, consulting, and outsourcing solutions.",
     "filter": {"company": "Infosys"}},
    {"question": "What is Maruti Suzuki's core business?",
     "ground_truth": "Maruti Suzuki is India's leading automobile manufacturer producing passenger vehicles.",
     "filter": {"company": "Maruti Suzuki"}},
    {"question": "What technology areas is HCL Technologies focused on?",
     "ground_truth": "HCL Technologies provides IT services including digital transformation, engineering, and technology services.",
     "filter": {"company": "HCL Technologies"}},
    {"question": "What businesses does ITC operate in?",
     "ground_truth": "ITC operates across FMCG, hotels, paperboards, packaging, and agribusiness.",
     "filter": {"company": "ITC"}},
    {"question": "What is Wipro's primary service offering?",
     "ground_truth": "Wipro provides IT services, consulting, and business process services globally.",
     "filter": {"company": "Wipro"}},
]


def build_ragas_dataset(chain, test_data, verbose=True):
    results = []
    print(f"Running {len(test_data)} questions...\n")

    for i, item in enumerate(test_data, 1):
        question     = item["question"]
        ground_truth = item["ground_truth"]
        where_filter = item.get("filter")

        if verbose:
            print(f"[{i:02d}/{len(test_data)}] {question[:65]}")

        try:
            result   = chain.query(question=question, where_filter=where_filter)
            contexts = [c["text"] for c in result["chunks"]]
            answered = "insufficient information" not in result["answer"].lower()

            results.append({
                "question":     question,
                "answer":       result["answer"],
                "contexts":     contexts,
                "ground_truth": ground_truth,
                "filter":       str(where_filter),
                "top_score":    result["chunks"][0].get("hybrid_score", 0) if result["chunks"] else 0,
                "tokens_used":  result.get("tokens_used", 0),
            })

            if verbose:
                print(f"         [{'OK' if answered else 'NO'}]  score={results[-1]['top_score']:.3f}")

        except Exception as e:
            logger.error(f"Q{i} failed: {e}")
            results.append({"question": question, "answer": f"ERROR: {e}",
                            "contexts": [], "ground_truth": ground_truth,
                            "filter": str(where_filter), "top_score": 0, "tokens_used": 0})

    return results


def print_ragas_report(scores, ragas_dataset):
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION REPORT")
    print("=" * 60)
    thresholds = {
        "faithfulness": 0.80, "answer_relevancy": 0.75,
        "context_precision": 0.70, "context_recall": 0.65,
    }
    for metric, threshold in thresholds.items():
        score  = scores.get(metric, 0)
        status = "PASS" if score >= threshold else "NEEDS WORK"
        bar    = "█" * int(score * 20)
        print(f"\n  {metric}")
        print(f"  {score:.3f}  [{bar:<20}]  {status}")
    print("=" * 60)