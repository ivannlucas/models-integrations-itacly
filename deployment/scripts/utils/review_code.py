# scripts/review_code.py
# Purpose: Analyze the quality of a code file and return the analysis as text.

import argparse
import sys

from files import get_files_to_process, read_file_content, ALLOWED_EXTENSIONS, load_agent_instructions
from ia import setup_generative_model, process_file_with_ia, MAX_TOTAL_TOKENS
from ui import log_message, Colors, STATUS_COLORS

# --- AGENT INSTRUCTIONS ---
AGENT_INSTRUCTIONS = {
    "python": load_agent_instructions("review_code_python.txt"),
    "sql": load_agent_instructions("review_code_sql.txt"),
    "java": load_agent_instructions("review_code_java.txt")
}


def quality_report(all_results, max_tokens_reached) -> None:
    # --- FINAL REPORT ---
    log_message("CODE QUALITY REPORT", "HEADER")
    total_score = 0
    num_scores = 0

    for result in all_results:
        filepath = result.get('filepath', 'Unknown')
        analysis = result.get('analisis_detallado', {})
        score = result.get('puntuacion_calidad', 0)
        suggestions = result.get('sugerencias_mejora', [])

        print(f"\n\n{Colors.HEADER}File: {Colors.UNDERLINE}{filepath}{Colors.ENDC}")
        print(f"{Colors.BLUE}Quality Score: {score}/10{Colors.ENDC}")

        print(f"\n{Colors.CYAN}--- Detailed Analysis ---{Colors.ENDC}")
        for category, details in analysis.items():
            if isinstance(details, dict):
                status = details.get("estado", "N/A").upper()
                observation = details.get("observacion", "")
                color = STATUS_COLORS.get(status, STATUS_COLORS["DEFAULT"])

                print(
                    f"  {color}{Colors.BOLD}{category.replace('_', ' ').title()}:{Colors.ENDC} {color}[{status}]{Colors.ENDC} {observation}")
            else:
                print(f"  {Colors.BOLD}{category.replace('_', ' ').title()}:{Colors.ENDC} {details}")

        if suggestions:
            print(f"\n{Colors.CYAN}--- Improvement Suggestions ---{Colors.ENDC}")
            for i, sug in enumerate(suggestions, 1):
                print(f"  {i}. {sug.get('sugerencia', 'N/A')}")

        if score is not None:
            try:
                total_score += float(score)
                num_scores += 1
            except (ValueError, TypeError):
                log_message(f"Invalid score '{score}' for file {filepath}.", "WARNING")

    # --- SUMMARY AND EXIT STATUS ---
    log_message("FINAL SUMMARY", "HEADER")
    if num_scores > 0:
        final_score = total_score / num_scores
        score_message = f"Final average score: {final_score:.2f}/10."

        if final_score < 6.0:
            log_message(f"{score_message} LOW QUALITY.", "ERROR")
            sys.exit(1)
        elif max_tokens_reached:
            log_message(f"{score_message} ACCEPTABLE QUALITY, but not all files could be reviewed.", "ERROR")
            sys.exit(1)
        else:
            log_message(f"{score_message} ACCEPTABLE QUALITY.", "SUCCESS")
    else:
        log_message("No quality scores could be calculated.", "WARNING")


# --- MAIN FUNCTION ---
def main():
    parser = argparse.ArgumentParser(
        description="Analyzes code quality using a generative AI model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("path", nargs='?', default=None, type=str,
                        help="Path to the file/directory. If omitted, analyzes modified files in Git.")
    args = parser.parse_args()

    log_message("STARTING AI CODE REVIEWER", "HEADER")

    # Inicialización de modelo
    model = setup_generative_model()
    if not model:
        log_message("Failed to initialize model. Exiting.", "ERROR")
        sys.exit(1)

    # Obtención de archivos
    files_to_process = get_files_to_process(args.path)
    if not files_to_process:
        sys.exit(0)  # Salimos si no hay archivos

    log_message(f"{len(files_to_process)} file(s) will be analyzed.", "INFO")

    all_results = []
    total_tokens_used = 0
    consumption_total = 0
    max_tokens_reached = False
    file_number_processed = 1
    for filepath in files_to_process:
        if total_tokens_used >= MAX_TOTAL_TOKENS:
            log_message(f"Token limit reached ({MAX_TOTAL_TOKENS}). Stopping.", "WARNING")
            max_tokens_reached = True
            break

        log_message(
            f"Analyzing file: {Colors.UNDERLINE}{filepath}{Colors.ENDC}. File {file_number_processed} from {len(files_to_process)}",
            "INFO")
        file_number_processed += 1
        language = ALLOWED_EXTENSIONS.get(filepath.suffix)
        if not language:
            log_message(f"Unsupported file extension: {filepath.suffix}", "WARNING")
            break

        agent_instructions = AGENT_INSTRUCTIONS.get(language)
        if not agent_instructions:
            log_message(f"No agent instructions for language: {language}", "WARNING")
            break

        code_content = read_file_content(filepath)
        if not code_content:
            log_message(f"Skipping {filepath}: Could not read file content.", "WARNING")
            continue

        result, tokens_call, consumption_call = process_file_with_ia(model, agent_instructions, code_content,
                                                                     str(filepath))
        total_tokens_used += tokens_call
        consumption_total += consumption_call
        log_message(f"Tokens consumed: {tokens_call} | Total: {total_tokens_used} --> {round(consumption_total, 4)} $",
                    "INFO")

        all_results.append(result)

    if not all_results:
        log_message("Could not complete the analysis of any file.", "ERROR")
        sys.exit(1)
    quality_report(all_results, max_tokens_reached)


if __name__ == "__main__":
    main()
