import os
import logging
import json
import yaml
import pandas as pd
from tqdm import tqdm
import time
from functools import lru_cache
from typing import List, Tuple, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO, Entrez

logging.basicConfig(filename="blastn_and_parse.log", level=logging.INFO)


def read_config(file_path: str) -> Tuple[str, str]:
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            email = file.readline().strip()
            api_key = file.readline().strip()
        return email, api_key
    else:
        raise Exception(
            "Config file not found. Please provide a valid file path."
        )


def blastn(sequence: str) -> Any:
    result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
    blast_record = NCBIXML.read(result_handle)
    return blast_record


@lru_cache(maxsize=None)
def get_taxa_info(tax_id: str) -> Dict[str, str]:
    try:
        handle = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml")
        records = Entrez.read(handle)
        tax_info = records[0]["LineageEx"]
        return {i["Rank"]: i["ScientificName"] for i in tax_info}
    except Exception as e:
        logging.error(f"Failed to get taxa info for {tax_id}: {e}")
        return {"Error": str(e)}


def parse_results(
    blast_record: Any, fasta: SeqIO.SeqRecord, hit_seqs: int
) -> List[Dict[str, Union[str, int, float, Dict[str, str]]]]:
    if not blast_record.alignments:
        return [
            {
                "sequence_id": fasta.id,
                "message": "No results are available for this sequence.",
            }
        ]

    result_list = []
    for alignment in blast_record.alignments[:hit_seqs]:
        for hsp in alignment.hsps:
            species = " ".join(alignment.title.split(" ")[1:3])
            tax_id = alignment.title.split("|")[3]
            taxa_info = get_taxa_info(tax_id)
            result_dict = {
                "sequence_id": fasta.id,
                "sequence_description": fasta.description,
                "alignment_length": alignment.length,
                "e_value": hsp.expect,
                "identity_percentage": hsp.identities / hsp.align_length,
                "species": species,
                "taxa_info": taxa_info,
            }
            result_list.append(result_dict)
    return result_list


def blastn_sequences(
    executor: ThreadPoolExecutor,
    fasta_sequences: List[SeqIO.SeqRecord],
    start_index: int,
    end_index: int,
) -> Dict[Future, SeqIO.SeqRecord]:
    step_futures = {
        executor.submit(blastn, str(fasta.seq)): fasta
        for fasta in fasta_sequences[start_index:end_index]
    }
    return step_futures


def parse_and_save_results(
    step_futures: Dict[Future, SeqIO.SeqRecord],
    results: List[Dict[str, Union[str, int, float, Dict[str, str]]]],
    hit_seqs: int,
) -> List[Dict[str, Union[str, int, float, Dict[str, str]]]]:
    for future in as_completed(step_futures):
        fasta = step_futures[future]
        try:
            blast_record = future.result()
            parsed_results = parse_results(blast_record, fasta, hit_seqs)
            if parsed_results:
                results.extend(parsed_results)
        except Exception as e:
            logging.error(
                f"Failed to get results for sequence {fasta.id}: {e}"
            )
    return results


def save_results(
    results: List[Dict[str, Any]], file_format: str = "json"
) -> str:

    file_name = f"blast_results_{time.time()}.{file_format}"
    if file_format == "json":
        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)
    elif file_format == "yaml":
        with open(file_name, "w") as f:
            yaml.dump(results, f)
    elif file_format in ["csv", "tsv"]:
        df = pd.DataFrame(results)
        if file_format == "csv":
            df.to_csv(file_name, index=False)
        elif file_format == "tsv":
            df.to_csv(file_name, sep="\t", index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return file_name


def blastn_and_parse(
    file_path: str, hit_seqs: int, file_format: str = "json"
) -> str:

    if os.path.exists(file_path):  # added file existence check
        with open(file_path) as f:
            fasta_sequences = list(SeqIO.parse(f, "fasta"))
        total_seqs = len(fasta_sequences)
        results: List[Dict[str, Union[str, int, float, Dict[str, str]]]] = []

        # Compute total steps for tqdm, updated part
        total_steps = total_seqs // 10
        if total_seqs % 10 != 0:
            total_steps += 1

        # Create tqdm object for tracking progress, updated part
        pbar = tqdm(total=total_steps, ncols=60)

        with ThreadPoolExecutor() as executor:
            for i in range(0, total_seqs, 10):  # Change batch size to 10
                step_futures = blastn_sequences(
                    executor, fasta_sequences, i, i + 10
                )
                results = parse_and_save_results(
                    step_futures, results, hit_seqs
                )

                # Increment progress bar and update description, updated part
                pbar.update(1)
                pbar.set_description(f"Progress: {pbar.n}/{total_steps}")

        # Save final results
        final_file_name = save_results(results, file_format=file_format)
        return final_file_name

    else:
        raise FileNotFoundError(f"File {file_path} not found.")
