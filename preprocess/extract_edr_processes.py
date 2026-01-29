import json
import os


EDR_PROCESSES = [
    "NisSrv.exe",
    "MsMpEng.exe",
    "SecurityHealthService.exe",
    "SecurityHealthSystray.exe"
]

# EDR_PROCESSES = [
#     "CSFalconService.exe",
#     "CsSystemTray_7.15.18514.0.exe",
#     "CSFalconContainer.exe"
# ]

# EDR_PROCESSES = [
#     "avp.exe",
#     "avpui.exe"
# ]


def extract_edr_processes(input_file, output_file, append=False):
    """
    Extract records related to EDR processes from a JSONL input file.
    """
    extracted_count = 0
    total_count = 0

    mode = 'a' if append else 'w'

    print(f"Start processing file: {input_file}")
    print(f"Write mode: {'append' if append else 'overwrite'}")
    print(f"Target EDR processes: {', '.join(EDR_PROCESSES)}")
    print("-" * 60)

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, mode, encoding='utf-8') as outfile:

        for line in infile:
            total_count += 1

            if total_count % 10000 == 0:
                print(f"Processed {total_count} records, extracted {extracted_count} records...")

            try:
                data = json.loads(line.strip())

                process_path = data.get('ProcessName', '')
                # Extract basename from path
                process_name = process_path.split('\\')[-1] if '\\' in process_path else process_path

                plugin = data.get('Plugin', '')
                if plugin == 'sysret':
                    continue

                if process_name in EDR_PROCESSES:
                    outfile.write(line)
                    extracted_count += 1

            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error at line {total_count}: {e}")
            except Exception as e:
                print(f"Warning: Error processing line {total_count}: {e}")

    print("-" * 60)
    print(f"Finished processing file: {input_file}")
    print(f"Total records processed: {total_count}")
    print(f"EDR records extracted: {extracted_count}\n")


if __name__ == "__main__":

    base_dir = "../data/baseline"
    output_path = "../data/baseline/faclon_edr_only.jsonl"

    jsonl_files = sorted(
        f for f in os.listdir(base_dir)
        if f.lower().endswith(".jsonl")
    )
    first_file = True

    for filename in jsonl_files:
        input_path = os.path.join(base_dir, filename)

        extract_edr_processes(
            input_file=input_path,
            output_file=output_path,
            append=not first_file
        )

        first_file = False

    print(f"\n[SUCCESS] EDR process data has been successfully extracted to: {output_path}")
