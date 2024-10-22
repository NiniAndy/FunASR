def parse_file(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = lines[:-4]

    i = 0
    while i < len(lines):
        if lines[i].startswith("BAC"):
            file_id = lines[i].split('(')[0].strip()
            ref = lines[i + 1].strip().replace("ref:", "").strip()
            hyp = lines[i + 2].strip().replace("hyp:", "").strip()
            data[file_id] = {'ref': ref, 'hyp': hyp}
            i += 3  # Move to the next group
        else:
            i += 1  # Skip non-group lines
    return data


file_a_path = "/ssd/zhuang/code/FunASR/examples/aishell/hypformer/exp/baseline_hypformer_conformer_12e_6d_2048_256_zh_char_exp1/inference-model.pt.avg10/test_nar-err-ar-decoding/1best_recog/text.cer"
file_b_path = "/ssd/zhuang/code/FunASR/examples/aishell/hypformer/exp/baseline_hypformer_conformer_12e_6d_2048_256_zh_char_exp1/inference-model.pt.avg10/test_wenet_attn_re_search/1best_recog/text.cer"

# Parse both files
data_a = parse_file(file_a_path)
data_b = parse_file(file_b_path)

# Find differences
differences = []
for key in data_a.keys():
    if key in data_b:
        if data_a[key]['hyp'] != data_b[key]['hyp']:
            differences.append((key, data_a[key], data_b[key]))
    else:
        differences.append((key, data_a[key], "Not found in File B"))

# Display differences
for i in range(len(differences)):
    print (f"File ID: {differences[i][0]}")
    print (f"Ref                : {differences[i][1]['ref']}")
    print (f"nar-ar-decoding Hyp: {differences[i][1]['hyp']}")
    print (f"greedy-search Hyp  : {differences[i][2]['hyp']}")
    print ("-" * 50)


